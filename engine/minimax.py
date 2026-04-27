import abc
import argparse
import math
import time
from typing import List

import chess
from chess.polyglot import ZobristHasher, POLYGLOT_RANDOM_ARRAY

from chess_board_screen import ChessBoardScreen
from constants import LOGGER, PIECE_VALUES
from opening_book import OpeningBook
from order_moves_minimax import OrderMovesMinimax
from player import Player


class Minimax(Player):

    # Maximum additional plies the quiescence search can extend beyond `depth == 0`
    QS_MAX_DEPTH: int = 8

    # Maximum total check extensions allowed per branch (prevents explosion from series of checks)
    MAX_CHECK_EXTENSIONS: int = 3

    # Finite mate score used instead of raw +/-inf so the engine prefers faster mates
    # and delays unavoidable mate. Mate scores are adjusted by actual ply from root.
    MATE_SCORE: float = 1_000_000.0
    MATE_THRESHOLD: float = 990_000.0

    # Maximum number of entries in the transposition table before cleanup is triggered
    TT_MAX_SIZE: int = 1_000_000

    # Entries older than this many generations can be replaced regardless of depth
    TT_MAX_AGE: int = 2

    # Aspiration window half-width (in evaluation units, e.g. 0.50 = half a pawn)
    ASPIRATION_DELTA: float = 0.50

    # Reverse Futility Pruning: max depth to apply RFP, and margin per depth (in pawns).
    RFP_MAX_DEPTH: int = 3
    RFP_MARGIN_PER_DEPTH: float = 1.0

    # Forward Futility Pruning: max depth to apply futility, and margin per depth (in pawns).
    FUTILITY_MAX_DEPTH: int = 2
    FUTILITY_MARGIN_PER_DEPTH: float = 1.5

    # Late Move Pruning: at low depth, skip late quiet moves (after `4 + depth*depth` moves).
    LMP_MAX_DEPTH: int = 4

    # SEE Pruning in main search: at low depth, skip captures with negative SEE (losing trades).
    SEE_PRUNE_MAX_DEPTH: int = 3

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_minimax = OrderMovesMinimax(args, color)

        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black

        self.transposition_table = {}
        self._tt_generation: int = 0
        self.hasher = ZobristHasher(POLYGLOT_RANDOM_ARRAY)

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        """
        Chooses and plays a move for the current player using Iterative Deepening (ID).
        - In the opening phase, selects a move from the opening book randomly, weighted by book move weights.
        - After the opening phase, uses iterative deepening with minimax alpha-beta pruning:
            * Searches from depth 1 up to self.depth, reusing the transposition table (TT) across iterations.
            * Each iteration seeds the next with TT entries, enabling PV move prioritization for better ordering.
            * Early termination if checkmate is found.
        - Move ordering heuristics used:
            * TT move (PV move from previous iteration): always searched first.
            * Promotions are prioritized highest after TT move.
            * Captures are ordered by Most Valuable Victim - Least Valuable Aggressor (MVV-LVA).
            * Killer moves (moves that caused beta cutoffs in previous searches) are prioritized.
            * History heuristic: quiet moves that have historically caused cutoffs are boosted.
            * Checks are given a bonus.
        TT is preserved across moves (not cleared between turns) to retain useful cached positions.
        """
        if board.turn != self.color:
            raise ValueError(
                f'{type(self).__name__}.make_move called out of turn: '
                f'player={"WHITE" if self.color else "BLACK"}, '
                f'board.turn={"WHITE" if board.turn else "BLACK"}'
            )

        start_time: float = time.perf_counter()

        if self.opening_book.use_opening_book and self.opening_book.is_opening:
            if self.opening_book.make_move(board, start_time):
                return  # Move already made from an opening book

        # Clearing killer moves and history heuristic (TT is preserved across moves for ID)
        # Killer moves array sized to accommodate check extensions (extra slots for extended branches).
        killer_size = self.depth + self.MAX_CHECK_EXTENSIONS + 1
        self.order_moves_minimax.killer_moves = [[None, None] for _ in range(killer_size)]
        self.order_moves_minimax.history_heuristic_table = [[0] * 64 for _ in range(64)]

        # Increment TT generation for age-based replacement policy
        self._tt_generation += 1

        # Cleanup stale TT entries if table exceeds max size
        if len(self.transposition_table) > self.TT_MAX_SIZE:
            self.transposition_table = {
                k: v for k, v in self.transposition_table.items()
                if self._tt_generation - v['g'] < self.TT_MAX_AGE
            }

        internal_board: chess.Board = board.copy()

        legal_moves: List[chess.Move] = list(internal_board.legal_moves)
        if not legal_moves:
            return

        best_move: chess.Move | None = None
        best_value: float = -math.inf if self.color == chess.WHITE else math.inf
        is_maximizing: bool = self.color == chess.WHITE
        board_hash: int = self.hasher(internal_board)

        # Iterative Deepening: search from depth 1 to self.depth
        for current_depth in range(1, self.depth + 1):

            # Aspiration Windows: use a narrow window around the previous iteration's score
            if current_depth == 1:
                alpha = -math.inf
                beta = math.inf
            else:
                alpha = best_value - self.ASPIRATION_DELTA
                beta = best_value + self.ASPIRATION_DELTA

            # Get TT move from previous iteration for root position
            tt_entry = self.transposition_table.get(board_hash)
            tt_move = tt_entry.get('m') if tt_entry else None
            ordered_moves: List[chess.Move] = self.order_moves_minimax.order_moves(
                internal_board, legal_moves, ply=0, tt_move=tt_move)

            iteration_best_move, iteration_best_value = self.__search_root(
                internal_board, ordered_moves, current_depth, alpha, beta, is_maximizing)

            # Aspiration window fail-high or fail-low: re-search with full window
            if current_depth > 1 and (iteration_best_value <= alpha or iteration_best_value >= beta):
                alpha = -math.inf
                beta = math.inf
                tt_entry = self.transposition_table.get(board_hash)
                tt_move = tt_entry.get('m') if tt_entry else iteration_best_move
                ordered_moves = self.order_moves_minimax.order_moves(
                    internal_board, legal_moves, ply=0, tt_move=tt_move)
                iteration_best_move, iteration_best_value = self.__search_root(
                    internal_board, ordered_moves, current_depth, alpha, beta, is_maximizing)

            self.__store_root_tt_entry(board_hash, current_depth, iteration_best_value,
                                       iteration_best_move, alpha, beta)

            best_move = iteration_best_move
            best_value = iteration_best_value

            LOGGER.debug(
                f'{type(self).__name__}; ID iteration depth={current_depth}; '
                f'move: {best_move.uci() if best_move else "None"}; value: {best_value:.2f}'
            )

            # Early termination: checkmate found
            if self.__is_mate_score(best_value):
                break

        if best_move is not None:
            board.push(best_move)
        else:
            LOGGER.warning(f'{type(self).__name__}: No valid move found. Skipping push.')

        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        LOGGER.info(
            f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; depth: {self.depth}; time: {duration:.6f}s; move: {best_move.uci() if best_move else "None"}; ' +
            f'value: {best_value:.2f}'
        )

    def __search_root(self, board: chess.Board, ordered_moves: List[chess.Move],
                      depth: int, alpha: float, beta: float,
                      is_maximizing: bool) -> tuple:
        """
        Performs the root-level search loop for a single ID iteration.
        Returns (best_move, best_value) for this iteration.
        Extracted to allow re-use by aspiration window re-search.
        """
        iteration_best_move: chess.Move | None = None
        iteration_best_value: float = -math.inf if is_maximizing else math.inf

        for move in ordered_moves:
            board.push(move)
            # Check extension at root level
            extensions_left = self.MAX_CHECK_EXTENSIONS
            extension = 1 if extensions_left > 0 and board.is_check() else 0
            effective_extensions = extensions_left - extension
            board_value = self.__minimax_alphabeta(board, depth - 1 + extension, alpha, beta,
                                                   not is_maximizing, effective_extensions,
                                                   actual_ply=1)
            board.pop()

            if is_maximizing:
                if (board_value > iteration_best_value) or (
                        iteration_best_move is None and board_value == iteration_best_value):
                    iteration_best_value = board_value
                    iteration_best_move = move
                alpha = max(alpha, board_value)
                if beta <= alpha:
                    break
            else:
                if (board_value < iteration_best_value) or (
                        iteration_best_move is None and board_value == iteration_best_value):
                    iteration_best_value = board_value
                    iteration_best_move = move
                beta = min(beta, board_value)
                if beta <= alpha:
                    break

        return iteration_best_move, iteration_best_value

    def __store_root_tt_entry(self, board_hash: int, depth: int, value: float,
                              best_move: chess.Move | None, alpha: float, beta: float) -> None:
        """Stores the root search result in TT so the next ID iteration can prioritize its PV move."""
        if best_move is None:
            return

        flag: str = 'E'
        if value <= alpha:
            flag = 'U'
        elif value >= beta:
            flag = 'L'

        tt_entry = self.transposition_table.get(board_hash)
        if not tt_entry or depth >= tt_entry['d'] or self._tt_generation - tt_entry['g'] >= self.TT_MAX_AGE:
            self.transposition_table[board_hash] = {
                'v': self.__score_to_tt(value, actual_ply=0),
                'd': depth,
                'f': flag,
                'm': best_move,
                'g': self._tt_generation
            }

    def __is_mate_score(self, value: float) -> bool:
        """Returns True for finite mate-distance scores and raw infinities from legacy evaluations."""
        return abs(value) >= self.MATE_THRESHOLD or abs(value) == math.inf

    def __is_regular_bound(self, value: float) -> bool:
        """Returns True when a pruning bound is neither infinite nor in the mate-score range."""
        return abs(value) < math.inf and not self.__is_mate_score(value)

    def __mate_score(self, board: chess.Board, actual_ply: int) -> float:
        """
        Returns a finite checkmate score from White's perspective.
        If White is mated, the score is negative and less bad when mate is delayed.
        If Black is mated, the score is positive and better when mate is faster.
        """
        if board.turn == chess.WHITE:
            return -self.MATE_SCORE + actual_ply
        return self.MATE_SCORE - actual_ply

    def __normalize_evaluation_score(self, value: float, actual_ply: int) -> float:
        """Converts raw +/-inf evaluation values to finite mate-distance scores."""
        if value == math.inf:
            return self.MATE_SCORE - actual_ply
        if value == -math.inf:
            return -self.MATE_SCORE + actual_ply
        return value

    def __normalize_evaluator_score(self, value: float, actual_ply: int) -> float:
        """Converts evaluator scores to root-relative values, including finite leaf mate scores."""
        if value == math.inf:
            return self.MATE_SCORE - actual_ply
        if value == -math.inf:
            return -self.MATE_SCORE + actual_ply
        if value >= self.MATE_THRESHOLD:
            return max(self.MATE_THRESHOLD, value - actual_ply)
        if value <= -self.MATE_THRESHOLD:
            return min(-self.MATE_THRESHOLD, value + actual_ply)
        return value

    def __evaluate_board_score(self, board: chess.Board, actual_ply: int) -> float:
        """Evaluates a board and normalizes any raw mate infinities to mate-distance scores."""
        return self.__normalize_evaluator_score(self.evaluate_board(board), actual_ply)

    def __terminal_or_evaluation_score(self, board: chess.Board, actual_ply: int) -> float:
        """Returns a normalized terminal/evaluation score, using exact ply-aware mate scores."""
        if board.is_checkmate():
            return self.__mate_score(board, actual_ply)
        if (board.is_stalemate()
                or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()):
            return 0.0
        return self.__evaluate_board_score(board, actual_ply)

    def __score_to_tt(self, value: float, actual_ply: int) -> float:
        """Stores mate scores relative to the TT node so retrieval at a different ply remains correct."""
        value = self.__normalize_evaluation_score(value, actual_ply)
        if value >= self.MATE_THRESHOLD:
            return value + actual_ply
        if value <= -self.MATE_THRESHOLD:
            return value - actual_ply
        return value

    def __score_from_tt(self, value: float, actual_ply: int) -> float:
        """Converts TT mate scores back to the current root-relative ply."""
        value = self.__normalize_evaluation_score(value, actual_ply)
        if value >= self.MATE_THRESHOLD:
            return value - actual_ply
        if value <= -self.MATE_THRESHOLD:
            return value + actual_ply
        return value

    def __minimax_alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                            maximizing_player: bool, extensions_left: int,
                            actual_ply: int, can_null: bool = True) -> float:
        """
        Performs a recursive minimax search with alpha-beta pruning to evaluate the best achievable score
        from the current board position, assuming optimal play from both sides.

        How it works:
        - Uses alpha-beta pruning to eliminate branches that cannot affect the final decision, improving efficiency.
        - At each node, recursively explores legal moves, alternating between maximizing and minimizing player.
        - Uses a transposition table (TT) to cache and reuse previously computed positions:
            * Stores evaluation, depth, flag (Exact/Lower/Upper), and best move (PV move) per position.
            * On lookup, the TT best move is prioritized in move ordering for maximum cutoff efficiency.
            * TT entries from shallower Iterative Deepening iterations seed deeper searches,
              providing PV move ordering that significantly improves pruning.
        - Applies move ordering heuristics (TT move, promotions, captures, killer moves, history heuristic, checks)
          to search the most promising moves first, increasing pruning effectiveness.
        - Uses `actual_ply` (true distance from root, incremented per recursive call) to correctly
          index killer moves and history heuristic, even when check extensions keep `depth` constant
          across multiple plies.
        - Updates killer and history heuristics for quiet moves that cause cutoffs or improve bounds.
        - Returns the best evaluation found for the current player at this node.
        """
        # These automatic draws depend on halfmove clock / repetition history, which are not part
        # of the Polyglot hash used as the TT key. They must be handled before any TT lookup.
        if board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.0

        current_ply: int = actual_ply  # True ply from root (correct even with check extensions)
        original_alpha: float = alpha  # Store original alpha for TT flag and history update
        original_beta: float = beta  # Store original beta for history update

        board_hash: int = self.hasher(board)
        tt_entry = self.transposition_table.get(board_hash)

        # Transposition Table Lookup
        if tt_entry and tt_entry['d'] >= depth:
            tt_value = self.__score_from_tt(tt_entry['v'], actual_ply)
            if tt_entry['f'] == 'E':  # Flag: Exact
                return tt_value
            elif tt_entry['f'] == 'L':  # Flag: Lower bound
                alpha = max(alpha, tt_value)
            elif tt_entry['f'] == 'U':  # Flag: Upper bound
                beta = min(beta, tt_value)
            if beta <= alpha:
                return tt_value

        if board.is_game_over():
            return self.__terminal_or_evaluation_score(board, actual_ply)
        if depth == 0:
            return self.__quiescence_search(board, alpha, beta, maximizing_player, qs_depth=0,
                                            actual_ply=actual_ply)

        # Static evaluation — computed lazily, reused by RFP and forward futility pruning.
        static_eval: float | None = None

        # Reverse Futility Pruning (RFP / Static Null Move): if the static evaluation is so high
        # (for max) / low (for min) that even after losing a margin we'd still cause a cutoff,
        # we can prune without searching. Cheaper than NMP and very effective at low depth.
        # Disabled near mate scores (would otherwise mis-prune mate-related lines).
        if (depth <= self.RFP_MAX_DEPTH
                and not board.is_check()
                and not self.__is_zugzwang_risk(board)
                and self.__is_regular_bound(beta) and self.__is_regular_bound(alpha)):
            static_eval = self.__evaluate_board_score(board, actual_ply)
            rfp_margin = depth * self.RFP_MARGIN_PER_DEPTH
            if maximizing_player and static_eval - rfp_margin >= beta:
                return beta
            if not maximizing_player and static_eval + rfp_margin <= alpha:
                return alpha

        # Null Move Pruning: if giving the opponent a free move still results in a cutoff,
        # the position is so good that we can prune this branch without full search.
        # `can_null` prevents consecutive null moves (which would degenerate to evaluating
        # the position at depth - 2*(1+R) without any real play happening).
        if (can_null
                and depth >= 3
                and not board.is_check()
                and not self.__is_zugzwang_risk(board)):
            R: int = 2  # Reduction factor
            board.push(chess.Move.null())
            null_eval = self.__minimax_alphabeta(board, depth - 1 - R, alpha, beta,
                                                not maximizing_player, extensions_left,
                                                actual_ply=actual_ply + 1, can_null=False)
            board.pop()

            if maximizing_player and null_eval >= beta:
                return beta
            elif not maximizing_player and null_eval <= alpha:
                return alpha

        legal_moves: List[chess.Move] = list(board.legal_moves)
        tt_move = tt_entry.get('m') if tt_entry else None
        ordered_moves: List[chess.Move] = self.order_moves_minimax.order_moves(board, legal_moves, ply=current_ply,
                                                                               tt_move=tt_move)
        best_move: chess.Move | None = None

        if maximizing_player:
            max_evaluation = -math.inf
            searched_any = False
            for move_count, move in enumerate(ordered_moves):
                # Late Move Pruning: at low depth, skip late quiet non-checking moves.
                # Relies on good move ordering — assumes the best moves are tried first.
                if (depth <= self.LMP_MAX_DEPTH
                        and move_count >= 4 + depth * depth
                        and not board.is_check()
                        and not board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)):
                    continue

                # Forward Futility Pruning: at low depth, skip quiet non-checking moves
                # whose static evaluation + margin cannot improve alpha.
                if (depth <= self.FUTILITY_MAX_DEPTH
                        and not board.is_check()
                        and not board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)
                        and self.__is_regular_bound(alpha)):
                    if static_eval is None:
                        static_eval = self.__evaluate_board_score(board, actual_ply)
                    if static_eval + depth * self.FUTILITY_MARGIN_PER_DEPTH <= alpha:
                        continue

                # SEE Pruning: at low depth, skip captures that lose material on the swap-off.
                # Disabled for promotions (SEE underestimates promotion gain) and checks.
                if (depth <= self.SEE_PRUNE_MAX_DEPTH
                        and not board.is_check()
                        and board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)
                        and self.__static_exchange_evaluation(board, move) < 0):
                    continue

                searched_any = True
                board.push(move)
                # Check extension: extend search by 1 ply when the move gives check
                extension = 1 if extensions_left > 0 and board.is_check() else 0
                evaluation = self.__minimax_alphabeta(board, depth - 1 + extension, alpha, beta, False,
                                                     extensions_left - extension,
                                                     actual_ply=actual_ply + 1)
                board.pop()

                if (evaluation > max_evaluation) or (best_move is None and evaluation == max_evaluation):
                    max_evaluation = evaluation
                    best_move = move

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.order_moves_minimax.store_killer_move(current_ply, move)
                        self.order_moves_minimax.update_history_score(move, depth)
                    break

            # Selective pruning must never leave a non-terminal node with the sentinel value (-inf).
            # If LMP/futility/SEE skipped every legal move, force-search the best ordered move and
            # return it without storing a TT entry for this node (its bound semantics would be unsafe).
            if not searched_any:
                fallback_move = ordered_moves[0] if ordered_moves else None
                if fallback_move is None:
                    return self.__terminal_or_evaluation_score(board, actual_ply)
                board.push(fallback_move)
                extension = 1 if extensions_left > 0 and board.is_check() else 0
                fallback_evaluation = self.__minimax_alphabeta(
                    board, depth - 1 + extension, alpha, beta, False,
                    extensions_left - extension, actual_ply=actual_ply + 1)
                board.pop()
                return fallback_evaluation

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually raised alpha (if it was quiet)
                # Check if max_evaluation actually improved alpha from the original value
                if max_evaluation > original_alpha:
                    self.order_moves_minimax.update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if max_evaluation <= original_alpha:
                flag = 'U'  # Upper bound
            elif max_evaluation >= beta:
                flag = 'L'  # Lower bound
            # Store in Transposition Table. Re-fetch the current entry because recursive calls may have
            # written a fresher/deeper entry for the same position via transposition.
            current_tt_entry = self.transposition_table.get(board_hash)
            if (not current_tt_entry
                    or depth >= current_tt_entry['d']
                    or self._tt_generation - current_tt_entry['g'] >= self.TT_MAX_AGE):
                self.transposition_table[board_hash] = {'v': self.__score_to_tt(max_evaluation, actual_ply), 'd': depth, 'f': flag, 'm': best_move, 'g': self._tt_generation}

            return max_evaluation

        else:
            min_eval = math.inf
            searched_any = False
            for move_count, move in enumerate(ordered_moves):
                # Late Move Pruning: at low depth, skip late quiet non-checking moves.
                if (depth <= self.LMP_MAX_DEPTH
                        and move_count >= 4 + depth * depth
                        and not board.is_check()
                        and not board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)):
                    continue

                # Forward Futility Pruning: at low depth, skip quiet non-checking moves
                # whose static evaluation - margin cannot improve beta (lower it).
                if (depth <= self.FUTILITY_MAX_DEPTH
                        and not board.is_check()
                        and not board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)
                        and self.__is_regular_bound(beta)):
                    if static_eval is None:
                        static_eval = self.__evaluate_board_score(board, actual_ply)
                    if static_eval - depth * self.FUTILITY_MARGIN_PER_DEPTH >= beta:
                        continue

                # SEE Pruning: at low depth, skip captures that lose material on the swap-off.
                # Disabled for promotions (SEE underestimates promotion gain) and checks.
                if (depth <= self.SEE_PRUNE_MAX_DEPTH
                        and not board.is_check()
                        and board.is_capture(move) and move.promotion is None
                        and not board.gives_check(move)
                        and self.__static_exchange_evaluation(board, move) < 0):
                    continue

                searched_any = True
                board.push(move)
                # Check extension: extend search by 1 ply when the move gives check
                extension = 1 if extensions_left > 0 and board.is_check() else 0
                evaluation = self.__minimax_alphabeta(board, depth - 1 + extension, alpha, beta, True,
                                                     extensions_left - extension,
                                                     actual_ply=actual_ply + 1)
                board.pop()

                if (evaluation < min_eval) or (best_move is None and evaluation == min_eval):
                    min_eval = evaluation
                    best_move = move

                beta = min(beta, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.order_moves_minimax.store_killer_move(current_ply, move)
                        self.order_moves_minimax.update_history_score(move, depth)
                    break

            # Selective pruning must never leave a non-terminal node with the sentinel value (+inf).
            # If LMP/futility/SEE skipped every legal move, force-search the best ordered move and
            # return it without storing a TT entry for this node (its bound semantics would be unsafe).
            if not searched_any:
                fallback_move = ordered_moves[0] if ordered_moves else None
                if fallback_move is None:
                    return self.__terminal_or_evaluation_score(board, actual_ply)
                board.push(fallback_move)
                extension = 1 if extensions_left > 0 and board.is_check() else 0
                fallback_evaluation = self.__minimax_alphabeta(
                    board, depth - 1 + extension, alpha, beta, True,
                    extensions_left - extension, actual_ply=actual_ply + 1)
                board.pop()
                return fallback_evaluation

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually lowered beta (if it was quiet)
                # Check if min_eval actually improved beta from the original value
                if min_eval < original_beta:
                    self.order_moves_minimax.update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if min_eval >= original_beta:
                flag = 'L'  # Lower bound
            elif min_eval <= alpha:
                flag = 'U'  # Upper bound
            # Store in Transposition Table. Re-fetch the current entry because recursive calls may have
            # written a fresher/deeper entry for the same position via transposition.
            current_tt_entry = self.transposition_table.get(board_hash)
            if (not current_tt_entry
                    or depth >= current_tt_entry['d']
                    or self._tt_generation - current_tt_entry['g'] >= self.TT_MAX_AGE):
                self.transposition_table[board_hash] = {'v': self.__score_to_tt(min_eval, actual_ply), 'd': depth, 'f': flag, 'm': best_move, 'g': self._tt_generation}

            return min_eval

    def __is_zugzwang_risk(self, board: chess.Board) -> bool:
        """
        Simple heuristic to detect positions where null move pruning is unsafe.
        Returns True if the side to move has only king and pawns (no major/minor pieces),
        which makes zugzwang likely — passing the turn would genuinely be harmful.
        """
        side: chess.Color = board.turn
        return not bool(
            board.pieces(chess.KNIGHT, side) | board.pieces(chess.BISHOP, side) |
            board.pieces(chess.ROOK, side) | board.pieces(chess.QUEEN, side)
        )

    def __quiescence_search(self, board: chess.Board, alpha: float, beta: float,
                            maximizing_player: bool, qs_depth: int, actual_ply: int) -> float:
        """
        Quiescence Search (QS): after the main search reaches depth == 0, continues exploring
        only "noisy" moves (captures and promotions) until the position becomes quiet.
        This eliminates the horizon effect — the engine no longer evaluates a position in the
        middle of an active capture sequence, where a static evaluation would be misleading.

        Key elements:
        - Stand-pat: the static evaluation acts as a lower (max player) / upper (min player) bound,
          since the side to move is not forced to make a capture. If the stand-pat already causes
          a beta cutoff (or alpha cutoff for the minimizer), we return immediately.
          Stand-pat is **not** applied when the side to move is in check — passing is illegal.
        - Check evasion: when in check, ALL legal moves are searched (not just captures), since
          a quiet evasion (e.g. king move) may be the only legal reply. If no legal moves exist,
          the position is checkmate and we return ±inf.
        - Depth limit (`QS_MAX_DEPTH`): caps the recursion to avoid pathological cases with very
          long capture sequences.
        - SEE filtering: captures with negative Static Exchange Evaluation (i.e. "bad captures"
          that lose material on the swap-off) are skipped (only when not in check and not giving check).
          Promotions and capture-checks are always searched.
        - Captures are ordered by MVV-LVA so the most promising are tried first.
        """
        in_check: bool = board.is_check()

        if board.is_game_over():
            return self.__terminal_or_evaluation_score(board, actual_ply)

        # Stand-pat is illegal when in check (cannot pass). Skip the stand-pat bound update.
        if not in_check:
            stand_pat: float = self.__evaluate_board_score(board, actual_ply)

            if maximizing_player:
                if stand_pat >= beta:
                    return beta
                if stand_pat > alpha:
                    alpha = stand_pat
            else:
                if stand_pat <= alpha:
                    return alpha
                if stand_pat < beta:
                    beta = stand_pat

            # Depth cap for QS — prevents runaway recursion in very tactical positions.
            if qs_depth >= self.QS_MAX_DEPTH:
                return stand_pat
        else:
            # In check: cannot stand pat. Hard depth cap returns static eval as fallback.
            if qs_depth >= self.QS_MAX_DEPTH:
                return self.__terminal_or_evaluation_score(board, actual_ply)

        # Move generation:
        # - In check: search ALL legal moves (check evasions, including quiet replies).
        # - Otherwise: only captures and promotions (noisy moves), with SEE pruning.
        candidate_moves: List[chess.Move] = []
        if in_check:
            candidate_moves = list(board.legal_moves)
            if not candidate_moves:
                # Checkmate: side to move is mated.
                return self.__mate_score(board, actual_ply)
        else:
            for move in board.legal_moves:
                is_promotion: bool = move.promotion is not None
                is_capture: bool = board.is_capture(move)
                if not (is_capture or is_promotion):
                    continue
                # SEE filtering: skip clearly losing captures, but keep promotions and capture-checks.
                if is_capture and not is_promotion and not board.gives_check(move):
                    if self.__static_exchange_evaluation(board, move) < 0:
                        continue
                candidate_moves.append(move)

        # Order moves by MVV-LVA (with a small bonus for promotions) for better pruning.
        # For check evasions this still works: quiet evasions get score 0 and sort last.
        candidate_moves.sort(key=lambda m: self.__mvv_lva_score(board, m), reverse=True)

        for move in candidate_moves:
            board.push(move)
            evaluation = self.__quiescence_search(board, alpha, beta, not maximizing_player, qs_depth + 1,
                                                  actual_ply + 1)
            board.pop()

            if maximizing_player:
                if evaluation > alpha:
                    alpha = evaluation
                if alpha >= beta:
                    return beta
            else:
                if evaluation < beta:
                    beta = evaluation
                if beta <= alpha:
                    return alpha

        return alpha if maximizing_player else beta

    def __mvv_lva_score(self, board: chess.Board, move: chess.Move) -> float:
        """Scores a noisy move (capture/promotion) by MVV-LVA for quiescence search ordering."""
        score: float = 0.0
        if move.promotion is not None:
            score += 10_000 + PIECE_VALUES.get(move.promotion, 0)
        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES[victim.piece_type] if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
        score += victim_value - attacker_value / 10.0
        return score

    def __static_exchange_evaluation(self, board: chess.Board, move: chess.Move) -> float:
        """
        Static Exchange Evaluation (SEE): estimates the material gain/loss of a capture
        sequence on the destination square, assuming both sides recapture optimally with
        the least valuable attacker (LVA) at each step.

        Used by quiescence search to prune "bad captures" — captures where the swap-off
        loses material (SEE < 0). Such captures rarely improve the position, so skipping
        them dramatically reduces the QS branching factor.

        Algorithm:
        1. Build a "gains" array: gains[i] is the material gained at step i of the swap-off.
           Step 0 = the value of the originally captured piece.
           Step i (i > 0) = value of the piece captured at step i, minus gains[i-1].
        2. After collecting all swap steps, fold the array back via negamax:
           gains[i-1] = -max(-gains[i-1], gains[i]) — at each level, the side to move chooses
           whether to continue the exchange or stand pat (not recapture).
        3. The final gains[0] is the SEE value from the perspective of the side that moved first.

        Notes:
        - Promotions during the swap are simplified to queen promotions for the gain calculation.
        - Pinned attackers are filtered out via `board.is_legal()` at each step.
        - Uses push/pop on the board to simulate the swap without allocating a board copy.
        """
        to_sq: int = move.to_square

        # Initial victim value (the piece originally captured by `move`).
        if board.is_en_passant(move):
            initial_victim_value: float = PIECE_VALUES[chess.PAWN]
        else:
            captured = board.piece_at(to_sq)
            if captured is None:
                return 0.0
            initial_victim_value = PIECE_VALUES[captured.piece_type]

        # Simulate the swap-off using push/pop on the board directly.
        board.push(move)
        gains: List[float] = [initial_victim_value]
        moves_pushed: int = 1  # Track how many moves to pop at the end

        while True:
            side_to_move: chess.Color = board.turn
            attackers = board.attackers(side_to_move, to_sq)
            if not attackers:
                break

            # Find the least valuable attacker that can legally recapture (handles pins).
            sorted_attackers = sorted(
                attackers, key=lambda sq: PIECE_VALUES[board.piece_at(sq).piece_type])
            chosen_capture: chess.Move | None = None
            for from_sq in sorted_attackers:
                attacker_piece = board.piece_at(from_sq)
                promo = None
                if (attacker_piece.piece_type == chess.PAWN
                        and chess.square_rank(to_sq) in (0, 7)):
                    promo = chess.QUEEN
                candidate = chess.Move(from_sq, to_sq, promotion=promo)
                if board.is_legal(candidate):
                    chosen_capture = candidate
                    break
            if chosen_capture is None:
                break

            # The piece being captured next is whatever currently sits on `to_sq`.
            victim = board.piece_at(to_sq)
            if victim is None:
                break
            gains.append(PIECE_VALUES[victim.piece_type] - gains[-1])
            board.push(chosen_capture)
            moves_pushed += 1

        # Undo all pushed moves to restore the original board state.
        for _ in range(moves_pushed):
            board.pop()

        # Negamax fold of the swap list.
        for i in range(len(gains) - 1, 0, -1):
            gains[i - 1] = -max(-gains[i - 1], gains[i])
        return gains[0]

