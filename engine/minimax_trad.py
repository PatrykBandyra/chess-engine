import argparse
import bisect
import math
import random
import time
from typing import List

import chess
from chess.polyglot import ZobristHasher, POLYGLOT_RANDOM_ARRAY, open_reader, Entry

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from player import Player


class MinimaxTrad(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args)
        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black
        self.color: chess.Color = color

        self.transposition_table = {}
        self.hasher = ZobristHasher(POLYGLOT_RANDOM_ARRAY)

        self.use_opening_book: bool = args.opening_book
        self.is_opening: bool = True
        self.opening_book = open_reader('codekiddy.bin')

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3.05,
            chess.BISHOP: 3.33,
            chess.ROOK: 5.63,
            chess.QUEEN: 9.5,
            chess.KING: 100_000
        }

        # Stores 2 moves per ply that caused beta cutoffs - indexed by ply (0 = root, 1 = depth-1, etc.)
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]

        # Stores scores for non-capture moves based on success - indexed by [from_square][to_square]
        self.history_heuristic_table = [[0] * 64 for _ in range(64)]

        # Approximate values for move ordering heuristics
        self.move_ordering_promotion_bonus = 1_500_000
        self.move_ordering_capture_bonus = 1_000_000
        self.move_ordering_killer_bonus = 750_000
        self.move_ordering_check_bonus = 500_000

    PIECE_SQUARE_TABLE_MID = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
    }
    PIECE_SQUARE_TABLE_END = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10, 0, 0, -10, -20, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50
        ]
    }

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        """
        Chooses and plays a move for the current player.
        - In the opening phase, selects a move from the opening book randomly, weighted by book move weights.
        - After the opening phase, uses minimax with alpha-beta pruning to select the best move.
        - Move ordering heuristics used:
            * Promotions are prioritized highest.
            * Captures are ordered by Most Valuable Victim - Least Valuable Aggressor (MVV-LVA).
            * Killer moves (moves that caused beta cutoffs in previous searches) are prioritized.
            * History heuristic: quiet moves that have historically caused cutoffs are boosted.
            * Checks are given a bonus.
        This improves search efficiency and playing strength by exploring promising moves first.
        """
        start_time: float = time.perf_counter()

        if self.use_opening_book and self.is_opening:
            entries: List[Entry] = list(self.opening_book.find_all(board))
            if entries:
                weights = [entry.weight for entry in entries]
                weights_sum = sum(weights)
                if weights_sum > 0:
                    cum_weights = []
                    cum_sum = 0
                    for weight in weights:
                        cum_sum += weight
                        cum_weights.append(cum_sum)
                    r = random.uniform(0, weights_sum)
                    idx = bisect.bisect_left(cum_weights, r)
                    entry = entries[idx]
                    opening_book_best_move = entry.move
                else:
                    entry = random.choice(entries)
                    opening_book_best_move = entry.move
                board.push(opening_book_best_move)

                end_time: float = time.perf_counter()
                duration: float = end_time - start_time

                max_weight = max(weights)
                max_weight_move = next((e for e in entries if e.weight == max_weight), None).move.uci()
                LOGGER.info(
                    f'MINIMAX-TRAD - OPENING BOOK; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; ' +
                    f'move: {opening_book_best_move.uci()}; weight: {entry.weight}; ' +
                    f'max weight move: {max_weight_move}; max weight: {max_weight}')
                return
            else:
                self.is_opening = False
                LOGGER.info(f'MINIMAX-TRAD - OPENING BOOK; {"WHITE" if self.color else "BLACK"}; opening phase ended')

        # Clearing
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]
        self.history_heuristic_table = [[0] * 64 for _ in range(64)]

        internal_board: chess.Board = board.copy()

        legal_moves: List[chess.Move] = list(internal_board.legal_moves)
        if not legal_moves:
            return
        ordered_moves: List[chess.Move] = self.__order_moves(internal_board, legal_moves, ply=0)

        best_move: chess.Move | None = None
        best_value: float = -math.inf if self.color == chess.WHITE else math.inf
        is_maximizing: bool = self.color == chess.WHITE
        alpha: float = -math.inf
        beta: float = math.inf

        for move in ordered_moves:
            internal_board.push(move)
            board_value = self.__minimax_alphabeta(internal_board, self.depth - 1, alpha, beta, not is_maximizing)
            internal_board.pop()

            if is_maximizing:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
                alpha = max(alpha, board_value)
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
                beta = min(beta, board_value)

        board.push(best_move)

        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        LOGGER.info(
            f'MINIMAX-TRAD; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_move.uci()}; ' +
            f'value: {best_value}'
        )

    def __order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int) -> List[chess.Move]:
        """
        Orders a list of legal moves to improve alpha-beta pruning efficiency.

        How it works:
        - Assigns a score to each move based on several heuristics:
            * Promotions: Highest priority, large bonus.
            * Captures: Scored by Most Valuable Victim - Least Valuable Aggressor (MVV-LVA).
            * Killer moves: Moves that caused beta cutoffs at this ply in previous searches are prioritized.
            * History heuristic: Quiet moves that have historically caused cutoffs are boosted.
            * Checks: Moves that give check are given a bonus.
        - Moves are sorted in descending order of their score, so the most promising moves are searched first.
        - This ordering increases the likelihood of alpha-beta cutoffs, making the search more efficient and improving engine strength.

        Args:
            board: The current board state.
            moves: List of legal moves to order.
            ply: The current ply (search depth from root).
        Returns:
            List of moves sorted from best to worst according to the heuristics.
        """
        move_scores = []
        killers = self.killer_moves[ply] if 0 <= ply < len(self.killer_moves) else [None, None]

        for move in moves:
            score: float = 0
            is_promotion: bool = move.promotion is not None
            is_capture: bool = board.is_capture(move)

            # 1. Promotions
            if is_promotion:
                score += (self.move_ordering_promotion_bonus +
                          self.__get_piece_value(chess.Piece(move.promotion, board.turn)))

            # 2. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor)
            elif is_capture:  # If both promotion and capture, then prefer promotion score
                move_piece: chess.Piece = board.piece_at(move.from_square)
                captured_piece: chess.Piece | None = board.piece_at(move.to_square)
                captured_piece_value: float = 0
                if captured_piece:
                    captured_piece_value = self.__get_piece_value(captured_piece)
                elif board.is_en_passant(move):
                    captured_piece_value = self.piece_values[chess.PAWN]

                aggressor_value: float = self.__get_piece_value(move_piece)
                score += self.move_ordering_capture_bonus + (captured_piece_value - aggressor_value / 10)

            # 3. Quiet Moves (apply Killer and History Heuristics)
            else:
                is_killer: bool = move == killers[0] or move == killers[1]
                if is_killer:
                    score += self.move_ordering_killer_bonus

                history_score = self.history_heuristic_table[move.from_square][move.to_square]
                score += history_score

                # 4. Checks (check if the move puts the opponent in check)
                board.push(move)
                is_check: bool = board.is_check()
                board.pop()

                if is_check:
                    # Add check bonus, potentially less if already highly scored
                    # Check bonus might be less relevant now with history heuristic also boosting good quiet checks
                    if score < self.move_ordering_check_bonus * 1.5:  # Avoid excessive bonus stacking
                        score += self.move_ordering_check_bonus

            move_scores.append((score, move))

        # Sort moves in descending order of score
        move_scores.sort(key=lambda item: item[0], reverse=True)

        # Return just the moves in the sorted order
        return [move for _, move in move_scores]

    def __minimax_alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                            maximizing_player: bool) -> float:
        current_ply: int = self.depth - depth
        original_alpha: float = alpha  # Store original alpha for TT flag and history update
        original_beta: float = beta  # Store original beta for history update

        board_hash: int = self.hasher.hash_board(board)
        tt_entry = self.transposition_table.get(board_hash)

        # Transposition Table Lookup
        if tt_entry and tt_entry['d'] >= depth:
            if tt_entry['f'] == 'E':  # Flag: Exact
                return tt_entry['v']
            elif tt_entry['f'] == 'L':  # Flag: Lower bound
                alpha = max(alpha, tt_entry['v'])
            elif tt_entry['f'] == 'U':  # Flag: Upper bound
                beta = min(beta, tt_entry['v'])
            if beta <= alpha:
                return tt_entry['v']

        if depth == 0 or board.is_game_over():
            return self.__evaluate_board(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)
        ordered_moves: List[chess.Move] = self.__order_moves(board, legal_moves, ply=current_ply)
        best_move: chess.Move | None = None

        if maximizing_player:
            max_evaluation = -math.inf
            for move in ordered_moves:
                board.push(move)
                evaluation = self.__minimax_alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                if evaluation > max_evaluation:
                    max_evaluation = evaluation
                    best_move = move

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.__store_killer_move(current_ply, move)
                        self.__update_history_score(move, depth)
                    break

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually raised alpha (if it was quiet)
                # Check if max_evaluation actually improved alpha from the original value
                if max_evaluation > original_alpha:
                    self.__update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if max_evaluation <= original_alpha:
                flag = 'U'  # Upper bound
            elif max_evaluation >= beta:
                flag = 'L'  # Lower bound
            # Store in Transposition Table
            if not tt_entry or depth >= tt_entry['d']:
                self.transposition_table[board_hash] = {'v': max_evaluation, 'd': depth, 'f': flag}

            return max_evaluation

        else:
            min_eval = math.inf
            for move in ordered_moves:
                board.push(move)
                evaluation = self.__minimax_alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move

                beta = min(beta, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.__store_killer_move(current_ply, move)
                        self.__update_history_score(move, depth)
                    break

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually lowered beta (if it was quiet)
                # Check if min_eval actually improved beta from the original value
                if min_eval < original_beta:
                    self.__update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if min_eval >= beta:
                flag = 'L'  # Lower bound
            elif min_eval <= alpha:
                flag = 'U'  # Upper bound
            # Store in Transposition Table
            if not tt_entry or depth >= tt_entry['d']:
                self.transposition_table[board_hash] = {'v': min_eval, 'd': depth, 'f': flag}

            return min_eval

    def __get_piece_value(self, piece: chess.Piece | None) -> float:
        """Safely gets the value of a piece, returning 0 if None"""
        return self.piece_values.get(piece.piece_type, 0) if piece else 0

    def __store_killer_move(self, ply: int, move: chess.Move):
        """Stores a killer move for a given ply, keeping the two best"""
        if 0 <= ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move

    def __update_history_score(self, move: chess.Move, depth: int):
        """Increases the history score for a successful quiet move"""
        if move.promotion is None:  # Captures are implicitly excluded by where this is called
            bonus = depth * depth  # Weight bonus by depth squared
            self.history_heuristic_table[move.from_square][move.to_square] += bonus

    def __get_game_phase(self, board: chess.Board) -> float:
        """
        Returns a phase value in [0, 1]: 1.0 = middlegame, 0.0 = endgame, interpolated by non-pawn material.
        """
        non_pawn_material = 0.0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            non_pawn_material += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            non_pawn_material += len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        endgame_threshold = 20.0
        middlegame_threshold = 67.0
        if non_pawn_material >= middlegame_threshold:
            return 1.0
        elif non_pawn_material <= endgame_threshold:
            return 0.0
        else:
            return (non_pawn_material - endgame_threshold) / (middlegame_threshold - endgame_threshold)

    def __evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates the board using blended middlegame/endgame piece-square tables and material.
        """
        if board.is_checkmate():
            return -math.inf if board.turn == chess.WHITE else math.inf
        if board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.0
        phase = self.__get_game_phase(board)
        white_score = 0.0
        black_score = 0.0
        for piece_type in self.piece_values.keys():
            for color in [chess.WHITE, chess.BLACK]:
                squares = board.pieces(piece_type, color)
                for sq in squares:
                    pst_mid = self.PIECE_SQUARE_TABLE_MID[piece_type][
                        sq if color == chess.WHITE else chess.square_mirror(sq)]
                    pst_end = self.PIECE_SQUARE_TABLE_END[piece_type][
                        sq if color == chess.WHITE else chess.square_mirror(sq)]
                    value = self.piece_values[piece_type] + 0.01 * (phase * pst_mid + (1 - phase) * pst_end)
                    if color == chess.WHITE:
                        white_score += value
                    else:
                        black_score += value
        return white_score - black_score

    def __del__(self):
        if hasattr(self, 'opening_book') and self.opening_book is not None:
            self.opening_book.close()
