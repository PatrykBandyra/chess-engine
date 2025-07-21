import abc
import argparse
import math
import time
from typing import List

import chess
from chess.polyglot import ZobristHasher, POLYGLOT_RANDOM_ARRAY

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from opening_book import OpeningBook
from order_moves_minimax import OrderMovesMinimax
from player import Player


class Minimax(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_minimax = OrderMovesMinimax(args, color)

        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black

        self.transposition_table = {}
        self.hasher = ZobristHasher(POLYGLOT_RANDOM_ARRAY)

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

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

        if self.opening_book.use_opening_book and self.opening_book.is_opening:
            if self.opening_book.make_move(board, start_time):
                return  # Move already made from an opening book

        # Clearing
        self.transposition_table = {}
        self.order_moves_minimax.killer_moves = [[None, None] for _ in range(self.depth + 1)]
        self.order_moves_minimax.history_heuristic_table = [[0] * 64 for _ in range(64)]

        internal_board: chess.Board = board.copy()

        legal_moves: List[chess.Move] = list(internal_board.legal_moves)
        if not legal_moves:
            return
        ordered_moves: List[chess.Move] = self.order_moves_minimax.order_moves(internal_board, legal_moves, ply=0)

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
                if (board_value > best_value) or (best_move is None and board_value == best_value):
                    best_value = board_value
                    best_move = move
                alpha = max(alpha, board_value)
            else:
                if (board_value < best_value) or (best_move is None and board_value == best_value):
                    best_value = board_value
                    best_move = move
                beta = min(beta, board_value)

        if best_move is not None:
            board.push(best_move)
        else:
            LOGGER.warning('MINIMAX-TRAD: No valid move found. Skipping push.')

        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        LOGGER.info(
            f'MINIMAX-TRAD; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_move.uci() if best_move else "None"}; ' +
            f'value: {best_value:.2f}'
        )

    def __minimax_alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                            maximizing_player: bool) -> float:
        """
        Performs a recursive minimax search with alpha-beta pruning to evaluate the best achievable score
        from the current board position, assuming optimal play from both sides.

        How it works:
        - Uses alpha-beta pruning to eliminate branches that cannot affect the final decision, improving efficiency.
        - At each node, recursively explores legal moves, alternating between maximizing and minimizing player.
        - Uses a transposition table to cache and reuse previously computed positions, further speeding up the search.
        - Applies move ordering heuristics (promotions, captures, killer moves, history heuristic, checks) to search the most promising moves first, increasing pruning effectiveness.
        - Updates killer and history heuristics for quiet moves that cause cutoffs or improve bounds.
        - Returns the best evaluation found for the current player at this node.
        """
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
            return self.evaluate_board(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)
        ordered_moves: List[chess.Move] = self.order_moves_minimax.order_moves(board, legal_moves, ply=current_ply)
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
                        self.order_moves_minimax.store_killer_move(current_ply, move)
                        self.order_moves_minimax.update_history_score(move, depth)
                    break

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
                        self.order_moves_minimax.store_killer_move(current_ply, move)
                        self.order_moves_minimax.update_history_score(move, depth)
                    break

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually lowered beta (if it was quiet)
                # Check if min_eval actually improved beta from the original value
                if min_eval < original_beta:
                    self.order_moves_minimax.update_history_score(best_move, depth)

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
