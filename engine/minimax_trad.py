import argparse
import math
import time

import chess

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from player import Player


class MinimaxTrad(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args)
        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black
        self.color: chess.Color = color

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3.05,
            chess.BISHOP: 3.33,
            chess.ROOK: 5.63,
            chess.QUEEN: 9.5,
            chess.KING: 100_000
        }

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        start_time = time.perf_counter()

        internal_board: chess.Board = board.copy()

        legal_moves = list(internal_board.legal_moves)
        if not legal_moves:
            return
        # Optional: Implement move ordering here to improve pruning efficiency
        # (e.g., captures first, then checks, then quiet moves)

        best_move: chess.Move | None = None
        best_value: float = -math.inf if self.color == chess.WHITE else math.inf
        is_maximizing: bool = self.color == chess.WHITE

        for move in legal_moves:
            internal_board.push(move)
            board_value = self.minimax_alphabeta(internal_board, self.depth - 1, -math.inf, math.inf, not is_maximizing)
            internal_board.pop()

            if is_maximizing:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move

        board.push(best_move)

        end_time = time.perf_counter()
        duration = end_time - start_time
        LOGGER.info(
            f'MINIMAX-TRAD; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_move.uci()}; ' +
            f'value: {best_value}'
        )

    def minimax_alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                          maximizing_player: bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)
        # Optional: Implement move ordering here to improve pruning efficiency
        # (e.g., captures first, then checks, then quiet moves)

        if maximizing_player:
            max_evaluation = -math.inf
            for move in legal_moves:
                board.push(move)
                evaluation = self.minimax_alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_evaluation = max(max_evaluation, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_evaluation
        else:
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                evaluation = self.minimax_alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -math.inf
            else:
                return math.inf
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0

        white_material = 0.0
        black_material = 0.0

        for piece_type in self.piece_values.keys():
            white_material += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]

        return white_material - black_material
