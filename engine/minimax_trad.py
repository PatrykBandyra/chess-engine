import argparse

import chess

from board_evaluator_trad import BoardEvaluatorTrad
from minimax import Minimax


class MinimaxTrad(Minimax):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.board_evaluator = BoardEvaluatorTrad(args, color)

    def evaluate_board(self, board: chess.Board) -> float:
        return self.board_evaluator.evaluate_board(board)
