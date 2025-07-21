import argparse

import chess

from board_evaluator_nn import BoardEvaluatorNN
from player import Player


class MCTSNN(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.board_evaluator = BoardEvaluatorNN(args, color)

    def evaluate_board(self, board: chess.Board) -> float:
        return self.board_evaluator.evaluate_board(board)
