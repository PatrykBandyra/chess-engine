import abc
import argparse

import chess


class BoardEvaluator(abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.debug: bool = args.debug
        self.color: chess.Color = color

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass
