import abc
import argparse

import chess

from chess_board_screen import ChessBoardScreen


class Player(abc.ABC):

    def __init__(self, args: argparse.Namespace):
        self.args: argparse.Namespace = args

    @abc.abstractmethod
    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        pass
