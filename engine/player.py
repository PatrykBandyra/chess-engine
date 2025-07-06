import abc
import argparse

import chess

from chess_board_screen import ChessBoardScreen


class Player(abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.args: argparse.Namespace = args
        self.color: chess.Color = color

    @abc.abstractmethod
    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        pass
