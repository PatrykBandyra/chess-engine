import abc
import argparse

import chess

from chess_board_screen import ChessBoardScreen


class Player(abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.args: argparse.Namespace = args
        self.color: chess.Color = color
        self.debug: bool = args.debug

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3.05,
        chess.BISHOP: 3.33,
        chess.ROOK: 5.63,
        chess.QUEEN: 9.5,
        chess.KING: 100_000
    }

    @abc.abstractmethod
    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        pass
