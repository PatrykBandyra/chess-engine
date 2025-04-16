import abc

import chess

from chess_board_screen import ChessBoardScreen


class Player(abc.ABC):

    @abc.abstractmethod
    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        pass
