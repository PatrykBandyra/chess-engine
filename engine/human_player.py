import chess

from chess_board_screen import ChessBoardScreen
from player import Player


class HumanPlayer(Player):

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        move_uci: str = screen.get_move_uci()
        board.push_uci(move_uci)
