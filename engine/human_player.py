import chess

from chess_board_screen import ChessBoardScreen
from player import Player


class HumanPlayer(Player):

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        move: chess.Move = screen.get_move()
        board.push(move)
