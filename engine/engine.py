import argparse
import threading

import chess

from chess_board_screen import ChessBoardScreen
from human_player import HumanPlayer
from minimax_nn import MinimaxNN
from minimax_trad import MinimaxTrad
from mode import Mode
from player import Player
from player_type import PlayerType


class Engine:

    def __init__(self, args: argparse.Namespace):
        self.screen_ready_event = threading.Event()
        self.screen = None

        white_player = self.__get_player(args.white)
        black_player = self.__get_player(args.black)
        self.white_player = white_player
        self.black_player = black_player
        self.board = chess.Board()
        self.is_graphic_mode = True if args.mode == Mode.G.value else False

        self.screen = ChessBoardScreen(self.board, self.screen_ready_event) if self.is_graphic_mode else None

        self.engine_thread = threading.Thread(target=self.__run)
        self.engine_thread.daemon = True
        self.engine_thread.start()

        self.screen.run()

    @staticmethod
    def __get_player(player_type: str) -> Player:
        match player_type:
            case PlayerType.HUMAN.value:
                return HumanPlayer()
            case PlayerType.MINI_MAX_TRAD.value:
                return MinimaxTrad()
            case PlayerType.MINI_MAX_NN.value:
                return MinimaxNN()

    def __run(self):
        if self.is_graphic_mode:
            self.screen_ready_event.wait()

        while not self.board.is_game_over():
            self.white_player.make_move(self.board, self.screen)
            # white_move_uci: str = self.board.move_stack[-1].uci()
            # self.screen.make_move_uci(white_move_uci)
            if self.board.is_game_over():
                break
            self.black_player.make_move(self.board, self.screen)
            # black_move_uci: str = self.board.move_stack[-1].uci()
            # self.screen.make_move_uci(black_move_uci)

        self.__handle_game_over()

    def __handle_game_over(self):
        pass

    def reset(self):
        pass
