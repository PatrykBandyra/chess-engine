import argparse
import logging
import threading

import chess
import chess.pgn

from chess_board_screen import ChessBoardScreen
from constants import LOG_FILE_NAME
from human_player import HumanPlayer
from minimax_nn import MinimaxNN
from minimax_trad import MinimaxTrad
from mode import Mode
from player import Player
from player_type import PlayerType
from stockfishplayer import StockfishPlayer

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
LOGGER.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(FORMATTER)
LOGGER.addHandler(file_handler)


class Engine:

    def __init__(self, args: argparse.Namespace):
        self.args: argparse.Namespace = args

        self.screen_ready_event = threading.Event()
        self.screen = None

        white_player = self.__get_player(args.white, chess.WHITE)
        black_player = self.__get_player(args.black, chess.BLACK)
        self.white_player = white_player
        self.black_player = black_player
        self.board = chess.Board()
        self.is_graphic_mode = True if args.mode == Mode.G.value else False

        self.screen = ChessBoardScreen(self.board, self.screen_ready_event) if self.is_graphic_mode else None

        if self.screen is not None:
            self.engine_thread = threading.Thread(target=self.__run)
            self.engine_thread.daemon = True
            self.engine_thread.start()

            self.screen.run()
        else:
            self.__run()

    def __get_player(self, player_type: str, color: chess.Color) -> Player:
        match player_type:
            case PlayerType.HUMAN.value:
                return HumanPlayer(self.args)
            case PlayerType.STOCKFISH.value:
                return StockfishPlayer(self.args, color)
            case PlayerType.MINI_MAX_TRAD.value:
                return MinimaxTrad(self.args)
            case PlayerType.MINI_MAX_NN.value:
                return MinimaxNN(self.args)

    def __run(self) -> None:
        if self.is_graphic_mode:
            self.screen_ready_event.wait()

        while not self.board.is_game_over():
            self.white_player.make_move(self.board, self.screen)
            LOGGER.info(f'White move: {self.board.move_stack[-1].uci()}')
            if self.board.is_game_over():
                break
            self.black_player.make_move(self.board, self.screen)
            LOGGER.info(f'Black move: {self.board.move_stack[-1].uci()}')

        self.__handle_game_over()

    def __handle_game_over(self) -> None:
        LOGGER.info(f'GAME OVER - {self.__get_game_status()}')
        self.__save_moves_to_file()
        if self.screen is not None:
            self.screen.running = False

    def __get_game_status(self) -> str | None:
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return 'Black wins by checkmate'
            else:
                return 'White wins by checkmate'
        elif self.board.is_stalemate():
            return 'Draw by stalemate'
        elif self.board.is_insufficient_material():
            return 'Draw by insufficient material'
        elif self.board.can_claim_fifty_moves():
            return 'Draw by the fifty-move rule'
        elif self.board.can_claim_threefold_repetition():
            return 'Draw by threefold repetition'
        elif self.board.is_fivefold_repetition():
            return 'Draw by the fivefold repetition'
        elif self.board.is_seventyfive_moves():
            return 'Draw by the seventy-five-move rule'
        return None

    def __save_moves_to_file(self, filename: str = 'out/game_1.txt') -> None:
        with open(filename, mode='w', encoding='utf-8') as file:
            file.write(f'Result: {self.__get_game_status()}\n\n')
            for i, move in enumerate(self.board.move_stack):
                file.write(f'{i + 1}: {move.uci()}\n')
