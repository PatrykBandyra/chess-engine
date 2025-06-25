import argparse
import threading

import chess
import chess.pgn

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from human_player import HumanPlayer
from minimax_nn import MinimaxNN
from minimax_trad import MinimaxTrad
from mode import Mode
from player import Player
from player_type import PlayerType
from stockfishplayer import StockfishPlayer


class Engine:

    def __init__(self, args: argparse.Namespace):
        self.args: argparse.Namespace = args
        self.is_graphic_mode = True if args.mode == Mode.G.value else False
        self.is_settings_mode = True if args.mode == Mode.S.value else False

        self.screen_ready_event = threading.Event()
        self.screen = None

        self.white_player: Player | None = None
        self.black_player: Player | None = None

        if not self.is_settings_mode:
            self.white_player = self.__get_player(args.white, chess.WHITE)
            self.black_player = self.__get_player(args.black, chess.BLACK)

        self.board: chess.Board = chess.Board()
        if args.input:
            self.board = chess.Board(fen=self.__load_fen_from_file())
        if self.is_settings_mode and args.empty:
            self.board = chess.Board(fen=None)

        if self.is_graphic_mode or self.is_settings_mode:
            self.screen = ChessBoardScreen(self.board, self.screen_ready_event)

        if self.screen is not None:
            if self.is_graphic_mode:
                self.engine_thread = threading.Thread(target=self.__run)
                self.engine_thread.daemon = True
                self.engine_thread.start()

                self.screen.run()

            elif self.is_settings_mode:
                self.screen.run_settings()
        else:
            self.__run()

        if self.args.output:
            self.__save_board_to_fen_file()

    def __get_player(self, player_type: str, color: chess.Color) -> Player:
        match player_type:
            case PlayerType.HUMAN.value:
                return HumanPlayer(self.args)
            case PlayerType.STOCKFISH.value:
                return StockfishPlayer(self.args, color)
            case PlayerType.MINIMAX_TRAD.value:
                return MinimaxTrad(self.args, color)
            case PlayerType.MINIMAX_NN.value:
                return MinimaxNN(self.args)

    def __run(self) -> None:
        if self.is_graphic_mode:
            self.screen_ready_event.wait()

        while not self.board.is_game_over():
            self.white_player.make_move(self.board, self.screen)
            LOGGER.info(f'White move: {self.board.move_stack[-1].uci() if len(self.board.move_stack) > 0 else "None"}')
            if self.board.is_game_over():
                break
            self.black_player.make_move(self.board, self.screen)
            LOGGER.info(f'Black move: {self.board.move_stack[-1].uci() if len(self.board.move_stack) > 0 else "None"}')

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

    def __save_moves_to_file(self) -> None:
        with open(f'out/{self.args.game}', mode='w', encoding='utf-8') as file:
            file.write(f'Result: {self.__get_game_status()}\n\n')
            for i, move in enumerate(self.board.move_stack):
                file.write(f'{i + 1}: {move.uci()}\n')

    def __save_board_to_fen_file(self) -> None:
        with open(f'out/{self.args.output}', mode='w', encoding='utf-8') as file:
            file.write(self.board.fen())

    def __load_fen_from_file(self) -> str:
        with open(f'out/{self.args.input}', mode='r', encoding='utf-8') as file:
            return file.read().strip()
