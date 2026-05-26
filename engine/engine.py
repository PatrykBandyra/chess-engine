import argparse
import json
import threading
import time

import chess
import chess.pgn

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from human_player import HumanPlayer
from mcts_nn import MCTSNN
from mcts_trad import MCTSTrad
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

        self.json_log_file = open(f'out/{args.json_log}', 'w', encoding='utf-8') if args.json_log else None

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
                return HumanPlayer(self.args, color)
            case PlayerType.STOCKFISH.value:
                return StockfishPlayer(self.args, color)
            case PlayerType.MINIMAX_TRAD.value:
                return MinimaxTrad(self.args, color)
            case PlayerType.MINIMAX_NN.value:
                return MinimaxNN(self.args, color)
            case PlayerType.MCTS_TRAD.value:
                return MCTSTrad(self.args, color)
            case PlayerType.MCTS_NN.value:
                return MCTSNN(self.args, color)
            case _:
                raise ValueError(
                    f'Unknown player type: {player_type}. Available types: {[p.value for p in PlayerType]}')

    def __run(self) -> None:
        if self.is_graphic_mode:
            self.screen_ready_event.wait()

        total_time_white = 0.0
        total_time_black = 0.0
        white_moves = 0
        black_moves = 0

        while not self.__is_game_over_or_draw_claim_available():
            white_move_number = self.board.fullmove_number
            stack_len_before = len(self.board.move_stack)
            t0 = time.perf_counter()
            self.white_player.make_move(self.board, self.screen)
            move_time = time.perf_counter() - t0
            last_move = self.board.move_stack[-1] if len(self.board.move_stack) > stack_len_before else None
            LOGGER.info(f'move_number: {white_move_number}; White move: {last_move.uci() if last_move else "None"}')
            if last_move:
                total_time_white += move_time
                white_moves += 1
                self.__log_move_json(white_move_number, 'WHITE', last_move, self.white_player, move_time)
            if self.__is_game_over_or_draw_claim_available():
                break

            black_move_number = self.board.fullmove_number
            stack_len_before = len(self.board.move_stack)
            t0 = time.perf_counter()
            self.black_player.make_move(self.board, self.screen)
            move_time = time.perf_counter() - t0
            last_move = self.board.move_stack[-1] if len(self.board.move_stack) > stack_len_before else None
            LOGGER.info(f'move_number: {black_move_number}; Black move: {last_move.uci() if last_move else "None"}')
            if last_move:
                total_time_black += move_time
                black_moves += 1
                self.__log_move_json(black_move_number, 'BLACK', last_move, self.black_player, move_time)

        self.__handle_game_over(total_time_white, total_time_black, white_moves, black_moves)

    def __is_game_over_or_draw_claim_available(self) -> bool:
        return (self.board.is_game_over()
                or self.board.can_claim_threefold_repetition()
                or self.board.can_claim_fifty_moves())

    def __log_move_json(self, move_number: int, side: str, move: chess.Move,
                        player: 'Player', move_time: float) -> None:
        if not self.json_log_file:
            return
        move_log = {
            'move_number': move_number,
            'side': side,
            'move': move.uci(),
            'eval': getattr(player, 'last_eval', None),
            'time_s': round(move_time, 4),
            'phase': getattr(player, 'last_phase', None),
            'algorithm_stats': getattr(player, 'stats', {}),
        }
        self.json_log_file.write(json.dumps(move_log) + '\n')
        self.json_log_file.flush()

    def __handle_game_over(self, total_time_white: float = 0.0, total_time_black: float = 0.0,
                           white_moves: int = 0, black_moves: int = 0) -> None:
        LOGGER.info(f'GAME OVER - {self.__get_game_status()}')
        self.__save_moves_to_file()

        if self.json_log_file:
            termination_reason = self.__get_termination_reason()
            outcome = self.board.outcome(claim_draw=True)
            if outcome is None:
                result = '1/2-1/2'
            elif outcome.winner == chess.WHITE:
                result = '1-0'
            elif outcome.winner == chess.BLACK:
                result = '0-1'
            else:
                result = '1/2-1/2'
            game_summary = {
                'type': 'game_summary',
                'result': result,
                'total_moves': len(self.board.move_stack),
                'termination': termination_reason,
                'total_time_white': round(total_time_white, 4),
                'total_time_black': round(total_time_black, 4),
                'avg_time_white': round(total_time_white / white_moves, 4) if white_moves > 0 else 0.0,
                'avg_time_black': round(total_time_black / black_moves, 4) if black_moves > 0 else 0.0,
            }
            self.json_log_file.write(json.dumps(game_summary) + '\n')
            self.json_log_file.close()
            self.json_log_file = None

        if self.screen is not None:
            self.screen.running = False

    def __get_termination_reason(self) -> str:
        if self.board.is_checkmate():
            return 'checkmate'
        elif self.board.is_stalemate():
            return 'stalemate'
        elif self.board.is_seventyfive_moves():
            return 'draw_75'
        elif self.board.is_fivefold_repetition():
            return 'draw_5fold'
        elif self.board.is_insufficient_material():
            return 'draw_insufficient'
        elif self.board.can_claim_fifty_moves():
            return 'draw_50_claim'
        elif self.board.can_claim_threefold_repetition():
            return 'draw_3fold_claim'
        return 'unknown'

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
        elif self.board.is_fivefold_repetition():
            return 'Draw by the fivefold repetition'
        elif self.board.is_seventyfive_moves():
            return 'Draw by the seventy-five-move rule'
        elif self.board.can_claim_fifty_moves():
            return 'Draw by the fifty-move rule'
        elif self.board.can_claim_threefold_repetition():
            return 'Draw by threefold repetition'
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
