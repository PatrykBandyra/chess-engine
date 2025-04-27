import argparse
import time

import chess
from stockfish import Stockfish

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from constants import STOCKFISH_PATH
from player import Player


class StockfishPlayer(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args)
        self.color: chess.Color = color
        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black
        self.skill_level: int = args.skill_white if color == chess.WHITE else args.skill_black
        self.stockfish = Stockfish(STOCKFISH_PATH)
        self.stockfish.set_depth(self.depth)
        self.stockfish.set_skill_level(self.skill_level)

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        start_time = time.perf_counter()

        self.stockfish.set_fen_position(board.fen())
        move_uci: str = self.stockfish.get_best_move()
        board.push_uci(move_uci)

        end_time = time.perf_counter()
        duration = end_time - start_time
        self.stockfish.set_fen_position(board.fen())
        evaluation = self.stockfish.get_evaluation()['value']
        LOGGER.info(
            f'STOCKFISH {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {move_uci}; ' +
            f'centipawns: {evaluation}'
        )
