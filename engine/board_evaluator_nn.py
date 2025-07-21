import argparse
import time

import chess
from stockfish import Stockfish

from board_evaluator import BoardEvaluator
from constants import STOCKFISH_PATH, LOGGER


class BoardEvaluatorNN(BoardEvaluator):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.depth_stockfish: int = args.depth_white_stockfish if color == chess.WHITE else args.depth_black_stockfish
        self.skill: int = args.skill_white if color == chess.WHITE else args.skill_black
        self.stockfish_path: str = args.stockfish_path if args.stockfish_path is not None else STOCKFISH_PATH

        self.stockfish = Stockfish(self.stockfish_path, parameters={'Threads': 10})
        if self.skill is not None:
            self.stockfish.set_skill_level(self.skill)
        if self.depth_stockfish is not None:
            self.stockfish.set_depth(self.depth_stockfish)

        self.__eval_times = []
        self.__eval_count = 0

    def evaluate_board(self, board: chess.Board) -> float:
        if self.debug:
            start = time.perf_counter()

        self.stockfish.set_fen_position(board.fen())
        value = self.stockfish.get_evaluation()['value'] / 100.0  # Convert centipawns to pawns for consistency

        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times.append(elapsed)
            self.__eval_count += 1
            if self.__eval_count % 200 == 0:
                avg = sum(self.__eval_times) / len(self.__eval_times)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} BoardEvaluatorNN.evaluate_board() average time after {self.__eval_count} calls: {avg:.6f}s')
        return value
