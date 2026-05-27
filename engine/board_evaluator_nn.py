import argparse
import math
import time

import chess
from stockfish import Stockfish

from board_evaluator import BoardEvaluator
from constants import STOCKFISH_PATH, LOGGER


class BoardEvaluatorNN(BoardEvaluator):

    # Keep these values in sync with Minimax.MATE_SCORE / Minimax.MATE_THRESHOLD.
    # BoardEvaluatorNN does not import Minimax to avoid coupling the evaluator to the search layer.
    MATE_SCORE: float = 1_000_000.0
    MATE_THRESHOLD: float = 990_000.0

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.depth_stockfish: int = args.depth_white_stockfish if color == chess.WHITE else args.depth_black_stockfish
        self.skill: int = args.skill_white if color == chess.WHITE else args.skill_black
        self.stockfish_path: str = args.stockfish_path if args.stockfish_path is not None else STOCKFISH_PATH

        # Threads=1: avoid CPU overcommit when running multiple BoardEvaluatorNN instances
        # in parallel (e.g. 6+ pairs running concurrently). Single-threaded Stockfish is only
        # ~10-20% slower for shallow evaluations (depth 10-15) but eliminates context switching.
        # Hash=128MB: larger transposition table speeds up repeated position lookups within
        # iterative deepening, especially at the same depth searched many times per move.
        self.stockfish = Stockfish(self.stockfish_path, parameters={'Threads': 1, 'Hash': 128})
        if self.skill is not None:
            self.stockfish.set_skill_level(self.skill)
        if self.depth_stockfish is not None:
            self.stockfish.set_depth(self.depth_stockfish)

        self.__eval_times = []
        self.__eval_count = 0

    def evaluate_board(self, board: chess.Board) -> float:
        start = time.perf_counter() if self.debug else 0.0

        if board.is_checkmate():
            value = -math.inf if board.turn == chess.WHITE else math.inf
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            value = 0.0
        else:
            self.stockfish.set_fen_position(board.fen())
            evaluation = self.stockfish.get_evaluation()
            if evaluation['type'] == 'mate':
                mate_in: int = int(evaluation['value'])
                max_mate_distance = int(self.MATE_SCORE - self.MATE_THRESHOLD)
                mate_distance = min(max(abs(mate_in), 1), max_mate_distance)
                value = (self.MATE_SCORE - mate_distance if mate_in > 0
                         else -self.MATE_SCORE + mate_distance if mate_in < 0
                         else 0.0)
            else:
                value = evaluation['value'] / 100.0  # Convert centipawns to pawns for consistency

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
