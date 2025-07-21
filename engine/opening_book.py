import argparse
import bisect
import random
import time
from typing import List

import chess
from chess.polyglot import open_reader, Entry

from constants import LOGGER


class OpeningBook:

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.color: chess.Color = color
        self.use_opening_book: bool = args.opening_book
        self.is_opening: bool = True
        self.opening_book = open_reader('codekiddy.bin')

    def make_move(self, board: chess.Board, start_time: float) -> bool:
        entries: List[Entry] = list(self.opening_book.find_all(board))
        if entries:
            weights = [entry.weight for entry in entries]
            weights_sum = sum(weights)
            if weights_sum > 0:
                cum_weights = []
                cum_sum = 0
                for weight in weights:
                    cum_sum += weight
                    cum_weights.append(cum_sum)
                r = random.uniform(0, weights_sum)
                idx = bisect.bisect_left(cum_weights, r)
                entry = entries[idx]
                opening_book_best_move = entry.move
            else:
                entry = random.choice(entries)
                opening_book_best_move = entry.move
            board.push(opening_book_best_move)

            end_time: float = time.perf_counter()
            duration: float = end_time - start_time

            max_weight = max(weights)
            max_weight_move = next((e for e in entries if e.weight == max_weight), None).move.uci()
            LOGGER.info(
                f'OPENING BOOK; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; ' +
                f'move: {opening_book_best_move.uci()}; weight: {entry.weight}; ' +
                f'max weight move: {max_weight_move}; max weight: {max_weight}')
            return self.is_opening
        else:
            self.is_opening = False
            LOGGER.info(f'OPENING BOOK; {"WHITE" if self.color else "BLACK"}; opening phase ended')
            return self.is_opening

    def __del__(self):
        if hasattr(self, 'opening_book') and self.opening_book is not None:
            self.opening_book.close()
