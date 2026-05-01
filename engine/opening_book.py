import argparse
import bisect
import random
import time
from typing import List

import chess
from chess.polyglot import open_reader, Entry

from constants import LOGGER


class OpeningBook:

    # Entries with `weight < MIN_WEIGHT_RATIO * max_weight` for the current position
    # are discarded before the weighted random pick. Prevents selecting marginal lines
    # (e.g. weight 3 when max is 10) which historically caused early-game material loss.
    MIN_WEIGHT_RATIO: float = 0.25

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.color: chess.Color = color
        self.use_opening_book: bool = args.opening_book
        # Deterministic mode — always pick the max-weight entry. Useful for
        # reproducible self-play / regression runs. Defaults to False (random).
        self.strict: bool = bool(getattr(args, 'opening_book_strict', False))
        self.is_opening: bool = True
        self.opening_book = open_reader('codekiddy.bin')

    def make_move(self, board: chess.Board, start_time: float) -> bool:
        move_number: int = board.fullmove_number
        entries: List[Entry] = list(self.opening_book.find_all(board))
        if entries:
            weights = [entry.weight for entry in entries]
            max_weight = max(weights)

            # Filter marginal entries before the random pick. Fallback to all entries
            # if filtering would discard everything (e.g. all weights == 0 and max == 0).
            if max_weight > 0:
                threshold = self.MIN_WEIGHT_RATIO * max_weight
                filtered = [(e, w) for e, w in zip(entries, weights) if w >= threshold]
                if not filtered:
                    filtered = list(zip(entries, weights))
            else:
                filtered = list(zip(entries, weights))

            filtered_entries = [e for e, _ in filtered]
            filtered_weights = [w for _, w in filtered]
            weights_sum = sum(filtered_weights)

            if self.strict:
                # Deterministic: pick the heaviest among the filtered set.
                idx_max = max(range(len(filtered_weights)), key=lambda i: filtered_weights[i])
                entry = filtered_entries[idx_max]
                opening_book_best_move = entry.move
            elif weights_sum > 0:
                cum_weights = []
                cum_sum = 0
                for weight in filtered_weights:
                    cum_sum += weight
                    cum_weights.append(cum_sum)
                r = random.uniform(0, weights_sum)
                idx = bisect.bisect_left(cum_weights, r)
                entry = filtered_entries[idx]
                opening_book_best_move = entry.move
            else:
                entry = random.choice(filtered_entries)
                opening_book_best_move = entry.move
            board.push(opening_book_best_move)

            end_time: float = time.perf_counter()
            duration: float = end_time - start_time

            max_weight_move = next((e for e in entries if e.weight == max_weight), None).move.uci()
            share = (entry.weight / weights_sum) if weights_sum > 0 else 0.0
            LOGGER.info(
                f'OPENING BOOK; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; time: {duration:.6f}s; ' +
                f'move: {opening_book_best_move.uci()}; weight: {entry.weight}; share: {share:.3f}; ' +
                f'max weight move: {max_weight_move}; max weight: {max_weight}; ' +
                f'entries: {len(entries)}; filtered_entries: {len(filtered_entries)}; weights_sum: {weights_sum}; ' +
                f'strict: {self.strict}')
            return self.is_opening
        else:
            self.is_opening = False
            LOGGER.info(f'OPENING BOOK; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; opening phase ended')
            return self.is_opening

    def __del__(self):
        if hasattr(self, 'opening_book') and self.opening_book is not None:
            self.opening_book.close()
