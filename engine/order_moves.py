import abc
import argparse
from typing import List

import chess


class OrderMoves(abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black

        # Approximate values for move ordering heuristics
        self.move_ordering_promotion_bonus = 1_500_000
        self.move_ordering_capture_bonus = 1_000_000
        self.move_ordering_check_bonus = 500_000

    @abc.abstractmethod
    def order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int | None) -> List[chess.Move]:
        pass
