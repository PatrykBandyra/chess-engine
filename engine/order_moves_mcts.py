import argparse
from typing import List

import chess

from constants import PIECE_VALUES, get_piece_value
from order_moves import OrderMoves


class OrderMovesMCTS(OrderMoves):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

    def order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int | None) -> List[chess.Move]:
        """
        Orders moves using heuristics: promotions, captures (MVV-LVA), checks.
        """
        move_scores = []
        for move in moves:
            score: float = 0
            is_promotion: bool = move.promotion is not None
            is_capture: bool = board.is_capture(move)

            # 1. Promotions
            if is_promotion:
                score += self.move_ordering_promotion_bonus + get_piece_value(
                    chess.Piece(move.promotion, board.turn))

            # 2. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor)
            elif is_capture:
                move_piece: chess.Piece = board.piece_at(move.from_square)
                captured_piece: chess.Piece | None = board.piece_at(move.to_square)
                captured_piece_value: float = 0
                if captured_piece:
                    captured_piece_value = get_piece_value(captured_piece)
                elif board.is_en_passant(move):
                    captured_piece_value = PIECE_VALUES[chess.PAWN]
                aggressor_value: float = get_piece_value(move_piece)
                score += self.move_ordering_capture_bonus + (captured_piece_value - aggressor_value / 10)

            # 3. Checks (check if the move puts the opponent in check)
            board.push(move)
            is_check: bool = board.is_check()
            board.pop()
            if is_check:
                score += self.move_ordering_check_bonus
            move_scores.append((score, move))
        move_scores.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in move_scores]
