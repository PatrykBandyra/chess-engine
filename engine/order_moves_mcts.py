import argparse
from typing import List, Tuple

import chess

from constants import PIECE_VALUES, get_piece_value
from order_moves import OrderMoves


class OrderMovesMCTS(OrderMoves):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

    def score_moves(self, board: chess.Board, moves: List[chess.Move], ply: int | None,
                    tt_move: chess.Move | None = None) -> List[Tuple[float, chess.Move]]:
        """
        Score moves with the same heuristic used for ordering (promotions,
        MVV-LVA captures, checks). Returns (score, move) pairs sorted desc.
        """
        move_scores: List[Tuple[float, chess.Move]] = []
        for move in moves:
            score: float = 0
            is_promotion: bool = move.promotion is not None
            is_capture: bool = board.is_capture(move)

            if is_promotion:
                score += self.move_ordering_promotion_bonus + get_piece_value(
                    chess.Piece(move.promotion, board.turn))
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

            # Native check detection — much faster than push/is_check/pop.
            if board.gives_check(move):
                score += self.move_ordering_check_bonus
            move_scores.append((score, move))
        move_scores.sort(key=lambda item: item[0], reverse=True)
        return move_scores

    def order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int | None,
                    tt_move: chess.Move | None = None) -> List[chess.Move]:
        """Orders moves using heuristics: promotions, captures (MVV-LVA), checks."""
        return [move for _, move in self.score_moves(board, moves, ply, tt_move)]
