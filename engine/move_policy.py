from collections.abc import Iterable
from typing import List

import chess


def queen_promotions_only(moves: Iterable[chess.Move]) -> List[chess.Move]:
    """Return moves with underpromotions removed, keeping queen promotions.

    The engine treats underpromotions as outside the default search policy.
    Human-entered moves are not passed through this helper, so manual play can
    still use any legal promotion if support is added at the UI/input layer.
    """
    return [move for move in moves if move.promotion is None or move.promotion == chess.QUEEN]


def force_queen_promotion(board: chess.Board, move: chess.Move | None) -> chess.Move | None:
    """Convert a selected underpromotion to the equivalent queen promotion.

    This is a final safety net at the root: search code should normally filter
    underpromotions before they are explored, but this keeps persisted/reused or
    manually provided candidate moves from reaching ``board.push()`` unchanged.
    """
    if move is None or move.promotion is None or move.promotion == chess.QUEEN:
        return move

    queen_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return queen_move if queen_move in board.legal_moves else move

