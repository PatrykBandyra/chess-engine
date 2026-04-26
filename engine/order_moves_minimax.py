import argparse
from typing import List

import chess

from constants import get_piece_value, PIECE_VALUES
from order_moves import OrderMoves


class OrderMovesMinimax(OrderMoves):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        # Stores 2 moves per ply that caused beta cutoffs - indexed by ply (0 = root, 1 = depth-1, etc.)
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]

        # Stores scores for non-capture moves based on success - indexed by [from_square][to_square]
        self.history_heuristic_table = [[0] * 64 for _ in range(64)]

        # Approximate values for move ordering heuristics
        self.move_ordering_killer_bonus = 750_000
        self.move_ordering_tt_move_bonus = 2_000_000

    def order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int | None,
                    tt_move: chess.Move | None = None) -> List[chess.Move]:
        """
        Orders a list of legal moves to improve alpha-beta pruning efficiency.

        How it works:
        - Assigns a score to each move based on several heuristics (in priority order):
            * TT move (PV move): The best move from the transposition table (found in a previous Iterative
              Deepening iteration or earlier search) is always searched first with the highest bonus.
            * Promotions: Very high priority, scored by promoted piece value.
            * Captures: Scored by Most Valuable Victim - Least Valuable Aggressor (MVV-LVA).
            * Killer moves: Moves that caused beta cutoffs at this ply in previous searches are prioritized.
            * History heuristic: Quiet moves that have historically caused cutoffs are boosted.
            * Checks: Moves that give check receive a bonus (applied to all move types: promotions, captures, and quiet).
        - Moves are sorted in descending order of their score, so the most promising moves are searched first.
        - This ordering increases the likelihood of alpha-beta cutoffs, making the search more efficient
          and improving engine strength. Combined with Iterative Deepening, TT move prioritization ensures
          the PV move from shallower iterations seeds deeper searches for optimal move ordering.

        Args:
            board: The current board state.
            moves: List of legal moves to order.
            ply: The current ply (search depth from root).
            tt_move: The best move from the transposition table for this position (PV move from a previous
                     iteration or search). If provided, this move receives the highest ordering bonus.
        Returns:
            List of moves sorted from best to worst according to the heuristics.
        """
        move_scores = []
        killers = self.killer_moves[ply] if 0 <= ply < len(self.killer_moves) else [None, None]

        for move in moves:
            score: float = 0

            # 0. TT move (PV move from previous iteration) — always searched first
            if tt_move is not None and move == tt_move:
                score += self.move_ordering_tt_move_bonus
                move_scores.append((score, move))
                continue

            is_promotion: bool = move.promotion is not None
            is_capture: bool = board.is_capture(move)

            # 1. Check bonus (applied to all move types: promotions, captures, and quiet moves)
            if board.gives_check(move):
                score += self.move_ordering_check_bonus

            # 2. Promotions
            if is_promotion:
                score += (self.move_ordering_promotion_bonus +
                          get_piece_value(chess.Piece(move.promotion, board.turn)))

            # 3. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor)
            elif is_capture:  # If both promotion and capture, then prefer promotion score
                move_piece: chess.Piece = board.piece_at(move.from_square)
                captured_piece: chess.Piece | None = board.piece_at(move.to_square)
                captured_piece_value: float = 0
                if captured_piece:
                    captured_piece_value = get_piece_value(captured_piece)
                elif board.is_en_passant(move):
                    captured_piece_value = PIECE_VALUES[chess.PAWN]

                aggressor_value: float = get_piece_value(move_piece)
                score += self.move_ordering_capture_bonus + (captured_piece_value - aggressor_value / 10)

            # 4. Quiet Moves (apply Killer and History Heuristics)
            else:
                is_killer: bool = move == killers[0] or move == killers[1]
                if is_killer:
                    score += self.move_ordering_killer_bonus

                history_score = self.history_heuristic_table[move.from_square][move.to_square]
                score += history_score


            move_scores.append((score, move))

        # Sort moves in descending order of score
        move_scores.sort(key=lambda item: item[0], reverse=True)

        # Return just the moves in the sorted order
        return [move for _, move in move_scores]

    def store_killer_move(self, ply: int, move: chess.Move) -> None:
        """Stores a killer move for a given ply, keeping the two best"""
        if 0 <= ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move

    def update_history_score(self, move: chess.Move, depth: int) -> None:
        """Increases the history score for a successful quiet move"""
        if move.promotion is None:  # Captures are implicitly excluded by where this is called
            bonus = depth * depth  # Weight bonus by depth squared
            self.history_heuristic_table[move.from_square][move.to_square] += bonus
