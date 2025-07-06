import abc
import argparse
import bisect
import math
import random
import time
from typing import List

import chess
from chess.polyglot import open_reader, Entry

from constants import LOGGER
from player import Player


class PlayerTrad(Player, abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.use_opening_book: bool = args.opening_book
        self.is_opening: bool = True
        self.opening_book = open_reader('codekiddy.bin')

        self.__eval_times = []
        self.__eval_times_evaluate_mobility_and_activity = []
        self.__eval_times_evaluate_king_safety = []
        self.__eval_times_evaluate_pawn_structure = []

        self.__eval_count = 0
        self.__eval_count_evaluate_mobility_and_activity = 0
        self.__eval_count_evaluate_king_safety = 0
        self.__eval_count_evaluate_pawn_structure = 0

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3.05,
        chess.BISHOP: 3.33,
        chess.ROOK: 5.63,
        chess.QUEEN: 9.5,
        chess.KING: 100_000
    }

    PIECE_SQUARE_TABLE_MID = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
    }
    PIECE_SQUARE_TABLE_END = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10, 0, 0, -10, -20, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50
        ]
    }

    def make_move_from_opening_book(self, board: chess.Board, start_time: float) -> bool:
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

    def get_piece_value(self, piece: chess.Piece | None) -> float:
        """Safely gets the value of a piece, returning 0 if None"""
        return self.PIECE_VALUES.get(piece.piece_type, 0) if piece else 0

    def __get_game_phase(self, board: chess.Board) -> float:
        """
        Returns a phase value in [0, 1]: 1.0 = middlegame, 0.0 = endgame, interpolated by non-pawn material.
        """
        non_pawn_material = 0.0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            non_pawn_material += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            non_pawn_material += len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]
        endgame_threshold = 20.0
        middlegame_threshold = 67.0
        if non_pawn_material >= middlegame_threshold:
            return 1.0
        elif non_pawn_material <= endgame_threshold:
            return 0.0
        else:
            return (non_pawn_material - endgame_threshold) / (middlegame_threshold - endgame_threshold)

    def __evaluate_pawn_structure(self, board: chess.Board) -> float:
        """
        Evaluates pawn structure for both sides, considering doubled, isolated, and passed pawns.
        Returns a score (positive for White, negative for Black).
        """
        if self.debug:
            start = time.perf_counter()

        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            # Doubled pawns
            doubled_penalty = sum(files.count(f) - 1 for f in set(files) if files.count(f) > 1)
            # Isolated pawns
            isolated_penalty = 0
            for sq in pawns:
                file = chess.square_file(sq)
                has_left = any(chess.square_file(p) == file - 1 for p in pawns)
                has_right = any(chess.square_file(p) == file + 1 for p in pawns)
                if not has_left and not has_right:
                    isolated_penalty += 1
            # Passed pawns
            passed_bonus = 0
            for sq in pawns:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                is_passed = True
                for opp_sq in board.pieces(chess.PAWN, not color):
                    opp_file = chess.square_file(opp_sq)
                    opp_rank = chess.square_rank(opp_sq)
                    if abs(opp_file - file) <= 1:
                        if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                            is_passed = False
                            break
                if is_passed:
                    passed_bonus += 1
            # Penalties and bonuses
            penalty = -0.25 * doubled_penalty - 0.25 * isolated_penalty
            bonus = 0.3 * passed_bonus
            if color == chess.WHITE:
                score += penalty + bonus
            else:
                score -= penalty + bonus
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times_evaluate_pawn_structure.append(elapsed)
            self.__eval_count_evaluate_pawn_structure += 1
            if self.__eval_count_evaluate_pawn_structure % 200 == 0:
                avg = sum(self.__eval_times_evaluate_pawn_structure) / len(
                    self.__eval_times_evaluate_pawn_structure)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} __evaluate_pawn_structure average time after {self.__eval_count_evaluate_pawn_structure} calls: {avg:.6f}s')
        return score

    def __evaluate_king_safety(self, board: chess.Board) -> float:
        """
        Evaluates king safety for both sides.
        Considers pawn shield and open files near the king.
        Returns a score (positive for White, negative for Black).
        """
        if self.debug:
            start = time.perf_counter()

        def king_zone_squares(king_sq: int) -> list[int]:
            zone = [king_sq]
            rank = chess.square_rank(king_sq)
            file = chess.square_file(king_sq)
            # Iterate over the 3x3 grid centered on the king
            for dr in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    if dr == 0 and df == 0:
                        continue  # Skip the king's own square
                    neighbor_rank = rank + dr
                    neighbor_file = file + df
                    if 0 <= neighbor_rank < 8 and 0 <= neighbor_file < 8:
                        neighbor_sq = chess.square(neighbor_file, neighbor_rank)
                        zone.append(neighbor_sq)
            return zone

        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue
            zone = king_zone_squares(king_sq)
            # Pawn shield: count friendly pawns in front of king (3 squares in front)
            shield = 0
            rank = chess.square_rank(king_sq)
            file = chess.square_file(king_sq)
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f < 8:
                    if color == chess.WHITE and rank < 7:
                        sq = chess.square(f, rank + 1)
                    elif color == chess.BLACK and rank > 0:
                        sq = chess.square(f, rank - 1)
                    else:
                        continue
                    if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(
                            sq).color == color:
                        shield += 1
            # Open files near king: penalty for open/semi-open files
            open_file_penalty = 0.0
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f < 8:
                    pawns_on_file = any(
                        chess.square_file(sq) == f and board.piece_at(sq) and board.piece_at(
                            sq).piece_type == chess.PAWN and board.piece_at(sq).color == color
                        for sq in chess.SQUARES
                    )
                    opp_pawns_on_file = any(
                        chess.square_file(sq) == f and board.piece_at(sq) and board.piece_at(
                            sq).piece_type == chess.PAWN and board.piece_at(sq).color != color
                        for sq in chess.SQUARES
                    )
                    if not pawns_on_file:
                        if not opp_pawns_on_file:
                            open_file_penalty += 0.4  # open file
                        else:
                            open_file_penalty += 0.2  # semi-open file
            # King in zone attacked by enemy pieces
            attack_penalty = 0.0
            for sq in zone:
                for attacker in board.attackers(not color, sq):
                    piece = board.piece_at(attacker)
                    if piece:
                        attack_penalty += self.get_piece_value(piece) * 0.1
            # Combine: reward pawn shield, penalize open files and attacks
            king_safety = 0.3 * shield - open_file_penalty - attack_penalty
            if color == chess.WHITE:
                score += king_safety
            else:
                score -= king_safety
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times_evaluate_king_safety.append(elapsed)
            self.__eval_count_evaluate_king_safety += 1
            if self.__eval_count_evaluate_king_safety % 200 == 0:
                avg = sum(self.__eval_times_evaluate_king_safety) / len(
                    self.__eval_times_evaluate_king_safety)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} __evaluate_king_safety average time after {self.__eval_count_evaluate_king_safety} calls: {avg:.6f}s')
        return score

    def __evaluate_mobility_and_activity(self, board: chess.Board) -> float:
        """
        Evaluates piece mobility and activity for both sides.
        Mobility: Number of legal moves for each piece type (except pawns and kings).
        Activity: Bonus for pieces on advanced ranks and controlling central squares.
        Returns a score (positive for White, negative for Black).
        """
        if self.debug:
            start = time.perf_counter()

        mobility_weights = {
            chess.KNIGHT: 0.08,
            chess.BISHOP: 0.10,
            chess.ROOK: 0.07,
            chess.QUEEN: 0.04
        }
        activity_bonus = 0.05  # Bonus for a piece on advanced rank or central square
        central_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            color_sign = 1 if color == chess.WHITE else -1
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, color):
                    # Mobility: count legal moves for this piece
                    mobility = 0
                    for move in board.legal_moves:
                        if move.from_square == sq:
                            mobility += 1
                    score += color_sign * mobility_weights[piece_type] * mobility
                    # Activity: advanced rank or central control
                    rank = chess.square_rank(sq)
                    if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
                        score += color_sign * activity_bonus
                    if sq in central_squares:
                        score += color_sign * activity_bonus
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times_evaluate_mobility_and_activity.append(elapsed)
            self.__eval_count_evaluate_mobility_and_activity += 1
            if self.__eval_count_evaluate_mobility_and_activity % 200 == 0:
                avg = sum(self.__eval_times_evaluate_mobility_and_activity) / len(
                    self.__eval_times_evaluate_mobility_and_activity)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} __evaluate_mobility_and_activity average time after {self.__eval_count_evaluate_mobility_and_activity} calls: {avg:.6f}s')
        return score

    def evaluate_board(self, board: chess.Board) -> float:
        if self.debug:
            start = time.perf_counter()

        if board.is_checkmate():
            result = -math.inf if board.turn == chess.WHITE else math.inf
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            result = 0.0
        else:
            phase = self.__get_game_phase(board)
            white_score = 0.0
            black_score = 0.0
            for piece_type in self.PIECE_VALUES.keys():
                for color in [chess.WHITE, chess.BLACK]:
                    squares = board.pieces(piece_type, color)
                    for sq in squares:
                        pst_mid = self.PIECE_SQUARE_TABLE_MID[piece_type][
                            sq if color == chess.WHITE else chess.square_mirror(sq)]
                        pst_end = self.PIECE_SQUARE_TABLE_END[piece_type][
                            sq if color == chess.WHITE else chess.square_mirror(sq)]
                        value = self.PIECE_VALUES[piece_type] + 0.01 * (phase * pst_mid + (1 - phase) * pst_end)
                        if color == chess.WHITE:
                            white_score += value
                        else:
                            black_score += value
            pawn_structure_score = self.__evaluate_pawn_structure(board)
            king_safety_score = self.__evaluate_king_safety(board)
            mobility_activity_score = self.__evaluate_mobility_and_activity(board)
            result = (white_score - black_score) + pawn_structure_score + king_safety_score + mobility_activity_score
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times.append(elapsed)
            self.__eval_count += 1
            if self.__eval_count % 200 == 0:
                avg = sum(self.__eval_times) / len(self.__eval_times)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} __evaluate_board average time after {self.__eval_count} calls: {avg:.6f}s')
        return result
