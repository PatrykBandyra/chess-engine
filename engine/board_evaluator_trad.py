import argparse
import math
import time

import chess

from board_evaluator import BoardEvaluator
from constants import LOGGER
from constants import PIECE_VALUES, get_piece_value


class BoardEvaluatorTrad(BoardEvaluator):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.__eval_times = []
        self.__eval_times_evaluate_mobility_and_activity = []
        self.__eval_times_evaluate_king_safety = []
        self.__eval_times_evaluate_pawn_structure = []

        self.__eval_count = 0
        self.__eval_count_evaluate_mobility_and_activity = 0
        self.__eval_count_evaluate_king_safety = 0
        self.__eval_count_evaluate_pawn_structure = 0

    ENDGAME_MATERIAL_THRESHOLD = 20.0
    MIDDLEGAME_MATERIAL_THRESHOLD = 67.0
    PST_SCALE = 0.01

    DOUBLED_PAWN_PENALTY = 0.25
    ISOLATED_PAWN_PENALTY = 0.25
    PASSED_PAWN_BONUS_BY_ADVANCEMENT = [0.0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 0.0]
    PASSED_PAWN_ENDGAME_MULTIPLIER = 0.5
    PASSED_PAWN_BLOCKED_MULTIPLIER = 0.55
    PASSED_PAWN_SUPPORTED_BY_PAWN_BONUS = 0.15
    PASSED_PAWN_SUPPORTED_BY_PAWN_ENDGAME_BONUS = 0.10
    PASSED_PAWN_SUPPORTED_BY_PIECE_BONUS = 0.05
    CONNECTED_PASSED_PAWN_BONUS = 0.15
    CONNECTED_PASSED_PAWN_ENDGAME_BONUS = 0.10
    PASSED_PAWN_KING_RACE_BONUS = 0.04
    PASSED_PAWN_KING_RACE_BONUS_LIMIT = 0.15

    KING_SHIELD_BONUS = 0.30
    KING_OPEN_FILE_PENALTY = 0.40
    KING_SEMI_OPEN_FILE_PENALTY = 0.20
    KING_ATTACK_WEIGHT = 0.10
    KING_ATTACKED_ZONE_SQUARE_WEIGHT = 0.03
    KING_ATTACKED_ZONE_SQUARE_PENALTY_LIMIT = 0.24
    KING_ATTACK_PENALTY_LIMIT = 1.50

    BISHOP_PAIR_BASE_BONUS = 0.25
    BISHOP_PAIR_OPENNESS_BONUS = 0.15
    BISHOP_PAIR_PHASE_BASE = 0.90
    BISHOP_PAIR_ENDGAME_BONUS = 0.10

    ROOK_ACTIVITY_PHASE_BASE = 0.90
    ROOK_ACTIVITY_ENDGAME_BONUS = 0.20
    ROOK_OPEN_FILE_BONUS = 0.25
    ROOK_SEMI_OPEN_FILE_BONUS = 0.15
    ROOK_SEVENTH_RANK_BONUS = 0.25
    ROOK_DOUBLED_BONUS = 0.10

    HANGING_PIECE_VALUE_WEIGHT = 0.15
    HANGING_PIECE_PENALTY_LIMIT = 0.75
    HANGING_MAJOR_PIECE_EXTRA_PENALTY = 0.10
    PAWN_ATTACKED_PIECE_VALUE_WEIGHT = 0.10
    PAWN_ATTACKED_PIECE_PENALTY_LIMIT = 0.75
    LOWER_VALUE_ATTACK_WEIGHT = 0.08
    LOWER_VALUE_ATTACK_PENALTY_LIMIT = 0.60

    ENDGAME_KING_CENTER_BONUS = 0.05
    ENDGAME_KING_ENEMY_PAWN_PRESSURE_BONUS = 0.015
    ENDGAME_KING_ENEMY_PAWN_PRESSURE_LIMIT = 0.18
    ENDGAME_KING_OWN_PASSER_SUPPORT_BONUS = 0.025
    ENDGAME_KING_ENEMY_PASSER_STOP_BONUS = 0.02

    MOBILITY_WEIGHTS = {
        chess.KNIGHT: 0.08,
        chess.BISHOP: 0.10,
        chess.ROOK: 0.07,
        chess.QUEEN: 0.04
    }
    ACTIVITY_BONUS = 0.05
    CENTRAL_CONTROL_BONUS = 0.03
    CENTRAL_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}
    CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

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
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            0, 0, 5, 5, 5, 5, 0, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30
        ]
    }
    PIECE_SQUARE_TABLE_END = {
        chess.PAWN: [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -10, -10, 10, 10, 5,
            5, 0, 0, 5, 5, 0, 0, 5,
            5, 5, 10, 20, 20, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            20, 20, 35, 45, 45, 35, 20, 20,
            60, 60, 70, 80, 80, 70, 60, 60,
            0, 0, 0, 0, 0, 0, 0, 0
        ],
        chess.KNIGHT: [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ],
        chess.BISHOP: [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ],
        chess.ROOK: [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        chess.QUEEN: [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            0, 0, 5, 5, 5, 5, 0, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ],
        chess.KING: [
            -50, -30, -30, -30, -30, -30, -30, -50,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -20, -10, 0, 0, -10, -20, -30,
            -50, -40, -30, -20, -20, -30, -40, -50
        ]
    }

    def __get_game_phase(self, board: chess.Board) -> float:
        """
        Returns a phase value in [0, 1]: 1.0 = middlegame, 0.0 = endgame, interpolated by non-pawn material.
        """
        non_pawn_material = 0.0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            non_pawn_material += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            non_pawn_material += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        endgame_threshold = self.ENDGAME_MATERIAL_THRESHOLD
        middlegame_threshold = self.MIDDLEGAME_MATERIAL_THRESHOLD
        if non_pawn_material >= middlegame_threshold:
            return 1.0
        elif non_pawn_material <= endgame_threshold:
            return 0.0
        else:
            return (non_pawn_material - endgame_threshold) / (middlegame_threshold - endgame_threshold)

    def __pst_value(self, piece_type: chess.PieceType, square: chess.Square, color: chess.Color,
                    phase: float) -> float:
        """
        Returns the phase-interpolated piece-square table value in pawn units.

        The PST arrays are stored in python-chess square order from a1 to h8, from White's
        perspective. White pieces can use the square index directly. Black pieces are
        mirrored vertically, giving symmetric values for equivalent White/Black placements.
        """
        pst_index = square if color == chess.WHITE else chess.square_mirror(square)
        pst_mid = self.PIECE_SQUARE_TABLE_MID[piece_type][pst_index]
        pst_end = self.PIECE_SQUARE_TABLE_END[piece_type][pst_index]
        return self.PST_SCALE * (phase * pst_mid + (1 - phase) * pst_end)

    def __is_passed_pawn(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        for opp_sq in board.pieces(chess.PAWN, not color):
            opp_file = chess.square_file(opp_sq)
            opp_rank = chess.square_rank(opp_sq)
            if abs(opp_file - file) <= 1:
                if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                    return False
        return True

    def __passed_pawns(self, board: chess.Board, color: chess.Color) -> list[chess.Square]:
        return [sq for sq in board.pieces(chess.PAWN, color) if self.__is_passed_pawn(board, sq, color)]

    def __evaluate_pawn_structure(self, board: chess.Board, phase: float) -> float:
        """
        Evaluates pawn structure for both sides, considering doubled, isolated, and passed pawns.
        Returns a score (positive for White, negative for Black).
        """
        start = time.perf_counter() if self.debug else 0.0

        endgame_weight = 1.0 - phase
        def evaluate_passed_pawn(square: chess.Square, color: chess.Color,
                                 pawns: chess.SquareSet, passed_pawns: list[chess.Square]) -> float:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            advancement = rank if color == chess.WHITE else 7 - rank
            bonus = self.PASSED_PAWN_BONUS_BY_ADVANCEMENT[advancement] * (
                    1.0 + self.PASSED_PAWN_ENDGAME_MULTIPLIER * endgame_weight)

            forward_rank = rank + (1 if color == chess.WHITE else -1)
            if 0 <= forward_rank < 8:
                blocker_sq = chess.square(file, forward_rank)
                if board.piece_at(blocker_sq) is not None:
                    bonus *= self.PASSED_PAWN_BLOCKED_MULTIPLIER

            friendly_pawn_attackers = board.attackers(color, square) & pawns
            if friendly_pawn_attackers:
                bonus += self.PASSED_PAWN_SUPPORTED_BY_PAWN_BONUS + \
                         self.PASSED_PAWN_SUPPORTED_BY_PAWN_ENDGAME_BONUS * endgame_weight
            elif board.attackers(color, square):
                bonus += self.PASSED_PAWN_SUPPORTED_BY_PIECE_BONUS

            if any(other_sq != square and abs(chess.square_file(other_sq) - file) == 1 for other_sq in passed_pawns):
                bonus += self.CONNECTED_PASSED_PAWN_BONUS + self.CONNECTED_PASSED_PAWN_ENDGAME_BONUS * endgame_weight

            own_king_sq = board.king(color)
            enemy_king_sq = board.king(not color)
            if advancement >= 4 and own_king_sq is not None and enemy_king_sq is not None:
                promotion_sq = chess.square(file, 7 if color == chess.WHITE else 0)
                own_king_distance = chess.square_distance(own_king_sq, promotion_sq)
                enemy_king_distance = chess.square_distance(enemy_king_sq, promotion_sq)
                king_race_bonus = self.PASSED_PAWN_KING_RACE_BONUS * (
                        enemy_king_distance - own_king_distance) * endgame_weight
                bonus += max(-self.PASSED_PAWN_KING_RACE_BONUS_LIMIT,
                             min(self.PASSED_PAWN_KING_RACE_BONUS_LIMIT, king_race_bonus))

            return max(0.0, bonus)

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
            passed_pawns = self.__passed_pawns(board, color)
            passed_bonus = sum(evaluate_passed_pawn(sq, color, pawns, passed_pawns) for sq in passed_pawns)
            # Penalties and bonuses
            penalty = -self.DOUBLED_PAWN_PENALTY * doubled_penalty - self.ISOLATED_PAWN_PENALTY * isolated_penalty
            bonus = passed_bonus
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
                    f'{"WHITE" if self.color else "BLACK"} BoardEvaluatorTrad.__evaluate_pawn_structure() average time after {self.__eval_count_evaluate_pawn_structure} calls: {avg:.6f}s')
        return score

    def __evaluate_king_safety(self, board: chess.Board) -> float:
        """
        Evaluates king safety for both sides.
        Considers pawn shield and open files near the king.
        Returns a score (positive for White, negative for Black).
        """
        start = time.perf_counter() if self.debug else 0.0

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
            own_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
            opp_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f < 8:
                    if f not in own_pawn_files:
                        if f not in opp_pawn_files:
                            open_file_penalty += self.KING_OPEN_FILE_PENALTY
                        else:
                            open_file_penalty += self.KING_SEMI_OPEN_FILE_PENALTY
            # King in zone attacked by enemy pieces
            unique_attackers = set()
            attacked_zone_squares = set()
            for sq in zone:
                attackers = board.attackers(not color, sq)
                if attackers:
                    attacked_zone_squares.add(sq)
                    unique_attackers.update(attackers)
            unique_attacker_penalty = 0.0
            for attacker in unique_attackers:
                piece = board.piece_at(attacker)
                if piece:
                    unique_attacker_penalty += get_piece_value(piece) * self.KING_ATTACK_WEIGHT
            coverage_penalty = min(self.KING_ATTACKED_ZONE_SQUARE_PENALTY_LIMIT,
                                   len(attacked_zone_squares) * self.KING_ATTACKED_ZONE_SQUARE_WEIGHT)
            attack_penalty = min(self.KING_ATTACK_PENALTY_LIMIT, unique_attacker_penalty + coverage_penalty)
            # Combine: reward pawn shield, penalize open files and attacks
            king_safety = self.KING_SHIELD_BONUS * shield - open_file_penalty - attack_penalty
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
                    f'{"WHITE" if self.color else "BLACK"} BoardEvaluatorTrad.__evaluate_king_safety() average time after {self.__eval_count_evaluate_king_safety} calls: {avg:.6f}s')
        return score

    def __evaluate_minor_piece_features(self, board: chess.Board, phase: float) -> float:
        """
        Evaluates lightweight minor-piece positional features.
        Currently rewards the bishop pair, with a slightly larger bonus in open/endgame positions.
        Returns a score (positive for White, negative for Black).
        """
        total_pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
        openness = (16 - total_pawns) / 16.0
        endgame_weight = 1.0 - phase

        bishop_pair_bonus = (self.BISHOP_PAIR_BASE_BONUS + self.BISHOP_PAIR_OPENNESS_BONUS * openness) * (
                self.BISHOP_PAIR_PHASE_BASE + self.BISHOP_PAIR_ENDGAME_BONUS * endgame_weight)

        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                if color == chess.WHITE:
                    score += bishop_pair_bonus
                else:
                    score -= bishop_pair_bonus
        return score

    def __evaluate_rook_activity(self, board: chess.Board, phase: float) -> float:
        """
        Evaluates rook activity: open/semi-open files, rooks on the 7th/2nd rank,
        and doubled rooks on useful files. Returns a score (positive for White, negative for Black).
        """
        endgame_weight = 1.0 - phase
        activity_scale = self.ROOK_ACTIVITY_PHASE_BASE + self.ROOK_ACTIVITY_ENDGAME_BONUS * endgame_weight

        open_file_bonus = self.ROOK_OPEN_FILE_BONUS * activity_scale
        semi_open_file_bonus = self.ROOK_SEMI_OPEN_FILE_BONUS * activity_scale
        seventh_rank_bonus = self.ROOK_SEVENTH_RANK_BONUS * activity_scale
        doubled_rooks_bonus = self.ROOK_DOUBLED_BONUS * activity_scale

        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            color_sign = 1 if color == chess.WHITE else -1
            rooks = board.pieces(chess.ROOK, color)
            useful_rook_files = set()

            for sq in rooks:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                own_pawns_on_file = any(chess.square_file(pawn_sq) == file for pawn_sq in board.pieces(chess.PAWN, color))
                enemy_pawns_on_file = any(chess.square_file(pawn_sq) == file for pawn_sq in board.pieces(chess.PAWN, not color))

                if not own_pawns_on_file and not enemy_pawns_on_file:
                    score += color_sign * open_file_bonus
                    useful_rook_files.add(file)
                elif not own_pawns_on_file and enemy_pawns_on_file:
                    score += color_sign * semi_open_file_bonus
                    useful_rook_files.add(file)

                if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                    score += color_sign * seventh_rank_bonus

            for file in useful_rook_files:
                rooks_on_file = sum(1 for rook_sq in rooks if chess.square_file(rook_sq) == file)
                if rooks_on_file >= 2:
                    score += color_sign * doubled_rooks_bonus

        return score

    def __evaluate_threats_and_hanging_pieces(self, board: chess.Board) -> float:
        """
        Evaluates lightweight tactical static features: hanging pieces, pieces attacked by pawns,
        and higher-value pieces attacked by lower-value pieces. Returns a score from White's
        perspective (positive for White, negative for Black).
        """
        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            color_sign = 1 if color == chess.WHITE else -1
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, color):
                    attackers = board.attackers(not color, sq)
                    if not attackers:
                        continue

                    defenders = board.attackers(color, sq)
                    piece_value = PIECE_VALUES[piece_type]
                    penalty = 0.0

                    if not defenders:
                        penalty += min(self.HANGING_PIECE_PENALTY_LIMIT,
                                       self.HANGING_PIECE_VALUE_WEIGHT * piece_value)
                        if piece_type in [chess.ROOK, chess.QUEEN]:
                            penalty += self.HANGING_MAJOR_PIECE_EXTRA_PENALTY

                    pawn_attackers = [attacker_sq for attacker_sq in attackers
                                      if board.piece_at(attacker_sq)
                                      and board.piece_at(attacker_sq).piece_type == chess.PAWN]
                    if pawn_attackers:
                        penalty += min(self.PAWN_ATTACKED_PIECE_PENALTY_LIMIT,
                                       self.PAWN_ATTACKED_PIECE_VALUE_WEIGHT * piece_value)

                    weakest_attacker_value = min(
                        get_piece_value(board.piece_at(attacker_sq)) for attacker_sq in attackers)
                    if not pawn_attackers and weakest_attacker_value < piece_value:
                        penalty += min(self.LOWER_VALUE_ATTACK_PENALTY_LIMIT,
                                       self.LOWER_VALUE_ATTACK_WEIGHT * (piece_value - weakest_attacker_value))

                    score -= color_sign * penalty
        return score

    def __evaluate_endgame_king_activity(self, board: chess.Board, phase: float) -> float:
        """
        Rewards active kings in endgames: centralization, attacking enemy pawns, supporting own
        passed pawns, and stopping enemy passed pawns. Returns a score from White's perspective.
        """
        endgame_weight = 1.0 - phase
        if endgame_weight <= 0.0:
            return 0.0

        center_squares = self.CENTER_SQUARES

        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue

            activity = 0.0
            color_sign = 1 if color == chess.WHITE else -1

            center_distance = min(chess.square_distance(king_sq, center_sq) for center_sq in center_squares)
            activity += self.ENDGAME_KING_CENTER_BONUS * max(0, 4 - center_distance)

            enemy_pawn_pressure = 0.0
            for enemy_pawn_sq in board.pieces(chess.PAWN, not color):
                distance = chess.square_distance(king_sq, enemy_pawn_sq)
                enemy_pawn_pressure += self.ENDGAME_KING_ENEMY_PAWN_PRESSURE_BONUS * max(0, 4 - distance)
            activity += min(self.ENDGAME_KING_ENEMY_PAWN_PRESSURE_LIMIT, enemy_pawn_pressure)

            own_passed_pawns = self.__passed_pawns(board, color)
            for pawn_sq in own_passed_pawns:
                distance = chess.square_distance(king_sq, pawn_sq)
                activity += self.ENDGAME_KING_OWN_PASSER_SUPPORT_BONUS * max(0, 4 - distance)

            enemy_passed_pawns = self.__passed_pawns(board, not color)
            for pawn_sq in enemy_passed_pawns:
                distance_to_pawn = chess.square_distance(king_sq, pawn_sq)
                promotion_sq = chess.square(chess.square_file(pawn_sq), 0 if color == chess.WHITE else 7)
                distance_to_promotion = chess.square_distance(king_sq, promotion_sq)
                activity += self.ENDGAME_KING_ENEMY_PASSER_STOP_BONUS * max(
                    0, 4 - min(distance_to_pawn, distance_to_promotion))

            score += color_sign * activity * endgame_weight
        return score

    def __evaluate_mobility_and_activity(self, board: chess.Board) -> float:
        """
        Evaluates piece mobility and activity for both sides.
        Mobility: Number of attacked squares for each piece type (except pawns and kings),
        evaluated symmetrically for both colors without changing board.turn.
        Activity: Bonus for pieces on advanced ranks and controlling central squares.
        Returns a score (positive for White, negative for Black).
        """
        start = time.perf_counter() if self.debug else 0.0

        mobility_weights = self.MOBILITY_WEIGHTS
        activity_bonus = self.ACTIVITY_BONUS
        central_control_bonus = self.CENTRAL_CONTROL_BONUS
        central_squares = self.CENTRAL_SQUARES
        score = 0.0
        for color in [chess.WHITE, chess.BLACK]:
            color_sign = 1 if color == chess.WHITE else -1
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, color):
                    # Mobility: count attacked squares not occupied by friendly pieces. This remains
                    # symmetric and independent of whose turn it is. Do not mutate board.turn here:
                    # minimax relies on evaluate_board() preserving the exact board state it receives.
                    attacks = board.attacks(sq)
                    mobility = len(attacks & ~board.occupied_co[color])
                    score += color_sign * mobility_weights[piece_type] * mobility
                    # Activity: advanced rank, central placement and actual central control
                    rank = chess.square_rank(sq)
                    if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
                        score += color_sign * activity_bonus
                    if sq in central_squares:
                        score += color_sign * activity_bonus
                    central_control = sum(1 for central_sq in central_squares if central_sq in attacks)
                    score += color_sign * central_control_bonus * central_control
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times_evaluate_mobility_and_activity.append(elapsed)
            self.__eval_count_evaluate_mobility_and_activity += 1
            if self.__eval_count_evaluate_mobility_and_activity % 200 == 0:
                avg = sum(self.__eval_times_evaluate_mobility_and_activity) / len(
                    self.__eval_times_evaluate_mobility_and_activity)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} BoardEvaluatorTrad.__evaluate_mobility_and_activity() average time after {self.__eval_count_evaluate_mobility_and_activity} calls: {avg:.6f}s')
        return score

    def evaluate_board(self, board: chess.Board) -> float:
        start = time.perf_counter() if self.debug else 0.0

        if board.is_checkmate():
            result = -math.inf if board.turn == chess.WHITE else math.inf
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            result = 0.0
        else:
            phase = self.__get_game_phase(board)
            white_score = 0.0
            black_score = 0.0
            for piece_type in PIECE_VALUES.keys():
                for color in [chess.WHITE, chess.BLACK]:
                    squares = board.pieces(piece_type, color)
                    for sq in squares:
                        material_value = 0.0 if piece_type == chess.KING else PIECE_VALUES[piece_type]
                        value = material_value + self.__pst_value(piece_type, sq, color, phase)
                        if color == chess.WHITE:
                            white_score += value
                        else:
                            black_score += value
            pawn_structure_score = self.__evaluate_pawn_structure(board, phase)
            king_safety_score = phase * self.__evaluate_king_safety(board)
            minor_piece_score = self.__evaluate_minor_piece_features(board, phase)
            rook_activity_score = self.__evaluate_rook_activity(board, phase)
            threats_score = self.__evaluate_threats_and_hanging_pieces(board)
            endgame_king_activity_score = self.__evaluate_endgame_king_activity(board, phase)
            mobility_activity_score = self.__evaluate_mobility_and_activity(board)
            result = ((white_score - black_score) + pawn_structure_score + king_safety_score +
                      minor_piece_score + rook_activity_score + threats_score + endgame_king_activity_score +
                      mobility_activity_score)
        if self.debug:
            end = time.perf_counter()
            elapsed = end - start
            self.__eval_times.append(elapsed)
            self.__eval_count += 1
            if self.__eval_count % 200 == 0:
                avg = sum(self.__eval_times) / len(self.__eval_times)
                LOGGER.debug(
                    f'{"WHITE" if self.color else "BLACK"} BoardEvaluatorTrad.evaluate_board() average time after {self.__eval_count} calls: {avg:.6f}s')
        return result
