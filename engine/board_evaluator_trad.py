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

    def __get_game_phase(self, board: chess.Board) -> float:
        """
        Returns a phase value in [0, 1]: 1.0 = middlegame, 0.0 = endgame, interpolated by non-pawn material.
        """
        non_pawn_material = 0.0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            non_pawn_material += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            non_pawn_material += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        endgame_threshold = 20.0
        middlegame_threshold = 67.0
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

        The PST arrays are stored in the usual chess-engine order from rank 8 to rank 1.
        python-chess squares are indexed from a1 to h8, so White pieces must be mirrored
        before indexing. Black pieces can use the square index directly, giving symmetric
        values for equivalent White/Black placements.
        """
        pst_index = chess.square_mirror(square) if color == chess.WHITE else square
        pst_mid = self.PIECE_SQUARE_TABLE_MID[piece_type][pst_index]
        pst_end = self.PIECE_SQUARE_TABLE_END[piece_type][pst_index]
        return 0.01 * (phase * pst_mid + (1 - phase) * pst_end)

    def __evaluate_pawn_structure(self, board: chess.Board, phase: float) -> float:
        """
        Evaluates pawn structure for both sides, considering doubled, isolated, and passed pawns.
        Returns a score (positive for White, negative for Black).
        """
        if self.debug:
            start = time.perf_counter()

        endgame_weight = 1.0 - phase
        passed_pawn_bonus_by_advancement = [0.0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 0.0]

        def is_passed_pawn(square: chess.Square, color: chess.Color) -> bool:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            for opp_sq in board.pieces(chess.PAWN, not color):
                opp_file = chess.square_file(opp_sq)
                opp_rank = chess.square_rank(opp_sq)
                if abs(opp_file - file) <= 1:
                    if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                        return False
            return True

        def evaluate_passed_pawn(square: chess.Square, color: chess.Color,
                                 pawns: chess.SquareSet, passed_pawns: list[chess.Square]) -> float:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            advancement = rank if color == chess.WHITE else 7 - rank
            bonus = passed_pawn_bonus_by_advancement[advancement] * (1.0 + 0.5 * endgame_weight)

            forward_rank = rank + (1 if color == chess.WHITE else -1)
            if 0 <= forward_rank < 8:
                blocker_sq = chess.square(file, forward_rank)
                if board.piece_at(blocker_sq) is not None:
                    bonus *= 0.55

            friendly_pawn_attackers = board.attackers(color, square) & pawns
            if friendly_pawn_attackers:
                bonus += 0.15 + 0.10 * endgame_weight
            elif board.attackers(color, square):
                bonus += 0.05

            if any(other_sq != square and abs(chess.square_file(other_sq) - file) == 1 for other_sq in passed_pawns):
                bonus += 0.15 + 0.10 * endgame_weight

            own_king_sq = board.king(color)
            enemy_king_sq = board.king(not color)
            if advancement >= 4 and own_king_sq is not None and enemy_king_sq is not None:
                promotion_sq = chess.square(file, 7 if color == chess.WHITE else 0)
                own_king_distance = chess.square_distance(own_king_sq, promotion_sq)
                enemy_king_distance = chess.square_distance(enemy_king_sq, promotion_sq)
                king_race_bonus = 0.04 * (enemy_king_distance - own_king_distance) * endgame_weight
                bonus += max(-0.15, min(0.15, king_race_bonus))

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
            passed_pawns = [sq for sq in pawns if is_passed_pawn(sq, color)]
            passed_bonus = sum(evaluate_passed_pawn(sq, color, pawns, passed_pawns) for sq in passed_pawns)
            # Penalties and bonuses
            penalty = -0.25 * doubled_penalty - 0.25 * isolated_penalty
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
                        attack_penalty += get_piece_value(piece) * 0.1
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

        bishop_pair_bonus = (0.25 + 0.15 * openness) * (0.90 + 0.10 * endgame_weight)

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
        activity_scale = 0.90 + 0.20 * endgame_weight

        open_file_bonus = 0.25 * activity_scale
        semi_open_file_bonus = 0.15 * activity_scale
        seventh_rank_bonus = 0.25 * activity_scale
        doubled_rooks_bonus = 0.10 * activity_scale

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
                        penalty += min(0.75, 0.15 * piece_value)
                        if piece_type in [chess.ROOK, chess.QUEEN]:
                            penalty += 0.10

                    pawn_attackers = [attacker_sq for attacker_sq in attackers
                                      if board.piece_at(attacker_sq)
                                      and board.piece_at(attacker_sq).piece_type == chess.PAWN]
                    if pawn_attackers:
                        penalty += min(0.75, 0.10 * piece_value)

                    weakest_attacker_value = min(
                        get_piece_value(board.piece_at(attacker_sq)) for attacker_sq in attackers)
                    if weakest_attacker_value < piece_value:
                        penalty += min(0.60, 0.08 * (piece_value - weakest_attacker_value))

                    score -= color_sign * penalty
        return score

    def __evaluate_mobility_and_activity(self, board: chess.Board) -> float:
        """
        Evaluates piece mobility and activity for both sides.
        Mobility: Number of attacked squares for each piece type (except pawns and kings),
        evaluated symmetrically for both colors without changing board.turn.
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
        central_control_bonus = 0.03
        central_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
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
            mobility_activity_score = self.__evaluate_mobility_and_activity(board)
            result = ((white_score - black_score) + pawn_structure_score + king_safety_score +
                      minor_piece_score + rook_activity_score + threats_score + mobility_activity_score)
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
