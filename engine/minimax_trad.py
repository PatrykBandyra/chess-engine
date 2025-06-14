import argparse
import math
import time
from typing import List

import chess
from chess.polyglot import ZobristHasher, POLYGLOT_RANDOM_ARRAY

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from player import Player


class MinimaxTrad(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args)
        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black
        self.color: chess.Color = color

        self.transposition_table = {}
        self.hasher = ZobristHasher(POLYGLOT_RANDOM_ARRAY)

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3.05,
            chess.BISHOP: 3.33,
            chess.ROOK: 5.63,
            chess.QUEEN: 9.5,
            chess.KING: 100_000
        }

        # Stores 2 moves per ply that caused beta cutoffs - indexed by ply (0 = root, 1 = depth-1, etc.)
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]

        # Stores scores for non-capture moves based on success - indexed by [from_square][to_square]
        self.history_heuristic_table = [[0] * 64 for _ in range(64)]

        # Approximate values for move ordering heuristics
        self.move_ordering_promotion_bonus = 1_500_000
        self.move_ordering_capture_bonus = 1_000_000
        self.move_ordering_killer_bonus = 750_000
        self.move_ordering_check_bonus = 500_000

    # Piece-square tables for both white and black (all values declared statically)
    PAWN_PSQ_WHITE = [  # Indexing: 0 - a1, 1 - b1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        5.0, 10.0, 10.0, -20.0, -20.0, 10.0, 10.0, 5.0,
        5.0, -5.0, -10.0, 0.0, 0.0, -10.0, -5.0, 5.0,
        0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0,
        5.0, 5.0, 10.0, 25.0, 25.0, 10.0, 5.0, 5.0,
        10.0, 10.0, 20.0, 30.0, 30.0, 20.0, 10.0, 10.0,
        50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    PAWN_PSQ_BLACK = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        10.0, 10.0, 20.0, 30.0, 30.0, 20.0, 10.0, 10.0,
        5.0, 5.0, 10.0, 25.0, 25.0, 10.0, 5.0, 5.0,
        0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0,
        5.0, -5.0, -10.0, 0.0, 0.0, -10.0, -5.0, 5.0,
        5.0, 10.0, 10.0, -20.0, -20.0, 10.0, 10.0, 5.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    KNIGHT_PSQ_WHITE = [
        -50.0, -40.0, -30.0, -30.0, -30.0, -30.0, -40.0, -50.0,
        -40.0, -20.0, 0.0, 0.0, 0.0, 0.0, -20.0, -40.0,
        -30.0, 0.0, 10.0, 15.0, 15.0, 10.0, 0.0, -30.0,
        -30.0, 5.0, 15.0, 20.0, 20.0, 15.0, 5.0, -30.0,
        -30.0, 0.0, 15.0, 20.0, 20.0, 15.0, 0.0, -30.0,
        -30.0, 5.0, 10.0, 15.0, 15.0, 10.0, 5.0, -30.0,
        -40.0, -20.0, 0.0, 5.0, 5.0, 0.0, -20.0, -40.0,
        -50.0, -40.0, -30.0, -30.0, -30.0, -30.0, -40.0, -50.0
    ]
    KNIGHT_PSQ_BLACK = [
        -50.0, -40.0, -30.0, -30.0, -30.0, -30.0, -40.0, -50.0,
        -40.0, -20.0, 0.0, 5.0, 5.0, 0.0, -20.0, -40.0,
        -30.0, 5.0, 10.0, 15.0, 15.0, 10.0, 5.0, -30.0,
        -30.0, 0.0, 15.0, 20.0, 20.0, 15.0, 0.0, -30.0,
        -30.0, 5.0, 15.0, 20.0, 20.0, 15.0, 5.0, -30.0,
        -30.0, 0.0, 10.0, 15.0, 15.0, 10.0, 0.0, -30.0,
        -40.0, -20.0, 0.0, 0.0, 0.0, 0.0, -20.0, -40.0,
        -50.0, -40.0, -30.0, -30.0, -30.0, -30.0, -40.0, -50.0
    ]
    BISHOP_PSQ_WHITE = [
        -20.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -20.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -10.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0, -10.0,
        -10.0, 5.0, 5.0, 10.0, 10.0, 5.0, 5.0, -10.0,
        -10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, -10.0,
        -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -10.0,
        -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, -10.0,
        -20.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -20.0
    ]
    BISHOP_PSQ_BLACK = [
        -20.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -20.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, -10.0,
        -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -10.0,
        -10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, -10.0,
        -10.0, 5.0, 5.0, 10.0, 10.0, 5.0, 5.0, -10.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -20.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -20.0
    ]
    ROOK_PSQ_WHITE = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0
    ]
    ROOK_PSQ_BLACK = [
        0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0,
        5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    QUEEN_PSQ_WHITE = [
        -20.0, -10.0, -10.0, -5.0, -5.0, -10.0, -10.0, -20.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -10.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, -10.0,
        -5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, -5.0,
        0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, -5.0,
        -10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, -10.0,
        -10.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -20.0, -10.0, -10.0, -5.0, -5.0, -10.0, -10.0, -20.0
    ]
    QUEEN_PSQ_BLACK = [
        -20.0, -10.0, -10.0, -5.0, -5.0, -10.0, -10.0, -20.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -10.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, -10.0,
        0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, -5.0,
        -5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, -5.0,
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0,
        -20.0, -10.0, -10.0, -5.0, -5.0, -10.0, -10.0, -20.0
    ]
    KING_PSQ_WHITE = [
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -20.0, -30.0, -30.0, -40.0, -40.0, -30.0, -30.0, -20.0,
        -10.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -10.0,
        20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0,
        20.0, 30.0, 10.0, 0.0, 0.0, 10.0, 30.0, 20.0
    ]
    KING_PSQ_BLACK = [
        20.0, 30.0, 10.0, 0.0, 0.0, 10.0, 30.0, 20.0,
        20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0,
        -10.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -10.0,
        -20.0, -30.0, -30.0, -40.0, -40.0, -30.0, -30.0, -20.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0,
        -30.0, -40.0, -40.0, -50.0, -50.0, -40.0, -40.0, -30.0
    ]

    PIECE_SQUARE_TABLES_WHITE = {
        chess.PAWN: PAWN_PSQ_WHITE,
        chess.KNIGHT: KNIGHT_PSQ_WHITE,
        chess.BISHOP: BISHOP_PSQ_WHITE,
        chess.ROOK: ROOK_PSQ_WHITE,
        chess.QUEEN: QUEEN_PSQ_WHITE,
        chess.KING: KING_PSQ_WHITE
    }
    PIECE_SQUARE_TABLES_BLACK = {
        chess.PAWN: PAWN_PSQ_BLACK,
        chess.KNIGHT: KNIGHT_PSQ_BLACK,
        chess.BISHOP: BISHOP_PSQ_BLACK,
        chess.ROOK: ROOK_PSQ_BLACK,
        chess.QUEEN: QUEEN_PSQ_BLACK,
        chess.KING: KING_PSQ_BLACK
    }

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        start_time: float = time.perf_counter()

        # Clearing
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]
        self.history_heuristic_table = [[0] * 64 for _ in range(64)]

        internal_board: chess.Board = board.copy()

        legal_moves: List[chess.Move] = list(internal_board.legal_moves)
        if not legal_moves:
            return
        ordered_moves: List[chess.Move] = self.__order_moves(internal_board, legal_moves, ply=0)

        best_move: chess.Move | None = None
        best_value: float = -math.inf if self.color == chess.WHITE else math.inf
        is_maximizing: bool = self.color == chess.WHITE
        alpha: float = -math.inf
        beta: float = math.inf

        for move in ordered_moves:
            internal_board.push(move)
            board_value = self.__minimax_alphabeta(internal_board, self.depth - 1, alpha, beta, not is_maximizing)
            internal_board.pop()

            if is_maximizing:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
                alpha = max(alpha, board_value)
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
                beta = min(beta, board_value)

        board.push(best_move)

        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        LOGGER.info(
            f'MINIMAX-TRAD; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_move.uci()}; ' +
            f'value: {best_value}'
        )

    def __order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int) -> List[chess.Move]:
        """
        Orders moves based on heuristics to improve alpha-beta pruning efficiency.
        Prioritizes: Promotions, Captures (MVV-LVA), Killer Moves, History Heuristic, Checks, Quiet Moves.

        Args:
            board: The current board state
            moves: A list of legal moves to order
            ply: The current ply depth (0 for root, 1 for the next level, etc.).

        Returns:
            A list of moves sorted from potentially best to worst
        """
        move_scores = []
        killers = self.killer_moves[ply] if 0 <= ply < len(self.killer_moves) else [None, None]

        for move in moves:
            score: float = 0
            is_promotion: bool = move.promotion is not None
            is_capture: bool = board.is_capture(move)

            # 1. Promotions
            if is_promotion:
                score += (self.move_ordering_promotion_bonus +
                          self.__get_piece_value(chess.Piece(move.promotion, board.turn)))

            # 2. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor)
            elif is_capture:  # If both promotion and capture, then prefer promotion score
                move_piece: chess.Piece = board.piece_at(move.from_square)
                captured_piece: chess.Piece | None = board.piece_at(move.to_square)
                captured_piece_value: float = 0
                if captured_piece:
                    captured_piece_value = self.__get_piece_value(captured_piece)
                elif board.is_en_passant(move):
                    captured_piece_value = self.piece_values[chess.PAWN]

                aggressor_value: float = self.__get_piece_value(move_piece)
                score += self.move_ordering_capture_bonus + (captured_piece_value - aggressor_value / 10)

            # 3. Quiet Moves (apply Killer and History Heuristics)
            else:
                is_killer: bool = move == killers[0] or move == killers[1]
                if is_killer:
                    score += self.move_ordering_killer_bonus

                history_score = self.history_heuristic_table[move.from_square][move.to_square]
                score += history_score

                # 4. Checks (check if the move puts the opponent in check)
                board.push(move)
                is_check: bool = board.is_check()
                board.pop()

                if is_check:
                    # Add check bonus, potentially less if already highly scored
                    # Check bonus might be less relevant now with history heuristic also boosting good quiet checks
                    if score < self.move_ordering_check_bonus * 1.5:  # Avoid excessive bonus stacking
                        score += self.move_ordering_check_bonus

            move_scores.append((score, move))

        # Sort moves in descending order of score
        move_scores.sort(key=lambda item: item[0], reverse=True)

        # Return just the moves in the sorted order
        return [move for _, move in move_scores]

    def __minimax_alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                            maximizing_player: bool) -> float:
        current_ply: int = self.depth - depth
        original_alpha: float = alpha  # Store original alpha for TT flag and history update
        original_beta: float = beta  # Store original beta for history update

        board_hash: int = self.hasher.hash_board(board)
        tt_entry = self.transposition_table.get(board_hash)

        # Transposition Table Lookup
        if tt_entry and tt_entry['d'] >= depth:
            if tt_entry['f'] == 'E':  # Flag: Exact
                return tt_entry['v']
            elif tt_entry['f'] == 'L':  # Flag: Lower bound
                alpha = max(alpha, tt_entry['v'])
            elif tt_entry['f'] == 'U':  # Flag: Upper bound
                beta = min(beta, tt_entry['v'])
            if beta <= alpha:
                return tt_entry['v']

        if depth == 0 or board.is_game_over():
            return self.__evaluate_board(board)

        legal_moves: List[chess.Move] = list(board.legal_moves)
        ordered_moves: List[chess.Move] = self.__order_moves(board, legal_moves, ply=current_ply)
        best_move: chess.Move | None = None

        if maximizing_player:
            max_evaluation = -math.inf
            for move in ordered_moves:
                board.push(move)
                evaluation = self.__minimax_alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()

                if evaluation > max_evaluation:
                    max_evaluation = evaluation
                    best_move = move

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.__store_killer_move(current_ply, move)
                        self.__update_history_score(move, depth)
                    break

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually raised alpha (if it was quiet)
                # Check if max_evaluation actually improved alpha from the original value
                if max_evaluation > original_alpha:
                    self.__update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if max_evaluation <= original_alpha:
                flag = 'U'  # Upper bound
            elif max_evaluation >= beta:
                flag = 'L'  # Lower bound
            # Store in Transposition Table
            if not tt_entry or depth >= tt_entry['d']:
                self.transposition_table[board_hash] = {'v': max_evaluation, 'd': depth, 'f': flag}

            return max_evaluation

        else:
            min_eval = math.inf
            for move in ordered_moves:
                board.push(move)
                evaluation = self.__minimax_alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()

                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move

                beta = min(beta, evaluation)
                if beta <= alpha:
                    if not board.is_capture(move) and move.promotion is None:
                        self.__store_killer_move(current_ply, move)
                        self.__update_history_score(move, depth)
                    break

            # After checking all moves, if no cutoff occurred, update history for the best move found
            if beta > alpha and best_move and not board.is_capture(best_move) and best_move.promotion is None:
                # Update history for the move that actually lowered beta (if it was quiet)
                # Check if min_eval actually improved beta from the original value
                if min_eval < original_beta:
                    self.__update_history_score(best_move, depth)

            # Determine TT flag
            flag: str = 'E'  # Exact
            if min_eval >= beta:
                flag = 'L'  # Lower bound
            elif min_eval <= alpha:
                flag = 'U'  # Upper bound
            # Store in Transposition Table
            if not tt_entry or depth >= tt_entry['d']:
                self.transposition_table[board_hash] = {'v': min_eval, 'd': depth, 'f': flag}

            return min_eval

    def __get_piece_value(self, piece: chess.Piece | None) -> float:
        """Safely gets the value of a piece, returning 0 if None"""
        return self.piece_values.get(piece.piece_type, 0) if piece else 0

    def __store_killer_move(self, ply: int, move: chess.Move):
        """Stores a killer move for a given ply, keeping the two best"""
        if 0 <= ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move

    def __update_history_score(self, move: chess.Move, depth: int):
        """Increases the history score for a successful quiet move"""
        if move.promotion is None:  # Captures are implicitly excluded by where this is called
            bonus = depth * depth  # Weight bonus by depth squared
            self.history_heuristic_table[move.from_square][move.to_square] += bonus

    def __evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -math.inf
            else:
                return math.inf
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0

        white_material = 0.0
        black_material = 0.0
        white_positional = 0.0
        black_positional = 0.0

        for piece_type in self.piece_values.keys():
            white_squares = board.pieces(piece_type, chess.WHITE)
            black_squares = board.pieces(piece_type, chess.BLACK)
            white_material += len(white_squares) * self.piece_values[piece_type]
            black_material += len(black_squares) * self.piece_values[piece_type]

            pst_white = self.PIECE_SQUARE_TABLES_WHITE[piece_type]
            pst_black = self.PIECE_SQUARE_TABLES_BLACK[piece_type]
            for sq in white_squares:
                white_positional += pst_white[sq]
            for sq in black_squares:
                black_positional += pst_black[sq]

        # Scale positional score to match material scale
        positional_scale = 0.01
        white_score = white_material + positional_scale * white_positional
        black_score = black_material + positional_scale * black_positional
        return white_score - black_score
