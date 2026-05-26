import abc
import argparse

import chess

PHASE_PIECE_WEIGHTS = {chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0}
PHASE_ENDGAME_THRESHOLD = 20.0
PHASE_MIDDLEGAME_THRESHOLD = 67.0


class BoardEvaluator(abc.ABC):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        self.debug: bool = args.debug
        self.color: chess.Color = color

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

    def get_game_phase(self, board: chess.Board) -> float:
        non_pawn = 0.0
        for pt, w in PHASE_PIECE_WEIGHTS.items():
            non_pawn += (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * w
        if non_pawn >= PHASE_MIDDLEGAME_THRESHOLD:
            return 1.0
        if non_pawn <= PHASE_ENDGAME_THRESHOLD:
            return 0.0
        return (non_pawn - PHASE_ENDGAME_THRESHOLD) / (PHASE_MIDDLEGAME_THRESHOLD - PHASE_ENDGAME_THRESHOLD)
