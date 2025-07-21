import logging

import chess

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
GREEN = (204, 255, 153)
GOLDEN = (253, 220, 92)

BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_HEIGHT // BOARD_SIZE

PIECE_NAME_TO_IMAGE_NAME = {
    'p': 'black-pawn', 'r': 'black-rook', 'n': 'black-knight', 'b': 'black-bishop', 'q': 'black-queen',
    'k': 'black-king',
    'P': 'white-pawn', 'R': 'white-rook', 'N': 'white-knight', 'B': 'white-bishop', 'Q': 'white-queen',
    'K': 'white-king'
}

STOCKFISH_PATH = '../stockfish_ai/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
LOGGER.addHandler(stream_handler)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.05,
    chess.BISHOP: 3.33,
    chess.ROOK: 5.63,
    chess.QUEEN: 9.5,
    chess.KING: 100_000
}


def get_piece_value(piece: chess.Piece | None) -> float:
    """Safely gets the value of a piece, returning 0 if None"""
    return PIECE_VALUES.get(piece.piece_type, 0) if piece else 0
