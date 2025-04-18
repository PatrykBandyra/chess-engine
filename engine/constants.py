import logging

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

LOG_FILE_NAME = 'out/logs.log'

STOCKFISH_PATH = '../stockfish_ai/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(FORMATTER)
LOGGER.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(FORMATTER)
LOGGER.addHandler(file_handler)
