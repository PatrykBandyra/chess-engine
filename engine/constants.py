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
