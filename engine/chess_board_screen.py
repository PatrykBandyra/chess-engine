import threading
from typing import Dict, Tuple, List

import chess
import pygame as pg

from constants import SCREEN_WIDTH, SCREEN_HEIGHT, BOARD_SIZE, SQUARE_SIZE, WHITE, BLACK, PIECE_NAME_TO_IMAGE_NAME, \
    GREEN, GOLDEN


class ChessBoardScreen:
    def __init__(self, board: chess.Board, screen_ready_event: threading.Event):
        self.new_move: str = ''
        self.should_get_new_move = False
        self.new_move_ready_event = threading.Event()

        self.board = board
        self.screen_ready_event = screen_ready_event
        pg.init()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pg.display.set_caption('Chess Engine')

        self.piece_images = self.__load_pieces()

        self.selected_piece_square: chess.Square | None = None
        self.possible_moves: List[chess.Move] = []
        self.highlighted_squares: List[str] = []

    @staticmethod
    def __load_pieces() -> Dict[str, pg.Surface]:
        return {piece_name: pg.image.load(f'assets/{image_name}.png') for piece_name, image_name in
                PIECE_NAME_TO_IMAGE_NAME.items()}

    def __draw_chess_board(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pg.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        for square_name in self.highlighted_squares:
            col = ord(square_name[0]) - ord('a')
            row = (BOARD_SIZE - 1) - int(square_name[1]) + 1
            pg.draw.rect(self.screen, GREEN, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        if self.selected_piece_square is not None:
            col = self.selected_piece_square % BOARD_SIZE
            row = (BOARD_SIZE - 1) - (self.selected_piece_square // BOARD_SIZE)
            pg.draw.rect(self.screen, GOLDEN, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def __draw_chess_pieces(self) -> None:
        for square, piece in self.board.piece_map().items():
            row = (BOARD_SIZE - 1) - (square // BOARD_SIZE)
            col = square % BOARD_SIZE
            piece_symbol = piece.symbol()
            piece_image = self.piece_images[piece_symbol]

            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

            piece_rect = piece_image.get_rect(center=(center_x, center_y))

            self.screen.blit(piece_image, piece_rect)

    def run(self):
        self.screen_ready_event.set()
        running = True
        while running:
            for event in pg.event.get():
                if self.should_get_new_move and event.type == pg.MOUSEBUTTONDOWN:
                    pos: Tuple[int, int] = pg.mouse.get_pos()
                    self.__handle_mouse_click(pos)
                if event.type == pg.QUIT:
                    running = False

            self.__draw_chess_board()
            self.__draw_chess_pieces()

            pg.display.flip()
        pg.quit()

    def __handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        clicked_square = self.__get_square_index_from_pos(pos)
        if clicked_square is not None:
            if self.selected_piece_square is None:
                piece = self.board.piece_at(clicked_square)
                if piece and (
                        (self.board.turn == chess.WHITE and piece.color == chess.WHITE) or
                        (self.board.turn == chess.BLACK and piece.color == chess.BLACK)
                ):
                    self.selected_piece_square = clicked_square
                    self.possible_moves = [
                        move for move in self.board.legal_moves if move.from_square == clicked_square
                    ]
                    self.highlighted_squares = [chess.square_name(move.to_square) for move in self.possible_moves]
                else:
                    self.highlighted_squares = []
            else:
                target_square = clicked_square
                move = chess.Move(self.selected_piece_square, target_square)
                if move in self.possible_moves:
                    self.new_move = move.uci()
                    self.should_get_new_move = False
                    self.new_move_ready_event.set()

                self.selected_piece_square = None
                self.possible_moves = []
                self.highlighted_squares = []

    @staticmethod
    def __get_square_index_from_pos(pos: Tuple[int, int]) -> chess.Square | None:
        col = pos[0] // SQUARE_SIZE
        row = (SCREEN_HEIGHT - pos[1]) // SQUARE_SIZE
        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
            return row * BOARD_SIZE + col
        return None

    def get_move_uci(self) -> str:
        self.should_get_new_move = True
        self.new_move_ready_event.wait()
        self.new_move_ready_event.clear()
        return self.new_move
