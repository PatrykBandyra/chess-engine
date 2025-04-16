from typing import Dict, Tuple, Optional

import chess
import pygame as pg
import threading

from constants import SCREEN_WIDTH, SCREEN_HEIGHT, BOARD_SIZE, SQUARE_SIZE, WHITE, BLACK, PIECE_NAME_TO_IMAGE_NAME


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

        self.selected_piece_square: Optional[chess.Square] = None
        self.possible_moves: list[chess.Move] = []
        self.highlighted_squares: list[str] = []

        self.screen_ready_event.set()

        self.__run()

    @staticmethod
    def __load_pieces() -> Dict[str, pg.Surface]:
        return {piece_name: pg.transform.scale(pg.image.load(f'assets/{image_name}.png'), (SQUARE_SIZE, SQUARE_SIZE))
                for piece_name, image_name in
                PIECE_NAME_TO_IMAGE_NAME.items()}

    def __draw_chess_board(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pg.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        # Highlight possible moves
        for square_name in self.highlighted_squares:
            col = ord(square_name[0]) - ord('a')
            row = (BOARD_SIZE - 1) - int(square_name[1]) + 1
            pg.draw.rect(self.screen, (204, 255, 153), (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        # Highlight selected piece
        if self.selected_piece_square is not None:
            col = self.selected_piece_square % BOARD_SIZE
            row = (BOARD_SIZE - 1) - (self.selected_piece_square // BOARD_SIZE)
            pg.draw.rect(self.screen, (173, 216, 230), (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def __draw_chess_pieces(self) -> None:
        for square, piece in self.board.piece_map().items():
            row = (BOARD_SIZE - 1) - (square // BOARD_SIZE)
            col = square % BOARD_SIZE
            piece_symbol = piece.symbol()
            piece_image = self.piece_images[piece_symbol]

            self.screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def __get_square_from_pos(self, pos: Tuple[int, int]) -> Optional[chess.Square]:
        """Converts pixel coordinates to a chess square index."""
        col = pos[0] // SQUARE_SIZE
        row = (SCREEN_HEIGHT - pos[1]) // SQUARE_SIZE
        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
            return row * BOARD_SIZE + col
        return None

    def __handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        clicked_square = self.__get_square_from_pos(pos)
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
                    self.board.push(move)
                    self.selected_piece_square = None
                    self.possible_moves = []
                    self.highlighted_squares = []
                else:
                    # Deselect if clicking an invalid target or empty square
                    self.selected_piece_square = None
                    self.possible_moves = []
                    self.highlighted_squares = []

    def __run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    pos = pg.mouse.get_pos()
                    self.__handle_mouse_click(pos)

            self.__draw_chess_board()
            self.__draw_chess_pieces()

            pg.display.flip()
        pg.quit()

    def make_move_uci(self, move: str) -> None:
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
            else:
                print(f"Invalid move received: {move}")
        except ValueError:
            print(f"Invalid UCI format: {move}")
        self.selected_piece_square = None
        self.possible_moves = []
        self.highlighted_squares = []

    def get_move_uci(self) -> str:
        self.should_get_new_move = True
        self.new_move_ready_event.wait()
        self.new_move_ready_event.clear()
        return self.new_move

if __name__ == '__main__':
    # Example usage:
    initial_board = chess.Board()
    screen_ready = threading.Event()
    chess_screen = ChessBoardScreen(initial_board, screen_ready)

    # You can interact with the chess_screen object here if needed
    # For example, to make a move programmatically:
    # chess_screen.make_move_uci("e2e4")

    # The game loop in __run__ will handle user interaction.