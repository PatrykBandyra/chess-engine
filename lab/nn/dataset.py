import os
from typing import Tuple

import chess
import pandas as pd
import torch
from torch.utils.data import Dataset

from download_dataset import download_chess_evaluations_dataset


def _algebraic2index(c: int) -> Tuple[int, int]:
    return 8 - int(c[1]), ord(c[0]) - 97


class ChessPositionEvaluationsDataset(Dataset):
    def __init__(self):
        self.csv_file_name = 'chessData.csv'
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        self.algebraic2index = _algebraic2index

        if not os.path.isfile(self.csv_file_name):
            download_chess_evaluations_dataset()
        self.df = pd.read_csv(self.csv_file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int | torch.Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fen = self.df.iloc[idx]['FEN']
        evaluation_str = self.df.iloc[idx]['Evaluation']

        if evaluation_str.startswith('#'):  # Forced checkmate in X moves
            if evaluation_str[1] == '-':  # Black checkmating
                evaluation = -10000.0
            else:  # White checkmating
                evaluation = 10000.0
        else:
            evaluation = float(evaluation_str)

        board_tensor, active_player, half_move_clock = self.fen_to_tensor(fen)

        sample = {
            'board_tensor': board_tensor,  # 13x8x8 tensor
            'active_player': active_player,  # For conditional normalization
            'half_move_clock': half_move_clock,  # To be passed to FC layers
            'evaluation': torch.tensor(evaluation, dtype=torch.float32)  # Evaluation as target
        }

        return sample

    def fen_to_tensor(self, fen):
        """ Converts fen to 13x8x8 tensor (8x8 board for all 12 pieces + en passant) """
        board = chess.Board(fen=fen)
        board_tensor = torch.zeros((13, 8, 8), dtype=torch.float32)  # 13 channels, 8x8 board

        d = {'K': (7, 6), 'Q': (7, 2), 'k': (0, 6), 'q': (0, 2)}  # mapping dictionary for castling squares

        # Fill the tensor based on piece positions
        for square, piece in board.piece_map().items():
            row = 7 - chess.square_rank(square)  # Flip rows
            col = chess.square_file(square)
            index = self.piece_to_index[piece.symbol()]
            board_tensor[index, row, col] = 1  # Mark piece with 1 in the appropriate channel

        split = fen.split(' ')
        active_player = 1 if split[1] == 'w' else 0  # 1 for white, 0 for black
        en_passant = split[3]
        castle = split[2]
        half_move_clock = int(split[4]) / 100.0  # Normalize by 100
        # Encoding en passant and castling information on the same channel
        if en_passant != '-':
            r, c = self.algebraic2index(en_passant)
            board_tensor[12, r, c] = 1  # Encode en_passant square with 1 if there is any
        if castle != '-':
            for piece in castle:
                r, c = d[piece]
                board_tensor[12, r, c] = 1  # Castling squares

        return board_tensor, active_player, half_move_clock
