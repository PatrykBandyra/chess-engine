from enum import Enum


class PlayerType(Enum):
    HUMAN = 'H'
    MINIMAX_TRAD = 'MINIMAX_TRAD'
    MINIMAX_NN = 'MINIMAX_NN'
    MCTS_TRAD = 'MCTS_TRAD'
    MCTS_NN = 'MCTS_NN'
    STOCKFISH = 'STOCKFISH'
