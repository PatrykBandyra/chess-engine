from enum import Enum


class PlayerType(Enum):
    HUMAN = 'H'
    MINIMAX_TRAD = 'MINIMAX_TRAD'
    MINIMAX_NN = 'MINIMAX_NN'
    MONTE_CARLO_TREE_SEARCH = 'MONTE_CARLO_TREE_SEARCH'
    STOCKFISH = 'STOCKFISH'
