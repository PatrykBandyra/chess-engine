from enum import Enum


class PlayerType(Enum):
    HUMAN = 'Human'
    MINI_MAX_TRAD = 'Minimax Trad'
    MINI_MAX_NN = 'Minimax NN'
    MONTE_CARLO_TREE_SEARCH = 'MCTS Heuristics Rollout'
    STOCKFISH = 'Stockfish'
