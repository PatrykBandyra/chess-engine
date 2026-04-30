import abc
import argparse
import math
import time
from collections import OrderedDict
from typing import List, Optional

import chess
import chess.polyglot

from chess_board_screen import ChessBoardScreen
from constants import LOGGER
from opening_book import OpeningBook
from order_moves_mcts import OrderMovesMCTS
from player import Player


class MCTSNode:
    __slots__ = ('board', 'parent', 'move', 'children', 'untried_moves',
                 'visits', 'value', 'player', 'is_terminal', '_moves_sorted')

    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, _copy: bool = True):
        self.board = board.copy(stack=False) if _copy else board
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.untried_moves = list(board.legal_moves)
        self.visits = 0
        self.value = 0.0
        self.player = board.turn
        self.is_terminal: bool = board.is_game_over()
        self._moves_sorted: bool = False

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=math.sqrt(2)):
        """
        Selects the best child node based on the UCT (Upper Confidence Bound for Trees) formula.
        Values are normalized to [0, 1], so the standard c = sqrt(2) is used.
        Single-pass with precomputed log(parent.visits).
        """
        log_parent_visits = math.log(self.visits)
        best = self.children[0]
        best_score = (best.value / best.visits) + c_param * math.sqrt(log_parent_visits / best.visits)
        for child in self.children[1:]:
            score = (child.value / child.visits) + c_param * math.sqrt(log_parent_visits / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)


class MCTS(Player):
    EVAL_CACHE_MAX_SIZE = 500_000
    # Threshold for sigmoid input: math.exp overflows for arguments > ~709.78.
    # Beyond this raw value the sigmoid saturates to 0/1 anyway, so we short-circuit
    # to avoid OverflowError. With finite mate scores from BoardEvaluatorNN
    # (~±1_000_000), the unguarded sigmoid would crash on mates for Black (raw ~ -1e6).
    SIGMOID_RAW_LIMIT = 2800.0  # ~ 4.0 * 700, leaves margin below math.exp overflow

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_mcts = OrderMovesMCTS(args, color)

        self.mcts_time_budget: float = args.mcts_time_white if color == chess.WHITE else args.mcts_time_black
        self.__eval_cache: OrderedDict[int, float] = OrderedDict()
        self.__root: Optional[MCTSNode] = None
        self.__last_best_child: Optional[MCTSNode] = None

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        move_number: int = board.fullmove_number
        start_time: float = time.perf_counter()
        if self.opening_book.use_opening_book and self.opening_book.is_opening:
            if self.opening_book.make_move(board, start_time):
                self.__root = None
                self.__last_best_child = None
                return  # Move already made from an opening book
        LOGGER.info(
            f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; '
            f'starting search; time_budget: {self.mcts_time_budget:.6f}s'
        )
        self.__run_mcts(board, start_time, move_number)

    def __run_mcts(self, board: chess.Board, start_time: float, move_number: int) -> None:
        root = self.__get_or_create_root(board)
        end_time = time.perf_counter() + self.mcts_time_budget
        iterations = 0
        while True:
            if iterations & 127 == 0 and time.perf_counter() >= end_time:
                break
            node = self.__select(root)
            if node.is_terminal and node.visits > 0:
                # Already evaluated terminal node, skip simulate/backpropagate.
                # Increment iterations anyway so the periodic time check (every 128
                # iterations) still fires; otherwise UCT can keep selecting the same
                # winning terminal forever without iterations ever advancing.
                iterations += 1
                continue  # already evaluated terminal node, skip
            if node.untried_moves:
                node = self.__expand(node)
            value = self.__simulate(node)
            self.__backpropagate(node, value)
            iterations += 1
        if root.children:
            best_child = root.most_visited_child()
            board.push(best_child.move)
            self.__root = root
            self.__last_best_child = best_child
            duration = time.perf_counter() - start_time
            mean_value = best_child.value / best_child.visits if best_child.visits > 0 else 0.0
            LOGGER.info(
                f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; time: {duration:.6f}s; move: {best_child.move.uci()}; value: {mean_value:.4f}; visits: {best_child.visits}; iterations: {iterations}'
            )
        else:
            self.__root = None
            self.__last_best_child = None
            LOGGER.warning(f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; No valid move found. Skipping push.')

    def __get_or_create_root(self, board: chess.Board) -> MCTSNode:
        if self.__last_best_child is not None:
            opponent_move = board.move_stack[-1] if board.move_stack else None
            if opponent_move:
                for child in self.__last_best_child.children:
                    if child.move == opponent_move:
                        child.parent = None  # detach from old tree for GC
                        self.__root = None
                        self.__last_best_child = None
                        return child
        # Fallback: create a new root
        self.__root = None
        self.__last_best_child = None
        return MCTSNode(board)

    def __select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal and node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def __expand(self, node: MCTSNode) -> MCTSNode:
        if not node._moves_sorted:
            sorted_moves = self.order_moves_mcts.order_moves(node.board, node.untried_moves, None)
            sorted_moves.reverse()  # best moves at end for O(1) pop
            node.untried_moves = sorted_moves
            node._moves_sorted = True
        move = node.untried_moves.pop()  # O(1), pops best remaining move
        next_board = node.board.copy(stack=False)
        next_board.push(move)
        child_node = MCTSNode(next_board, parent=node, move=move, _copy=False)
        node.children.append(child_node)
        return child_node

    def __simulate(self, node: MCTSNode) -> float:
        board_hash = chess.polyglot.zobrist_hash(node.board)
        if board_hash in self.__eval_cache:
            self.__eval_cache.move_to_end(board_hash)
            v = self.__eval_cache[board_hash]
        else:
            raw = self.evaluate_board(node.board)
            # Sigmoid normalization to [0, 1] (white perspective). Saturate explicitly
            # outside the safe range for math.exp to avoid OverflowError on finite
            # mate scores returned by BoardEvaluatorNN (raw ~ ±1_000_000).
            if raw >= self.SIGMOID_RAW_LIMIT:
                v = 1.0
            elif raw <= -self.SIGMOID_RAW_LIMIT:
                v = 0.0
            else:
                v = 1.0 / (1.0 + math.exp(-raw / 4.0))
            self.__eval_cache[board_hash] = v
            if len(self.__eval_cache) > self.EVAL_CACHE_MAX_SIZE:
                self.__eval_cache.popitem(last=False)  # evict least recently used
        return v if node.player == chess.BLACK else 1.0 - v  # convert to parent's perspective

    def __backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            value = 1.0 - value  # complement: flip perspective for the next level
            node = node.parent
