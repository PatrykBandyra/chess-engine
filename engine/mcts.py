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
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board.copy(stack=False)
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.untried_moves = list(board.legal_moves)
        self.visits = 0
        self.value = 0.0
        self.player = board.turn
        self.is_terminal: bool = board.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=math.sqrt(2)):
        """
        Selects the best child node based on the UCT (Upper Confidence Bound for Trees) formula.
        Values are normalized to [0, 1], so the standard c = sqrt(2) is used.
        """
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)


class MCTS(Player):
    EVAL_CACHE_MAX_SIZE = 500_000

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_mcts = OrderMovesMCTS(args, color)

        self.mcts_time_budget: float = args.mcts_time
        self.__eval_cache: OrderedDict[int, float] = OrderedDict()
        self.__root: Optional[MCTSNode] = None
        self.__last_best_child: Optional[MCTSNode] = None

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        start_time: float = time.perf_counter()
        if self.opening_book.use_opening_book and self.opening_book.is_opening:
            if self.opening_book.make_move(board, start_time):
                self.__root = None
                self.__last_best_child = None
                return  # Move already made from an opening book
        self.__run_mcts(board, start_time)

    def __run_mcts(self, board: chess.Board, start_time: float) -> None:
        root = self.__get_or_create_root(board)
        end_time = time.perf_counter() + self.mcts_time_budget
        while time.perf_counter() < end_time:
            node = self.__select(root)
            if node.is_terminal and node.visits > 0:
                continue  # already evaluated terminal node, skip
            if node.untried_moves:
                node = self.__expand(node)
            value = self.__simulate(node)
            self.__backpropagate(node, value)
        if root.children:
            best_child = root.most_visited_child()
            board.push(best_child.move)
            self.__root = root
            self.__last_best_child = best_child
            duration = time.perf_counter() - start_time
            LOGGER.info(
                f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_child.move.uci()}; visits: {best_child.visits}'
            )
        else:
            self.__root = None
            self.__last_best_child = None
            LOGGER.warning(f'{type(self).__name__}: No valid move found. Skipping push.')

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
        move = self.order_moves_mcts.order_moves(node.board, node.untried_moves, None)[0]
        node.untried_moves.remove(move)
        next_board = node.board.copy(stack=False)
        next_board.push(move)
        child_node = MCTSNode(next_board, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def __simulate(self, node: MCTSNode) -> float:
        board_hash = chess.polyglot.zobrist_hash(node.board)
        if board_hash in self.__eval_cache:
            self.__eval_cache.move_to_end(board_hash)
            v = self.__eval_cache[board_hash]
        else:
            raw = self.evaluate_board(node.board)
            v = 1.0 / (1.0 + math.exp(-raw / 4.0))  # sigmoid normalization to [0, 1], white perspective
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
