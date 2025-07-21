import abc
import argparse
import math
import random
import time
from typing import List, Optional

import chess

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

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        """
        Selects the best child node based on the UCT (Upper Confidence Bound for Trees) formula.
        """
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)


class MCTS(Player):

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_mcts = OrderMovesMCTS(args, color)

        self.depth: int = args.depth_white if color == chess.WHITE else args.depth_black

    @abc.abstractmethod
    def evaluate_board(self, board: chess.Board) -> float:
        pass

    def make_move(self, board: chess.Board, screen: ChessBoardScreen) -> None:
        start_time: float = time.perf_counter()
        if self.opening_book.use_opening_book and self.opening_book.is_opening:
            if super().make_move(board, start_time):
                return  # Move already made from an opening book
        self.__run_mcts(board, start_time)

    def __run_mcts(self, board: chess.Board, start_time: float) -> None:
        SIMULATION_TIME = 20.0  # seconds
        root = MCTSNode(board)
        end_time = time.perf_counter() + SIMULATION_TIME
        while time.perf_counter() < end_time:
            node = self.__select(root)
            if node.untried_moves:
                node = self.__expand(node)
            value = self.__simulate(node)
            self.__backpropagate(node, value)
        if root.children:
            best_child = root.most_visited_child()
            board.push(best_child.move)
            duration = time.perf_counter() - start_time
            LOGGER.info(
                f'MCTS-TRAD; {"WHITE" if self.color else "BLACK"}; time: {duration:.6f}s; move: {best_child.move.uci()}; visits: {best_child.visits}'
            )
        else:
            LOGGER.warning('MCTS-TRAD: No valid move found. Skipping push.')

    def __select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and node.children:
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
        sim_board = node.board.copy(stack=False)
        moves_played = 0
        while not sim_board.is_game_over() and moves_played < self.depth:
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            sim_board.push(move)
            moves_played += 1
        return self.evaluate_board(sim_board)

    def __backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            if node.player == self.color:
                node.value += value
            else:
                node.value -= value
            node = node.parent
