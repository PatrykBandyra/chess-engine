import abc
import argparse
import math
import time
from collections import OrderedDict
from typing import List, Optional

import chess
import chess.polyglot

from chess_board_screen import ChessBoardScreen
from constants import LOGGER, PIECE_VALUES
from move_policy import force_queen_promotion, queen_promotions_only
from opening_book import OpeningBook
from order_moves_mcts import OrderMovesMCTS
from player import Player


class MCTSNode:
    __slots__ = ('board', 'parent', 'move', 'children', 'untried_moves',
                 'visits', 'value', 'player', 'is_terminal', '_moves_sorted',
                 'proven_value', 'prior', '_untried_priors')

    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, _copy: bool = True):
        self.board = board.copy(stack=False) if _copy else board
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.untried_moves = queen_promotions_only(board.legal_moves)
        self.visits = 0
        self.value = 0.0
        self.player = board.turn
        self.is_terminal: bool = board.is_game_over()
        self._moves_sorted: bool = False
        # Prior probability assigned by the parent at expansion time (PUCT).
        self.prior: float = 0.0
        # Parallel to ``untried_moves`` after first sort; popped in sync.
        self._untried_priors: List[float] = []
        # Proven game-theoretic value from the move-maker's perspective (matches
        # ``self.value``): 1.0 = win, 0.0 = loss, 0.5 = draw, None = unknown.
        self.proven_value: Optional[float] = None
        if self.is_terminal:
            outcome = self.board.outcome(claim_draw=False)
            if outcome is None or outcome.winner is None:
                self.proven_value = 0.5  # stalemate / 75-move / 5-fold / insufficient
            else:
                self.proven_value = 1.0  # checkmate: move-maker won

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    @staticmethod
    def q_value(node: 'MCTSNode') -> float:
        """Mean value used by PUCT. Unvisited children receive neutral Q."""
        return node.value / node.visits if node.visits > 0 else 0.5

    def best_child(self, c_puct: float = 2.0):
        """
        PUCT child selection (AlphaZero-style):
            score = Q + c_puct * prior * sqrt(parent.visits) / (1 + child.visits)
        Proof-aware: a proven-winning child is taken immediately, proven-losing
        children are skipped.
        """
        candidates = []
        for child in self.children:
            pv = child.proven_value
            if pv == 1.0:
                return child
            if pv == 0.0:
                continue
            candidates.append(child)
        if not candidates:
            return max(self.children, key=lambda c: c.visits)
        sqrt_parent = math.sqrt(self.visits)
        best = candidates[0]
        best_score = self.q_value(best) + c_puct * best.prior * sqrt_parent / (1 + best.visits)
        for child in candidates[1:]:
            score = self.q_value(child) + c_puct * child.prior * sqrt_parent / (1 + child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)


class MCTS(Player):
    EVAL_CACHE_MAX_SIZE = 500_000
    # Sigmoid mapping raw eval (pawns, white perspective) → [0, 1]:
    #   v = 1 / (1 + exp(-raw / SIGMOID_SCALE))
    # Calibration targets: raw=1 → ~0.66, raw=3 → ~0.88, raw=9 → ~0.998.
    # SIGMOID_RAW_LIMIT short-circuits the formula for finite mate scores
    # (~±1e6) before math.exp would overflow (~±709.78).
    SIGMOID_SCALE = 1.5
    SIGMOID_RAW_LIMIT = SIGMOID_SCALE * 700.0
    # Below this many visits on the most-visited root child, the visit count is
    # too noisy to trust; fall back to best mean Q for the played move.
    MIN_ROOT_VISITS = 32
    # Quiescence search: limits ply depth and uses a delta-pruning margin (pawns).
    QS_MAX_DEPTH = 6
    QS_DELTA_MARGIN = 2.0
    QS_MATE_SCORE = 1_000_000.0
    # PUCT exploration constant and softmax temperature for prior derivation
    # from move-ordering scores. Tau acts on min-max-rescaled scores in [0, 1].
    C_PUCT = 2.0
    PRIOR_SOFTMAX_TAU = 0.4

    def __init__(self, args: argparse.Namespace, color: chess.Color):
        super().__init__(args, color)

        self.opening_book = OpeningBook(args, color)
        self.order_moves_mcts = OrderMovesMCTS(args, color)

        self.mcts_time_budget: float = args.mcts_time_white if color == chess.WHITE else args.mcts_time_black
        self.__eval_cache: OrderedDict[tuple[int, int], float] = OrderedDict()
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
                self.stats = {'from_book': True}
                self.last_eval = None
                self.last_phase = None
                return
        LOGGER.info(
            f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; '
            f'starting search; time_budget: {self.mcts_time_budget:.6f}s'
        )
        self.__run_mcts(board, start_time, move_number)

    def __run_mcts(self, board: chess.Board, start_time: float, move_number: int) -> None:
        root = self.__get_or_create_root(board)
        # Snapshot inherited visit counts (subtree reuse) so we can later report
        # how many visits were actually added during *this* turn.
        inherited_root_visits = root.visits
        inherited_child_visits = {c.move: c.visits for c in root.children}

        self.stats = {
            'iterations': 0,
            'skipped_terminals': 0,
            'nodes_created': 0,
            'max_depth': 0,
            'eval_calls': 0,
            'eval_cache_hits': 0,
            'reused_visits': root.visits,
            'root_children_count': 0,
            'best_child_visits': 0,
            'root_visit_entropy': 0.0,
            'convergence_point': 1.0,
            'avg_backprop_depth': 0.0,
            'c_puct': self.C_PUCT,
        }
        self._backprop_total_depth = 0
        self._backprop_count = 0
        convergence_iteration = None
        current_best_move = None

        end_time = time.perf_counter() + self.mcts_time_budget
        iterations = 0
        while True:
            if iterations & 127 == 0 and time.perf_counter() >= end_time:
                break
            # Forced result already known — no further search can change it.
            if root.proven_value is not None:
                break
            node = self.__select(root)
            # Skip already-evaluated proven nodes to avoid re-simulating them.
            if node.proven_value is not None and node.visits > 0:
                self.stats['skipped_terminals'] += 1
                iterations += 1
                continue
            if node.untried_moves:
                node = self.__expand(node)
            if node.proven_value is not None:
                # Use exact proven value instead of the static evaluator.
                value = node.proven_value
            else:
                value = self.__simulate(node)
            self.__backpropagate(node, value)
            if node.proven_value is not None:
                self.__propagate_proof(node.parent)
            if root.children:
                top_child = max(root.children, key=lambda c: c.visits)
                if top_child.move != current_best_move:
                    current_best_move = top_child.move
                    convergence_iteration = iterations
            iterations += 1
        self.stats['iterations'] = iterations
        if root.children:
            best_child, policy = self.__select_root_move(root)
            self.stats['root_children_count'] = len(root.children)
            self.stats['best_child_visits'] = best_child.visits
            total_visits = sum(c.visits for c in root.children) or 1
            entropy = 0.0
            for c in root.children:
                if c.visits > 0:
                    p = c.visits / total_visits
                    entropy -= p * math.log(p)
            self.stats['root_visit_entropy'] = round(entropy, 4)
            self.stats['convergence_point'] = round(
                (convergence_iteration or 0) / max(iterations, 1), 4
            )
            if self._backprop_count > 0:
                self.stats['avg_backprop_depth'] = round(
                    self._backprop_total_depth / self._backprop_count, 2
                )
            self.last_eval = best_child.value / max(best_child.visits, 1)
            self.last_phase = self.board_evaluator.get_game_phase(board) if hasattr(self, 'board_evaluator') else None
            best_move = force_queen_promotion(board, best_child.move)
            assert best_move is not None
            board.push(best_move)
            if best_move == best_child.move:
                self.__root = root
                self.__last_best_child = best_child
            else:
                # Safety fallback only: if a selected underpromotion was
                # normalized at the root, the old child board no longer matches
                # the real game state, so subtree reuse must be disabled.
                self.__root = None
                self.__last_best_child = None
            duration = time.perf_counter() - start_time
            mean_value = best_child.value / best_child.visits if best_child.visits > 0 else 0.0
            new_visits = best_child.visits - inherited_child_visits.get(best_child.move, 0)
            # Diagnostic fields are always emitted (even with sentinel values) so
            # downstream grep / log-parsers can rely on a stable schema.
            proven_root_str = (
                'none' if root.proven_value is None else f'{root.proven_value:.1f}'
            )
            # ``root.proven_value`` follows the same move-maker/parent-perspective
            # convention as non-root nodes for proof propagation. The root has no
            # parent, so expose an explicit side-to-move view for diagnostics.
            proven_root_stm_str = (
                'none' if root.proven_value is None else f'{1.0 - root.proven_value:.1f}'
            )
            top3 = sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]
            top3_str = ', '.join(
                f'{c.move.uci()}:vt={c.visits},'
                f'vn={c.visits - inherited_child_visits.get(c.move, 0)},'
                f'q={(c.value / c.visits if c.visits > 0 else 0.0):.3f},'
                f'p={c.prior:.3f}'
                f'{(",pv=" + format(c.proven_value, ".1f")) if c.proven_value is not None else ""}'
                for c in top3
            )
            LOGGER.info(
                f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; '
                f'time: {duration:.6f}s; move: {best_move.uci()}; value: {mean_value:.4f}; '
                f'new_visits: {new_visits}; total_visits: {best_child.visits}; iterations: {iterations}; '
                f'reused: {inherited_root_visits}; policy: {policy}; proven_root: {proven_root_str}; '
                f'proven_root_stm: {proven_root_stm_str}; '
                f'top3: [{top3_str}]'
            )
        else:
            self.__root = None
            self.__last_best_child = None
            LOGGER.warning(f'{type(self).__name__}; {"WHITE" if self.color else "BLACK"}; move_number: {move_number}; No valid move found. Skipping push.')

    def __select_root_move(self, root: MCTSNode) -> tuple[MCTSNode, str]:
        """
        Pick the move to play from the root, in priority order:
          1. proven win for side-to-move,
          2. skip proven losses (unless every move loses),
          3. robust child (max visits, tie-break by mean Q),
          4. low-visit safety: fall back to best mean Q if visits < MIN_ROOT_VISITS.
        Returns (chosen_child, policy_name) for diagnostic logging.
        """
        for child in root.children:
            if child.proven_value == 1.0:
                return child, 'proven_win'
        candidates = [c for c in root.children if c.proven_value != 0.0]
        if not candidates:
            return max(root.children, key=lambda c: c.visits), 'all_losses'
        best_visits = max(c.visits for c in candidates)
        if best_visits < self.MIN_ROOT_VISITS:
            chosen = max(
                candidates,
                key=lambda c: (c.value / c.visits) if c.visits > 0 else -1.0
            )
            return chosen, 'best_q'
        chosen = max(
            candidates,
            key=lambda c: (c.visits, (c.value / c.visits) if c.visits > 0 else 0.0)
        )
        return chosen, 'robust'

    def __get_or_create_root(self, board: chess.Board) -> MCTSNode:
        if self.__last_best_child is not None:
            opponent_move = board.move_stack[-1] if board.move_stack else None
            if opponent_move:
                current_state = self.__reuse_state_key(board)
                for child in self.__last_best_child.children:
                    if child.move == opponent_move and self.__reuse_state_key(child.board) == current_state:
                        child.parent = None  # detach from old tree for GC
                        self.__root = None
                        self.__last_best_child = None
                        return child
        # Fallback: create a new root
        self.__root = None
        self.__last_best_child = None
        return MCTSNode(board)

    @staticmethod
    def __reuse_state_key(board: chess.Board) -> tuple[str, chess.Color, int, Optional[int], int, int]:
        """Lightweight board-state key for safe tree reuse. Deliberately avoids
        move-stack history while checking all state that affects legal moves and
        draw counters relevant to the stored stackless node."""
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            board.ep_square,
            board.halfmove_clock,
            board.fullmove_number,
        )

    def __select(self, node: MCTSNode) -> MCTSNode:
        depth = 0
        while not node.is_terminal and node.children:
            if node.untried_moves:
                self.__prepare_untried_moves(node)
                best_child = node.best_child(self.C_PUCT)
                if self.__best_untried_score(node) >= self.__puct_child_score(node, best_child):
                    break
                node = best_child
                depth += 1
            elif node.is_fully_expanded():
                node = node.best_child(self.C_PUCT)
                depth += 1
        if hasattr(self, 'stats') and depth > self.stats.get('max_depth', 0):
            self.stats['max_depth'] = depth
        return node

    def __prepare_untried_moves(self, node: MCTSNode) -> None:
        if not node._moves_sorted:
            scored = self.order_moves_mcts.score_moves(node.board, node.untried_moves, None)
            moves = [m for _, m in scored]
            priors = self.__softmax_priors([s for s, _ in scored])
            moves.reverse()    # best at the end for O(1) pop
            priors.reverse()
            node.untried_moves = moves
            node._untried_priors = priors
            node._moves_sorted = True

    def __puct_child_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        pv = child.proven_value
        if pv == 1.0:
            return math.inf
        if pv == 0.0:
            return -math.inf
        return (MCTSNode.q_value(child)
                + self.C_PUCT * child.prior * math.sqrt(parent.visits) / (1 + child.visits))

    def __best_untried_score(self, node: MCTSNode) -> float:
        if not node.untried_moves:
            return -math.inf
        prior = node._untried_priors[-1] if node._untried_priors else (
            1.0 / max(1, len(node.children) + 1)
        )
        # Values are stored in [0, 1], so 0.5 is the neutral Q estimate for an
        # unvisited virtual child. This lets PUCT compare expansion against
        # descending into already-expanded children instead of expanding blindly.
        return 0.5 + self.C_PUCT * prior * math.sqrt(node.visits)

    def __expand(self, node: MCTSNode) -> MCTSNode:
        self.__prepare_untried_moves(node)
        move = node.untried_moves.pop()
        prior = node._untried_priors.pop() if node._untried_priors else (
            1.0 / max(1, len(node.children) + 1)
        )
        next_board = node.board.copy(stack=False)
        next_board.push(move)
        child_node = MCTSNode(next_board, parent=node, move=move, _copy=False)
        child_node.prior = prior
        node.children.append(child_node)
        if hasattr(self, 'stats'):
            self.stats['nodes_created'] += 1
        return child_node

    @staticmethod
    def __softmax_priors(scores: List[float]) -> List[float]:
        """Softmax over scores rescaled to [0, 1] (temperature ``PRIOR_SOFTMAX_TAU``).
        Returns a probability distribution; uniform when all scores are equal."""
        n = len(scores)
        if n == 0:
            return []
        max_s = max(scores)
        min_s = min(scores)
        if max_s == min_s:
            return [1.0 / n] * n
        span = max_s - min_s
        rescaled = [(s - min_s) / span for s in scores]
        tau = MCTS.PRIOR_SOFTMAX_TAU
        exps = [math.exp(r / tau) for r in rescaled]
        total = sum(exps)
        return [e / total for e in exps]

    def __simulate(self, node: MCTSNode) -> float:
        board = node.board
        cache_key = (chess.polyglot.zobrist_hash(board), board.halfmove_clock)
        if hasattr(self, 'stats'):
            self.stats['eval_calls'] += 1
        if cache_key in self.__eval_cache:
            if hasattr(self, 'stats'):
                self.stats['eval_cache_hits'] += 1
            self.__eval_cache.move_to_end(cache_key)
            v = self.__eval_cache[cache_key]
        else:
            # Quiescence is computed in side-to-move perspective; convert back
            # to white-perspective for the existing sigmoid mapping.
            qs_stm = self.__qs_negamax(board, self.QS_MAX_DEPTH, -math.inf, math.inf)
            raw_white = qs_stm if board.turn == chess.WHITE else -qs_stm
            if raw_white >= self.SIGMOID_RAW_LIMIT:
                v = 1.0
            elif raw_white <= -self.SIGMOID_RAW_LIMIT:
                v = 0.0
            else:
                v = 1.0 / (1.0 + math.exp(-raw_white / self.SIGMOID_SCALE))
            self.__eval_cache[cache_key] = v
            if len(self.__eval_cache) > self.EVAL_CACHE_MAX_SIZE:
                self.__eval_cache.popitem(last=False)
        return v if node.player == chess.BLACK else 1.0 - v  # convert to parent's perspective

    def __qs_negamax(self, board: chess.Board, depth_left: int,
                     alpha: float, beta: float) -> float:
        """Quiescence search (negamax, side-to-move perspective). Explores all
        legal evasions while in check; otherwise explores only captures and
        promotions, with SEE-based pruning of losing captures and delta-pruning.
        Returns the leaf value in pawn units."""
        if board.is_checkmate():
            return -self.QS_MATE_SCORE
        if board.is_game_over(claim_draw=False):
            return 0.0  # stalemate / 75-move / 5-fold / insufficient material

        if board.is_check():
            # In check there is no legal "stand pat" option: the side to move
            # must answer the check, and the evasion can be a quiet king move or
            # interposition. Therefore search all legal moves and skip capture-
            # only, SEE, and delta-pruning filters in this branch.
            if depth_left <= 0:
                raw_white = self.evaluate_board(board)
                return raw_white if board.turn == chess.WHITE else -raw_white
            best = -math.inf
            for move in queen_promotions_only(board.legal_moves):
                board.push(move)
                score = -self.__qs_negamax(board, depth_left - 1, -beta, -alpha)
                board.pop()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return best

        raw_white = self.evaluate_board(board)
        stand_pat = raw_white if board.turn == chess.WHITE else -raw_white
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if depth_left <= 0:
            return stand_pat
        best = stand_pat
        for move in queen_promotions_only(board.legal_moves):
            if not (board.is_capture(move) or move.promotion):
                continue
            # Skip clearly losing captures (negative SEE). Promotions always
            # tried; the gain of promotion can rescue them.
            if not move.promotion and self.__see_move(board, move) < 0:
                continue
            # Delta pruning: even the most optimistic gain wouldn't reach alpha.
            cap_value = self.__capture_value(board, move)
            if stand_pat + cap_value + self.QS_DELTA_MARGIN < alpha:
                continue
            board.push(move)
            score = -self.__qs_negamax(board, depth_left - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    @staticmethod
    def __capture_value(board: chess.Board, move: chess.Move) -> float:
        """Value (pawns) of the piece captured by ``move``; 0 for non-captures."""
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        captured = board.piece_at(move.to_square)
        return PIECE_VALUES[captured.piece_type] if captured else 0.0

    def __see_move(self, board: chess.Board, move: chess.Move) -> float:
        """Static Exchange Evaluation for a single capture, in pawn units.
        Positive = winning capture, negative = losing capture. Promotion gain
        is included. Implemented via simulated swap-off (push/pop) — accurate
        but not the fastest; OK for filtering quiescence captures."""
        if not (board.is_capture(move) or move.promotion):
            return 0.0
        captured_value = self.__capture_value(board, move)
        promo_gain = (PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
                      ) if move.promotion else 0.0
        board.push(move)
        opp_gain = self.__see_at(board, move.to_square)
        board.pop()
        return captured_value + promo_gain - opp_gain

    def __see_at(self, board: chess.Board, target_sq: int) -> float:
        """Best material the side to move can gain by initiating a sequence of
        recaptures on ``target_sq`` (pawn units, ≥ 0). Side may also pass."""
        lva_move = None
        lva_val = math.inf
        for m in queen_promotions_only(board.legal_moves):
            if m.to_square != target_sq or not board.is_capture(m):
                continue
            attacker = board.piece_at(m.from_square)
            if attacker is None:
                continue
            v = PIECE_VALUES[attacker.piece_type]
            if v < lva_val:
                lva_val = v
                lva_move = m
        if lva_move is None:
            return 0.0
        target_piece = board.piece_at(target_sq)
        if target_piece is None:
            return 0.0
        captured_value = PIECE_VALUES[target_piece.piece_type]
        board.push(lva_move)
        opp_gain = self.__see_at(board, target_sq)
        board.pop()
        return max(0.0, captured_value - opp_gain)

    def __backpropagate(self, node: MCTSNode, value: float) -> None:
        depth = 0
        while node is not None:
            node.visits += 1
            node.value += value
            value = 1.0 - value
            node = node.parent
            depth += 1
        if hasattr(self, '_backprop_total_depth'):
            self._backprop_total_depth += depth
            self._backprop_count += 1

    def __propagate_proof(self, node: Optional[MCTSNode]) -> None:
        """
        Negamax-style proof propagation up the tree.

        ``child.proven_value`` is in ``node.player`` perspective; ``node.proven_value``
        is in the parent's (move-maker's) perspective, so signs flip:
          * any winning child (1.0)            → node is a LOSS for parent (0.0),
          * all children proven, none winning  → DRAW (0.5) if any draw, else WIN (1.0).
        """
        while node is not None and node.proven_value is None:
            has_winning_move = False
            all_proven = True
            has_draw = False
            for c in node.children:
                pv = c.proven_value
                if pv is None:
                    all_proven = False
                elif pv == 1.0:
                    has_winning_move = True
                    break
                elif pv == 0.5:
                    has_draw = True
            if has_winning_move:
                node.proven_value = 0.0
            elif not node.untried_moves and all_proven:
                node.proven_value = 0.5 if has_draw else 1.0
            else:
                return
            node = node.parent

