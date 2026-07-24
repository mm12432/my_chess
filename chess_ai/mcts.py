"""Minimal AlphaZero-style MCTS implementation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Protocol

from my_chess.chess_ai.action_space import move_to_action_id
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


class PolicyValueFn(Protocol):
    """Callable interface used by MCTS for policy/value inference."""

    def __call__(
        self,
        board: Chessboard,
        legal_moves: list[Move] | None = None,
    ) -> tuple[Dict[int, float], float]:
        """Returns legal-action priors and a scalar value for the current board."""


@dataclass
class EdgeStats:
    """Per-action search statistics stored on a tree edge."""

    prior: float
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        """Returns the running average value for this edge."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


@dataclass
class SearchNode:
    """A node in the search tree (board state lives on the shared path board)."""

    edges: Dict[int, EdgeStats] = field(default_factory=dict)
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    legal_moves: Dict[int, Move] = field(default_factory=dict)
    is_expanded: bool = False

    def total_visits(self) -> int:
        """Returns total visits over outgoing edges."""
        return sum(edge.visits for edge in self.edges.values())


@dataclass(frozen=True)
class MCTSConfig:
    """Config for the search budget and exploration strength."""

    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class MCTS:
    """Minimal PUCT-based tree search with make/unmake path simulation."""

    def __init__(
        self, policy_value_fn: PolicyValueFn, config: MCTSConfig | None = None
    ):
        self.policy_value_fn = policy_value_fn
        self.config = config or MCTSConfig()
        self._root_legal_moves: Dict[int, Move] = {}

    def run(self, board: Chessboard) -> Dict[int, float]:
        """Runs search from the given board and returns root visit distribution."""
        distribution, _ = self.run_with_value(board)
        return distribution

    def run_with_value(self, board: Chessboard) -> tuple[Dict[int, float], float]:
        """Runs search and also returns the root network/heuristic value."""
        # One clone per search; simulations make/unmake on this working board.
        working = board.clone()
        root = SearchNode()
        root_value = self._expand(root, working, add_dirichlet_noise=True)
        self._root_legal_moves = dict(root.legal_moves)

        for _ in range(self.config.num_simulations):
            self._simulate(root, working)

        total = root.total_visits()
        if total == 0:
            return {}, root_value
        distribution = {
            action_id: edge.visits / total for action_id, edge in root.edges.items()
        }
        return distribution, root_value

    def _simulate(self, node: SearchNode, board: Chessboard) -> float:
        """Runs one simulation from a node and backpropagates the result."""
        fast_winner = board.get_fast_winner()
        if fast_winner is not None:
            return self._winner_to_value(fast_winner, board.is_red_turn)

        if not node.is_expanded:
            return self._expand(node, board)

        if not node.edges:
            # Expanded with no legal moves => stalemate loss for side to move.
            return -1.0

        action_id = self._select_action(node)
        move = node.legal_moves[action_id]
        child = node.children.get(action_id)
        if child is None:
            child = SearchNode()
            node.children[action_id] = child

        undo = board.push_move(move)
        try:
            child_value = self._simulate(child, board)
        finally:
            board.pop_move(undo)

        # child_value is from the opponent's side-to-move view after the move.
        my_value = -child_value
        edge = node.edges[action_id]
        edge.visits += 1
        edge.total_value += my_value
        return my_value

    def _expand(
        self,
        node: SearchNode,
        board: Chessboard,
        add_dirichlet_noise: bool = False,
    ) -> float:
        """Expands a leaf node using the policy/value function."""
        fast_winner = board.get_fast_winner()
        if fast_winner is not None:
            node.is_expanded = True
            return self._winner_to_value(fast_winner, board.is_red_turn)

        legal_list = board.get_legal_moves()
        if not legal_list:
            # Stalemate: side to move loses under this engine's rules.
            node.is_expanded = True
            return -1.0

        legal_moves = {move_to_action_id(move): move for move in legal_list}
        priors, value = self.policy_value_fn(board, legal_list)
        if add_dirichlet_noise and len(legal_moves) > 1:
            priors = self._apply_dirichlet_noise(priors, legal_moves.keys())

        node.edges = {
            action_id: EdgeStats(prior=priors.get(action_id, 0.0))
            for action_id in legal_moves
        }
        node.legal_moves = legal_moves
        node.is_expanded = True
        return value

    def _apply_dirichlet_noise(
        self, priors: Dict[int, float], action_ids
    ) -> Dict[int, float]:
        """Mixes Dirichlet noise into root priors for exploration."""
        action_ids = list(action_ids)
        epsilon = self.config.dirichlet_epsilon
        if epsilon <= 0.0:
            return priors
        noise = random.gammavariate
        alpha = self.config.dirichlet_alpha
        raw = [noise(alpha, 1.0) for _ in action_ids]
        total = sum(raw) or 1.0
        mixed: Dict[int, float] = {}
        for action_id, sample in zip(action_ids, raw):
            prior = priors.get(action_id, 0.0)
            mixed[action_id] = (1.0 - epsilon) * prior + epsilon * (sample / total)
        return mixed

    def select_action(self, board: Chessboard) -> int:
        """Runs search and returns the most-visited root action id."""
        distribution = self.run(board)
        if not distribution:
            raise RuntimeError("No action available for the given board")
        return max(distribution.items(), key=lambda item: item[1])[0]

    def action_id_to_legal_move(self, board: Chessboard, action_id: int) -> Move:
        """Resolves an action id back to the concrete legal move object."""
        cached = self._root_legal_moves.get(action_id)
        if cached is not None:
            return cached
        for move in board.get_legal_moves():
            if move_to_action_id(move) == action_id:
                return move
        raise ValueError(f"Action id {action_id} is not legal in the current board")

    def _select_action(self, node: SearchNode) -> int:
        """Selects the next edge according to a PUCT-style score."""
        total_visits = max(1, node.total_visits())
        best_action_id = -1
        best_score = float("-inf")
        sqrt_total = math.sqrt(total_visits)
        c_puct = self.config.c_puct

        for action_id, edge in node.edges.items():
            exploration = c_puct * edge.prior * sqrt_total / (1 + edge.visits)
            score = edge.mean_value + exploration
            if score > best_score:
                best_score = score
                best_action_id = action_id

        if best_action_id < 0:
            raise RuntimeError("Search node has no legal actions to select")
        return best_action_id

    @staticmethod
    def _winner_to_value(winner: str, player_is_red: bool) -> float:
        if winner == "Red":
            return 1.0 if player_is_red else -1.0
        if winner == "Black":
            return -1.0 if player_is_red else 1.0
        return 0.0


def uniform_policy_value_fn(
    board: Chessboard,
    legal_moves: list[Move] | None = None,
) -> tuple[Dict[int, float], float]:
    """Fallback evaluator used before a trained model is available.

    It assigns equal prior to each legal move and returns neutral value.
    """
    if legal_moves is None:
        legal_moves = board.get_legal_moves()
    if not legal_moves:
        return {}, 0.0

    prior = 1.0 / len(legal_moves)
    return ({move_to_action_id(move): prior for move in legal_moves}, 0.0)
