"""Minimal AlphaZero-style MCTS implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Protocol

from my_chess.chess_ai.action_space import move_to_action_id
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


class PolicyValueFn(Protocol):
    """Callable interface used by MCTS for policy/value inference."""

    def __call__(self, board: Chessboard) -> tuple[Dict[int, float], float]:
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
    """A node in the search tree."""

    board: Chessboard
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


class MCTS:
    """Minimal PUCT-based tree search."""

    def __init__(self, policy_value_fn: PolicyValueFn, config: MCTSConfig | None = None):
        self.policy_value_fn = policy_value_fn
        self.config = config or MCTSConfig()

    def run(self, board: Chessboard) -> Dict[int, float]:
        """Runs search from the given board and returns root visit distribution."""
        root = SearchNode(board=board.clone())
        self._expand(root)

        for _ in range(self.config.num_simulations):
            self._simulate(root)

        total = root.total_visits()
        if total == 0:
            return {}
        return {
            action_id: edge.visits / total
            for action_id, edge in root.edges.items()
        }

    def _simulate(self, node: SearchNode) -> float:
        """Runs one simulation from a node and backpropagates the result."""
        if node.board.is_end():
            winner = node.board.get_winner()
            if winner is None:
                return 0.0
            current_player_wins = (winner == "Red" and node.board.is_red_turn) or (
                winner == "Black" and not node.board.is_red_turn
            )
            return 1.0 if current_player_wins else -1.0

        if not node.is_expanded:
            return self._expand(node)

        action_id = self._select_action(node)
        child = node.children.get(action_id)
        if child is None:
            move = node.legal_moves[action_id]
            child_board = node.board.clone()
            child_board.apply_move(move)
            child = SearchNode(board=child_board)
            node.children[action_id] = child

        child_value = self._simulate(child)
        edge = node.edges[action_id]
        edge.visits += 1
        edge.total_value += child_value
        return -child_value

    def _expand(self, node: SearchNode) -> float:
        """Expands a leaf node using the policy/value function."""
        priors, value = self.policy_value_fn(node.board)
        legal_moves = {
            move_to_action_id(move): move for move in node.board.get_legal_moves()
        }
        node.edges = {
            action_id: EdgeStats(prior=priors.get(action_id, 0.0))
            for action_id in legal_moves
        }
        node.legal_moves = legal_moves
        node.is_expanded = True
        return value

    def select_action(self, board: Chessboard) -> int:
        """Runs search and returns the most-visited root action id."""
        distribution = self.run(board)
        if not distribution:
            raise RuntimeError("No action available for the given board")
        return max(distribution.items(), key=lambda item: item[1])[0]

    @staticmethod
    def action_id_to_legal_move(board: Chessboard, action_id: int) -> Move:
        """Resolves an action id back to the concrete legal move object."""
        for move in board.get_legal_moves():
            if move_to_action_id(move) == action_id:
                return move
        raise ValueError(f"Action id {action_id} is not legal in the current board")

    def _select_action(self, node: SearchNode) -> int:
        """Selects the next edge according to a PUCT-style score."""
        total_visits = max(1, node.total_visits())
        best_action_id = -1
        best_score = float("-inf")

        for action_id, edge in node.edges.items():
            exploration = (
                self.config.c_puct
                * edge.prior
                * math.sqrt(total_visits)
                / (1 + edge.visits)
            )
            score = edge.mean_value + exploration
            if score > best_score:
                best_score = score
                best_action_id = action_id

        if best_action_id < 0:
            raise RuntimeError("Search node has no legal actions to select")
        return best_action_id


def uniform_policy_value_fn(board: Chessboard) -> tuple[Dict[int, float], float]:
    """Fallback evaluator used before a trained model is available.

    It assigns equal prior to each legal move and returns neutral value.
    """
    legal_moves = board.get_legal_moves()
    if not legal_moves:
        return {}, 0.0

    prior = 1.0 / len(legal_moves)
    return ({move_to_action_id(move): prior for move in legal_moves}, 0.0)
