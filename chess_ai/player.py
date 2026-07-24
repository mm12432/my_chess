"""High-level AI player helpers for GUI/CLI integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch

from my_chess.chess_ai.evaluator import (
    HybridPolicyValueEvaluator,
    TorchPolicyValueEvaluator,
)
from my_chess.chess_ai.mcts import MCTS, MCTSConfig, uniform_policy_value_fn
from my_chess.chess_ai.model import TORCH_AVAILABLE, build_model
from my_chess.chess_ai.online_learning import MistakeMemoryStore
from my_chess.chess_ai.train import DEFAULT_CHECKPOINT_PATH, load_checkpoint
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


@dataclass(frozen=True)
class CandidateAnalysis:
    """Human-readable analysis for one candidate move."""

    move: Move
    selection_probability: float
    win_rate_after_move: float


@dataclass(frozen=True)
class PositionAnalysis:
    """AI summary for the current board."""

    current_win_rate: float
    candidates: list[CandidateAnalysis]


class AIPlayer:
    """A simple MCTS-backed AI player for one side."""

    def __init__(
        self,
        is_red: bool = False,
        num_simulations: int = 32,
        checkpoint_path: str | Path | None = None,
        mistake_memory: dict[str, dict[int, int]] | None = None,
    ) -> None:
        self.is_red = is_red
        self.num_simulations = num_simulations
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.mistake_memory = mistake_memory or MistakeMemoryStore().load()

        if TORCH_AVAILABLE:
            if checkpoint_path.exists():
                model = load_checkpoint(checkpoint_path, device=self.device)
                self.model_source = "trained"
                # Pure network so self-play / online learning actually show up in play.
                self.evaluator = TorchPolicyValueEvaluator(model, device=self.device)
            else:
                model = build_model()
                self.model_source = "random_init"
                # Untrained nets need light tactical bias; heuristic no longer clones.
                self.evaluator = HybridPolicyValueEvaluator(
                    model,
                    device=self.device,
                    model_prior_weight=0.5,
                    model_value_weight=0.5,
                )
            self.mcts = MCTS(self.evaluator, MCTSConfig(num_simulations=num_simulations))
        else:
            self.evaluator = uniform_policy_value_fn
            self.mcts = MCTS(
                uniform_policy_value_fn,
                MCTSConfig(num_simulations=num_simulations),
            )
            self.model_source = "uniform_fallback"

    def should_move(self, board: Chessboard) -> bool:
        """Returns whether it is this AI player's turn."""
        return board.is_red_turn == self.is_red

    def choose_move(self, board: Chessboard) -> Move:
        """Chooses the next move from the current board state."""
        action_id = self.mcts.select_action(board)
        return self.mcts.action_id_to_legal_move(board, action_id)

    def choose_move_with_distribution(
        self, board: Chessboard
    ) -> tuple[Move, dict[int, float], float]:
        """Chooses a move and returns the root visit distribution and root value."""
        visit_distribution, root_value = self.mcts.run_with_value(board)
        if not visit_distribution:
            raise RuntimeError("AI found no legal move")
        visit_distribution = self._apply_mistake_penalties(board, visit_distribution)
        selected_action_id = max(
            visit_distribution.items(), key=lambda item: item[1]
        )[0]
        selected_move = self.mcts.action_id_to_legal_move(board, selected_action_id)
        return selected_move, visit_distribution, root_value

    def choose_move_with_analysis(
        self, board: Chessboard, top_k: int = 3
    ) -> tuple[Move, PositionAnalysis, dict[int, float]]:
        """Chooses the next move and returns a compact analysis summary.

        Avoids re-evaluating top candidates with extra network forwards; visit
        share is used as a cheap proxy for candidate quality.
        """
        selected_move, visit_distribution, root_value = (
            self.choose_move_with_distribution(board)
        )
        current_win_rate = self._value_to_win_rate(root_value)

        sorted_candidates = sorted(
            visit_distribution.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
        candidates: list[CandidateAnalysis] = []
        for action_id, probability in sorted_candidates:
            move = self.mcts.action_id_to_legal_move(board, action_id)
            # Map visit mass into a rough "after-move" confidence for the UI.
            candidate_win_rate = max(
                0.0,
                min(1.0, current_win_rate * 0.5 + probability * 0.5),
            )
            candidates.append(
                CandidateAnalysis(
                    move=move,
                    selection_probability=probability,
                    win_rate_after_move=candidate_win_rate,
                )
            )

        return (
            selected_move,
            PositionAnalysis(
                current_win_rate=current_win_rate,
                candidates=candidates,
            ),
            visit_distribution,
        )

    @staticmethod
    def _value_to_win_rate(value: float) -> float:
        """Maps [-1, 1] value output to an intuitive [0, 1] win-rate estimate."""
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    def _apply_mistake_penalties(
        self, board: Chessboard, distribution: dict[int, float]
    ) -> dict[int, float]:
        """Downweights actions that previously led to losses in the same position."""
        state_penalties = self.mistake_memory.get(board.to_fen())
        if not state_penalties:
            return distribution

        adjusted = {}
        for action_id, probability in distribution.items():
            penalty_count = state_penalties.get(action_id, 0)
            if penalty_count <= 0:
                adjusted[action_id] = probability
                continue
            adjusted[action_id] = probability * math.exp(-0.7 * penalty_count)

        total = sum(adjusted.values())
        if total <= 0:
            return distribution
        return {
            action_id: probability / total
            for action_id, probability in adjusted.items()
        }
