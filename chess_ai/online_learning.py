"""Human-vs-AI experience capture and incremental learning helpers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from my_chess.chess_ai.action_space import (
    ACTION_SPACE_SIZE,
    HORIZONTAL_FLIP_INDICES,
    canonical_action_id,
    move_to_action_id,
)
from my_chess.chess_ai.dataset import ReplayBuffer, TrainingSample
from my_chess.chess_ai.encoding import (
    ENCODING_VERSION,
    encode_board,
    mirror_encoded_state,
)
from my_chess.chess_ai.self_play import dense_policy_target as _dense_canonical
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move

DEFAULT_EXPERIENCE_DIR = Path("artifacts") / "experience"
DEFAULT_HUMAN_REPLAY_PATH = DEFAULT_EXPERIENCE_DIR / "human_replay.pkl"
DEFAULT_MISTAKE_MEMORY_PATH = DEFAULT_EXPERIENCE_DIR / "mistake_memory.pkl"


@dataclass(frozen=True)
class PendingExperience:
    """Intermediate move record stored until the game result is known."""

    state: List[List[List[int]]]
    policy_target: List[float]
    player_is_red: bool
    is_human_move: bool = False


class OnlineLearningSession:
    """Collects one game's experiences and finalizes them after game end."""

    def __init__(self) -> None:
        self._records: list[PendingExperience] = []
        self._ai_actions: list[tuple[str, int, bool]] = []

    def reset(self) -> None:
        """Clears any in-progress game records."""
        self._records.clear()
        self._ai_actions.clear()

    def discard_last_record(self) -> None:
        """Drops the most recent pending record (failed human move)."""
        if self._records:
            self._records.pop()

    def record_distribution(
        self,
        board: Chessboard,
        sparse_distribution: dict[int, float],
    ) -> None:
        """Records one move using a sparse action distribution."""
        dense = dense_policy_target(sparse_distribution, board.is_red_turn)
        self._records.append(
            PendingExperience(
                state=encode_board(board),
                policy_target=dense,
                player_is_red=board.is_red_turn,
                is_human_move=False,
            )
        )

    def record_selected_move(self, board: Chessboard, move: Move) -> None:
        """Records one human move as a one-hot policy target."""
        dense = one_hot_policy_target(move, board.is_red_turn)
        self._records.append(
            PendingExperience(
                state=encode_board(board),
                policy_target=dense,
                player_is_red=board.is_red_turn,
                is_human_move=True,
            )
        )

    def append_record(self, record: PendingExperience) -> None:
        """Appends a prebuilt move record after move validation succeeds."""
        self._records.append(record)

    def record_ai_action(self, board: Chessboard, action_id: int) -> None:
        """Records the exact AI action chosen in a board position."""
        self._ai_actions.append((board.to_fen(), action_id, board.is_red_turn))

    def finalize(self, winner: str | None) -> list[TrainingSample]:
        """Converts in-progress records into labeled training samples.

        Winning human moves receive a boosted policy weight (3.0x) and are
        augmented with horizontal symmetry so the AI strongly absorbs the
        winning human play style, while losing moves get minimal policy weight
        (0.15x) so the AI does not reinforce blunders.
        """
        samples: list[TrainingSample] = []
        value_weight = 0.0 if winner is None else 1.0
        for record in self._records:
            value_target = self._winner_to_value(winner, record.player_is_red)
            if value_target > 0.0:
                policy_weight = 3.0 if record.is_human_move else 1.5
            elif value_target < 0.0:
                policy_weight = 0.15
            else:
                policy_weight = 1.0

            samples.append(
                TrainingSample(
                    state=record.state,
                    policy_target=record.policy_target,
                    value_target=value_target,
                    value_weight=value_weight,
                    policy_weight=policy_weight,
                )
            )
            if winner is not None:
                mirrored_policy = [
                    record.policy_target[src_idx] for src_idx in HORIZONTAL_FLIP_INDICES
                ]
                samples.append(
                    TrainingSample(
                        state=mirror_encoded_state(record.state),
                        policy_target=mirrored_policy,
                        value_target=value_target,
                        value_weight=value_weight,
                        policy_weight=policy_weight,
                    )
                )
        self.reset()
        return samples

    def finalize_ai_mistakes(
        self, winner: str | None, ai_is_red: bool
    ) -> list[tuple[str, int]]:
        """Returns the losing AI actions as (state_key, action_id) pairs."""
        if winner is None:
            self._ai_actions.clear()
            return []

        ai_lost = (winner == "Red" and not ai_is_red) or (
            winner == "Black" and ai_is_red
        )
        if not ai_lost:
            self._ai_actions.clear()
            return []

        mistakes = [
            (state_key, action_id)
            for state_key, action_id, player_is_red in self._ai_actions
            if player_is_red == ai_is_red
        ]
        self._ai_actions.clear()
        return mistakes

    @staticmethod
    def _winner_to_value(winner: str | None, player_is_red: bool) -> float:
        if winner is None or winner == "Draw":
            return 0.0
        if winner == "Red":
            return 1.0 if player_is_red else -1.0
        if winner == "Black":
            return -1.0 if player_is_red else 1.0
        raise ValueError(f"Unexpected winner label: {winner}")


class PersistentReplayBuffer:
    """On-disk replay buffer for human-vs-AI games."""

    def __init__(
        self, path: str | Path = DEFAULT_HUMAN_REPLAY_PATH, capacity: int = 8192
    ):
        self.path = Path(path)
        self.capacity = capacity

    def load(self) -> ReplayBuffer:
        """Loads the replay buffer from disk if it exists."""
        buffer = ReplayBuffer(capacity=self.capacity)
        if not self.path.exists():
            return buffer

        with self.path.open("rb") as file:
            payload = pickle.load(file)

        if payload.get("encoding_version") != ENCODING_VERSION:
            # Old-encoding samples would teach the network wrong board views.
            return buffer
        for item in payload.get("samples", []):
            buffer.add(
                TrainingSample(
                    state=item["state"],
                    policy_target=item["policy_target"],
                    value_target=item["value_target"],
                    value_weight=item.get("value_weight", 1.0),
                    policy_weight=item.get("policy_weight", 1.0),
                )
            )
        return buffer

    def save(self, samples: Iterable[TrainingSample]) -> None:
        """Persists samples to disk, truncating to capacity."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trimmed = list(samples)[-self.capacity :]
        payload = {
            "encoding_version": ENCODING_VERSION,
            "samples": [
                {
                    "state": sample.state,
                    "policy_target": sample.policy_target,
                    "value_target": sample.value_target,
                    "value_weight": sample.value_weight,
                    "policy_weight": sample.policy_weight,
                }
                for sample in trimmed
            ],
        }
        with self.path.open("wb") as file:
            pickle.dump(payload, file)


class MistakeMemoryStore:
    """Persistent store of penalized state-action pairs."""

    def __init__(
        self,
        path: str | Path = DEFAULT_MISTAKE_MEMORY_PATH,
        max_states: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.max_states = max_states

    def load(self) -> dict[str, dict[int, int]]:
        """Loads mistake counts from disk."""
        if not self.path.exists():
            return {}
        with self.path.open("rb") as file:
            payload = pickle.load(file)
        return {
            str(state_key): {
                int(action_id): int(count) for action_id, count in actions.items()
            }
            for state_key, actions in payload.get("mistakes", {}).items()
        }

    def save(self, memory: dict[str, dict[int, int]]) -> None:
        """Saves mistake counts to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trimmed_items = list(memory.items())[-self.max_states :]
        payload = {"mistakes": dict(trimmed_items)}
        with self.path.open("wb") as file:
            pickle.dump(payload, file)

    def add_mistakes(
        self, mistakes: Iterable[tuple[str, int]]
    ) -> dict[str, dict[int, int]]:
        """Increments mistake counts and persists them."""
        memory = self.load()
        for state_key, action_id in mistakes:
            state_memory = memory.setdefault(state_key, {})
            state_memory[action_id] = state_memory.get(action_id, 0) + 1
        self.save(memory)
        return memory


def dense_policy_target(
    sparse_distribution: dict[int, float], is_red_turn: bool
) -> list[float]:
    """Converts sparse absolute-id probabilities into a dense canonical vector."""
    return _dense_canonical(sparse_distribution, is_red_turn)


def one_hot_policy_target(move: Move, is_red_turn: bool) -> list[float]:
    """Builds a canonical one-hot policy target for a selected move."""
    dense = [0.0] * ACTION_SPACE_SIZE
    dense[canonical_action_id(move_to_action_id(move), is_red_turn)] = 1.0
    return dense
