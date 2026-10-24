"""Training-sample structures for self-play and supervised updates."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.encoding import NUM_CHANNELS

Tensor3D = List[List[List[int]]]


@dataclass(frozen=True)
class TrainingSample:
    """One training sample for policy/value learning.

    Attributes:
        state: Encoded board tensor with shape C x H x W.
        policy_target: Dense action distribution with length ACTION_SPACE_SIZE.
        value_target: Final game outcome from the current player's perspective.
        value_weight: 1.0 when the game reached a real result, 0.0 when it was
            truncated (unknown outcome must not be taught as a draw).
        policy_weight: Relative weight for the policy loss of this sample
            (e.g. boosted for winning human moves, reduced/zeroed for losing moves).
    """

    state: Tensor3D
    policy_target: List[float]
    value_target: float
    value_weight: float = 1.0
    policy_weight: float = 1.0

    def validate(self) -> None:
        """Checks sample shape invariants early."""
        if len(self.state) != NUM_CHANNELS:
            raise ValueError(
                f"Expected {NUM_CHANNELS} state channels, got {len(self.state)}"
            )
        if len(self.policy_target) != ACTION_SPACE_SIZE:
            raise ValueError(
                "Policy target length mismatch: "
                f"expected {ACTION_SPACE_SIZE}, got {len(self.policy_target)}"
            )
        if not -1.0 <= self.value_target <= 1.0:
            raise ValueError(
                f"value_target must be within [-1, 1], got {self.value_target}"
            )


class ReplayBuffer:
    """Simple in-memory replay buffer for generated training samples."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._capacity = capacity
        self._samples: List[TrainingSample] = []

    @property
    def capacity(self) -> int:
        """Returns the maximum number of samples retained."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: TrainingSample) -> None:
        """Adds a sample and evicts the oldest one if needed."""
        sample.validate()
        self._samples.append(sample)
        if len(self._samples) > self._capacity:
            self._samples.pop(0)

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        """Adds many samples in order."""
        for sample in samples:
            self.add(sample)

    def as_lists(self) -> Tuple[List[Tensor3D], List[List[float]], List[float]]:
        """Returns plain Python lists suitable for framework conversion."""
        states = [sample.state for sample in self._samples]
        policy_targets = [sample.policy_target for sample in self._samples]
        value_targets = [sample.value_target for sample in self._samples]
        return states, policy_targets, value_targets

    def samples(self) -> Sequence[TrainingSample]:
        """Returns a read-only view of retained samples."""
        return tuple(self._samples)

    def sample_batch(self, batch_size: int) -> Sequence[TrainingSample]:
        """Returns a random batch, or all samples if the buffer is smaller."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._samples) <= batch_size:
            return tuple(self._samples)
        return tuple(random.sample(self._samples, batch_size))
