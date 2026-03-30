"""Tests for training-sample and replay-buffer helpers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.dataset import ReplayBuffer, TrainingSample
from my_chess.chess_ai.encoding import empty_tensor


class TestDatasetHelpers(unittest.TestCase):
    """Tests sample validation and replay-buffer behavior."""

    def _sample(self, value_target: float = 1.0) -> TrainingSample:
        policy_target = [0.0] * ACTION_SPACE_SIZE
        policy_target[0] = 1.0
        return TrainingSample(
            state=empty_tensor(),
            policy_target=policy_target,
            value_target=value_target,
        )

    def test_training_sample_validate(self):
        sample = self._sample(0.5)
        sample.validate()

    def test_training_sample_rejects_bad_value_target(self):
        sample = self._sample(2.0)
        with self.assertRaises(ValueError):
            sample.validate()

    def test_replay_buffer_evicts_oldest(self):
        buffer = ReplayBuffer(capacity=2)
        sample_a = self._sample(-1.0)
        sample_b = self._sample(0.0)
        sample_c = self._sample(1.0)

        buffer.add(sample_a)
        buffer.add(sample_b)
        buffer.add(sample_c)

        self.assertEqual(len(buffer), 2)
        self.assertEqual(
            [sample.value_target for sample in buffer.samples()],
            [0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
