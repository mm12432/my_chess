"""Tests for self-play sample generation."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.encoding import NUM_CHANNELS
from my_chess.chess_ai.self_play import SelfPlayGenerator


class TestSelfPlay(unittest.TestCase):
    """Tests that self-play produces structurally valid samples."""

    def test_play_game_returns_valid_samples(self):
        generator = SelfPlayGenerator()

        samples = generator.play_game(max_moves=4)

        self.assertTrue(samples)
        sample = samples[0]
        self.assertEqual(len(sample.state), NUM_CHANNELS)
        self.assertEqual(len(sample.policy_target), ACTION_SPACE_SIZE)
        self.assertTrue(-1.0 <= sample.value_target <= 1.0)


if __name__ == "__main__":
    unittest.main()
