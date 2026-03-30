"""Tests for optional model definitions."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.encoding import NUM_CHANNELS
from my_chess.chess_ai.model import NetworkConfig, TORCH_AVAILABLE, require_torch


class TestModelConfig(unittest.TestCase):
    """Tests framework-independent parts of the model module."""

    def test_default_network_config(self):
        config = NetworkConfig()
        self.assertEqual(config.input_channels, NUM_CHANNELS)
        self.assertEqual(config.policy_size, ACTION_SPACE_SIZE)

    def test_require_torch_matches_runtime_dependency(self):
        if TORCH_AVAILABLE:
            require_torch()
        else:
            with self.assertRaises(RuntimeError):
                require_torch()


if __name__ == "__main__":
    unittest.main()
