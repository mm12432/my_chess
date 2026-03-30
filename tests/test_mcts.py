"""Tests for minimal MCTS search behavior."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.mcts import MCTS, MCTSConfig, uniform_policy_value_fn
from my_chess.chess_core.chessboard import Chessboard


class TestMCTS(unittest.TestCase):
    """Tests search output shape and probability normalization."""

    def test_run_returns_normalized_root_distribution(self):
        board = Chessboard("test")
        board.init_board()
        mcts = MCTS(uniform_policy_value_fn, MCTSConfig(num_simulations=8))

        distribution = mcts.run(board)

        self.assertTrue(distribution)
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=6)
        for probability in distribution.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
