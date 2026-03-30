"""Tests for the torch-backed policy/value evaluator."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.evaluator import TorchPolicyValueEvaluator
from my_chess.chess_ai.model import TORCH_AVAILABLE, build_model
from my_chess.chess_core.chessboard import Chessboard


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for evaluator tests")
class TestEvaluator(unittest.TestCase):
    """Tests legal-prior extraction from model outputs."""

    def test_evaluator_returns_normalized_legal_priors(self):
        board = Chessboard("test")
        board.init_board()
        evaluator = TorchPolicyValueEvaluator(build_model())

        priors, value = evaluator(board)

        self.assertTrue(priors)
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=5)
        self.assertTrue(-1.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
