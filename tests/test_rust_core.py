"""Tests for the native Rust rules engine and batched MCTS bridge."""

import os
import random
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import move_to_action_id
from my_chess.chess_ai.encoding import (
    BOARD_COLS,
    BOARD_ROWS,
    NUM_CHANNELS,
    _fill_planes,
)
from my_chess.chess_ai.evaluator import TorchPolicyValueEvaluator
from my_chess.chess_ai.mcts import MCTS, MCTSConfig
from my_chess.chess_ai.model import TORCH_AVAILABLE, build_model
from my_chess.chess_ai.self_play import play_games_batched_rust
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.rust_bridge import (
    RUST_AVAILABLE,
    rust_encode_board_numpy,
    rust_gen_legal_action_ids,
)


class TestRustCore(unittest.TestCase):
    """Verifies bit-for-bit parity between the Rust core and Python rules."""

    def test_rust_core_is_loaded(self):
        self.assertTrue(RUST_AVAILABLE, "Rust core DLL should be loaded")

    def test_legal_moves_and_encoding_match_python_across_playout(self):
        rng = random.Random(42)
        board = Chessboard("parity")
        board.init_board()

        for _ in range(60):
            py_actions = sorted(move_to_action_id(m) for m in board.get_legal_moves())
            rs_actions = sorted(rust_gen_legal_action_ids(board))
            self.assertEqual(py_actions, rs_actions)

            py_enc = np.zeros((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
            _fill_planes(board, py_enc)
            rs_enc = rust_encode_board_numpy(board)
            self.assertTrue(np.array_equal(py_enc, rs_enc))

            if not py_actions or board.is_end():
                board.init_board()
            else:
                board.apply_move(rng.choice(board.get_legal_moves()))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for Rust MCTS tests")
    def test_rust_mcts_and_batched_self_play(self):
        board = Chessboard("mcts_rust")
        board.init_board()
        evaluator = TorchPolicyValueEvaluator(build_model(), device="cpu")
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=16, virtual_loss_batch=4))

        dist, root_val = mcts.run_with_value(board)
        self.assertTrue(dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=5)
        self.assertTrue(-1.0 <= root_val <= 1.0)

        samples, results = play_games_batched_rust(
            evaluator,
            num_games=2,
            num_simulations=8,
            max_moves=6,
            virtual_loss_batch=4,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(samples)
        for sample in samples:
            sample.validate()


if __name__ == "__main__":
    unittest.main()
