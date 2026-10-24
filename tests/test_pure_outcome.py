"""Tests for pure-outcome training: canonical encoding and value masking."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import (
    ACTION_SPACE_SIZE,
    canonical_action_id,
    flip_action_horizontal,
    mirror_action_id,
    move_to_action_id,
)
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.self_play import (
    SelfPlayGenerator,
    dense_policy_target,
    winner_to_value,
)
from my_chess.chess_core.chessboard import MAX_HALFMOVES_WITHOUT_CAPTURE, Chessboard
from my_chess.chess_core.rust_bridge import RUST_AVAILABLE, RustBoardHandle


class TestCanonical(unittest.TestCase):
    """Red and Black must see mirrored-identical positions identically."""

    def test_mirror_is_involution(self):
        for action_id in range(0, ACTION_SPACE_SIZE, 97):
            self.assertEqual(mirror_action_id(mirror_action_id(action_id)), action_id)
            self.assertEqual(
                flip_action_horizontal(flip_action_horizontal(action_id)), action_id
            )

    def test_symmetric_position_encodes_identically_for_both_sides(self):
        red_view = Chessboard.from_fen("4k4/9/9/9/9/9/9/9/9/R3K4 w - - 0 1")
        black_view = Chessboard.from_fen("r3k4/9/9/9/9/9/9/9/9/4K4 b - - 0 1")
        red_planes = encode_board(red_view)[:-1]
        black_planes = encode_board(black_view)[:-1]
        self.assertEqual(red_planes, black_planes)

    def test_policy_target_uses_canonical_index(self):
        board = Chessboard("t")
        board.init_board()
        board.apply_move(board.get_legal_moves()[0])  # now Black to move
        move = board.get_legal_moves()[0]
        action_id = move_to_action_id(move)
        target = dense_policy_target({action_id: 1.0}, board.is_red_turn)
        self.assertEqual(target[canonical_action_id(action_id, False)], 1.0)


class TestValueMasking(unittest.TestCase):
    """Truncated games must not be taught as draws."""

    def test_truncated_game_has_zero_value_weight(self):
        generator = SelfPlayGenerator()
        samples = generator.play_game(max_moves=4)
        self.assertEqual(generator.last_result, "truncated")
        self.assertTrue(all(s.value_weight == 0.0 for s in samples))

    def test_finished_game_has_full_value_weight(self):
        samples = SelfPlayGenerator._finalize_records([], "Red")
        self.assertEqual(samples, [])
        sample = TrainingSample(
            state=[], policy_target=[], value_target=1.0, value_weight=1.0
        )
        self.assertEqual(sample.value_weight, 1.0)
        self.assertEqual(winner_to_value("Draw", True), 0.0)
        self.assertEqual(winner_to_value("Draw", False), 0.0)

    def test_natural_move_limit_triggers_draw(self):
        board = Chessboard("draw_test")
        board.init_board()
        board.halfmove_clock = MAX_HALFMOVES_WITHOUT_CAPTURE
        self.assertEqual(board.get_winner(), "Draw")
        if RUST_AVAILABLE:
            rb = RustBoardHandle.from_python_board(board)
            try:
                self.assertEqual(rb.get_winner_label(), "Draw")
            finally:
                rb.close()


if __name__ == "__main__":
    unittest.main()
