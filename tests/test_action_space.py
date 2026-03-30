"""Tests for Xiangqi action-space helpers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import (
    ACTION_SPACE_SIZE,
    action_id_to_move,
    legal_moves_to_mask,
    move_to_action_id,
    square_to_index,
)
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


class TestActionSpace(unittest.TestCase):
    """Tests stable move-index mapping and legal masks."""

    def test_square_to_index(self):
        self.assertEqual(square_to_index(0, 0), 0)
        self.assertEqual(square_to_index(8, 0), 8)
        self.assertEqual(square_to_index(0, 1), 9)
        self.assertEqual(square_to_index(8, 9), 89)

    def test_move_round_trip(self):
        move = Move.from_coords(1, 0, 2, 2, "red_knight_left")
        action_id = move_to_action_id(move)
        recovered = action_id_to_move(action_id, piece_name="red_knight_left")

        self.assertEqual(recovered, move)

    def test_legal_moves_mask_marks_only_legal_actions(self):
        board = Chessboard("test")
        board.init_board()
        legal_moves = board.get_legal_moves()

        mask = legal_moves_to_mask(legal_moves)

        self.assertEqual(len(mask), ACTION_SPACE_SIZE)
        knight_move = Move.from_coords(1, 0, 2, 2, "red_knight_left")
        illegal_move = Move.from_coords(4, 0, 4, 9, "red_king")
        self.assertEqual(mask[move_to_action_id(knight_move)], 1)
        self.assertEqual(mask[move_to_action_id(illegal_move)], 0)


if __name__ == "__main__":
    unittest.main()
