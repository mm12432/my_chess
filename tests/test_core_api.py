"""Core board API tests for AI-facing move and clone helpers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


class TestCoreBoardApi(unittest.TestCase):
    """Tests the non-UI move API exposed by Chessboard."""

    def test_get_legal_moves_returns_opening_moves(self):
        """Opening position should expose legal moves as Move objects."""
        board = Chessboard("test")
        board.init_board()

        legal_moves = board.get_legal_moves()

        self.assertTrue(legal_moves, "开局应该存在合法走法")
        self.assertIn(
            Move.from_coords(1, 0, 2, 2, "red_knight_left"),
            legal_moves,
            "红左马应当能在开局走到 (2,2)",
        )

    def test_apply_move_preserves_existing_board_behavior(self):
        """Applying a Move should update board state and move history."""
        board = Chessboard("test")
        board.init_board()

        did_move = board.apply_move(
            Move.from_coords(1, 0, 2, 2, "red_knight_left")
        )

        self.assertTrue(did_move)
        self.assertIsNone(board.get_chessman(1, 0))
        moved_piece = board.get_chessman(2, 2)
        self.assertIsNotNone(moved_piece)
        self.assertEqual(moved_piece.name, "red_knight_left")
        self.assertFalse(board.is_red_turn)
        self.assertEqual(board.moves_history[-1], "马八进七")

    def test_clone_creates_independent_copy(self):
        """Cloning should isolate later mutations from the original board."""
        board = Chessboard("test")
        board.init_board()
        cloned = board.clone()

        did_move = cloned.apply_move(Move.from_coords(1, 0, 2, 2, "red_knight_left"))

        self.assertTrue(did_move)
        self.assertIsNotNone(board.get_chessman(1, 0))
        self.assertIsNone(board.get_chessman(2, 2))
        self.assertTrue(board.is_red_turn)
        self.assertFalse(cloned.is_red_turn)


if __name__ == "__main__":
    unittest.main()
