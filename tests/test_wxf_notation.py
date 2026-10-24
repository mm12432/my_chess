"""Tests for WXF notation conversions."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move
from my_chess.chess_core.wxf_notation import move_to_wxf, parse_wxf_move


class TestWxfNotation(unittest.TestCase):
    def test_red_opening_moves(self):
        board = Chessboard("t")
        board.init_board()
        red_pawn = board.get_chessman(0, 3)
        self.assertEqual(move_to_wxf(red_pawn, 0, 3, 0, 4), "P9+1")

        central_cannon = board.get_chessman(1, 2)
        self.assertEqual(move_to_wxf(central_cannon, 1, 2, 4, 2), "C8=5")

    def test_black_opening_moves(self):
        board = Chessboard("t")
        board.init_board()
        board.apply_move(Move.from_coords(0, 3, 0, 4))
        black_cannon = board.get_chessman(1, 7)
        self.assertEqual(move_to_wxf(black_cannon, 1, 7, 4, 7), "c2=5")

        parsed = parse_wxf_move(board, "c2=5")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed.from_point.x, parsed.from_point.y), (1, 7))
        self.assertEqual((parsed.to_point.x, parsed.to_point.y), (4, 7))

    def test_knight_move_parsing(self):
        board = Chessboard("t")
        board.init_board()
        board.apply_move(Move.from_coords(0, 3, 0, 4))
        # Black Horse at (1, 9) jumping to (2, 7) -> h2+3
        parsed = parse_wxf_move(board, "h2+3")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed.from_point.x, parsed.from_point.y), (1, 9))
        self.assertEqual((parsed.to_point.x, parsed.to_point.y), (2, 7))


if __name__ == "__main__":
    unittest.main()
