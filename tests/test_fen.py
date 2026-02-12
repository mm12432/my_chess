import unittest
import sys
import os

# Add project parent directory to path to allow importing MyChess as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MyChess.Chess_Core.Chessboard import Chessboard


class TestFEN(unittest.TestCase):
    def test_initial_fen(self):
        board = Chessboard("test")
        board.init_board()
        fen = board.to_fen()
        # Note: MyChess might define red/black sides differently than standard FEN?
        # In MyChess: Row 0 is Red (Bottom), Row 9 is Black (Top).
        # FEN usually goes from Rank 9 (Top/Black) to Rank 0 (Bottom/Red).
        # My implementation of to_fen iterates 9 down to 0.
        # So row 9 (Black pieces) should come first.
        # Black pieces: r n b a k a b n r
        # Row 9: rnbakabnr
        # Row 8: 9
        # Row 7: 1c5c1
        # Row 6: p1p1p1p1p
        # Row 5: 9
        # Row 4: 9
        # Row 3: P1P1P1P1P
        # Row 2: 1C5C1
        # Row 1: 9
        # Row 0: RNBAKABNR

        expected_fen_prefix = (
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
        )
        self.assertTrue(
            fen.startswith(expected_fen_prefix),
            f"Expected start with {expected_fen_prefix}, got {fen}",
        )

    def test_from_fen_initial(self):
        start_fen = (
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
        )
        board = Chessboard.from_fen(start_fen)

        # Check explicit positions
        # Row 9 (Top), Col 0: Black Rook 'r'
        p = board.get_chessman(0, 9)
        self.assertIsNotNone(p)
        self.assertEqual(p.fen_char, "r")
        self.assertFalse(p.is_red)

        # Row 0 (Bottom), Col 4: Red King 'K'
        p = board.get_chessman(4, 0)
        self.assertIsNotNone(p)
        self.assertEqual(p.fen_char, "K")
        self.assertTrue(p.is_red)

        # Center (4, 4) should be empty
        p = board.get_chessman(4, 4)
        self.assertIsNone(p)

        # Turn
        self.assertTrue(board.is_red_turn)

    def test_from_fen_custom(self):
        # A simple endgame FEN: Red King and Rook vs Black King
        # Red King at 4,0. Red Rook at 4,4. Black King at 4,9.
        # Rows:
        # 9: ....k.... -> 4k4
        # 8: 9
        # 7: 9
        # 6: 9
        # 5: 9
        # 4: ....R.... -> 4R4
        # 3: 9
        # 2: 9
        # 1: 9
        # 0: ....K.... -> 4K4

        fen = "4k4/9/9/9/9/4R4/9/9/9/4K4 b - - 0 1"
        board = Chessboard.from_fen(fen)

        self.assertFalse(board.is_red_turn)

        k = board.get_chessman(4, 9)
        self.assertEqual(k.fen_char, "k")

        R = board.get_chessman(4, 4)
        self.assertEqual(R.fen_char, "R")

        K = board.get_chessman(4, 0)
        self.assertEqual(K.fen_char, "K")

        self.assertIsNone(board.get_chessman(0, 0))


if __name__ == "__main__":
    unittest.main()
