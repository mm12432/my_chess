"""Tests for board-to-tensor encoding helpers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.encoding import (
    CHANNEL_NAMES,
    NUM_CHANNELS,
    encode_board,
    tensor_shape,
)
from my_chess.chess_core.chessboard import Chessboard


class TestEncoding(unittest.TestCase):
    """Tests stable board encoding for NN input."""

    def test_initial_board_encoding_shape(self):
        board = Chessboard("test")
        board.init_board()

        tensor = encode_board(board)

        self.assertEqual(tensor_shape(tensor), (NUM_CHANNELS, 10, 9))

    def test_initial_board_piece_planes(self):
        board = Chessboard("test")
        board.init_board()

        tensor = encode_board(board)
        red_rook_channel = CHANNEL_NAMES.index("red_rook")
        black_king_channel = CHANNEL_NAMES.index("black_king")

        self.assertEqual(tensor[red_rook_channel][0][0], 1)
        self.assertEqual(tensor[red_rook_channel][0][8], 1)
        self.assertEqual(tensor[black_king_channel][9][4], 1)

    def test_side_to_move_plane(self):
        board = Chessboard("test")
        board.init_board()

        tensor = encode_board(board)
        side_channel = CHANNEL_NAMES.index("side_to_move")
        self.assertEqual(tensor[side_channel][0][0], 1)
        self.assertEqual(tensor[side_channel][9][8], 1)

        board.apply_move(board.get_legal_moves()[0])
        tensor = encode_board(board)
        self.assertEqual(tensor[side_channel][0][0], 0)
        self.assertEqual(tensor[side_channel][9][8], 0)


if __name__ == "__main__":
    unittest.main()
