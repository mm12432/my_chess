"""Board encoding helpers for neural-network friendly Xiangqi input tensors.

Encoding is canonical (side-to-move perspective): when Black is to move the
board is mirrored vertically (row -> 9 - row) and colours are swapped, so the
network always "plays from the bottom". Red and Black experience is shared.
This is a pure board symmetry — no game knowledge is injected.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from my_chess.chess_core.chessboard import Chessboard

BOARD_COLS = 9
BOARD_ROWS = 10
# Bump when board/action encoding changes; older checkpoints/replays are unusable.
ENCODING_VERSION = "canonical_v1"

# Channel order is fixed to keep model I/O stable.
# Under canonical encoding "red_*" means "side to move" and "black_*" means
# "opponent"; names are kept for backward compatibility.
CHANNEL_NAMES = [
    "red_rook",
    "red_knight",
    "red_cannon",
    "red_elephant",
    "red_mandarin",
    "red_king",
    "red_pawn",
    "black_rook",
    "black_knight",
    "black_cannon",
    "black_elephant",
    "black_mandarin",
    "black_king",
    "black_pawn",
    "side_to_move",
]
NUM_CHANNELS = len(CHANNEL_NAMES)
SIDE_CHANNEL = NUM_CHANNELS - 1
OPPONENT_OFFSET = 7

PIECE_TYPE_INDEX = {"R": 0, "N": 1, "C": 2, "B": 3, "A": 4, "K": 5, "P": 6}


def _fill_planes(board: Chessboard, out: np.ndarray) -> None:
    """Writes canonical piece planes and side plane into a C x H x W array."""
    is_red_turn = board.is_red_turn
    for piece in board.chessmans_hash.values():
        channel = PIECE_TYPE_INDEX[piece.fen_char.upper()]
        if piece.is_red != is_red_turn:
            channel += OPPONENT_OFFSET
        row = piece.row_num if is_red_turn else BOARD_ROWS - 1 - piece.row_num
        out[channel, row, piece.col_num] = 1.0
    if is_red_turn:
        out[SIDE_CHANNEL, :, :] = 1.0


def empty_tensor() -> List[List[List[int]]]:
    """Builds an all-zero input tensor with stable shape C x H x W."""
    return [
        [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        for _ in range(NUM_CHANNELS)
    ]


def encode_board_numpy(board: Chessboard) -> np.ndarray:
    """Encodes a board into a canonical C x H x W float32 array."""
    arr = np.zeros((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    _fill_planes(board, arr)
    return arr


def encode_boards_batch(boards: Sequence[Chessboard]) -> np.ndarray:
    """Encodes boards into a single N x C x H x W float32 array."""
    batch = np.zeros(
        (len(boards), NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
    )
    for idx, board in enumerate(boards):
        _fill_planes(board, batch[idx])
    return batch


def encode_board(board: Chessboard) -> List[List[List[int]]]:
    """Encodes a board into a canonical multi-channel 0/1 nested list.

    Tensor layout:
    - Channels: own piece planes, opponent piece planes, side-to-move
    - Rows: 0..9 (row 0 = side-to-move's back rank)
    - Cols: 0..8
    """
    return encode_board_numpy(board).astype(np.int8).tolist()


def tensor_shape(tensor: List[List[List[int]]]) -> Tuple[int, int, int]:
    """Returns the shape of a C x H x W tensor."""
    channels = len(tensor)
    rows = len(tensor[0]) if channels else 0
    cols = len(tensor[0][0]) if rows else 0
    return channels, rows, cols
