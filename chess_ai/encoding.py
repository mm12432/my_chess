"""Board encoding helpers for neural-network friendly Xiangqi input tensors."""

from __future__ import annotations

from typing import Dict, List, Tuple

from my_chess.chess_core.chessboard import Chessboard

BOARD_COLS = 9
BOARD_ROWS = 10

# Channel order is fixed to keep model I/O stable.
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

PIECE_CHANNEL_BY_NAME: Dict[str, int] = {
    "Rook:True": 0,
    "Knight:True": 1,
    "Cannon:True": 2,
    "Elephant:True": 3,
    "Mandarin:True": 4,
    "King:True": 5,
    "Pawn:True": 6,
    "Rook:False": 7,
    "Knight:False": 8,
    "Cannon:False": 9,
    "Elephant:False": 10,
    "Mandarin:False": 11,
    "King:False": 12,
    "Pawn:False": 13,
}


def empty_tensor() -> List[List[List[int]]]:
    """Builds an all-zero input tensor with stable shape C x H x W."""
    return [
        [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        for _ in range(NUM_CHANNELS)
    ]


def encode_board(board: Chessboard) -> List[List[List[int]]]:
    """Encodes a board into a multi-channel 0/1 tensor.

    Tensor layout:
    - Channels: piece-type/color planes plus side-to-move
    - Rows: 0..9
    - Cols: 0..8
    """
    tensor = empty_tensor()

    for piece in board.chessmans_hash.values():
        key = f"{type(piece).__name__}:{piece.is_red}"
        channel = PIECE_CHANNEL_BY_NAME[key]
        tensor[channel][piece.row_num][piece.col_num] = 1

    side_value = 1 if board.is_red_turn else 0
    side_channel = NUM_CHANNELS - 1
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            tensor[side_channel][row][col] = side_value

    return tensor


def tensor_shape(tensor: List[List[List[int]]]) -> Tuple[int, int, int]:
    """Returns the shape of a C x H x W tensor."""
    channels = len(tensor)
    rows = len(tensor[0]) if channels else 0
    cols = len(tensor[0][0]) if rows else 0
    return channels, rows, cols
