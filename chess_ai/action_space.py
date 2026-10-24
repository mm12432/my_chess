"""Action-space helpers for mapping Xiangqi moves to stable policy indices."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from my_chess.chess_core.move import Move
from my_chess.chess_core.point import Point

BOARD_COLS = 9
BOARD_ROWS = 10
NUM_SQUARES = BOARD_COLS * BOARD_ROWS
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES


def square_to_index(col: int, row: int) -> int:
    """Maps a board coordinate to a stable square index."""
    if not (0 <= col < BOARD_COLS and 0 <= row < BOARD_ROWS):
        raise ValueError(f"Square out of bounds: ({col}, {row})")
    return row * BOARD_COLS + col


def index_to_point(index: int) -> Point:
    """Maps a square index back to a board coordinate."""
    if not (0 <= index < NUM_SQUARES):
        raise ValueError(f"Square index out of bounds: {index}")
    row, col = divmod(index, BOARD_COLS)
    return Point(col, row)


def move_to_action_id(move: Move) -> int:
    """Maps a move to a stable action-space index."""
    from_index = square_to_index(move.from_point.x, move.from_point.y)
    to_index = square_to_index(move.to_point.x, move.to_point.y)
    return from_index * NUM_SQUARES + to_index


def action_id_to_move(action_id: int, piece_name: str | None = None) -> Move:
    """Maps an action-space index back to a move value object."""
    if not (0 <= action_id < ACTION_SPACE_SIZE):
        raise ValueError(f"Action id out of bounds: {action_id}")
    from_index, to_index = divmod(action_id, NUM_SQUARES)
    return Move(
        from_point=index_to_point(from_index),
        to_point=index_to_point(to_index),
        piece_name=piece_name,
    )


def _mirror_square(index: int) -> int:
    row, col = divmod(index, BOARD_COLS)
    return (BOARD_ROWS - 1 - row) * BOARD_COLS + col


_MIRROR_SQUARE = [_mirror_square(i) for i in range(NUM_SQUARES)]


def mirror_action_id(action_id: int) -> int:
    """Mirrors an action vertically (row -> 9 - row); an involution."""
    from_index, to_index = divmod(action_id, NUM_SQUARES)
    return _MIRROR_SQUARE[from_index] * NUM_SQUARES + _MIRROR_SQUARE[to_index]


def canonical_action_id(action_id: int, is_red_turn: bool) -> int:
    """Maps an absolute action id to the side-to-move network index (and back)."""
    return action_id if is_red_turn else mirror_action_id(action_id)


def _flip_square_horizontal(index: int) -> int:
    row, col = divmod(index, BOARD_COLS)
    return row * BOARD_COLS + (BOARD_COLS - 1 - col)


_FLIP_SQUARE_H = [_flip_square_horizontal(i) for i in range(NUM_SQUARES)]


def flip_action_horizontal(action_id: int) -> int:
    """Mirrors an action horizontally (col -> 8 - col); an involution."""
    from_index, to_index = divmod(action_id, NUM_SQUARES)
    return _FLIP_SQUARE_H[from_index] * NUM_SQUARES + _FLIP_SQUARE_H[to_index]


HORIZONTAL_FLIP_INDICES = np.array(
    [flip_action_horizontal(i) for i in range(ACTION_SPACE_SIZE)], dtype=np.int32
)


def legal_moves_to_mask(legal_moves: Iterable[Move]) -> List[int]:
    """Builds a dense 0/1 legal-action mask from legal moves."""
    mask = [0] * ACTION_SPACE_SIZE
    for move in legal_moves:
        mask[move_to_action_id(move)] = 1
    return mask
