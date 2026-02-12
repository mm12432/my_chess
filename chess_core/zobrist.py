"""
This module implements Zobrist Hashing for Chinese Chess.
It provides a way to hash the board state into a unique integer to efficiently check for
repeated positions (draws) and enabling transposition tables for AI.
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_chess.chess_core import chessboard, chessman


class Zobrist:
    """
    Implements Zobrist Hashing for Chinese Chess board states.
    """

    def __init__(self):
        """Initializes Zobrist keys for pieces and turn."""
        # 14 piece types (7 * 2 colors)
        # Positions 0..89 (9x10)
        # piece_keys[piece_index][position_index]
        self.piece_keys = [
            [random.getrandbits(64) for _ in range(90)] for _ in range(14)
        ]
        self.turn_key = random.getrandbits(64)

        # Map piece name/type to index 0..13
        # Red: K=0, A=1, B=2, N=3, R=4, C=5, P=6
        # Black: k=7, a=8, b=9, n=10, r=11, c=12, p=13
        self.piece_map = {
            "red_king": 0,
            "red_mandarin": 1,
            "red_elephant": 2,
            "red_knight": 3,
            "red_rook": 4,
            "red_cannon": 5,
            "red_pawn": 6,
            "black_king": 7,
            "black_mandarin": 8,
            "black_elephant": 9,
            "black_knight": 10,
            "black_rook": 11,
            "black_cannon": 12,
            "black_pawn": 13,
        }

    def get_piece_index(self, piece: "chessman.Chessman") -> int:
        """
        Calculates the unique index (0-13) for a piece type and color.
        """
        # Determine index based on piece name or type
        # Assuming piece.name contains standard identifiers
        # But piece.name_en is "red_rook_1", etc.
        # We need the base name.
        # Let's rely on type and color.

        base_idx = 0
        if not piece.is_red:
            base_idx = 7

        p_char = piece.fen_char.upper()
        type_offset = 0
        if p_char == "K":
            type_offset = 0
        elif p_char == "A":
            type_offset = 1
        elif p_char == "B":
            type_offset = 2
        elif p_char == "N":
            type_offset = 3
        elif p_char == "R":
            type_offset = 4
        elif p_char == "C":
            type_offset = 5
        elif p_char == "P":
            type_offset = 6

        return base_idx + type_offset

    def get_position_index(self, col: int, row: int) -> int:
        """
        Maps 2D board coordinates (col, row) to a 1D index (0-89).
        """
        # Row 0..9, Col 0..8
        return row * 9 + col

    def hash_board(self, chessboard: "chessboard.Chessboard") -> int:
        """
        Computes the Zobrist hash for the entire board state.
        """
        h = 0
        if chessboard.is_red_turn:
            h ^= self.turn_key

        for col in range(9):
            for row in range(10):
                piece = chessboard.chessmans[col][row]
                if piece:
                    p_idx = self.get_piece_index(piece)
                    pos_idx = self.get_position_index(col, row)
                    h ^= self.piece_keys[p_idx][pos_idx]
        return h

    def update_hash(
        self,
        current_hash,
        piece,
        old_col,
        old_row,
        new_col,
        new_row,
        captured_piece=None,
    ) -> int:
        """
        Incrementally updates the hash based on a move.
        """
        # XOR out old position
        p_idx = self.get_piece_index(piece)
        old_pos_idx = self.get_position_index(old_col, old_row)
        current_hash ^= self.piece_keys[p_idx][old_pos_idx]

        # XOR in new position
        new_pos_idx = self.get_position_index(new_col, new_row)
        current_hash ^= self.piece_keys[p_idx][new_pos_idx]

        # If capture, XOR out captured piece
        if captured_piece:
            cap_idx = self.get_piece_index(captured_piece)
            # Captured piece is at new_col, new_row
            current_hash ^= self.piece_keys[cap_idx][new_pos_idx]

        # Toggle turn
        current_hash ^= self.turn_key

        return current_hash
