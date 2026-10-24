"""WXF (World Xiangqi Federation) notation conversion and parsing."""

from __future__ import annotations

import re
from typing import Optional

from my_chess.chess_core import chessman
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move

# WXF Piece letters: Red uppercase, Black lowercase
PIECE_TO_WXF = {
    "R": "R",
    "r": "r",
    "N": "H",
    "n": "h",  # Knight / Horse (H is standard WXF, N also used)
    "B": "E",
    "b": "e",  # Elephant (E is standard WXF, B also used)
    "A": "A",
    "a": "a",  # Advisor
    "K": "K",
    "k": "k",  # King
    "C": "C",
    "c": "c",  # Cannon
    "P": "P",
    "p": "p",  # Pawn
}


def move_to_wxf(
    piece: chessman.Chessman, from_col: int, from_row: int, to_col: int, to_row: int
) -> str:
    """Converts a move to standard WXF notation (e.g. 'P9+1', 'c2=5', 'H8+7').

    Coordinates:
    - col: 0 to 8 (0 = left, 8 = right from Red's perspective)
    - row: 0 to 9 (0 = bottom Red back rank, 9 = top Black back rank)
    WXF File numbering:
    - Red: 1 to 9 from Red's right to left (col 8 -> 1, col 0 -> 9). file = 9 - col
    - Black: 1 to 9 from Black's right to left (col 0 -> 1, col 8 -> 9). file = col + 1
    """
    p_char = piece.fen_char
    letter = PIECE_TO_WXF.get(p_char, p_char)
    is_red = piece.is_red

    from_file = (9 - from_col) if is_red else (from_col + 1)
    dy = to_row - from_row

    if dy == 0:
        action = "="
        to_dest = (9 - to_col) if is_red else (to_col + 1)
    elif (is_red and dy > 0) or (not is_red and dy < 0):
        action = "+"
        if p_char.upper() in ("R", "C", "P", "K"):
            to_dest = abs(dy)
        else:
            to_dest = (9 - to_col) if is_red else (to_col + 1)
    else:
        action = "-"
        if p_char.upper() in ("R", "C", "P", "K"):
            to_dest = abs(dy)
        else:
            to_dest = (9 - to_col) if is_red else (to_col + 1)

    return f"{letter}{from_file}{action}{to_dest}"


def normalize_wxf(notation: str) -> str:
    """Normalizes variations in WXF notation (case, N->H, B->E)."""
    s = notation.strip()
    # Normalize knight/elephant letters
    if len(s) >= 4:
        first = s[0]
        if first == "N":
            first = "H"
        elif first == "n":
            first = "h"
        elif first == "B":
            first = "E"
        elif first == "b":
            first = "e"
        s = first + s[1:]
    return s


def parse_wxf_move(board: Chessboard, wxf_str: str) -> Optional[Move]:
    """Finds the legal move on `board` that matches `wxf_str`."""
    norm_target = normalize_wxf(wxf_str).lower()
    legal_moves = board.get_legal_moves()

    for move in legal_moves:
        piece = board.get_chessman(move.from_point.x, move.from_point.y)
        if piece is None:
            continue
        gen_wxf = move_to_wxf(
            piece,
            move.from_point.x,
            move.from_point.y,
            move.to_point.x,
            move.to_point.y,
        )
        if normalize_wxf(gen_wxf).lower() == norm_target:
            return move

    # Front/back disambiguation (e.g. +n+6, -n+5, +p+1)
    s = wxf_str.strip()
    if len(s) >= 4 and s[0] in ("+", "-"):
        front = s[0] == "+"
        p_type = s[1].upper()
        if p_type == "H":
            p_type = "N"
        if p_type == "E":
            p_type = "B"
        action = s[2]
        try:
            dest_val = int(s[3])
        except ValueError:
            return None

        is_red = board.is_red_turn
        matching_moves = []
        for move in legal_moves:
            piece = board.get_chessman(move.from_point.x, move.from_point.y)
            if piece is None or piece.fen_char.upper() != p_type:
                continue
            to_col = move.to_point.x
            dy = move.to_point.y - move.from_point.y
            to_file = (9 - to_col) if is_red else (to_col + 1)
            dist = abs(dy)
            act = (
                "="
                if dy == 0
                else ("+" if ((is_red and dy > 0) or (not is_red and dy < 0)) else "-")
            )
            target = dist if p_type in ("R", "C", "P", "K") and act != "=" else to_file
            if act == action and target == dest_val:
                matching_moves.append((move, move.from_point.y))

        if matching_moves:
            matching_moves.sort(key=lambda item: item[1], reverse=is_red)
            return matching_moves[0][0] if front else matching_moves[-1][0]

    return None
