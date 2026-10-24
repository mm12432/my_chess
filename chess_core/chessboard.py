"""
This module defines the Chessboard class, which represents the game state, board configuration,
and game logic. It handles moving pieces, turn management, history tracking, and game end conditions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from my_chess.chess_core import chessman
from my_chess.chess_core.move import Move

if TYPE_CHECKING:
    from my_chess.chess_core.chessman import Chessman as ChessmanType

# 自然限着：连续 120 步（60 回合）无吃子判和
MAX_HALFMOVES_WITHOUT_CAPTURE = 120


@dataclass
class MoveUndo:
    """Snapshot used to reverse a fast search move."""

    piece: "ChessmanType"
    from_x: int
    from_y: int
    to_x: int
    to_y: int
    captured: Optional["ChessmanType"]
    prev_hash: int
    prev_halfmove_clock: int
    history_side: str
    history_chessman: object
    history_last_pos: Optional[Tuple[int, int]]
    history_repeat: int


class Chessboard:
    """
    Represents the chess board and game state.
    Manages pieces, turns, history, and game logic.
    """

    def __init__(self, name: str) -> None:
        self.__name = name
        self._is_red_turn = True
        # Initialize 9x10 board with None
        self.__chessmans: List[List[Optional[ChessmanType]]] = [
            ([None] * 10) for _ in range(9)
        ]
        self.__chessmans_hash: Dict[str, ChessmanType] = {}
        # 用于长将/长捉检测的历史记录
        self.__history: Dict[str, Dict] = {
            "red": {"chessman": None, "last_pos": None, "repeat": 0},
            "black": {"chessman": None, "last_pos": None, "repeat": 0},
        }
        self.moves_history: List[str] = []

        from my_chess.chess_core.zobrist import Zobrist

        self.zobrist = Zobrist()
        self.current_hash = 0
        self.hash_history: Dict[int, int] = {}
        self.halfmove_clock: int = 0

    @property
    def is_red_turn(self) -> bool:
        """Returns True if it is currently Red's turn."""
        return self._is_red_turn

    @property
    def name(self) -> str:
        """Returns the name of the chessboard."""
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def chessmans(self) -> List[List[Optional[ChessmanType]]]:
        """Returns the 2D grid of chess pieces."""
        return self.__chessmans

    @property
    def chessmans_hash(self) -> Dict[str, ChessmanType]:
        """Returns a dictionary of all active chess pieces by name."""
        return self.__chessmans_hash

    def init_board(self) -> None:
        """Initializes the board with the standard layout of pieces."""
        # Red Pieces
        chessman.Rook(" 车l红 ", "red_rook_left", True, self).add_to_board(0, 0)
        chessman.Rook(" 车r红 ", "red_rook_right", True, self).add_to_board(8, 0)
        chessman.Knight(" 马l红 ", "red_knight_left", True, self).add_to_board(1, 0)
        chessman.Knight(" 马r红 ", "red_knight_right", True, self).add_to_board(7, 0)
        chessman.Cannon(" 炮l红 ", "red_cannon_left", True, self).add_to_board(1, 2)
        chessman.Cannon(" 炮r红 ", "red_cannon_right", True, self).add_to_board(7, 2)
        chessman.Elephant(" 相l红 ", "red_elephant_left", True, self).add_to_board(2, 0)
        chessman.Elephant(" 相r红 ", "red_elephant_right", True, self).add_to_board(
            6, 0
        )
        chessman.Mandarin(" 仕l红 ", "red_mandarin_left", True, self).add_to_board(3, 0)
        chessman.Mandarin(" 仕r红 ", "red_mandarin_right", True, self).add_to_board(
            5, 0
        )
        chessman.King(" 帅 红 ", "red_king", True, self).add_to_board(4, 0)
        chessman.Pawn(" 兵1红 ", "red_pawn_1", True, self).add_to_board(0, 3)
        chessman.Pawn(" 兵2红 ", "red_pawn_2", True, self).add_to_board(2, 3)
        chessman.Pawn(" 兵3红 ", "red_pawn_3", True, self).add_to_board(4, 3)
        chessman.Pawn(" 兵4红 ", "red_pawn_4", True, self).add_to_board(6, 3)
        chessman.Pawn(" 兵5红 ", "red_pawn_5", True, self).add_to_board(8, 3)

        # Black Pieces
        chessman.Rook(" 车l黑 ", "black_rook_left", False, self).add_to_board(0, 9)
        chessman.Rook(" 车r黑 ", "black_rook_right", False, self).add_to_board(8, 9)
        chessman.Knight(" 马l黑 ", "black_knight_left", False, self).add_to_board(1, 9)
        chessman.Knight(" 马r黑 ", "black_knight_right", False, self).add_to_board(7, 9)
        chessman.Cannon(" 炮l黑 ", "black_cannon_left", False, self).add_to_board(1, 7)
        chessman.Cannon(" 炮r黑 ", "black_cannon_right", False, self).add_to_board(7, 7)
        chessman.Elephant(" 象l黑 ", "black_elephant_left", False, self).add_to_board(
            2, 9
        )
        chessman.Elephant(" 象r黑 ", "black_elephant_right", False, self).add_to_board(
            6, 9
        )
        chessman.Mandarin(" 仕l黑 ", "black_mandarin_left", False, self).add_to_board(
            3, 9
        )
        chessman.Mandarin(" 仕r黑 ", "black_mandarin_right", False, self).add_to_board(
            5, 9
        )
        chessman.King(" 将 黑 ", "black_king", False, self).add_to_board(4, 9)
        chessman.Pawn(" 卒1黑 ", "black_pawn_1", False, self).add_to_board(0, 6)
        chessman.Pawn(" 卒2黑 ", "black_pawn_2", False, self).add_to_board(2, 6)
        chessman.Pawn(" 卒3黑 ", "black_pawn_3", False, self).add_to_board(4, 6)
        chessman.Pawn(" 卒4黑 ", "black_pawn_4", False, self).add_to_board(6, 6)
        chessman.Pawn(" 卒5黑 ", "black_pawn_5", False, self).add_to_board(8, 6)

        # Initialize Hash
        self.current_hash = self.zobrist.hash_board(self)
        self.hash_history = {self.current_hash: 1}
        self.halfmove_clock = 0

    def add_chessman(self, piece: ChessmanType, col_num: int, row_num: int) -> None:
        """Adds a piece to the board at the specified coordinates."""
        self.chessmans[col_num][row_num] = piece
        if piece.name not in self.__chessmans_hash:
            self.__chessmans_hash[piece.name] = piece

    def remove_chessman_target(self, col_num: int, row_num: int) -> None:
        """Removes a piece from the board at the target coordinates (capture)."""
        chessman_old = self.get_chessman(col_num, row_num)
        if chessman_old is not None:
            self.__chessmans_hash.pop(chessman_old.name)

    def remove_chessman_source(self, col_num: int, row_num: int) -> None:
        """Removes a piece from the board at the source coordinates (move away)."""
        self.chessmans[col_num][row_num] = None

    def calc_chessmans_moving_list(self) -> None:
        """Calculates valid moves for all pieces of the current turn's color."""
        for piece in self.__chessmans_hash.values():
            if piece.is_red == self._is_red_turn:
                piece.calc_moving_list()

    def clear_chessmans_moving_list(self) -> None:
        """Clears the valid moves list for all pieces."""
        for piece in self.__chessmans_hash.values():
            piece.clear_moving_list()

    def get_legal_moves(self) -> List[Move]:
        """Returns all currently legal moves as value objects.

        This API is intended for engine and AI code that should not depend on
        mutable `Chessman` state or UI-specific call paths.
        """
        from my_chess.chess_core.point import Point

        legal_moves: List[Move] = []
        self.clear_chessmans_moving_list()
        self.calc_chessmans_moving_list()
        for piece in self.__chessmans_hash.values():
            if piece.is_red != self._is_red_turn:
                continue
            # Copy coordinates only — deepcopy of Point objects is too slow for MCTS.
            from_point = Point(piece.position.x, piece.position.y)
            for target in piece.moving_list:
                legal_moves.append(
                    Move(
                        from_point=from_point,
                        to_point=Point(target.x, target.y),
                        piece_name=piece.name,
                    )
                )
        return legal_moves

    def apply_move(self, move: Move) -> bool:
        """Applies a move object while preserving existing board behavior."""
        piece = self.get_chessman(move.from_point.x, move.from_point.y)
        if piece is None:
            return False
        if move.piece_name is not None and piece.name != move.piece_name:
            return False
        if piece.is_red != self._is_red_turn:
            return False

        self.clear_chessmans_moving_list()
        piece.calc_moving_list()
        did_move = piece.move(move.to_point.x, move.to_point.y)
        if did_move:
            self.clear_chessmans_moving_list()
        return did_move

    def push_move(self, move: Move) -> MoveUndo:
        """Applies a known-legal move for search and returns an undo token.

        Skips notation and moving-list validation so MCTS can make/unmake
        without cloning the full board on every expansion.
        """
        piece = self.get_chessman(move.from_point.x, move.from_point.y)
        if piece is None:
            raise ValueError("push_move requires a piece on the from-square")
        if piece.is_red != self._is_red_turn:
            raise ValueError("push_move called on the wrong side to move")

        from_x, from_y = move.from_point.x, move.from_point.y
        to_x, to_y = move.to_point.x, move.to_point.y
        captured = self.get_chessman(to_x, to_y)
        history_side = self.red_or_black(piece)
        history = self.__history[history_side]
        undo = MoveUndo(
            piece=piece,
            from_x=from_x,
            from_y=from_y,
            to_x=to_x,
            to_y=to_y,
            captured=captured,
            prev_hash=self.current_hash,
            prev_halfmove_clock=self.halfmove_clock,
            history_side=history_side,
            history_chessman=history["chessman"],
            history_last_pos=history["last_pos"],
            history_repeat=int(history["repeat"]),
        )

        self.__chessmans[from_x][from_y] = None
        self.update_history(piece, to_x, to_y)
        self.current_hash = self.zobrist.update_hash(
            self.current_hash,
            piece,
            from_x,
            from_y,
            to_x,
            to_y,
            captured,
        )
        self.hash_history[self.current_hash] = (
            self.hash_history.get(self.current_hash, 0) + 1
        )
        if captured is not None:
            self.__chessmans_hash.pop(captured.name, None)
            captured.is_alive = False
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        piece.update_position(to_x, to_y)
        self.__chessmans[to_x][to_y] = piece
        self.__chessmans_hash[piece.name] = piece
        self._is_red_turn = not self._is_red_turn
        return undo

    def pop_move(self, undo: MoveUndo) -> None:
        """Reverses a previous push_move on this board."""
        count = self.hash_history.get(self.current_hash, 0)
        if count <= 1:
            self.hash_history.pop(self.current_hash, None)
        else:
            self.hash_history[self.current_hash] = count - 1
        self.current_hash = undo.prev_hash
        self.halfmove_clock = undo.prev_halfmove_clock

        piece = undo.piece
        self.__chessmans[undo.to_x][undo.to_y] = None
        if undo.captured is not None:
            undo.captured.is_alive = True
            self.__chessmans[undo.to_x][undo.to_y] = undo.captured
            self.__chessmans_hash[undo.captured.name] = undo.captured

        piece.update_position(undo.from_x, undo.from_y)
        self.__chessmans[undo.from_x][undo.from_y] = piece
        self.__chessmans_hash[piece.name] = piece
        self._is_red_turn = not self._is_red_turn

        history = self.__history[undo.history_side]
        history["chessman"] = undo.history_chessman
        history["last_pos"] = undo.history_last_pos
        history["repeat"] = undo.history_repeat

    def get_fast_winner(self) -> Optional[str]:
        """Terminal check without stalemate scan (for MCTS leaves)."""
        if self.hash_history.get(self.current_hash, 0) >= 3:
            return "Red" if self._is_red_turn else "Black"
        if self.get_chessman_by_name("red_king") is None:
            return "Black"
        if self.get_chessman_by_name("black_king") is None:
            return "Red"
        if self.halfmove_clock >= MAX_HALFMOVES_WITHOUT_CAPTURE:
            return "Draw"
        return None

    def clone(self) -> "Chessboard":
        """Creates a deep copy of the board for simulation/search."""
        return copy.deepcopy(self)

    def move_chessman(self, piece: ChessmanType, col_num: int, row_num: int) -> bool:
        """Moves a piece to the target coordinates, handling captures and turn switching."""
        if piece.is_red == self._is_red_turn:
            # Record move BEFORE moving
            from my_chess.chess_core.move_notation import MoveNotation

            move_str = MoveNotation.get_move_name(
                piece, piece.col_num, piece.row_num, col_num, row_num
            )

            captured = self.get_chessman(col_num, row_num)

            # Update Hash
            self.current_hash = self.zobrist.update_hash(
                self.current_hash,
                piece,
                piece.col_num,
                piece.row_num,
                col_num,
                row_num,
                captured,
            )
            self.hash_history[self.current_hash] = (
                self.hash_history.get(self.current_hash, 0) + 1
            )

            self.moves_history.append(move_str)
            if captured is not None:
                self.halfmove_clock = 0
            else:
                self.halfmove_clock += 1

            self.remove_chessman_target(col_num, row_num)
            self.add_chessman(piece, col_num, row_num)
            self._is_red_turn = not self._is_red_turn
            return True
        else:
            print("the wrong turn")
            return False

    def get_winner(self) -> Optional[str]:
        """Checks if there is a winner (returns 'Red', 'Black', or None).

        判定规则：
        1. 三次重复局面 → 造成重复的一方（长将/长捉方）判负
        2. 将/帅被吃 → 对方胜
        3. 困毙（当前走棋方无任何合法走法）→ 对方胜
        """
        # 长将/长捉检测：三次重复局面，造成重复的一方判负
        # move_chessman() 走完后已翻转回合，所以当前 _is_red_turn 是下一步走棋方
        # 造成重复的是刚走完的一方（即对方），该方判负
        if self.hash_history.get(self.current_hash, 0) >= 3:
            # 当前轮到谁走 = 无辜方 → 无辜方胜
            return "Red" if self._is_red_turn else "Black"

        red_king = self.get_chessman_by_name("red_king")
        black_king = self.get_chessman_by_name("black_king")

        if not red_king:
            return "Black"
        if not black_king:
            return "Red"

        if self.halfmove_clock >= MAX_HALFMOVES_WITHOUT_CAPTURE:
            return "Draw"

        # 绝杀/困毙检测：优先使用 Rust 引擎严格应将合法走子判定，
        # 若当前走棋方走任何一步都会送将（无严格合法走法），立即判对方获胜
        from my_chess.chess_core.rust_bridge import RUST_AVAILABLE, RustBoardHandle

        if RUST_AVAILABLE:
            rb = RustBoardHandle.from_python_board(self)
            try:
                rust_winner = rb.get_winner_label()
                if rust_winner is not None:
                    return rust_winner
            finally:
                rb.close()

        # 困毙检测：当前走棋方所有棋子都没有合法走法
        if self._is_stalemated():
            return "Black" if self._is_red_turn else "Red"

        return None

    def _is_stalemated(self) -> bool:
        """检查当前走棋方是否被困毙（无任何合法走法）。"""
        for piece in self.__chessmans_hash.values():
            if piece.is_red == self._is_red_turn:
                piece.calc_moving_list()
                if piece.moving_list:
                    return False
        return True

    def is_end(self) -> bool:
        """Checks if the game has ended."""
        return self.get_winner() is not None

    def update_history(self, piece: ChessmanType, col_num: int, row_num: int) -> None:
        """Updates the move history for repetition detection."""
        red_or_black_key = self.red_or_black(piece)
        history_chessman = self.__history[red_or_black_key]["chessman"]
        history_pos = self.__history[red_or_black_key]["last_pos"]
        if (
            history_chessman == piece
            and history_pos is not None
            and history_pos[0] == col_num
            and history_pos[1] == row_num
        ):
            self.__history[red_or_black_key]["repeat"] += 1
        else:
            self.__history[red_or_black_key]["repeat"] = 0
        self.__history[red_or_black_key]["chessman"] = piece
        self.__history[red_or_black_key]["last_pos"] = (
            piece.col_num,
            piece.row_num,
        )

    def red_or_black(self, piece: ChessmanType) -> str:
        """Returns 'red' or 'black' string based on piece color."""
        if piece.is_red:
            return "red"
        else:
            return "black"

    def get_chessman(self, col_num: int, row_num: int) -> Optional[ChessmanType]:
        """Returns the chessman at the specified coordinates."""
        return self.__chessmans[col_num][row_num]

    def get_chessman_by_name(self, name: str) -> Optional[ChessmanType]:
        """Returns a chessman by its unique name."""
        if name in self.__chessmans_hash:
            return self.__chessmans_hash[name]
        return None

    def get_top_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the first chessman above the given position."""
        for i in range(row_num + 1, 10, 1):
            current = self.chessmans[col_num][i]
            if current is not None:
                return current
        return None

    def get_bottom_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the first chessman below the given position."""
        for i in range(row_num - 1, -1, -1):
            current = self.chessmans[col_num][i]
            if current is not None:
                return current
        return None

    def get_left_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the first chessman to the left of the given position."""
        for i in range(col_num - 1, -1, -1):
            current = self.chessmans[i][row_num]
            if current is not None:
                return current
        return None

    def get_right_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the first chessman to the right of the given position."""
        for i in range(col_num + 1, 9, 1):
            current = self.chessmans[i][row_num]
            if current is not None:
                return current
        return None

    def get_top_second_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the second chessman above the given position."""
        count = 0
        for i in range(row_num + 1, 10, 1):
            current = self.chessmans[col_num][i]
            if current is not None:
                if count == 1:
                    return current
                else:
                    count += 1
        return None

    def get_bottom_second_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the second chessman below the given position."""
        count = 0
        for i in range(row_num - 1, -1, -1):
            current = self.chessmans[col_num][i]
            if current is not None:
                if count == 1:
                    return current
                else:
                    count += 1
        return None

    def get_left_second_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the second chessman to the left of the given position."""
        count = 0
        for i in range(col_num - 1, -1, -1):
            current = self.chessmans[i][row_num]
            if current is not None:
                if count == 1:
                    return current
                else:
                    count += 1
        return None

    def get_right_second_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        """Finds the second chessman to the right of the given position."""
        count = 0
        for i in range(col_num + 1, 9, 1):
            current = self.chessmans[i][row_num]
            if current is not None:
                if count == 1:
                    return current
                else:
                    count += 1
        return None

    def print_to_cl(self) -> None:
        """Prints the current board state to the command line."""
        screen = "\r\n"
        for i in range(9, -1, -1):
            for j in range(9):
                piece = self.__chessmans[j][i]
                if piece is not None:
                    screen += piece.name_cn
                else:
                    screen += "   .   "
            screen += "\r\n" * 3
        print(screen)

    def to_fen(self) -> str:
        """Serializes the board state to a FEN string."""
        fen_rows = []
        for row in range(9, -1, -1):
            empty_count = 0
            row_str = ""
            for col in range(9):
                piece = self.chessmans[col][row]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece.fen_char
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)

        board_fen = "/".join(fen_rows)
        turn_fen = "w" if self.is_red_turn else "b"
        return f"{board_fen} {turn_fen} - - 0 1"

    def clear_board(self) -> None:
        """Clears all pieces from the board."""
        self.__chessmans = [([None] * 10) for _ in range(9)]
        self.__chessmans_hash = {}
        self.__history = {
            "red": {"chessman": None, "last_pos": None, "repeat": 0},
            "black": {"chessman": None, "last_pos": None, "repeat": 0},
        }
        self.moves_history = []
        self.current_hash = 0
        self.hash_history = {}
        self.halfmove_clock = 0

    @classmethod
    def from_fen(cls, fen: str) -> "Chessboard":
        """Creates a Chessboard instance from a FEN string."""
        board = cls("FEN_Board")
        board.clear_board()

        parts = fen.split()
        board_fen = parts[0]
        turn_fen = parts[1] if len(parts) > 1 else "w"

        # 'w' or 'r' -> Red turn
        board._is_red_turn = turn_fen.lower() == "w" or turn_fen.lower() == "r"

        rows = board_fen.split("/")
        if len(rows) != 10:
            raise ValueError("Invalid FEN: Wrong number of rows")

        # Mapping FEN char to (Class, is_red, name_cn_base, name_en_base)
        char_map = {
            "R": (chessman.Rook, True, "车", "red_rook"),
            "r": (chessman.Rook, False, "车", "black_rook"),
            "N": (chessman.Knight, True, "马", "red_knight"),
            "n": (chessman.Knight, False, "马", "black_knight"),
            "C": (chessman.Cannon, True, "炮", "red_cannon"),
            "c": (chessman.Cannon, False, "炮", "black_cannon"),
            "B": (chessman.Elephant, True, "相", "red_elephant"),
            "b": (chessman.Elephant, False, "象", "black_elephant"),
            "A": (chessman.Mandarin, True, "仕", "red_mandarin"),
            "a": (chessman.Mandarin, False, "士", "black_mandarin"),
            "K": (chessman.King, True, "帅", "red_king"),
            "k": (chessman.King, False, "将", "black_king"),
            "P": (chessman.Pawn, True, "兵", "red_pawn"),
            "p": (chessman.Pawn, False, "卒", "black_pawn"),
        }

        counts = {}

        for r_idx, row_data in enumerate(rows):
            actual_row = 9 - r_idx
            current_col = 0
            for char in row_data:
                if char.isdigit():
                    current_col += int(char)
                else:
                    if char in char_map:
                        piece_cls, is_red, cn_base, en_base = char_map[char]

                        if en_base not in counts:
                            counts[en_base] = 0
                        counts[en_base] += 1

                        if en_base in ("red_king", "black_king"):
                            name_en = en_base
                        else:
                            name_en = f"{en_base}_{counts[en_base]}"
                        name_cn = (
                            f" {cn_base} "  # Padding to match existing style roughly
                        )

                        piece = piece_cls(name_cn, name_en, is_red, board)
                        piece.add_to_board(current_col, actual_row)
                        current_col += 1
                    else:
                        raise ValueError(f"Unknown FEN character: {char}")

        # Calculate initial hash for FEN
        board.current_hash = board.zobrist.hash_board(board)
        board.hash_history = {board.current_hash: 1}

        return board
