"""
This module defines the Chessboard class, which represents the game state, board configuration,
and game logic. It handles moving pieces, turn management, history tracking, and game end conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import copy

from MyChess.Chess_Core import chessman

if TYPE_CHECKING:
    from MyChess.Chess_Core.chessman import Chessman as ChessmanType


class Chessboard:
    def __init__(self, name: str) -> None:
        self.__name = name
        self.__is_red_turn = True
        # Initialize 9x10 board with None
        self.__chessmans: List[List[Optional[ChessmanType]]] = [
            ([None] * 10) for _ in range(9)
        ]
        self.__chessmans_hash: Dict[str, ChessmanType] = {}
        # Legacy history for "chasing"/"checking" detection (optional, kept for compatibility)
        self.__history: Dict[str, Dict[str, Any]] = {
            "red": {"chessman": None, "last_pos": None, "repeat": 0},
            "black": {"chessman": None, "last_pos": None, "repeat": 0},
        }

        self.moves_history: List[str] = []
        self.undo_stack: List[Any] = []

        from MyChess.Chess_Core.zobrist import Zobrist

        self.zobrist = Zobrist()
        self.current_hash = 0
        self.hash_history: Dict[int, int] = {}

    @property
    def is_red_turn(self) -> bool:
        return self.__is_red_turn

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @property
    def chessmans(self) -> List[List[Optional[ChessmanType]]]:
        return self.__chessmans

    @property
    def chessmans_hash(self) -> Dict[str, ChessmanType]:
        return self.__chessmans_hash

    def init_board(self) -> None:
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

    def add_chessman(self, piece: ChessmanType, col_num: int, row_num: int) -> None:
        self.chessmans[col_num][row_num] = piece
        if piece.name not in self.__chessmans_hash:
            self.__chessmans_hash[piece.name] = piece

    def remove_chessman_target(self, col_num: int, row_num: int) -> None:
        chessman_old = self.get_chessman(col_num, row_num)
        if chessman_old is not None:
            self.__chessmans_hash.pop(chessman_old.name)

    def remove_chessman_source(self, col_num: int, row_num: int) -> None:
        self.chessmans[col_num][row_num] = None

    def calc_chessmans_moving_list(self) -> None:
        for piece in self.__chessmans_hash.values():
            if piece.is_red == self.__is_red_turn:
                piece.calc_moving_list()

    def clear_chessmans_moving_list(self) -> None:
        for piece in self.__chessmans_hash.values():
            piece.clear_moving_list()

    def move_chessman(self, piece: ChessmanType, col_num: int, row_num: int) -> bool:
        if piece.is_red == self.__is_red_turn:
            # Record move BEFORE moving
            from MyChess.Chess_Core.move_notation import MoveNotation

            move_str = MoveNotation.get_move_name(
                piece, piece.col_num, piece.row_num, col_num, row_num
            )

            # Save undo info: (chessman, from_col, from_row, to_col, to_row, captured_piece, history_state, history_str)
            captured = self.get_chessman(col_num, row_num)
            history_copy = copy.deepcopy(self.__history)

            self.undo_stack.append(
                {
                    "chessman": piece,
                    "from_col": piece.col_num,
                    "from_row": piece.row_num,
                    "to_col": col_num,
                    "to_row": row_num,
                    "captured": captured,
                    "history_state": history_copy,
                    "move_str": move_str,
                }
            )

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

            self.remove_chessman_target(col_num, row_num)
            self.add_chessman(piece, col_num, row_num)
            self.__is_red_turn = not self.__is_red_turn
            return True
        else:
            print("the wrong turn")
            return False

    def undo_move(self) -> bool:
        if not self.undo_stack:
            return False

        last_move = self.undo_stack.pop()
        chessman = last_move["chessman"]
        from_col = last_move["from_col"]
        from_row = last_move["from_row"]
        to_col = last_move["to_col"]
        to_row = last_move["to_row"]
        captured = last_move["captured"]
        history_state = last_move["history_state"]

        # Decrement hash count for current state
        if self.current_hash in self.hash_history:
            self.hash_history[self.current_hash] -= 1
            if self.hash_history[self.current_hash] <= 0:
                del self.hash_history[self.current_hash]

        # Revert Hash (Apply XOR again to go back)
        self.current_hash = self.zobrist.update_hash(
            self.current_hash, chessman, from_col, from_row, to_col, to_row, captured
        )

        # 1. Move chessman back
        self.remove_chessman_source(to_col, to_row)

        # Update internal position
        chessman._Chessman__position.x = from_col
        chessman._Chessman__position.y = from_row

        # Place back on board
        self.chessmans[from_col][from_row] = chessman

        # 2. Restore captured piece
        if captured:
            captured._Chessman__position.x = to_col
            captured._Chessman__position.y = to_row
            self.chessmans[to_col][to_row] = captured
            self.__chessmans_hash[captured.name] = captured
            captured.is_alive = True

        # 3. Restore history
        self.__history = history_state
        if self.moves_history:
            self.moves_history.pop()

        # 4. Flip turn
        self.__is_red_turn = not self.__is_red_turn

        return True

    def get_winner(self) -> Optional[str]:
        # Check Repetition Draw
        if self.hash_history.get(self.current_hash, 0) >= 3:
            return "Draw"

        red_king = self.get_chessman_by_name("red_king")
        black_king = self.get_chessman_by_name("black_king")

        if not red_king:
            return "Black"
        if not black_king:
            return "Red"

        return None

    def is_end(self) -> bool:
        return self.get_winner() is not None

    def update_history(self, piece: ChessmanType, col_num: int, row_num: int) -> None:
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
        if piece.is_red:
            return "red"
        else:
            return "black"

    def get_chessman(self, col_num: int, row_num: int) -> Optional[ChessmanType]:
        return self.__chessmans[col_num][row_num]

    def get_chessman_by_name(self, name: str) -> Optional[ChessmanType]:
        if name in self.__chessmans_hash:
            return self.__chessmans_hash[name]
        return None

    def get_top_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        for i in range(row_num + 1, 10, 1):
            current = self.chessmans[col_num][i]
            if current is not None:
                return current
        return None

    def get_bottom_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        for i in range(row_num - 1, -1, -1):
            current = self.chessmans[col_num][i]
            if current is not None:
                return current
        return None

    def get_left_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        for i in range(col_num - 1, -1, -1):
            current = self.chessmans[i][row_num]
            if current is not None:
                return current
        return None

    def get_right_first_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
        for i in range(col_num + 1, 9, 1):
            current = self.chessmans[i][row_num]
            if current is not None:
                return current
        return None

    def get_top_second_chessman(
        self, col_num: int, row_num: int
    ) -> Optional[ChessmanType]:
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
        self.undo_stack = []
        self.current_hash = 0
        self.hash_history = {}

    @classmethod
    def from_fen(cls, fen: str) -> "Chessboard":
        """Creates a Chessboard instance from a FEN string."""
        board = cls("FEN_Board")
        board.clear_board()

        parts = fen.split()
        board_fen = parts[0]
        turn_fen = parts[1] if len(parts) > 1 else "w"

        # 'w' or 'r' -> Red turn
        board._Chessboard__is_red_turn = (
            turn_fen.lower() == "w" or turn_fen.lower() == "r"
        )

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
