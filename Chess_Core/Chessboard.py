from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from MyChess.Chess_Core import Chessman

if TYPE_CHECKING:
    from MyChess.Chess_Core.Chessman import Chessman as ChessmanType


class Chessboard:
    def __init__(self, name: str) -> None:
        self.__name = name
        self.__is_red_turn = True
        # Initialize 9x10 board with None
        self.__chessmans: List[List[Optional[ChessmanType]]] = [
            ([None] * 10) for _ in range(9)
        ]
        self.__chessmans_hash: Dict[str, ChessmanType] = {}
        self.__history: Dict[str, Dict[str, Any]] = {
            "red": {"chessman": None, "last_pos": None, "repeat": 0},
            "black": {"chessman": None, "last_pos": None, "repeat": 0},
        }

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
        Chessman.Rook(" 车l红 ", "red_rook_left", True, self).add_to_board(0, 0)
        Chessman.Rook(" 车r红 ", "red_rook_right", True, self).add_to_board(8, 0)
        Chessman.Knight(" 马l红 ", "red_knight_left", True, self).add_to_board(1, 0)
        Chessman.Knight(" 马r红 ", "red_knight_right", True, self).add_to_board(7, 0)
        Chessman.Cannon(" 炮l红 ", "red_cannon_left", True, self).add_to_board(1, 2)
        Chessman.Cannon(" 炮r红 ", "red_cannon_right", True, self).add_to_board(7, 2)
        Chessman.Elephant(" 相l红 ", "red_elephant_left", True, self).add_to_board(2, 0)
        Chessman.Elephant(" 相r红 ", "red_elephant_right", True, self).add_to_board(
            6, 0
        )
        Chessman.Mandarin(" 仕l红 ", "red_mandarin_left", True, self).add_to_board(3, 0)
        Chessman.Mandarin(" 仕r红 ", "red_mandarin_right", True, self).add_to_board(
            5, 0
        )
        Chessman.King(" 帅 红 ", "red_king", True, self).add_to_board(4, 0)
        Chessman.Pawn(" 兵1红 ", "red_pawn_1", True, self).add_to_board(0, 3)
        Chessman.Pawn(" 兵2红 ", "red_pawn_2", True, self).add_to_board(2, 3)
        Chessman.Pawn(" 兵3红 ", "red_pawn_3", True, self).add_to_board(4, 3)
        Chessman.Pawn(" 兵4红 ", "red_pawn_4", True, self).add_to_board(6, 3)
        Chessman.Pawn(" 兵5红 ", "red_pawn_5", True, self).add_to_board(8, 3)

        # Black Pieces
        Chessman.Rook(" 车l黑 ", "black_rook_left", False, self).add_to_board(0, 9)
        Chessman.Rook(" 车r黑 ", "black_rook_right", False, self).add_to_board(8, 9)
        Chessman.Knight(" 马l黑 ", "black_knight_left", False, self).add_to_board(1, 9)
        Chessman.Knight(" 马r黑 ", "black_knight_right", False, self).add_to_board(7, 9)
        Chessman.Cannon(" 炮l黑 ", "black_cannon_left", False, self).add_to_board(1, 7)
        Chessman.Cannon(" 炮r黑 ", "black_cannon_right", False, self).add_to_board(7, 7)
        Chessman.Elephant(" 象l黑 ", "black_elephant_left", False, self).add_to_board(
            2, 9
        )
        Chessman.Elephant(" 象r黑 ", "black_elephant_right", False, self).add_to_board(
            6, 9
        )
        Chessman.Mandarin(" 仕l黑 ", "black_mandarin_left", False, self).add_to_board(
            3, 9
        )
        Chessman.Mandarin(" 仕r黑 ", "black_mandarin_right", False, self).add_to_board(
            5, 9
        )
        Chessman.King(" 将 黑 ", "black_king", False, self).add_to_board(4, 9)
        Chessman.Pawn(" 卒1黑 ", "black_pawn_1", False, self).add_to_board(0, 6)
        Chessman.Pawn(" 卒2黑 ", "black_pawn_2", False, self).add_to_board(2, 6)
        Chessman.Pawn(" 卒3黑 ", "black_pawn_3", False, self).add_to_board(4, 6)
        Chessman.Pawn(" 卒4黑 ", "black_pawn_4", False, self).add_to_board(6, 6)
        Chessman.Pawn(" 卒5黑 ", "black_pawn_5", False, self).add_to_board(8, 6)

    def add_chessman(self, chessman: ChessmanType, col_num: int, row_num: int) -> None:
        self.chessmans[col_num][row_num] = chessman
        if chessman.name not in self.__chessmans_hash:
            self.__chessmans_hash[chessman.name] = chessman

    def remove_chessman_target(self, col_num: int, row_num: int) -> None:
        chessman_old = self.get_chessman(col_num, row_num)
        if chessman_old is not None:
            self.__chessmans_hash.pop(chessman_old.name)

    def remove_chessman_source(self, col_num: int, row_num: int) -> None:
        self.chessmans[col_num][row_num] = None

    def calc_chessmans_moving_list(self) -> None:
        for chessman in self.__chessmans_hash.values():
            if chessman.is_red == self.__is_red_turn:
                chessman.calc_moving_list()

    def clear_chessmans_moving_list(self) -> None:
        for chessman in self.__chessmans_hash.values():
            chessman.clear_moving_list()

    def move_chessman(self, chessman: ChessmanType, col_num: int, row_num: int) -> bool:
        if chessman.is_red == self.__is_red_turn:
            self.remove_chessman_target(col_num, row_num)
            self.add_chessman(chessman, col_num, row_num)
            self.__is_red_turn = not self.__is_red_turn
            return True
        else:
            print("the wrong turn")
            return False

    def update_history(
        self, chessman: ChessmanType, col_num: int, row_num: int
    ) -> None:
        red_or_black_key = self.red_or_black(chessman)
        history_chessman = self.__history[red_or_black_key]["chessman"]
        history_pos = self.__history[red_or_black_key]["last_pos"]
        if (
            history_chessman == chessman
            and history_pos is not None
            and history_pos[0] == col_num
            and history_pos[1] == row_num
        ):
            self.__history[red_or_black_key]["repeat"] += 1
        else:
            self.__history[red_or_black_key]["repeat"] = 0
        self.__history[red_or_black_key]["chessman"] = chessman
        self.__history[red_or_black_key]["last_pos"] = (
            chessman.col_num,
            chessman.row_num,
        )

    def red_or_black(self, chessman: ChessmanType) -> str:
        if chessman.is_red:
            return "red"
        else:
            return "black"

    def is_end(self) -> bool:
        return self.who_is_victor(6)

    def who_is_victor(self, repeat_num: int) -> bool:
        whos_turn = "red" if self.__is_red_turn else "black"
        other_turn = "red" if not self.__is_red_turn else "black"
        chessman = self.get_chessman_by_name("{0}_king".format(whos_turn))
        if chessman is not None:
            if self.__history[other_turn]["repeat"] == repeat_num:
                print("{0} is victor".format(whos_turn))
                return True
            else:
                return False
        else:
            print("{0} is victor".format(other_turn))
            return True

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
                chessman = self.__chessmans[j][i]
                if chessman is not None:
                    screen += chessman.name_cn
                else:
                    screen += "   .   "
            screen += "\r\n" * 3
        print(screen)
