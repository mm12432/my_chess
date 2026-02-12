"""
This module defines the Chessman class and its subclasses (Rook, Knight, Cannon, etc.),
representing the pieces in Chinese Chess. It handles piece movement logic and validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from my_chess.chess_core import point as point_lib

if TYPE_CHECKING:
    from my_chess.chess_core.chessboard import Chessboard


def num_between(max_num: int, min_num: int, current: int) -> bool:
    """Checks if a number is between max and min (inclusive)."""
    return current >= min_num and current <= max_num


def creat_points(
    list_points: List[point_lib.Point], list_vs: tuple, list_hs: tuple
) -> None:
    """Helper to create Point objects from coordinate tuples and add to list."""
    for v in list_vs:
        for h in list_hs:
            list_points.append(point_lib.Point(v, h))


class Chessman:
    """
    Base class for all chess pieces.
    Handles common properties like position, color, and movement validation.
    """

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        self.__name = name
        self.__is_red = is_red
        self.__chessboard = chessboard
        self._position = point_lib.Point(0, 0)  # Initialize with dummy values
        self.__moving_list: List[point_lib.Point] = []
        self._top = 9
        self._bottom = 0
        self._left = 0
        self._right = 8
        self.__is_alive = True
        self.__name_cn = name_cn

    @property
    def row_num(self) -> int:
        """Returns the row number of the piece."""
        return self._position.y

    @property
    def col_num(self) -> int:
        """Returns the column number of the piece."""
        return self._position.x

    def update_position(self, col: int, row: int) -> None:
        """Updates the position of the piece."""
        self._position.x = col
        self._position.y = row

    @property
    def is_alive(self) -> bool:
        """Returns whether the piece is alive (not captured)."""
        return self.__is_alive

    @is_alive.setter
    def is_alive(self, is_alive: bool) -> None:
        """Sets the alive status of the piece."""
        self.__is_alive = is_alive

    @property
    def chessboard(self) -> Chessboard:
        """Returns the chessboard this piece belongs to."""
        return self.__chessboard

    @property
    def is_red(self) -> bool:
        """Returns True if the piece is Red, False if Black."""
        return self.__is_red

    @property
    def name(self) -> str:
        """Returns the unique name of the piece."""
        return self.__name

    @property
    def fen_char(self) -> str:
        """Returns the FEN character for this piece (Uppercase for Red, Lowercase for Black)"""
        return ""

    @property
    def name_cn(self) -> str:
        """Returns the Chinese name of the piece."""
        return self.__name_cn

    @property
    def position(self) -> point_lib.Point:
        """Returns the current position/point of the piece."""
        return self._position

    @property
    def moving_list(self) -> List[point_lib.Point]:
        """Returns the list of valid moving points for the piece."""
        return self.__moving_list

    def clear_moving_list(self) -> None:
        """Clears the list of valid moving points."""
        self.__moving_list = []

    def add_to_board(self, col_num: int, row_num: int) -> None:
        """Adds the piece to the board at the specified position."""
        if self.border_check(col_num, row_num):
            self._position.x = col_num
            self._position.y = row_num
            self.__chessboard.add_chessman(self, col_num, row_num)
        else:
            print("the wrong position")

    def move(self, col_num: int, row_num: int) -> bool:
        """Moves the piece to the specified position if valid."""
        if self.in_moving_list(col_num, row_num):
            self.__chessboard.remove_chessman_source(self._position.x, self._position.y)
            self.__chessboard.update_history(self, col_num, row_num)
            self._position.x = col_num
            self._position.y = row_num
            return self.__chessboard.move_chessman(self, col_num, row_num)

        print("the wrong target_position")
        return False

    def in_moving_list(self, col_num: int, row_num: int) -> bool:
        """Checks if the target position is in the piece's moving list."""
        for pt in self.__moving_list:
            if pt.x == col_num and pt.y == row_num:
                return True
        return False

    def calc_moving_list(self) -> None:
        """Calculates the list of valid moving points. To be implemented by subclasses."""

    def border_check(self, col_num: int, row_num: int) -> bool:
        """Checks if the given position is within the valid board area for this piece."""
        return num_between(self._top, self._bottom, row_num) and num_between(
            self._right, self._left, col_num
        )

    def calc_moving_path(
        self,
        direction_chessman: Optional[Chessman],
        direction_vertical_coordinate: Optional[int],
        current_vertical_coordinate: int,
        direction_parallel_coordinate: int,
        direction: int,
        border_vertical_coordinate: int,
        h_or_v: bool,
        ignore_color: bool = False,
    ) -> None:
        """
        Calculates moving path in a specific direction.
        Helper method for calculating moves for linear pieces (Rook, Cannon).
        """
        if direction_chessman is not None:
            if (
                direction_chessman.is_red == self.is_red or ignore_color
            ) and direction_vertical_coordinate is not None:
                for i in range(
                    direction_vertical_coordinate + direction,
                    current_vertical_coordinate,
                    direction,
                ):
                    self.__moving_list.append(
                        point_lib.Point(i, direction_parallel_coordinate)
                        if h_or_v
                        else point_lib.Point(direction_parallel_coordinate, i)
                    )

            elif direction_vertical_coordinate is not None:
                for i in range(
                    direction_vertical_coordinate,
                    current_vertical_coordinate,
                    direction,
                ):
                    self.__moving_list.append(
                        point_lib.Point(i, direction_parallel_coordinate)
                        if h_or_v
                        else point_lib.Point(direction_parallel_coordinate, i)
                    )
        else:
            for i in range(
                border_vertical_coordinate, current_vertical_coordinate, direction
            ):
                self.__moving_list.append(
                    point_lib.Point(i, direction_parallel_coordinate)
                    if h_or_v
                    else point_lib.Point(direction_parallel_coordinate, i)
                )

    def add_from_probable_points(
        self, probable_moving_points: List[point_lib.Point], current_color: bool
    ) -> None:
        """Filters probable points and adds valid ones to the moving list."""
        for pt in probable_moving_points:
            if self.border_check(pt.x, pt.y):
                chessman = self.chessboard.get_chessman(pt.x, pt.y)
                if chessman is None or chessman.is_red != current_color:
                    self.moving_list.append(pt)


class Rook(Chessman):
    """Represents the Rook (Chariot) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._top = 9
        self._bottom = 0
        self._left = 0
        self._right = 8

    @property
    def fen_char(self) -> str:
        return "R" if self.is_red else "r"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        left = self.chessboard.get_left_first_chessman(current_v_c, current_h_c)
        right = self.chessboard.get_right_first_chessman(current_v_c, current_h_c)
        top = self.chessboard.get_top_first_chessman(current_v_c, current_h_c)
        bottom = self.chessboard.get_bottom_first_chessman(current_v_c, current_h_c)

        self.calc_moving_path(
            left,
            (left.position.x if left is not None else None),
            current_v_c,
            current_h_c,
            1,
            0,
            True,
        )
        self.calc_moving_path(
            right,
            (right.position.x if right is not None else None),
            current_v_c,
            current_h_c,
            -1,
            8,
            True,
        )
        self.calc_moving_path(
            top,
            (top.position.y if top is not None else None),
            current_h_c,
            current_v_c,
            -1,
            9,
            False,
        )
        self.calc_moving_path(
            bottom,
            (bottom.position.y if bottom is not None else None),
            current_h_c,
            current_v_c,
            1,
            0,
            False,
        )


class Knight(Chessman):
    """Represents the Knight (Horse) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._top = 9
        self._bottom = 0
        self._left = 0
        self._right = 8

    @property
    def fen_char(self) -> str:
        return "N" if self.is_red else "n"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_obstacle_points: List[point_lib.Point] = []
        probable_moving_points: List[point_lib.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        creat_points(probable_obstacle_points, vs2, hs2)
        current_color = self.is_red
        for pt in probable_obstacle_points:
            if self.border_check(pt.x, pt.y):
                chessman = self.chessboard.get_chessman(pt.x, pt.y)
                if chessman is None:
                    if pt.x == current_v_c:
                        probable_moving_points.append(
                            point_lib.Point(pt.x + 1, 2 * pt.y - current_h_c)
                        )
                        probable_moving_points.append(
                            point_lib.Point(pt.x - 1, 2 * pt.y - current_h_c)
                        )
                    else:
                        probable_moving_points.append(
                            point_lib.Point(2 * pt.x - current_v_c, pt.y + 1)
                        )
                        probable_moving_points.append(
                            point_lib.Point(2 * pt.x - current_v_c, pt.y - 1)
                        )
        self.add_from_probable_points(probable_moving_points, current_color)


class Cannon(Chessman):
    """Represents the Cannon piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._top = 9
        self._bottom = 0
        self._left = 0
        self._right = 8

    @property
    def fen_char(self) -> str:
        return "C" if self.is_red else "c"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        left = self.chessboard.get_left_first_chessman(current_v_c, current_h_c)
        right = self.chessboard.get_right_first_chessman(current_v_c, current_h_c)
        top = self.chessboard.get_top_first_chessman(current_v_c, current_h_c)
        bottom = self.chessboard.get_bottom_first_chessman(current_v_c, current_h_c)
        tar_left = self.chessboard.get_left_second_chessman(current_v_c, current_h_c)
        tar_right = self.chessboard.get_right_second_chessman(current_v_c, current_h_c)
        tar_top = self.chessboard.get_top_second_chessman(current_v_c, current_h_c)
        tar_bottom = self.chessboard.get_bottom_second_chessman(
            current_v_c, current_h_c
        )
        self.calc_moving_path(
            left,
            (left.position.x if left is not None else None),
            current_v_c,
            current_h_c,
            1,
            0,
            True,
            True,
        )
        self.calc_moving_path(
            right,
            (right.position.x if right is not None else None),
            current_v_c,
            current_h_c,
            -1,
            8,
            True,
            True,
        )
        self.calc_moving_path(
            top,
            (top.position.y if top is not None else None),
            current_h_c,
            current_v_c,
            -1,
            9,
            False,
            True,
        )
        self.calc_moving_path(
            bottom,
            (bottom.position.y if bottom is not None else None),
            current_h_c,
            current_v_c,
            1,
            0,
            False,
            True,
        )
        current_color = self.is_red
        if tar_left is not None and tar_left.is_red != current_color:
            self.moving_list.append(
                point_lib.Point(tar_left.position.x, tar_left.position.y)
            )
        if tar_right is not None and tar_right.is_red != current_color:
            self.moving_list.append(
                point_lib.Point(tar_right.position.x, tar_right.position.y)
            )
        if tar_top is not None and tar_top.is_red != current_color:
            self.moving_list.append(
                point_lib.Point(tar_top.position.x, tar_top.position.y)
            )
        if tar_bottom is not None and tar_bottom.is_red != current_color:
            self.moving_list.append(
                point_lib.Point(tar_bottom.position.x, tar_bottom.position.y)
            )


class Mandarin(Chessman):
    """Represents the Mandarin (Advisor/Guard) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._top = 2
            self._bottom = 0
            self._left = 3
            self._right = 5
        else:
            self._top = 9
            self._bottom = 7
            self._left = 3
            self._right = 5

    @property
    def fen_char(self) -> str:
        return "A" if self.is_red else "a"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[point_lib.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        current_color = self.is_red

        self.add_from_probable_points(probable_moving_points, current_color)


class Elephant(Chessman):
    """Represents the Elephant (Bishop) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._top = 4
            self._bottom = 0
            self._left = 0
            self._right = 8
        else:
            self._top = 9
            self._bottom = 5
            self._left = 0
            self._right = 8

    @property
    def fen_char(self) -> str:
        return "B" if self.is_red else "b"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_obstacle_points: List[point_lib.Point] = []
        probable_moving_points: List[point_lib.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        current_color = self.is_red
        for pt in probable_obstacle_points:
            if self.border_check(pt.x, pt.y):
                chessman = self.chessboard.get_chessman(pt.x, pt.y)
                if chessman is None:
                    probable_moving_points.append(
                        point_lib.Point(2 * pt.x - current_v_c, 2 * pt.y - current_h_c)
                    )
        self.add_from_probable_points(probable_moving_points, current_color)


class Pawn(Chessman):
    """Represents the Pawn (Soldier) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._top = 9
            self._bottom = 3
            self._left = 0
            self._right = 8
            self.__direction = 1
            self.__river = 5
        else:
            self._top = 6
            self._bottom = 0
            self._left = 0
            self._right = 8
            self.__direction = -1
            self.__river = 4

    @property
    def fen_char(self) -> str:
        return "P" if self.is_red else "p"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[point_lib.Point] = []
        current_color = self.is_red
        probable_moving_points.append(
            point_lib.Point(current_v_c, current_h_c + self.__direction)
        )
        if current_h_c * self.__direction >= self.__river * self.__direction:
            probable_moving_points.append(point_lib.Point(current_v_c + 1, current_h_c))
            probable_moving_points.append(point_lib.Point(current_v_c - 1, current_h_c))
        self.add_from_probable_points(probable_moving_points, current_color)


class King(Chessman):
    """Represents the King (General) piece."""

    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._top = 2
            self._bottom = 0
            self._left = 3
            self._right = 5
        else:
            self._top = 9
            self._bottom = 7
            self._left = 3
            self._right = 5

    @property
    def fen_char(self) -> str:
        return "K" if self.is_red else "k"

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[point_lib.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        creat_points(probable_moving_points, vs2, hs2)
        current_color = self.is_red
        self.add_from_probable_points(probable_moving_points, current_color)
