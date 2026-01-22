from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from MyChess.Chess_Core import Point

if TYPE_CHECKING:
    from MyChess.Chess_Core.Chessboard import Chessboard


def num_between(max_num: int, min_num: int, current: int) -> bool:
    return current >= min_num and current <= max_num


def creat_points(
    list_points: List[Point.Point], list_vs: tuple, list_hs: tuple
) -> None:
    for v in list_vs:
        for h in list_hs:
            list_points.append(Point.Point(v, h))


class Chessman:
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        self.__name = name
        self.__is_red = is_red
        self.__chessboard = chessboard
        self.__position = Point.Point(0, 0)  # Initialize with dummy values
        self.__moving_list: List[Point.Point] = []
        self.__top = 9
        self.__bottom = 0
        self.__left = 0
        self.__right = 8
        self.__is_alive = True
        self.__name_cn = name_cn

    @property
    def row_num(self) -> int:
        return self.__position.y

    @property
    def col_num(self) -> int:
        return self.__position.x

    @property
    def is_alive(self) -> bool:
        return self.__is_alive

    @is_alive.setter
    def is_alive(self, is_alive: bool) -> None:
        self.__is_alive = is_alive

    @property
    def chessboard(self) -> Chessboard:
        return self.__chessboard

    @property
    def is_red(self) -> bool:
        return self.__is_red

    @property
    def name(self) -> str:
        return self.__name

    @property
    def name_cn(self) -> str:
        return self.__name_cn

    @property
    def position(self) -> Point.Point:
        return self.__position

    @property
    def moving_list(self) -> List[Point.Point]:
        return self.__moving_list

    def clear_moving_list(self) -> None:
        self.__moving_list = []

    def add_to_board(self, col_num: int, row_num: int) -> None:
        if self.border_check(col_num, row_num):
            self.__position.x = col_num
            self.__position.y = row_num
            self.__chessboard.add_chessman(self, col_num, row_num)
        else:
            print("the wrong position")

    def move(self, col_num: int, row_num: int) -> bool:
        if self.in_moving_list(col_num, row_num):
            self.__chessboard.remove_chessman_source(
                self.__position.x, self.__position.y
            )
            self.__chessboard.update_history(self, col_num, row_num)
            self.__position.x = col_num
            self.__position.y = row_num
            return self.__chessboard.move_chessman(self, col_num, row_num)
        else:
            print("the wrong target_position")
            return False

    def in_moving_list(self, col_num: int, row_num: int) -> bool:
        for point in self.__moving_list:
            if point.x == col_num and point.y == row_num:
                return True
        return False

    def calc_moving_list(self) -> None:
        pass

    def border_check(self, col_num: int, row_num: int) -> bool:
        return num_between(self.__top, self.__bottom, row_num) and num_between(
            self.__right, self.__left, col_num
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
                        Point.Point(i, direction_parallel_coordinate)
                        if h_or_v
                        else Point.Point(direction_parallel_coordinate, i)
                    )

            elif direction_vertical_coordinate is not None:
                for i in range(
                    direction_vertical_coordinate,
                    current_vertical_coordinate,
                    direction,
                ):
                    self.__moving_list.append(
                        Point.Point(i, direction_parallel_coordinate)
                        if h_or_v
                        else Point.Point(direction_parallel_coordinate, i)
                    )
        else:
            for i in range(
                border_vertical_coordinate, current_vertical_coordinate, direction
            ):
                self.__moving_list.append(
                    Point.Point(i, direction_parallel_coordinate)
                    if h_or_v
                    else Point.Point(direction_parallel_coordinate, i)
                )

    def add_from_probable_points(
        self, probable_moving_points: List[Point.Point], current_color: bool
    ) -> None:
        for point in probable_moving_points:
            if self.border_check(point.x, point.y):
                chessman = self.chessboard.get_chessman(point.x, point.y)
                if chessman is None or chessman.is_red != current_color:
                    self.moving_list.append(point)


class Rook(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

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
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_obstacle_points: List[Point.Point] = []
        probable_moving_points: List[Point.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        creat_points(probable_obstacle_points, vs2, hs2)
        current_color = self.is_red
        for point in probable_obstacle_points:
            if self.border_check(point.x, point.y):
                chessman = self.chessboard.get_chessman(point.x, point.y)
                if chessman is None:
                    if point.x == current_v_c:
                        probable_moving_points.append(
                            Point.Point(point.x + 1, 2 * point.y - current_h_c)
                        )
                        probable_moving_points.append(
                            Point.Point(point.x - 1, 2 * point.y - current_h_c)
                        )
                    else:
                        probable_moving_points.append(
                            Point.Point(2 * point.x - current_v_c, point.y + 1)
                        )
                        probable_moving_points.append(
                            Point.Point(2 * point.x - current_v_c, point.y - 1)
                        )
        self.add_from_probable_points(probable_moving_points, current_color)


class Cannon(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

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
                Point.Point(tar_left.position.x, tar_left.position.y)
            )
        if tar_right is not None and tar_right.is_red != current_color:
            self.moving_list.append(
                Point.Point(tar_right.position.x, tar_right.position.y)
            )
        if tar_top is not None and tar_top.is_red != current_color:
            self.moving_list.append(Point.Point(tar_top.position.x, tar_top.position.y))
        if tar_bottom is not None and tar_bottom.is_red != current_color:
            self.moving_list.append(
                Point.Point(tar_bottom.position.x, tar_bottom.position.y)
            )


class Mandarin(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._Chessman__top = 2
            self._Chessman__bottom = 0
            self._Chessman__left = 3
            self._Chessman__right = 5
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 7
            self._Chessman__left = 3
            self._Chessman__right = 5

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[Point.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        current_color = self.is_red

        self.add_from_probable_points(probable_moving_points, current_color)


class Elephant(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._Chessman__top = 4
            self._Chessman__bottom = 0
            self._Chessman__left = 0
            self._Chessman__right = 8
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 5
            self._Chessman__left = 0
            self._Chessman__right = 8

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_obstacle_points: List[Point.Point] = []
        probable_moving_points: List[Point.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        current_color = self.is_red
        for point in probable_obstacle_points:
            if self.border_check(point.x, point.y):
                chessman = self.chessboard.get_chessman(point.x, point.y)
                if chessman is None:
                    probable_moving_points.append(
                        Point.Point(
                            2 * point.x - current_v_c, 2 * point.y - current_h_c
                        )
                    )
        self.add_from_probable_points(probable_moving_points, current_color)


class Pawn(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._Chessman__top = 9
            self._Chessman__bottom = 3
            self._Chessman__left = 0
            self._Chessman__right = 8
            self.__direction = 1
            self.__river = 5
        else:
            self._Chessman__top = 6
            self._Chessman__bottom = 0
            self._Chessman__left = 0
            self._Chessman__right = 8
            self.__direction = -1
            self.__river = 4

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[Point.Point] = []
        current_color = self.is_red
        probable_moving_points.append(
            Point.Point(current_v_c, current_h_c + self.__direction)
        )
        if current_h_c * self.__direction >= self.__river * self.__direction:
            probable_moving_points.append(Point.Point(current_v_c + 1, current_h_c))
            probable_moving_points.append(Point.Point(current_v_c - 1, current_h_c))
        self.add_from_probable_points(probable_moving_points, current_color)


class King(Chessman):
    def __init__(
        self, name_cn: str, name: str, is_red: bool, chessboard: Chessboard
    ) -> None:
        super().__init__(name_cn, name, is_red, chessboard)
        if self.is_red:
            self._Chessman__top = 2
            self._Chessman__bottom = 0
            self._Chessman__left = 3
            self._Chessman__right = 5
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 7
            self._Chessman__left = 3
            self._Chessman__right = 5

    def calc_moving_list(self) -> None:
        current_v_c = self.position.x
        current_h_c = self.position.y
        probable_moving_points: List[Point.Point] = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        creat_points(probable_moving_points, vs2, hs2)
        current_color = self.is_red
        self.add_from_probable_points(probable_moving_points, current_color)
