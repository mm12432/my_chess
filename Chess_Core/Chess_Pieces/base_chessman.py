# coding:utf-8

from Chess_Core import Point


def num_between(max_num, min_num, current):
    return current >= min_num and current <= max_num


def creat_points(list_points, list_vs, list_hs):
    for v in list_vs:
        for h in list_hs:
            list_points.append(Point.Point(v, h))


class Chessman(object):

    def __init__(self, name_cn, name, is_red, chessboard):
        self.__name = name
        self.__is_red = is_red
        self.__chessboard = chessboard
        self.__position = Point.Point(None, None)
        self.__moving_list = []
        self.__top = 9
        self.__bottom = 0
        self.__left = 0
        self.__right = 8
        self.__is_alive = True
        self.__name_cn = name_cn

    @property
    def row_num(self):
        return self.__position.y

    @property
    def col_num(self):
        return self.__position.x

    @property
    def is_alive(self):
        return self.__is_alive

    @is_alive.setter
    def is_alive(self, is_alive):
        self.__is_alive = is_alive

    @property
    def chessboard(self):
        return self.__chessboard

    @property
    def is_red(self):
        return self.__is_red

    @property
    def name(self):
        return self.__name

    @property
    def name_cn(self):
        return self.__name_cn

    @property
    def position(self):
        return self.__position

    @property
    def moving_list(self):
        return self.__moving_list

    def clear_moving_list(self):
        self.__moving_list = []

    def add_to_board(self, col_num, row_num):
        if self.border_check(col_num, row_num):
            self.__position.x = col_num
            self.__position.y = row_num
            self.__chessboard.add_chessman(self, col_num, row_num)
        else:
            print("the worng postion")

    def move(self, col_num, row_num):
        if self.in_moving_list(col_num, row_num):
            self.__chessboard.remove_chessman_source(
                self.__position.x, self.__position.y)
            self.__chessboard.update_history(self, col_num, row_num)
            self.__position.x = col_num
            self.__position.y = row_num
            return self.__chessboard.move_chessman(self, col_num, row_num)
        else:
            print("the worng target_position")
            return False

    def in_moving_list(self, col_num, row_num):
        for point in self.__moving_list:
            if point.x == col_num and point.y == row_num:
                return True
        return False

    def calc_moving_list(self):
        pass

    def border_check(self, col_num, row_num):
        return num_between(self.__top, self.__bottom, row_num) and num_between(self.__right, self.__left, col_num)

    def calc_moving_path(self, direction_chessman, direction_vertical_coordinate, current_vertical_coordinate, direction_parallel_coordinate, direction, border_vertical_coordinate, h_or_v, ignore_color=False):
        if direction_chessman != None:
            if direction_chessman.is_red == self.is_red or ignore_color:
                for i in range(direction_vertical_coordinate + direction, current_vertical_coordinate, direction):
                    self.__moving_list.append(
                        Point.Point(i, direction_parallel_coordinate) if h_or_v else Point.Point(direction_parallel_coordinate, i))

            else:
                for i in range(direction_vertical_coordinate, current_vertical_coordinate, direction):
                    self.__moving_list.append(
                        Point.Point(i, direction_parallel_coordinate) if h_or_v else Point.Point(direction_parallel_coordinate, i))
        else:
            for i in range(border_vertical_coordinate, current_vertical_coordinate, direction):
                self.__moving_list.append(
                    Point.Point(i, direction_parallel_coordinate) if h_or_v else Point.Point(direction_parallel_coordinate, i))

    def add_from_probable_points(self, probable_moving_points, current_color):
        for point in probable_moving_points:
            if self.border_check(point.x, point.y):
                chessman = self.chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None or chessman.is_red != current_color:
                    self.moving_list.append(point)
