# coding:utf-8

from Chess_Core import Point
from Chess_Core.Chess_Pieces.base_chessman import Chessman, creat_points


class Knight(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Knight, self).__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Knight, self).position.x
        current_h_c = super(Knight, self).position.y
        probable_obstacle_points = []
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        creat_points(probable_obstacle_points, vs2, hs2)
        current_color = super(Knight, self).is_red
        for point in probable_obstacle_points:
            if super(Knight, self).border_check(point.x, point.y):
                chessman = super(Knight, self).chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None:
                    if point.x == current_v_c:
                        probable_moving_points.append(
                            Point.Point(point.x + 1, 2 * point.y - current_h_c))
                        probable_moving_points.append(
                            Point.Point(point.x - 1, 2 * point.y - current_h_c))
                    else:
                        probable_moving_points.append(
                            Point.Point(2 * point.x - current_v_c, point.y + 1))
                        probable_moving_points.append(
                            Point.Point(2 * point.x - current_v_c, point.y - 1))
        super(Knight, self).add_from_probable_points(
            probable_moving_points, current_color)
