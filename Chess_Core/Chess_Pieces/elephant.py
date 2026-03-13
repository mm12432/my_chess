# coding:utf-8

from Chess_Core import Point
from Chess_Core.Chess_Pieces.base_chessman import Chessman, creat_points


class Elephant(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Elephant, self).__init__(name_cn, name, is_red, chessboard)
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

    def calc_moving_list(self):
        current_v_c = super(Elephant, self).position.x
        current_h_c = super(Elephant, self).position.y
        probable_obstacle_points = []
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        current_color = super(Elephant, self).is_red
        for point in probable_obstacle_points:
            if super(Elephant, self).border_check(point.x, point.y):
                chessman = super(Elephant, self).chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None:
                    probable_moving_points.append(
                        Point.Point(2 * point.x - current_v_c, 2 * point.y - current_h_c))
        super(Elephant, self).add_from_probable_points(
            probable_moving_points, current_color)
