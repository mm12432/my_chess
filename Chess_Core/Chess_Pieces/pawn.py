# coding:utf-8

from Chess_Core import Point
from Chess_Core.Chess_Pieces.base_chessman import Chessman


class Pawn(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Pawn, self).__init__(name_cn, name, is_red, chessboard)
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

    def calc_moving_list(self):
        current_v_c = super(Pawn, self).position.x
        current_h_c = super(Pawn, self).position.y
        probable_moving_points = []
        current_color = super(Pawn, self).is_red
        probable_moving_points.append(
            Point.Point(current_v_c, current_h_c + self.__direction))
        if current_h_c * self.__direction >= self.__river * self.__direction:
            probable_moving_points.append(
                Point.Point(current_v_c + 1, current_h_c))
            probable_moving_points.append(
                Point.Point(current_v_c - 1, current_h_c))
        super(Pawn, self).add_from_probable_points(
            probable_moving_points, current_color)
