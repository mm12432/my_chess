# coding:utf-8

from Chess_Core.Chess_Pieces.base_chessman import Chessman, creat_points


class Mandarin(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Mandarin, self).__init__(name_cn, name, is_red, chessboard)
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

    def calc_moving_list(self):
        current_v_c = super(Mandarin, self).position.x
        current_h_c = super(Mandarin, self).position.y
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        current_color = super(Mandarin, self).is_red

        super(Mandarin, self).add_from_probable_points(
            probable_moving_points, current_color)
