# coding:utf-8

from Chess_Core.Chess_Pieces.base_chessman import Chessman


class Rook(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Rook, self).__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Rook, self).position.x
        current_h_c = super(Rook, self).position.y
        left = super(Rook, self).chessboard.get_left_first_chessman(
            current_v_c, current_h_c)
        right = super(Rook, self).chessboard.get_right_first_chessman(
            current_v_c, current_h_c)
        top = super(Rook, self).chessboard.get_top_first_chessman(
            current_v_c, current_h_c)
        bottom = super(Rook, self).chessboard.get_bottom_first_chessman(
            current_v_c, current_h_c)

        super(Rook, self).calc_moving_path(left, (left.position.x if left != None else None),
                                           current_v_c, current_h_c, 1, 0, True)
        super(Rook, self).calc_moving_path(right, (right.position.x if right != None else None),
                                           current_v_c, current_h_c, -1, 8, True)
        super(Rook, self).calc_moving_path(top, (top.position.y if top != None else None),
                                           current_h_c, current_v_c, -1, 9, False)
        super(Rook, self).calc_moving_path(bottom, (bottom.position.y if bottom != None else None),
                                           current_h_c, current_v_c, 1, 0, False)
