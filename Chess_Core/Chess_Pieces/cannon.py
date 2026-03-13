# coding:utf-8

from Chess_Core import Point
from Chess_Core.Chess_Pieces.base_chessman import Chessman


class Cannon(Chessman):

    def __init__(self, name_cn, name, is_red, chessboard):
        super(Cannon, self).__init__(name_cn, name, is_red, chessboard)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Cannon, self).position.x
        current_h_c = super(Cannon, self).position.y
        left = super(Cannon, self).chessboard.get_left_first_chessman(
            current_v_c, current_h_c)
        right = super(Cannon, self).chessboard.get_right_first_chessman(
            current_v_c, current_h_c)
        top = super(Cannon, self).chessboard.get_top_first_chessman(
            current_v_c, current_h_c)
        bottom = super(Cannon, self).chessboard.get_bottom_first_chessman(
            current_v_c, current_h_c)
        tar_left = super(Cannon, self).chessboard.get_left_second_chessman(
            current_v_c, current_h_c)
        tar_right = super(Cannon, self).chessboard.get_right_second_chessman(
            current_v_c, current_h_c)
        tar_top = super(Cannon, self).chessboard.get_top_second_chessman(
            current_v_c, current_h_c)
        tar_bottom = super(Cannon, self).chessboard.get_bottom_second_chessman(
            current_v_c, current_h_c)
        super(Cannon, self).calc_moving_path(left, (left.position.x if left != None else None),
                                             current_v_c, current_h_c, 1, 0, True, True)
        super(Cannon, self).calc_moving_path(right, (right.position.x if right != None else None),
                                             current_v_c, current_h_c, -1, 8, True, True)
        super(Cannon, self).calc_moving_path(top, (top.position.y if top != None else None),
                                             current_h_c, current_v_c, -1, 9, False, True)
        super(Cannon, self).calc_moving_path(bottom, (bottom.position.y if bottom != None else None),
                                             current_h_c, current_v_c, 1, 0, False, True)
        current_color = super(Cannon, self).is_red
        if tar_left != None and tar_left.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point.Point(tar_left.position.x, tar_left.position.y))
        if tar_right != None and tar_right.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point.Point(tar_right.position.x, tar_right.position.y))
        if tar_top != None and tar_top.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point.Point(tar_top.position.x, tar_top.position.y))
        if tar_bottom != None and tar_bottom.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point.Point(tar_bottom.position.x, tar_bottom.position.y))
