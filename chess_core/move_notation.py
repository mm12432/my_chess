"""
本模块负责生成标准的中国象棋记谱（如"炮二平五"）。
"""

from my_chess.chess_core import chessman


class MoveNotation:
    """中国象棋记谱辅助类。"""

    # 中文数字映射（零到九）
    _CHINESE_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    # 棋子 FEN 字符到中文名称的映射
    _PIECE_NAMES = {
        "K": "帅",
        "k": "将",
        "R": "车",
        "r": "车",
        "N": "马",
        "n": "马",
        "B": "相",
        "b": "象",
        "A": "仕",
        "a": "士",
        "C": "炮",
        "c": "炮",
        "P": "兵",
        "p": "卒",
    }

    @staticmethod
    def _get_piece_name(piece: chessman.Chessman) -> str:
        """根据棋子对象获取中文名称。"""
        return MoveNotation._PIECE_NAMES.get(piece.fen_char, "?")

    @staticmethod
    def _col_to_display(col: int, is_red: bool) -> str:
        """将内部列坐标(0-8)转换为记谱显示用的列号。

        红方：从右到左为一至九，即 col 0 -> 九, col 8 -> 一
              公式：display = 9 - col
        黑方：从右到左为1至9，即 col 0 -> 1, col 8 -> 9
              公式：display = col + 1
        """
        if is_red:
            display_num = 9 - col
            return MoveNotation._CHINESE_NUMS[display_num]
        display_num = col + 1
        return str(display_num)

    @staticmethod
    def _distance_display(distance: int, is_red: bool) -> str:
        """将移动距离转换为记谱显示格式。"""
        if is_red:
            return MoveNotation._CHINESE_NUMS[distance]
        return str(distance)

    @staticmethod
    def get_move_name(
        piece: chessman.Chessman,
        from_col: int,
        from_row: int,
        to_col: int,
        to_row: int,
    ) -> str:
        """生成标准的中国象棋记谱。

        格式：[棋子名][起始列号][动作][目标]
        例如：炮二平五（红方）、炮2平5（黑方）
        """
        # 获取棋子中文名称
        name = MoveNotation._get_piece_name(piece)
        is_red = piece.is_red

        # 起始列号（记谱显示用）
        s_from = MoveNotation._col_to_display(from_col, is_red)

        # 判断方向
        # y 向上增加（0=底部红方, 9=顶部黑方）
        dy = to_row - from_row

        if dy == 0:
            action = "平"
        elif (is_red and dy > 0) or (not is_red and dy < 0):
            action = "进"
        else:
            action = "退"

        # 计算目标值
        # 线性棋子（车、炮、兵、帅/将）：进/退时用移动距离
        # 斜线棋子（马、象/相、仕/士）：进/退时用目标列号
        # "平"时：一律用目标列号
        p_char = piece.fen_char.upper()
        is_linear = p_char in ["R", "C", "P", "K"]

        if action == "平":
            s_dest = MoveNotation._col_to_display(to_col, is_red)
        elif is_linear:
            s_dest = MoveNotation._distance_display(abs(dy), is_red)
        else:
            s_dest = MoveNotation._col_to_display(to_col, is_red)

        return f"{name}{s_from}{action}{s_dest}"
