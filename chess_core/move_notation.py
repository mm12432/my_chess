"""
This module handles the generation of standard Chinese Chess notation (e.g., "Cannon 2 moves 5").
"""

from my_chess.chess_core import chessman


class MoveNotation:
    """Helper class for generating move notations."""

    @staticmethod
    def get_chinese_num(num):
        """Convert number 0-9 to Chinese Characters.
        Red uses Chinese nums for cols: 一二三四五六七八九
        Black uses Arabic nums for cols: 1 2 3 4 5 6 7 8 9 (but in Chinese chars often??
        Standard Convention:
        Red cols: 九八七六五四三二一 (from right to left)
        Black cols: 1 2 3 4 5 6 7 8 9 (from right to left)
        """
        nums = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        return nums[num]

    @staticmethod
    def get_move_name(
        piece: chessman.Chessman,
        from_col: int,
        from_row: int,
        to_col: int,
        to_row: int,
    ) -> str:
        """
        Generates the standard Chinese Chess notation for a move.
        """
        # Determine strict name for the piece (e.g., "车", "马") without decorators
        name = piece.name_cn.strip()[1]  # e.g. " 车l红 " -> "车"

        is_red = piece.is_red

        # Calculate column numbers (files)
        # Red: 1 is at x=8, 9 is at x=0
        # Black: 1 is at x=0, 9 is at x=8
        if is_red:
            pass
        else:
            pass

        # Determine direction
        # y increases upwards (0 is bottom, 9 is top)
        dy = to_row - from_row

        # Movement type: 平 (Horizontal), 进 (Forward), 退 (Backward)
        # Red Forward: dy > 0
        # Black Forward: dy < 0

        if dy == 0:
            action = "平"
        elif (is_red and dy > 0) or (not is_red and dy < 0):
            action = "进"
        else:
            action = "退"

        # Calculate Distance or Target
        # For '平', usually target column.
        # For '进'/'退':
        #   - Linear pieces (Rook, Cannon, Pawn, King): Distance (abs(dy))
        #   - Diagonal pieces (Knight, Elephant, Mandarin): Target Column

        # Identify piece type by class name or char
        # Using fen_char or checking type is reliable
        p_char = piece.fen_char.upper()

        is_linear = p_char in ["R", "C", "P", "K"]  # Rook, Cannon, Pawn, King

        if action == "平":
            dest = to_col
        else:
            if is_linear:
                dest = abs(dy)
            else:
                dest = to_col

        # Formatting
        # Red uses Chinese numbers for Columns and Distances
        # Black uses Arabic numbers for Columns and Distances

        def fmt(n, use_chinese):
            if use_chinese:
                return MoveNotation.get_chinese_num(n)
            return str(n)

        # Structure: [Piece][FromCol][Action][Dest]
        # Example: 炮二平五 (Red), 炮2平5 (Black)

        s_piece = name
        s_from = fmt(from_col, is_red)
        s_action = action
        # Destination:
        # For Black, we use Arabic numbers. For Red, Chinese numbers.
        # Note: In standard notation, Red uses Chinese for everything. Black uses Arabic.
        s_dest = fmt(dest, is_red)

        return f"{s_piece}{s_from}{s_action}{s_dest}"
