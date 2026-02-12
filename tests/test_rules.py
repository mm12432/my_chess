"""走子合法性和规则引擎单元测试。"""
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_core.chessboard import Chessboard


class TestRookMoves(unittest.TestCase):
    """车(Rook)的走子规则测试。"""

    def test_rook_moves_straight(self):
        """车可以沿直线移动。"""
        # 只放一个红车和两个将帅
        fen = "4k4/9/9/9/9/9/9/9/9/4K3R w - - 0 1"
        board = Chessboard.from_fen(fen)
        rook = board.get_chessman(8, 0)
        self.assertIsNotNone(rook)
        self.assertEqual(rook.fen_char, "R")
        rook.calc_moving_list()
        moves = [(p.x, p.y) for p in rook.moving_list]
        # 车应该能向上移动 (8, 1) 到 (8, 9) 和向左移动 (5,0) 到 (7,0)
        self.assertIn((8, 1), moves)
        self.assertIn((8, 5), moves)
        self.assertIn((7, 0), moves)

    def test_rook_blocked(self):
        """车被己方棋子阻挡。"""
        # 红车在 (0,0), 红兵在 (0,3) 挡住向上路线
        fen = "4k4/9/9/9/9/9/p8/9/9/R3K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        rook = board.get_chessman(0, 0)
        self.assertIsNotNone(rook)
        rook.calc_moving_list()
        moves = [(p.x, p.y) for p in rook.moving_list]
        # 车能向右移动
        self.assertIn((1, 0), moves)
        self.assertIn((2, 0), moves)
        self.assertIn((3, 0), moves)


class TestKnightMoves(unittest.TestCase):
    """马(Knight)的走子规则测试。"""

    def test_knight_basic_move(self):
        """马走日字。"""
        # row 5 在 FEN 中是第5行 (r_idx=4, actual_row=5)
        fen = "4k4/9/9/9/4N4/9/9/9/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        knight = board.get_chessman(4, 5)
        self.assertIsNotNone(knight)
        self.assertEqual(knight.fen_char, "N")
        knight.calc_moving_list()
        moves = [(p.x, p.y) for p in knight.moving_list]
        # 马在 (4,5) 应该可以跳到日字位置
        expected_moves = [
            (3, 7), (5, 7),  # 上方
            (3, 3), (5, 3),  # 下方
            (2, 6), (2, 4),  # 左方
            (6, 6), (6, 4),  # 右方
        ]
        for move in expected_moves:
            self.assertIn(move, moves, f"马应该能走到 {move}")

    def test_knight_blocked_leg(self):
        """蹩马腿：马腿方向有棋子时不能跳。"""
        # 马在 (4,5), (4,6) 有棋子挡住上方马腿
        fen = "4k4/9/9/4p4/4N4/9/9/9/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        knight = board.get_chessman(4, 5)
        self.assertIsNotNone(knight)
        knight.calc_moving_list()
        moves = [(p.x, p.y) for p in knight.moving_list]
        # (4,6) 方向被堵，不能跳到 (3,7) 和 (5,7)
        self.assertNotIn((3, 7), moves)
        self.assertNotIn((5, 7), moves)


class TestCannonMoves(unittest.TestCase):
    """炮(Cannon)的走子规则测试。"""

    def test_cannon_move_no_capture(self):
        """炮不吃子时沿直线移动（同车）。"""
        fen = "4k4/9/9/9/9/9/9/4C4/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        cannon = board.get_chessman(4, 2)
        self.assertIsNotNone(cannon)
        self.assertEqual(cannon.fen_char, "C")
        cannon.calc_moving_list()
        moves = [(p.x, p.y) for p in cannon.moving_list]
        # 炮能向上、向左、向右移动
        self.assertIn((4, 3), moves)
        self.assertIn((4, 5), moves)
        self.assertIn((0, 2), moves)

    def test_cannon_capture_over_screen(self):
        """炮隔子吃子。"""
        # 炮在 (4,2), 架子在 (4,5), 目标在 (4,9)
        fen = "4k4/9/9/9/4p4/9/9/4C4/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        cannon = board.get_chessman(4, 2)
        self.assertIsNotNone(cannon)
        cannon.calc_moving_list()
        moves = [(p.x, p.y) for p in cannon.moving_list]
        # 炮应该能隔着 (4,5) 的棋子吃 (4,9) 的将
        self.assertIn((4, 9), moves)
        # 但不能走到架子位置
        self.assertNotIn((4, 5), moves)


class TestElephantMoves(unittest.TestCase):
    """象/相(Elephant)的走子规则测试。"""

    def test_elephant_moves_diagonal(self):
        """象走田字。"""
        fen = "4k4/9/9/9/9/9/9/9/9/2B1K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        elephant = board.get_chessman(2, 0)
        self.assertIsNotNone(elephant)
        self.assertEqual(elephant.fen_char, "B")
        elephant.calc_moving_list()
        moves = [(p.x, p.y) for p in elephant.moving_list]
        # 象在 (2,0) 只能走到 (0,2) 和 (4,2)
        self.assertIn((0, 2), moves)
        self.assertIn((4, 2), moves)

    def test_elephant_cannot_cross_river(self):
        """象不能过河。"""
        fen = "4k4/9/9/9/9/9/9/2B6/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        elephant = board.get_chessman(2, 2)
        self.assertIsNotNone(elephant)
        elephant.calc_moving_list()
        moves = [(p.x, p.y) for p in elephant.moving_list]
        # 象在 (2, 2) 向上最高到 (0,4) 或 (4,4)，但不能过河 (row >= 5)
        for _, row in moves:
            self.assertLess(row, 5, "红象不能过河")


class TestMandarinMoves(unittest.TestCase):
    """仕/士(Mandarin)的走子规则测试。"""

    def test_mandarin_moves_in_palace(self):
        """仕只能在九宫内斜走。"""
        fen = "4k4/9/9/9/9/9/9/9/9/3AK4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        mandarin = board.get_chessman(3, 0)
        self.assertIsNotNone(mandarin)
        self.assertEqual(mandarin.fen_char, "A")
        mandarin.calc_moving_list()
        moves = [(p.x, p.y) for p in mandarin.moving_list]
        # 仕在 (3,0) 只能走到 (4,1)
        self.assertIn((4, 1), moves)
        # 不能走出九宫
        for col, row in moves:
            self.assertTrue(3 <= col <= 5, f"仕不能走出九宫：col={col}")
            self.assertTrue(0 <= row <= 2, f"红仕不能走出九宫：row={row}")


class TestKingMoves(unittest.TestCase):
    """将/帅(King)的走子规则测试。"""

    def test_king_moves_in_palace(self):
        """帅只能在九宫内直线走一步。"""
        fen = "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        king = board.get_chessman(4, 0)
        self.assertIsNotNone(king)
        self.assertEqual(king.fen_char, "K")
        king.calc_moving_list()
        moves = [(p.x, p.y) for p in king.moving_list]
        # 帅在 (4,0) 可以走到 (3,0), (5,0), (4,1)
        self.assertIn((4, 1), moves)
        self.assertIn((3, 0), moves)
        self.assertIn((5, 0), moves)
        # 不能走出九宫
        for col, row in moves:
            self.assertTrue(3 <= col <= 5, f"帅不能走出九宫：col={col}")
            self.assertTrue(0 <= row <= 2, f"帅不能走出九宫：row={row}")


class TestPawnMoves(unittest.TestCase):
    """兵/卒(Pawn)的走子规则测试。"""

    def test_pawn_before_river(self):
        """兵过河前只能前进。"""
        fen = "4k4/9/9/9/9/9/4P4/9/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        pawn = board.get_chessman(4, 3)
        self.assertIsNotNone(pawn)
        self.assertEqual(pawn.fen_char, "P")
        pawn.calc_moving_list()
        moves = [(p.x, p.y) for p in pawn.moving_list]
        # 过河前(row < 5)只能前进
        self.assertIn((4, 4), moves)
        self.assertNotIn((3, 3), moves, "兵过河前不能横走")
        self.assertNotIn((5, 3), moves, "兵过河前不能横走")

    def test_pawn_after_river(self):
        """兵过河后可以前进和横走。"""
        fen = "4k4/9/9/9/4P4/9/9/9/9/4K4 w - - 0 1"
        board = Chessboard.from_fen(fen)
        pawn = board.get_chessman(4, 5)
        self.assertIsNotNone(pawn)
        pawn.calc_moving_list()
        moves = [(p.x, p.y) for p in pawn.moving_list]
        # 过河后(row >= 5)可以前进和横走
        self.assertIn((4, 6), moves, "兵过河后能前进")
        self.assertIn((3, 5), moves, "兵过河后能左走")
        self.assertIn((5, 5), moves, "兵过河后能右走")
        self.assertNotIn((4, 4), moves, "兵不能后退")


class TestStalemate(unittest.TestCase):
    """困毙规则测试。"""

    def test_stalemate_detection(self):
        """正常局面不应判定困毙。"""
        board = Chessboard("test")
        board.init_board()
        # 标准开局，双方都有大量合法走法
        winner = board.get_winner()
        self.assertIsNone(winner, "标准开局不应该有赢家")

    def test_king_captured_wins(self):
        """将/帅被吃则对方获胜。"""
        # 使用标准开局，然后移除黑将
        board = Chessboard("test")
        board.init_board()
        # 模拟黑将被吃掉：从哈希表和棋盘中移除
        black_king = board.get_chessman_by_name("black_king")
        self.assertIsNotNone(black_king)
        board.chessmans[4][9] = None
        board.chessmans_hash.pop("black_king")
        winner = board.get_winner()
        self.assertEqual(winner, "Red", "黑将不在棋盘上，红方应该获胜")

    def test_repetition_penalty(self):
        """三次重复局面，造成重复的一方判负。"""
        board = Chessboard("test")
        board.init_board()
        # 模拟：当前轮到红方走棋 (_is_red_turn=True)
        # 说明黑方刚走完造成了第3次重复，黑方判负，红方胜
        board.hash_history[board.current_hash] = 3
        winner = board.get_winner()
        self.assertEqual(winner, "Red", "黑方造成重复，应判红方胜")


class TestMoveNotation(unittest.TestCase):
    """记谱正确性测试。"""

    def test_red_cannon_notation(self):
        """红方炮二平五记谱正确。"""
        from my_chess.chess_core.move_notation import MoveNotation

        board = Chessboard("test")
        board.init_board()
        cannon = board.get_chessman(7, 2)  # 红右炮
        notation = MoveNotation.get_move_name(cannon, 7, 2, 4, 2)
        self.assertEqual(notation, "炮二平五")

    def test_black_pawn_notation(self):
        """黑方卒5进1记谱正确。"""
        from my_chess.chess_core.move_notation import MoveNotation

        board = Chessboard("test")
        board.init_board()
        pawn = board.get_chessman(4, 6)  # 黑中卒
        notation = MoveNotation.get_move_name(pawn, 4, 6, 4, 5)
        self.assertEqual(notation, "卒5进1")

    def test_red_knight_notation(self):
        """红方马八进七记谱正确。"""
        from my_chess.chess_core.move_notation import MoveNotation

        board = Chessboard("test")
        board.init_board()
        knight = board.get_chessman(1, 0)  # 红左马
        notation = MoveNotation.get_move_name(knight, 1, 0, 2, 2)
        self.assertEqual(notation, "马八进七")


if __name__ == "__main__":
    unittest.main()
