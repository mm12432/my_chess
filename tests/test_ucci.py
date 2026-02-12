import unittest
import sys
import os

# Add project parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from MyChess.Chess_Core.Point import Point


class TestUCCI(unittest.TestCase):
    def test_to_ucci(self):
        # Bottom-left (Red side)
        p = Point(0, 0)
        self.assertEqual(p.to_ucci(), "a0")

        # Bottom-right
        p = Point(8, 0)
        self.assertEqual(p.to_ucci(), "i0")

        # Top-left (Black side)
        p = Point(0, 9)
        self.assertEqual(p.to_ucci(), "a9")

        # Top-right
        p = Point(8, 9)
        self.assertEqual(p.to_ucci(), "i9")

        # Center
        p = Point(4, 4)
        self.assertEqual(p.to_ucci(), "e4")

    def test_from_ucci(self):
        p = Point.from_ucci("a0")
        self.assertEqual(p.x, 0)
        self.assertEqual(p.y, 0)

        p = Point.from_ucci("i9")
        self.assertEqual(p.x, 8)
        self.assertEqual(p.y, 9)

        p = Point.from_ucci("e4")
        self.assertEqual(p.x, 4)
        self.assertEqual(p.y, 4)

    def test_invalid_ucci(self):
        with self.assertRaises(ValueError):
            Point.from_ucci("j0")  # x out of bounds

        with self.assertRaises(ValueError):
            Point.from_ucci(
                "a10"
            )  # y out of bounds (length check will fail first or parsing)

        with self.assertRaises(ValueError):
            Point.from_ucci("z1")

        with self.assertRaises(ValueError):
            Point.from_ucci("a")


if __name__ == "__main__":
    unittest.main()
