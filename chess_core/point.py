from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def to_ucci(self) -> str:
        """Converts the point to UCCI coordinate string (e.g., a0, i9)."""
        return f"{chr(ord('a') + self.x)}{self.y}"

    @classmethod
    def from_ucci(cls, ucci: str) -> "Point":
        """Creates a Point from a UCCI coordinate string."""
        if len(ucci) != 2:
            raise ValueError(f"Invalid UCCI coordinate: {ucci}")
        x = ord(ucci[0]) - ord("a")
        y = int(ucci[1])
        if not (0 <= x <= 8 and 0 <= y <= 9):
            raise ValueError(f"Coordinate out of bounds: {ucci}")
        return cls(x, y)
