"""Move primitives shared by the rules engine, UI adapters, and AI layers."""

from __future__ import annotations

from dataclasses import dataclass

from my_chess.chess_core.point import Point


@dataclass(frozen=True)
class Move:
    """Represents a board move from one square to another."""

    from_point: Point
    to_point: Point
    piece_name: str | None = None

    @classmethod
    def from_coords(
        cls,
        from_col: int,
        from_row: int,
        to_col: int,
        to_row: int,
        piece_name: str | None = None,
    ) -> "Move":
        """Builds a move from raw board coordinates."""
        return cls(Point(from_col, from_row), Point(to_col, to_row), piece_name)

    def to_ucci(self) -> str:
        """Serializes the move as a simple UCCI coordinate pair."""
        return f"{self.from_point.to_ucci()}{self.to_point.to_ucci()}"
