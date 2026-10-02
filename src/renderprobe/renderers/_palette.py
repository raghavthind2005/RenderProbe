"""Shared color-name -> RGB map for renderers. Underscore-prefixed so it isn't
discovered as a renderer plugin."""
from __future__ import annotations

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "blue": (40, 90, 210),
    "green": (30, 160, 70),
    "orange": (240, 150, 20),
    "purple": (150, 60, 200),
    "yellow": (230, 205, 40),
    "pink": (235, 130, 180),
    "brown": (140, 80, 40),
    "gray": (140, 140, 140),
    "cyan": (40, 190, 200),
}


def rgb(color: str) -> tuple[int, int, int]:
    """Resolve a palette name to RGB; unknown names fall back to a neutral gray."""
    return PALETTE.get(color, (110, 110, 110))
