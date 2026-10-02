"""Shared perception-report handle.

A magenta ring that points at exactly one object so a report can ask about it WITHOUT
asking the model for pixel coordinates (VLMs localize poorly by coordinate - that
confound is what this avoids). The color is deliberately outside every scene palette,
so the ring can never be mistaken for an object.

Underscore-prefixed so the registry's autodiscover skips it - this is a helper, not a
plugin.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

HALO_RGB = (230, 0, 230)   # magenta - absent from every object palette


def ring(img: Image.Image, x: int, y: int, r: int, pad: int = 3, width: int = 3) -> Image.Image:
    """Return a copy of ``img`` with a magenta ring hugging the object at (x, y)."""
    out = img.copy()
    ImageDraw.Draw(out).ellipse(
        [x - r - pad, y - r - pad, x + r + pad, y + r + pad], outline=HALO_RGB, width=width
    )
    return out
