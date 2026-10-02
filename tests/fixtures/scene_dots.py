"""Smoke-slice scene: N colored dots on a white canvas, rendered with PIL. No Blender."""
from __future__ import annotations

import math
import random as _random

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene

_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
_CANVAS_SIZE = (400, 300)


def _dot_radius(n_dots: int) -> int:
    """Shrink dots as the count rises so N non-overlapping dots always fit - keeping
    the count recoverable in principle (GT trustable) while raising perceptual load."""
    if n_dots <= 8:
        return 18
    if n_dots <= 16:
        return 14
    if n_dots <= 26:
        return 11
    return 9


class _DotsSceneGenerator:
    name = "dots"
    params_schema = {
        "n_dots": {"type": "int", "min": 1, "max": 40, "default": 5},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        n_dots: int = int(params["n_dots"])
        rng = _random.Random(seed)

        img = Image.new("RGB", _CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(img)

        w, h = _CANVAS_SIZE
        radius = _dot_radius(n_dots)
        margin = radius + 5
        min_dist = 2 * radius + 4   # NON-OVERLAPPING: dots never merge -> count is exact
        placed: list[tuple[int, int]] = []
        dots: list[dict] = []
        for _ in range(n_dots):
            for _try in range(600):
                cx = rng.randint(margin, w - margin)
                cy = rng.randint(margin, h - margin)
                if all(math.hypot(cx - px, cy - py) >= min_dist for px, py in placed):
                    placed.append((cx, cy))
                    break
            else:
                placed.append((cx, cy))   # conservative radius makes this effectively unreachable
            color = rng.choice(_COLORS)
            dots.append({"x": cx, "y": cy, "color": color})
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=color,
                outline="black",
            )

        return Scene(
            id=f"dots_n{n_dots}_s{seed}",
            images=[img],
            # `count` is the task GT; `dots`/`dot_radius` are additive perceptual detail
            # (per-dot color + position) that lets the count probe's perception report
            # point at a single dot. Non-overlapping placement keeps every field exact.
            ground_truth={"count": n_dots, "dots": dots, "dot_radius": radius},
            factors={"n_dots": n_dots},
            meta={
                "generator": "dots",
                "seed": seed,
                "params": dict(params),
                # Few dots -> the ceiling tier (a valid oracle must be near-perfect here).
                "tier": "trivial" if n_dots <= 2 else "standard",
            },
        )


PLUGIN = _DotsSceneGenerator()
