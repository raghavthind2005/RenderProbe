"""Tabletop occlusion scene (PIL 2.5D, no Blender).

Renders a colored target object progressively hidden behind a gray occluder.
The controlled factor is ``occluder_frac`` ∈ [0.0, 1.0] - the fraction of the
target's area covered by the occluder. Sweeping it asks: at what occlusion
level do models fail to identify the target?

CONTRACT (locked - probes/analyzers depend on these exact keys; do not rename):

    ground_truth = {
        "target_color":  str,    # the target object's color name (e.g. "red")
        "target_shape":  str,    # "circle" | "square"
        "occluder_frac": float,  # fraction of target covered (mirrors factors)
    }
    factors = {"occluder_frac": float}
    meta    = {"generator": "tabletop_occlusion", "seed": int, "params": dict,
               "tier": "trivial" | "standard"}   # trivial = occluder_frac 0.0 (ceiling check)

Implementation notes:
- Draw target first (bottom layer), then occluder on top (2.5D compositing).
- Occluder is a gray rectangle; its width scales with occluder_frac so that
  occluder_frac=0.0 leaves the target fully visible and 1.0 covers it entirely.
- Target is always centered; occluder slides in from the right edge.
- Use a seeded random.Random for target color/shape selection.
"""
from __future__ import annotations

import random as _random

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene

_CANVAS_SIZE = (480, 360)
_TARGET_COLORS = ["red", "blue", "green", "orange", "purple"]
_TARGET_RADIUS = 60
_OCCLUDER_COLOR = (160, 160, 160)   # gray


def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: str) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="black", width=2)


def _draw_square(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: str) -> None:
    draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline="black", width=2)


class _TabletopOcclusionGenerator:
    name = "tabletop_occlusion"
    params_schema = {
        "occluder_frac": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        occluder_frac = float(params["occluder_frac"])
        occluder_frac = max(0.0, min(1.0, occluder_frac))

        rng = _random.Random(seed)
        color = rng.choice(_TARGET_COLORS)
        shape = rng.choice(["circle", "square"])

        img = Image.new("RGB", _CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(img)

        w, h = _CANVAS_SIZE
        cx, cy = w // 2, h // 2
        r = _TARGET_RADIUS

        # Layer 1: target
        if shape == "circle":
            _draw_circle(draw, cx, cy, r, color)
        else:
            _draw_square(draw, cx, cy, r, color)

        # Layer 2: occluder slides in from right, covering `occluder_frac` of diameter
        occ_width = int(2 * r * occluder_frac)
        if occ_width > 0:
            occ_x0 = cx + r - occ_width
            occ_y0 = cy - r - 4   # slightly taller than the target
            occ_x1 = cx + r + 4
            occ_y1 = cy + r + 4
            draw.rectangle([occ_x0, occ_y0, occ_x1, occ_y1], fill=_OCCLUDER_COLOR)

        return Scene(
            id=f"occ_f{occluder_frac:.2f}_s{seed}",
            images=[img],
            ground_truth={
                "target_color": color,
                "target_shape": shape,
                "occluder_frac": occluder_frac,
            },
            factors={"occluder_frac": occluder_frac},
            meta={
                "generator": "tabletop_occlusion",
                "seed": seed,
                "params": dict(params),
                # Fully visible target -> the ceiling tier for the oracle check.
                "tier": "trivial" if occluder_frac == 0.0 else "standard",
            },
        )


PLUGIN = _TabletopOcclusionGenerator()
