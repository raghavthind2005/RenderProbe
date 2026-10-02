"""Spatial-configuration scene (PIL 2.5D, no Blender).

Renders two always-present target objects (a red one and a blue one) plus
``n_distractors`` clutter shapes at known pixel positions on a canvas. The
controlled factor is ``n_distractors`` - sweeping it asks whether visual
clutter degrades a model's spatial reasoning.

CONTRACT (locked - probes/analyzers depend on these exact keys; do not rename):

    ground_truth = {
        "objects": [                # every drawn object, including the 2 targets
            {"color": str, "shape": str, "x": int, "y": int},
            ...
        ],
        "target_a": str,            # color of target A (e.g. "red")
        "target_b": str,            # color of target B (e.g. "blue")
        "relation": str,            # "left" | "right"  -> is A left/right of B (horizontal)
    }
    factors = {"n_distractors": int}
    meta    = {"generator": "spatial_config", "seed": int, "params": dict,
               "tier": "trivial" | "standard"}

The ``tier`` in meta is the difficulty label used by the oracle prompt-validity
(ceiling) check: on the "trivial" tier (0 distractors, just the two targets) a
correct oracle verbalization must let a competent reasoner near-ceiling, so a low
oracle accuracy there indicts the *prompt template*, not the model. Any scene
generator that wants to participate in that check should likewise stamp its
easiest configuration as ``meta["tier"] = "trivial"``.
"""
from __future__ import annotations

import math
import random as _random

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene

_CANVAS_SIZE = (480, 360)
_TARGET_A_COLOR = "red"
_TARGET_B_COLOR = "blue"
_DISTRACTOR_COLORS = ["green", "orange", "purple", "brown", "cyan", "gray"]
_RADIUS = 24
_MIN_DIST = 58       # minimum center-to-center distance between any two objects
_MIN_X_SEP = 80      # minimum horizontal separation between the two targets


def _place(rng: _random.Random, existing: list[tuple[int, int]]) -> tuple[int, int] | None:
    w, h = _CANVAS_SIZE
    margin = _RADIUS + 10
    for _ in range(300):
        x = rng.randint(margin, w - margin)
        y = rng.randint(margin, h - margin)
        if all(
            math.hypot(x - ox, y - oy) >= _MIN_DIST for ox, oy in existing
        ):
            return x, y
    return None


def _draw_circle(draw: ImageDraw.ImageDraw, x: int, y: int, color: str) -> None:
    r = _RADIUS
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="black", width=2)


def _draw_square(draw: ImageDraw.ImageDraw, x: int, y: int, color: str) -> None:
    r = _RADIUS
    draw.rectangle([x - r, y - r, x + r, y + r], fill=color, outline="black", width=2)


class _SpatialConfigGenerator:
    name = "spatial_config"
    params_schema = {
        "n_distractors": {"type": "int", "min": 0, "max": 14, "default": 0},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        n_distractors = int(params["n_distractors"])
        rng = _random.Random(seed)

        img = Image.new("RGB", _CANVAS_SIZE, "white")
        draw = ImageDraw.Draw(img)
        w, h = _CANVAS_SIZE
        margin = _RADIUS + 10
        placed: list[tuple[int, int]] = []
        objects = []

        # --- Target A (red) ---
        xa = rng.randint(margin, w - margin)
        ya = rng.randint(margin, h - margin)
        placed.append((xa, ya))

        # --- Target B (blue): enforce meaningful horizontal separation ---
        xb, yb = xa, ya  # sentinel
        for _ in range(500):
            pos = _place(rng, placed)
            if pos is None:
                break
            cx, cy = pos
            if abs(cx - xa) >= _MIN_X_SEP:
                xb, yb = cx, cy
                break
        placed.append((xb, yb))

        _draw_circle(draw, xa, ya, _TARGET_A_COLOR)
        _draw_circle(draw, xb, yb, _TARGET_B_COLOR)
        objects.append({"color": _TARGET_A_COLOR, "shape": "circle", "x": xa, "y": ya})
        objects.append({"color": _TARGET_B_COLOR, "shape": "circle", "x": xb, "y": yb})

        # --- Distractors ---
        distractor_shapes = ["circle", "square"]
        for _ in range(n_distractors):
            pos = _place(rng, placed)
            if pos is None:
                break
            dx, dy = pos
            placed.append((dx, dy))
            color = rng.choice(_DISTRACTOR_COLORS)
            shape = rng.choice(distractor_shapes)
            if shape == "circle":
                _draw_circle(draw, dx, dy, color)
            else:
                _draw_square(draw, dx, dy, color)
            objects.append({"color": color, "shape": shape, "x": dx, "y": dy})

        relation = "left" if xa < xb else "right"

        return Scene(
            id=f"spatial_n{n_distractors}_s{seed}",
            images=[img],
            ground_truth={
                "objects": objects,
                "target_a": _TARGET_A_COLOR,
                "target_b": _TARGET_B_COLOR,
                "relation": relation,
            },
            factors={"n_distractors": n_distractors},
            meta={
                "generator": "spatial_config",
                "seed": seed,
                "params": dict(params),
                # Easiest configuration (no clutter) -> the ceiling-check tier.
                "tier": "trivial" if n_distractors == 0 else "standard",
            },
        )


PLUGIN = _SpatialConfigGenerator()
