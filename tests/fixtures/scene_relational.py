"""Compositional relational scene - 2.5D PIL, keyless.

Renders ``n_objects`` colored shapes and a multi-hop referential chain question
whose answer is verified correct by construction (see :mod:`scenes._relational_core`).

CONTRACT (locked):

    ground_truth = {
        "question":          str,   # NL question (linear grammar)
        "answer":            str,   # single color word
        "scene_graph_text":  str,   # oracle verbalization of the full scene
        "objects": [                # all drawn objects
            {"color": str, "shape": str, "size": str, "x": int, "y": int},
            ...
        ],
    }
    factors = {"n_objects": int, "hop_depth": int}
    meta    = {"generator": "relational", "seed": int, "params": dict,
               "tier": "trivial" | "standard"}

``tier`` is "trivial" when n_objects==4 and hop_depth==1 (minimum load on both knobs).
Answer balance: target color = COLORS[seed % 6], cycling across seeds (P1 invariant).
"""
from __future__ import annotations

import math
import random as _random
from typing import Any

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene
from tests.fixtures._relational_core import (
    COLORS,
    SHAPES,
    SIZES,
    Obj,
    build_chain,
    render_question,
    scene_graph_to_text,
)

_CANVAS = (480, 360)
_MARGIN = 30

# Perceptual-difficulty ("clutter") levels -> (r_large, r_small, min_gap). Higher clutter
# shrinks objects and tightens spacing, so *seeing* the scene gets harder - but every
# level keeps a POSITIVE min_gap, so objects never overlap and a perfect perceiver can
# always disambiguate them. The ground truth is therefore never made ambiguous; only the
# perceptual load rises (spacing is the decisive factor in BlindTest, arXiv:2407.06581).
_CLUTTER: dict[int, tuple[int, int, int]] = {
    0: (26, 16, 12),   # spacious (original)
    1: (20, 13, 7),    # moderate
    2: (14, 10, 4),    # tight - small objects, near-touching (still non-overlapping)
}

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":    (220,  50,  50),
    "blue":   ( 50,  80, 220),
    "green":  ( 40, 170,  60),
    "orange": (240, 140,  20),
    "purple": (150,  50, 190),
    "yellow": (210, 200,  20),
}


def _clutter_cfg(clutter: int) -> tuple[int, int, int]:
    return _CLUTTER.get(clutter, _CLUTTER[0])


def _r(size: str, clutter: int) -> int:
    r_large, r_small, _ = _clutter_cfg(clutter)
    return r_large if size == "large" else r_small


def _place_objects(
    rng: _random.Random, attrs: list[tuple[str, str, str]], clutter: int
) -> list[Obj] | None:
    """Assign non-overlapping positions with distinct x AND distinct y (tie-free superlatives).

    attrs: list of (color, shape, size). Returns None on placement failure. Objects are
    kept a positive gap apart at every clutter level, so the layout is always resolvable
    in principle - clutter raises perceptual load, never ambiguity.
    """
    w, h = _CANVAS
    min_gap = _clutter_cfg(clutter)[2]
    placed: list[Obj] = []
    used_xs: set[int] = set()
    used_ys: set[int] = set()

    for color, shape, size in attrs:
        rad = _r(size, clutter)
        lo_x, hi_x = _MARGIN + rad, w - _MARGIN - rad
        lo_y, hi_y = _MARGIN + rad, h - _MARGIN - rad
        if lo_x >= hi_x or lo_y >= hi_y:
            return None

        for _ in range(800):
            x = rng.randrange(lo_x, hi_x, 2)
            y = rng.randrange(lo_y, hi_y, 2)
            if x in used_xs or y in used_ys:
                continue
            ok = all(
                math.hypot(x - o.x, y - o.y) >= _r(o.size, clutter) + rad + min_gap
                for o in placed
            )
            if not ok:
                continue
            placed.append(Obj(color=color, shape=shape, size=size, x=x, y=y))
            used_xs.add(x)
            used_ys.add(y)
            break
        else:
            return None

    return placed


def _draw_triangle(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, r: int,
    fill: tuple[int, int, int],
) -> None:
    draw.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)],
                 fill=fill, outline=(30, 30, 30))


def _render(objects: list[Obj], clutter: int) -> Image.Image:
    img = Image.new("RGB", _CANVAS, (248, 248, 248))
    draw = ImageDraw.Draw(img)
    for o in objects:
        fill = _COLOR_RGB[o.color]
        r = _r(o.size, clutter)
        if o.shape == "circle":
            draw.ellipse(
                [o.x - r, o.y - r, o.x + r, o.y + r],
                fill=fill, outline=(30, 30, 30), width=2,
            )
        elif o.shape == "square":
            draw.rectangle(
                [o.x - r, o.y - r, o.x + r, o.y + r],
                fill=fill, outline=(30, 30, 30), width=2,
            )
        else:
            _draw_triangle(draw, o.x, o.y, r, fill)
    return img


class _RelationalGenerator:
    name = "relational"
    params_schema = {
        "n_objects": {"type": "int", "min": 4, "max": 16, "default": 6},
        "hop_depth": {"type": "int", "min": 1, "max": 6,  "default": 2},
        # perceptual difficulty: 0 spacious, 1 moderate, 2 tight (small, near-touching)
        "clutter":   {"type": "int", "min": 0, "max": 2,  "default": 0},
    }

    _ATTRS_RETRIES = 6
    _PLACE_RETRIES = 8
    _CHAIN_RETRIES = 6

    def _build_scene(
        self, n_objects: int, hop_depth: int, seed: int, target_color: str,
        clutter: int,
    ) -> tuple[list[Obj] | None, "object | None"]:
        """Search deterministically for a (layout, verifiable chain) pair.

        Outer loop re-draws attributes + placement; inner loop retries chain
        construction. target_color is fixed so answer balance is preserved.
        Returns (objects, chain) or (None, None) if no feasible scene is found.
        """
        for a in range(self._ATTRS_RETRIES):
            attr_rng = _random.Random(seed * 1000 + a)
            attrs: list[tuple[str, str, str]] = [
                (target_color, attr_rng.choice(SHAPES), attr_rng.choice(SIZES))
            ]
            for _ in range(n_objects - 1):
                attrs.append((
                    attr_rng.choice(COLORS),
                    attr_rng.choice(SHAPES),
                    attr_rng.choice(SIZES),
                ))
            attr_rng.shuffle(attrs)

            objects: list[Obj] | None = None
            for p in range(self._PLACE_RETRIES):
                objects = _place_objects(
                    _random.Random(seed * 31 + p * 7 + a * 13), attrs, clutter
                )
                if objects is not None:
                    break
            if objects is None:
                continue

            for c in range(self._CHAIN_RETRIES):
                chain = build_chain(
                    objects,
                    _random.Random(seed * 97 + c * 101 + a * 17 + 1),
                    depth=hop_depth,
                    target_color=target_color,
                )
                if chain is not None:
                    return objects, chain
        return None, None

    def generate(self, params: dict, seed: int) -> Scene:
        n_objects = int(params["n_objects"])
        hop_depth = int(params["hop_depth"])
        clutter = int(params.get("clutter", 0))

        # Round-robin answer balance: the target color cycles by seed index and is
        # held FIXED across all retries below, so balance is never perturbed by the
        # search for a feasible scene.
        target_color = COLORS[seed % len(COLORS)]

        # Tight configurations (few objects, deep chains) occasionally admit no
        # verifiable chain for a given object layout. Rather than fail on
        # schema-valid params, re-draw the whole scene (attributes + placement)
        # under varied sub-seeds - deterministic per (seed, params) - and retry
        # chain construction, keeping target_color fixed.
        objects, chain = self._build_scene(
            n_objects, hop_depth, seed, target_color, clutter
        )
        if objects is None or chain is None:
            raise RuntimeError(
                f"relational: no verifiable scene after retries "
                f"(seed={seed} n_objects={n_objects} hop_depth={hop_depth} "
                f"clutter={clutter}). A depth-d chain needs more than d objects; "
                f"raise n_objects (>{hop_depth}) or reduce hop_depth/clutter."
            )

        # Store each object's RENDERED radius so the probe's highlight ring and any
        # consumer use the true pixel size (which depends on size AND clutter) - a
        # single source of truth, so the ring always encircles exactly the target.
        obj_dicts: list[dict[str, Any]] = [
            {"color": o.color, "shape": o.shape, "size": o.size,
             "x": o.x, "y": o.y, "r": _r(o.size, clutter)}
            for o in objects
        ]

        return Scene(
            id=f"rel_n{n_objects}_d{hop_depth}_c{clutter}_s{seed}",
            images=[_render(objects, clutter)],
            ground_truth={
                "question":         render_question(chain),
                "answer":           target_color,
                "scene_graph_text": scene_graph_to_text(objects),
                "objects":          obj_dicts,
            },
            factors={"n_objects": n_objects, "hop_depth": hop_depth, "clutter": clutter},
            meta={
                "generator": "relational",
                "seed":      seed,
                "params":    dict(params),
                "tier": (
                    "trivial"
                    if (n_objects == 4 and hop_depth == 1 and clutter == 0)
                    else "standard"
                ),
            },
        )


PLUGIN = _RelationalGenerator()
