"""Polycube shape-matching scene: 3D mental rotation over interlocking pieces.

An N x N x N cube is partitioned into K interlocking polycube pieces (see
:mod:`scenes._polycube_core`). One piece is shown again on its own, in a different
orientation and in neutral gray, as the TARGET. The model must say which of the K
colored candidates is the same piece.

WHY THIS QUESTION AND NOT "WHICH PIECE FILLS THE HOLE". The obvious version of this
task shows the assembly with one piece removed and asks which piece fills the cavity.
Measured against the renderer's own id pass, the cavity is fully visible from a single
view in only about 30% of scenes - often just 2 of 6 cells - so in most scenes the
hole's shape cannot be read off the image at all and the ground truth is not
recoverable. Standalone pieces have the opposite property: every piece has an
orientation showing all of its cubes, in 100% of scenes measured, and the scene picks
such an orientation for each. The question is therefore answerable by construction.

The four ways this question could be ill-posed are each closed and tested:

* two pieces the same shape would make the answer ambiguous - the decomposition is
  rejected unless the pieces are pairwise non-congruent;
* a piece with a unique cell count could be picked out by counting alone, and one with
  a unique bounding box by measuring alone, both without rotating anything - the answer
  is only ever a piece that shares BOTH with another piece;
* drawing the target in its own color would give the answer away - the target is
  always neutral gray, so color labels the candidates and carries no evidence.

The one factor is ``n_pieces``: it sets how many candidates there are, and so both the
chance floor (1/K) and how many near-identical shapes have to be told apart.
"""
from __future__ import annotations

import random as _random

from renderprobe.core.schema import GraphObject, Scene, SceneGraph
from renderprobe.renderers import render_graph
from renderprobe.scenes._polycube_core import (
    ROTATIONS,
    _apply,
    canonical,
    congruent,
    decompose,
    eligible_answers,
    normalize,
    orient_for_visibility,
    toward_camera,
)

_CANVAS = (820, 430)
# Six maximally separable hues. Orange is left out: under this lighting it reads
# close to yellow, and brown, and a color a model confuses is a measurement
# confound rather than a design detail.
_COLORS = ["red", "blue", "green", "purple", "cyan", "pink"]
_TARGET_COLOR = "gray"
_AZIMUTH, _ELEVATION = 32.0, 17.0
_UNIT_PX = 30.0             # screen px per unit cube
_GAP = 0.75                 # spacing between pieces, as a multiple of the largest
_TARGET_FORWARD = 3.0       # target set-back, as a multiple of the largest piece.
                            # Measured at grid 3: a fixed 5.0 cells hid a cube of the
                            # middle candidate in 6/6 seeds, 7.0 in 1/6, 9.0 in 0/6.


def _span(cells, rx: float = 1.0, rz: float = 0.0) -> float:
    """How wide a piece is ON SCREEN, along the horizontal axis of the view.

    The x-extent alone is not that width: under a three-quarter view a piece that runs
    deep in z also spreads sideways in the image, so spacing by x-extent leaves deep
    pieces overlapping their neighbors even though their x ranges are disjoint.
    Projecting each cell onto the screen-horizontal axis measures what actually has to
    fit.
    """
    proj = [c[0] * rx + c[2] * rz for c in cells]
    return max(proj) - min(proj) + 1.0


def _frame_distance(total_cells: float, forward: float, unit_px: float) -> float:
    """Camera distance that fits the whole layout in frame, with a margin.

    The sensor's 42-degree field of view spans about 0.384 * distance either side of
    the axis, so the distance has to grow with the widest thing being framed.
    """
    half_world = max(total_cells, forward) * unit_px / 100.0 / 2.0
    return max(9.0, half_world / 0.384 * 1.35 + 2.5)


def _screen_right(view: tuple[float, float, float]) -> tuple[float, float]:
    """The horizontal screen axis in grid coordinates, as (x, z).

    Pieces spread along this stay the same distance from the camera, so they all
    render at the same scale and the comparison between them is fair.
    """
    fx, fy, fz = (-view[0], -view[1], -view[2])          # into the scene
    rx, rz = (fy * 0.0 - fz, fx * 0.0)                   # cross(forward, up), y term drops
    rx, rz = -fz, fx
    mag = (rx * rx + rz * rz) ** 0.5 or 1.0
    return rx / mag, rz / mag


def _base_y(cells) -> float:
    """Grid-y offset that puts a piece's lowest cube on the floor."""
    return -min(c[1] for c in cells)


def _place(cells, ox: float, oy: float, oz: float, color: str, tag: str,
           unit_px: float = _UNIT_PX):
    """Lay one piece into the scene graph at a grid offset."""
    out = []
    for i, (x, y, z) in enumerate(cells):
        out.append(GraphObject(
            id=f"{tag}_{i}", shape="cube", color=color,
            position=(_CANVAS[0] / 2 + (ox + x) * unit_px,
                      _CANVAS[1] / 2 - (oy + y) * unit_px,
                      (oz + z) * unit_px),
            size=unit_px / 2))
    return out


class _PolycubeScene:
    name = "polycube"
    # The shapes are only readable under a true-3D renderer at this scene's camera
    # angle; a flat projection overlaps the pieces and the question stops being
    # answerable. Named here so `validate` knows which renderer to hold it to.
    default_renderer = "mitsuba_3d"
    params_schema = {
        # Fewer than 4 pieces on a 3-cube makes them bulky enough that many have no
        # readable orientation, so the floor is 4.
        "n_pieces": {"type": "int", "min": 4, "max": 6, "default": 5},
    }
    # `grid` is accepted as an advanced parameter, but only 3 is supported. At grid 4
    # the pieces are bulky enough that the line-up cannot be laid out without one piece
    # hiding part of another: measured over 18 scenes, grid 3 had an invisible cube in
    # 0 of them and grid 4 in 4, and widening the spacing did not help (it traded
    # occlusion for cropping). Rather than emit scenes whose ground truth is sometimes
    # not recoverable, the unverified value is refused.
    _SUPPORTED_GRIDS = (3,)

    def generate(self, params: dict, seed: int) -> Scene:
        rng = _random.Random(seed)
        k = int(params.get("n_pieces", 5))
        n = int(params.get("grid", 3))
        if n not in self._SUPPORTED_GRIDS:
            raise ValueError(
                f"grid={n} is not supported: only {self._SUPPORTED_GRIDS} lays out with "
                f"every cube visible. At larger grids the pieces are bulky enough that "
                f"one hides part of another, so some scenes would be unanswerable."
            )
        renderer = params.get("renderer", "mitsuba_3d")
        unit_px = _UNIT_PX * (3.0 / n)          # bigger grids draw smaller cubes
        view = toward_camera(_AZIMUTH, _ELEVATION)

        # A decomposition can contain a piece with no fully-visible orientation - a
        # 2x2x2 block always hides a corner, whatever you do with it - and such a piece
        # cannot be read off the image, so the scene would be asking an unanswerable
        # question. Draw another decomposition instead of failing; large pieces (small
        # k relative to the grid) are where this bites.
        for _attempt in range(40):
            pieces = decompose(n, k, rng)
            oriented = [orient_for_visibility(p, view, rng) for p in pieces]
            if all(o is not None for o in oriented):
                break
        else:
            raise ValueError(
                f"every decomposition of a {n}x{n}x{n} cube into {k} pieces contained a "
                f"piece with no fully-visible orientation - the pieces are too bulky to "
                f"read from one view; use more pieces or a larger grid"
            )
        eligible = eligible_answers(pieces)
        answer_idx = eligible[seed % len(eligible)]

        # Answer balance: the correct piece takes COLORS[seed % 6], so across seeds every
        # color is the answer equally often and a model cannot profit from a color prior.
        answer_color = _COLORS[seed % len(_COLORS)]
        others = [c for c in _COLORS if c != answer_color]
        rng.shuffle(others)
        colors: list[str] = []
        oi = 0
        for i in range(k):
            if i == answer_idx:
                colors.append(answer_color)
            else:
                colors.append(others[oi])
                oi += 1

        # The target is the answer piece under a DIFFERENT rotation, so the task is
        # mental rotation rather than picture matching.
        target = self._rotated_differently(oriented[answer_idx], view, rng)

        # Lay every piece along the screen-horizontal axis, which is perpendicular to
        # the view direction. A simpler row along grid x recedes from the camera, so
        # pieces further down the line render smaller and are measurably harder to
        # read - a perceptual handicap applied to some candidates and not others, which
        # would show up in the results as if it were a property of the model.
        rx, rz = _screen_right(view)
        widths = [_span(o, rx, rz) for o in oriented]
        # Spacing and the target's set-back scale with the largest piece. Tuned for one
        # grid size they stop working at another: at grid 4 the pieces are bulkier and a
        # fixed set-back let the target hide part of the line-up.
        bulk = max(max(widths), _span(target, rx, rz))
        gap = _GAP * bulk
        forward = _TARGET_FORWARD * bulk
        total = sum(widths) + gap * (k - 1)
        objs: list[GraphObject] = []
        cursor = -(total / 2.0)
        for i, o in enumerate(oriented):
            objs += _place(o, cursor * rx, _base_y(o), cursor * rz, colors[i],
                           f"cand{i}", unit_px)
            cursor += widths[i] + gap
        # The target sits in its own row, pulled toward the camera and centered, so it
        # reads as the question rather than as a sixth candidate. Its depth differs from
        # the line-up on purpose: it is never compared against them by size.
        tw = _span(target, rx, rz)
        objs += _place(
            target,
            -(tw / 2.0) * rx + view[0] * forward,
            _base_y(target),
            -(tw / 2.0) * rz + view[2] * forward,
            _TARGET_COLOR, "target", unit_px)

        graph = SceneGraph(
            objects=objs, canvas=_CANVAS, background=(255, 255, 255),
            meta={
                # Pull the camera back far enough to frame however wide the line-up
                # came out. With a fixed distance, widening the spacing to stop pieces
                # overlapping simply pushed the outer ones out of frame instead, which
                # the certifier reports the same way - measured 2, 4 and 5 failures in
                # 36 scenes as the gap grew, all of them cropping rather than occlusion.
                "camera": {"azimuth": _AZIMUTH, "elevation": _ELEVATION,
                           "distance": _frame_distance(total, forward, unit_px)},
                # world height at which a cube's bottom face rests, given _UNIT_PX
                "ground_y": -(unit_px / 2.0) / 100.0,
                # A piece is one solid object made of cubes, so its cubes partly hide
                # each other by nature. What the answer needs is which cubes are
                # PRESENT, not a separate percept of each, so full occlusion still
                # fails and partial occlusion is expected rather than warned.
                "occlusion": {"policy": "presence"},
                **dict(params.get("renderer_opts") or {}),
            },
        )
        images = render_graph(graph, renderer)

        candidates = [{"color": colors[i], "cells": [list(c) for c in oriented[i]],
                       "n_cubes": len(oriented[i])} for i in range(k)]
        # The candidate colors are named so the answer set is explicit. Without them a
        # model can answer "gray", the target's own color, which is never a candidate:
        # that is a misread of the question rather than a wrong choice among the
        # candidates, and it would be scored as if it were the latter. Naming them also
        # makes the answer set closed and known, which is what lets the balanced answer
        # colors stand in for a text-only baseline.
        listed = ", ".join(colors[:k])
        question = (
            "The gray piece standing on its own is the TARGET. The colored pieces are "
            f"the candidates: {listed}. Exactly one candidate is the same shape as the "
            "target, allowing for rotation (no mirroring). Which colored candidate is "
            "it?"
        )
        return Scene(
            id=f"polycube_n{n}_k{k}_s{seed}",
            images=images,
            ground_truth={
                "question": question,
                "answer": answer_color,
                "target_cells": [list(c) for c in target],
                "candidates": candidates,
                "n_pieces": k,
            },
            factors={"n_pieces": k, "grid": n},
            meta={"generator": "polycube", "seed": seed, "params": dict(params),
                  "tier": "trivial" if k <= 3 else "standard"},
            graph=graph,
        )

    @staticmethod
    def _rotated_differently(cells, view, rng):
        """A visible orientation of the same piece that is not the one on display.

        Matching two identical pictures is not mental rotation, so a target that came
        out in the candidate's own orientation is re-rolled. A piece with enough
        symmetry may have only one visible orientation; that is fine and simply makes
        an easier item, so the original is used rather than failing the scene.
        """
        options = []
        for r in ROTATIONS:
            cand = normalize(_apply(r, c) for c in cells)
            if cand != tuple(cells) and not _hidden(cand, view):
                options.append(cand)
        return rng.choice(options) if options else tuple(cells)


def _hidden(cells, view):
    from renderprobe.scenes._polycube_core import hidden_cells
    return hidden_cells(cells, view)


# Re-exported so the probe can verify a candidate without importing the core directly.
__all__ = ["PLUGIN", "canonical", "congruent"]

PLUGIN = _PolycubeScene()
