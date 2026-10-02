"""Graph-based demonstrator scene: colored spheres on a surface.

Unlike the other built-in scenes (which draw with PIL directly), this one builds a
renderer-INDEPENDENT ``SceneGraph``, derives its ground truth from the graph, then hands
the graph to whichever renderer the run selects:

    params['renderer'] = 'pil_2d'  (flat, default) | 'pil_25d' (depth-shaded) |
                         'mitsuba_3d' (true-3D CPU photorealism) | 'pyrender_3d' (GPU)
    params['renderer_opts'] = {'spp': 8}   # optional renderer hints (e.g. mitsuba samples)

Same graph -> same exact ground truth (the count), different realism. That is the point:
realism is a knob orthogonal to the task. Objects are placed non-overlapping at their base
size, and the 2D/2.5D renderers only shrink with depth, so the count stays trustable (every
object visible). NOTE: true-3D ``pyrender_3d`` can introduce occlusion - then a task must be
designed to stay answerable (the scientist's call).
"""
from __future__ import annotations

import math
import random as _random

from renderprobe.core.schema import GraphObject, Scene, SceneGraph
from renderprobe.renderers import render_graph

_CANVAS = (480, 360)
_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "pink", "brown"]
_RMIN, _RMAX = 18, 28
_MARGIN = 34
_MIN_DIST = 2 * _RMAX + 12   # non-overlapping at base size => count trustable in 2D/2.5D


class _BlocksScene:
    name = "blocks"
    params_schema = {
        "n_objects": {"type": "int", "min": 2, "max": 12, "default": 6},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        rng = _random.Random(seed)
        n = int(params["n_objects"])
        renderer = params.get("renderer", "pil_2d")   # extra param; not a UI slider
        w, h = _CANVAS

        placed: list[tuple[int, int]] = []
        ir: list[GraphObject] = []
        gt_objects: list[dict] = []
        dots: list[dict] = []
        for i in range(n):
            for _try in range(400):
                x = rng.randint(_MARGIN, w - _MARGIN)
                y = rng.randint(_MARGIN, h - _MARGIN)
                if all(math.hypot(x - px, y - py) >= _MIN_DIST for px, py in placed):
                    break
            else:
                continue   # couldn't place this one without overlap; skip (count stays exact)
            placed.append((x, y))
            z = rng.uniform(0.0, 3.0)
            color = rng.choice(_COLORS)
            size = rng.randint(_RMIN, _RMAX)
            ir.append(GraphObject(id=f"o{i}", shape="sphere", color=color,
                                  position=(float(x), float(y), z), size=float(size)))
            gt_objects.append({"color": color, "shape": "sphere", "x": x, "y": y,
                               "z": round(z, 3), "r": size, "size": size})
            dots.append({"x": x, "y": y, "color": color})

        # Renderer options (e.g. {"spp": 8} for mitsuba_3d) ride on graph.meta - the
        # renderer-specific hints bag - so a config can dial an expensive renderer's
        # cost/quality without touching the task. Cheap renderers simply ignore them.
        graph = SceneGraph(objects=ir, canvas=_CANVAS, background=(255, 255, 255),
                           meta=dict(params.get("renderer_opts") or {}))
        images = render_graph(graph, renderer)
        count = len(ir)
        return Scene(
            id=f"blocks_n{n}_{renderer}_s{seed}",
            images=images,
            # count is the task GT; objects/dots are additive perceptual detail (dots lets
            # the count probe's perception report point at one object).
            ground_truth={"count": count, "objects": gt_objects, "dots": dots,
                          "dot_radius": gt_objects[0]["size"] if gt_objects else 12},
            factors={"n_objects": n, "renderer": renderer},
            meta={"generator": "blocks", "seed": seed, "params": dict(params),
                  "tier": "trivial" if n <= 2 else "standard"},
            # Expose the renderer-independent graph so `renderprobe validate` can certify
            # every object stays visible (not occluded) under whichever renderer is chosen.
            graph=graph,
        )


PLUGIN = _BlocksScene()
