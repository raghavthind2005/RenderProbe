"""Flat 2D renderer: draws each object's silhouette with PIL, ignoring depth.

Reproduces the original 2D look from a scene graph with no extra dependencies. This is the
default renderer: fast, legible, exact ground truth, no realism.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from renderprobe.core.schema import GraphObject, SceneGraph
from renderprobe.renderers._palette import rgb


def _ordered(graph: SceneGraph) -> list[tuple[int, GraphObject]]:
    # (original index, object) sorted far-to-near so nearer objects paint over farther ones.
    # render() and render_ids() use the same order to keep their occlusion in sync.
    return sorted(enumerate(graph.objects), key=lambda t: -t[1].position[2])


def _silhouette(draw: ImageDraw.ImageDraw, obj: GraphObject, fill, outline=None, width=2) -> None:
    # An RGB fill for the color image, or fill=1 on a mask for the id map. Same geometry.
    x, y, _ = obj.position
    r = obj.size
    if obj.shape == "cube":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=width)
    elif obj.shape == "cone":
        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=fill, outline=outline)
    else:  # sphere / cylinder: circle silhouette
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=width)


class _Pil2DRenderer:
    name = "pil_2d"
    is_3d = False

    def render(self, graph: SceneGraph) -> list[Image.Image]:
        img = Image.new("RGB", graph.canvas, graph.background)
        draw = ImageDraw.Draw(img)
        for _idx, obj in _ordered(graph):
            _silhouette(draw, obj, fill=rgb(obj.color), outline=(0, 0, 0), width=2)
        return [img]

    def render_ids(self, graph: SceneGraph) -> np.ndarray:
        # Front-most object index per pixel (-1 = background), using the same order and
        # silhouettes as render() so it reflects the occlusion the color image shows.
        w, h = graph.canvas
        ids = np.full((h, w), -1, dtype=np.int32)
        for idx, obj in _ordered(graph):  # far to near; nearer overwrites
            mask = Image.new("1", (w, h), 0)
            _silhouette(ImageDraw.Draw(mask), obj, fill=1, outline=1)
            ids[np.asarray(mask, dtype=bool)] = idx
        return ids


PLUGIN = _Pil2DRenderer()
