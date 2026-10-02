"""COPY-ME TEMPLATE - a custom renderer.

A renderer turns a renderer-independent ``SceneGraph`` into pixels. The ground truth
lives with the graph, NOT here, so adding a renderer only changes how a scene LOOKS -
never what it means. That is the whole point: realism is a pluggable knob.

HOW TO USE
    1. Copy out, drop the underscore, rename the class and ``name``.
    2. Fill in ``render`` - read ``graph.objects`` (each has shape/color/position(x,y,z)/
       size) and draw to a PIL image of ``graph.canvas`` size.
    3. renderprobe validate my_renderers/my_renderer.py   (checks conformance)
    4. A scene requests it by name (e.g. params['renderer'] = 'my_renderer').

Keep the honesty invariant: if your renderer introduces occlusion (true 3D), make sure
the scene's task is still answerable from the image (e.g. the target stays visible), or
the ground truth stops being trustable.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from renderprobe.core.schema import SceneGraph
from renderprobe.renderers._palette import rgb


class _TemplateRenderer:
    name = "template_renderer"
    is_3d = False                       # True if you use depth (perspective/occlusion)

    def render(self, graph: SceneGraph) -> list[Image.Image]:
        img = Image.new("RGB", graph.canvas, graph.background)
        draw = ImageDraw.Draw(img)
        for obj in sorted(graph.objects, key=lambda o: -o.position[2]):  # far first
            x, y, _z = obj.position
            r = obj.size
            draw.ellipse([x - r, y - r, x + r, y + r], fill=rgb(obj.color), outline=(0, 0, 0))
        return [img]

    # -- OPTIONAL, but STRONGLY recommended if your renderer can occlude (any true 3D) --
    #
    # render_ids returns an (H, W) int array whose value is the INDEX (into graph.objects)
    # of the FRONT-MOST object at each pixel, -1 for background. It MUST use the SAME
    # projection and draw ORDER as render() so what it reports visible is exactly what
    # render() shows. `renderprobe validate` uses it to certify that every ground-truth
    # object stays visible (not hidden) under your renderer - the guarantee that keeps a
    # count/enumeration answer recoverable once you add occlusion. Omit it ONLY for a
    # renderer that can never occlude (validate then just warns it can't certify).
    def render_ids(self, graph: SceneGraph) -> np.ndarray:
        w, h = graph.canvas
        ids = np.full((h, w), -1, dtype=np.int32)
        # SAME order as render() (far -> near) so the near object overwrites -> front-most wins
        for idx, obj in sorted(enumerate(graph.objects), key=lambda t: -t[1].position[2]):
            x, y, _z = obj.position
            r = obj.size
            mask = Image.new("1", (w, h), 0)          # 1-bit silhouette of this one object
            ImageDraw.Draw(mask).ellipse([x - r, y - r, x + r, y + r], fill=1)
            ids[np.asarray(mask, dtype=bool)] = idx
        return ids


PLUGIN = _TemplateRenderer()
