"""Depth-shaded 2.5D renderer - pure PIL + numpy, no GPU, works headless everywhere.

Adds real 3D CUES to the SAME scene graph (so the ground truth is unchanged): objects
nearer the camera (smaller z) render larger and brighter; spheres get a Lambert-shaded
highlight; each casts a soft ground shadow; everything is drawn far-to-near (painter's
algorithm) over a subtle floor gradient. This is the "efficient realism" tier - a
genuine step up from flat 2D at negligible cost. For true perspective + occlusion, use
the optional photorealistic ``pyrender_3d``.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from renderprobe.core.schema import GraphObject, SceneGraph
from renderprobe.renderers._palette import rgb

# Directional key light (points toward the surface), normalized.
_LIGHT = np.array([-0.5, -0.6, 0.62])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _projected(graph: SceneGraph) -> list[tuple[int, GraphObject, float, float, int, float]]:
    """(original index, object, screen x, screen y, depth-scaled radius, brightness) in
    draw order (far -> near). render() and render_ids() BOTH consume this, so the silhouette
    an object occupies - and therefore the occlusion - is identical between them."""
    zs = [o.position[2] for o in graph.objects] or [0.0]
    zmin, zspan = min(zs), (max(zs) - min(zs)) or 1.0
    out = []
    for idx, obj in sorted(enumerate(graph.objects), key=lambda t: -t[1].position[2]):
        x, y, z = obj.position
        znorm = (z - zmin) / zspan
        scale = 1.0 - 0.35 * znorm            # nearer = larger
        bright = 1.0 - 0.35 * znorm           # nearer = brighter
        r = max(2, int(obj.size * scale))
        out.append((idx, obj, x, y, r, bright))
    return out


def _sphere_sprite(r: int, base: tuple[int, int, int], brightness: float) -> Image.Image:
    """A Lambert-shaded sphere as an RGBA sprite of diameter 2r."""
    r = max(2, int(r))
    d = 2 * r
    yy, xx = np.mgrid[0:d, 0:d]
    nx = (xx - r + 0.5) / r
    ny = (yy - r + 0.5) / r
    disc = nx * nx + ny * ny
    inside = disc <= 1.0
    nz = np.sqrt(np.clip(1.0 - disc, 0.0, 1.0))
    lambert = np.clip(nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2], 0.0, 1.0)
    inten = (0.30 + 0.70 * lambert) * brightness      # ambient + diffuse, dimmed by depth
    out = np.zeros((d, d, 4), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.clip(base[c] * inten, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _floor(canvas: tuple[int, int]) -> Image.Image:
    """A subtle top-to-bottom gradient so objects read as sitting on a surface."""
    w, h = canvas
    top, bot = np.array([250, 250, 252]), np.array([226, 228, 234])
    ramp = np.linspace(0, 1, h)[:, None]
    col = (top[None, :] * (1 - ramp) + bot[None, :] * ramp).astype(np.uint8)
    return Image.fromarray(np.repeat(col[:, None, :], w, axis=1), "RGB")


def _mask_shape(draw: ImageDraw.ImageDraw, shape: str, x: float, y: float, r: int) -> None:
    """Fill obj's silhouette on a 1-bit mask - the SAME footprint render() paints (the
    sphere sprite's ``inside`` disc is a filled circle of radius r; cube/cone match too)."""
    if shape == "cone":
        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=1)
    elif shape == "cube":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=1)
    else:  # sphere / cylinder
        draw.ellipse([x - r, y - r, x + r, y + r], fill=1)


class _Pil25DRenderer:
    name = "pil_25d"
    is_3d = True

    def render(self, graph: SceneGraph) -> list[Image.Image]:
        img = _floor(graph.canvas)
        draw = ImageDraw.Draw(img)

        for _idx, obj, x, y, r, bright in _projected(graph):   # far -> near
            base = rgb(obj.color)

            # soft ground shadow just below the object
            sx, sy = 1.7 * r, 0.45 * r
            draw.ellipse([x - sx, y + r * 0.55 - sy / 2, x + sx, y + r * 0.55 + sy / 2],
                         fill=(206, 208, 214))

            if obj.shape in ("sphere", "cylinder"):
                sprite = _sphere_sprite(r, base, bright)
                img.paste(sprite, (int(x - r), int(y - r)), sprite)
            elif obj.shape == "cone":
                col = tuple(int(c * bright) for c in base)
                draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=col)
            else:  # cube
                col = tuple(int(c * bright) for c in base)
                hi = tuple(min(255, int(c * 1.15)) for c in col)
                draw.rectangle([x - r, y - r, x + r, y + r], fill=col)
                draw.rectangle([x - r, y - r, x + r, y - r + max(2, r // 3)], fill=hi)

        return [img.convert("RGB")]

    def render_ids(self, graph: SceneGraph) -> np.ndarray:
        """(H, W) front-most object index per pixel (-1 = background), built from the SAME
        depth-scaled silhouettes render() paints (shadows are NOT objects, so they are
        excluded). See the Renderer Protocol docstring."""
        w, h = graph.canvas
        ids = np.full((h, w), -1, dtype=np.int32)
        for idx, obj, x, y, r, _bright in _projected(graph):   # far -> near; near wins
            mask = Image.new("1", (w, h), 0)
            _mask_shape(ImageDraw.Draw(mask), obj.shape, x, y, r)
            ids[np.asarray(mask, dtype=bool)] = idx
        return ids


PLUGIN = _Pil25DRenderer()
