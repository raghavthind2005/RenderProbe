"""Optional true-3D CPU photorealism via Mitsuba 3: perspective, soft shadows, global
illumination, and real occlusion, with no OpenGL/GPU/Metal.

Mitsuba renders on the CPU (Dr.Jit's LLVM backend), so unlike pyrender_3d it needs no GL
context and runs on headless machines including Apple Silicon. It is an optional heavy
dependency and is not installed by default; the import is lazy, so this module always
registers but Mitsuba is only imported when you render:

    pip install 'renderprobe[render3d-cpu]'

Ground truth is identical to the 2D/2.5D renderers (it comes from the scene graph, not
these pixels); only the realism changes. Since true perspective can make one object hide
another, render_ids (below) lets `renderprobe validate` check that a graph scene keeps
every object visible under this renderer.

Per-render options come from graph.meta:

    spp        samples per pixel (default 64). For a scene of 18 boxes at 820x430:
               ~1 s at 24, ~3 s at 64, ~13 s at 256. 256 is visually clean, so
               iterate at the default and raise it for a final render.
    rfilter    reconstruction filter (default "gaussian"); "box" trades antialiasing
               for bit exact pixels, which only helps if max_depth is also 1.
    camera     "front" (default, dead-on) | "three_quarter" | "elevated", or explicit
               {"azimuth": deg, "elevation": deg, "distance": units}.
    ground_y   world height of the floor. Defaults to just below the canvas, so a
               scene laid out in screen coordinates is never clipped by it.

The default camera and floor are chosen so the built-in screen-space scenes project
exactly as they did before this renderer grew a look; a scene genuinely arranged in
3D opts into the rest.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from renderprobe.core.schema import GraphObject, SceneGraph
from renderprobe.renderers._palette import rgb

# Screen pixels -> world units (origin at image center, y up, depth into -z), matching
# pyrender_3d so the two 3D tiers frame a scene the same way.
_UNIT = 100.0
_CAM_Z = 6.0
_FOV = 42.0
_BACKDROP_Z = -9.0          # far enough back to stay out of frame; still occludes
_DEFAULT_SPP = 64           # ~3 s per 820x430 frame; raise it in a config for a
                            # final run (256 is visually clean at ~13 s)
# Reconstruction filter. A wider filter splats each sample across neighboring pixels
# with atomic adds whose ordering varies with thread scheduling, so it does cost bit
# exact reproducibility: measured over 6 trials, gaussian is 6/6 nondeterministic,
# tent 3/6, box 0/6, and gaussian pinned to one thread 0/6, which identifies the
# cause. That was the reason to default to "box", but it no longer buys anything:
# the color render is nondeterministic regardless, because max_depth > 1 puts the
# path tracer's recorded loop in the same position (see `render`). Since the
# property is already gone, the default takes the antialiasing instead; box remains
# available through meta["rfilter"], and `render_ids` pins itself to box because its
# whole contract is one untouched sample per pixel.
_DEFAULT_RFILTER = "gaussian"
_GROUND_MARGIN = 0.02       # floor clearance below the canvas bottom (world units)
_BEVEL = 0.10               # edge radius as a fraction of a box's half-size
_ROUGHNESS = 0.18           # ggx alpha: low enough to catch a highlight, high enough
                            # that the highlight reads as a soft sheen, not a mirror

_variant_ready = False


def _mi():
    """Lazily import Mitsuba and select a CPU variant once. Raises a clear, actionable
    RuntimeError if the optional dependency is absent - never a bare ImportError."""
    try:
        import mitsuba as mi
    except Exception as exc:  # not installed
        raise RuntimeError(
            "mitsuba_3d needs the optional CPU 3D renderer - install it with "
            "`pip install 'renderprobe[render3d-cpu]'`. It is a headless CPU path tracer "
            "(no OpenGL/GPU/Metal). For zero-dependency depth-shaded output use pil_25d. "
            f"(import failed: {exc})"
        ) from exc
    global _variant_ready
    if not _variant_ready:
        variants = mi.variants()
        # llvm = JIT-compiled, multi-threaded CPU (fast); scalar = portable fallback.
        chosen = next((v for v in ("llvm_ad_rgb", "scalar_rgb") if v in variants), variants[-1])
        mi.set_variant(chosen)
        _variant_ready = True
    return mi


def _world(px: float, py: float, pz: float, canvas: tuple[int, int]) -> list[float]:
    w, h = canvas
    return [(px - w / 2) / _UNIT, (h / 2 - py) / _UNIT, -pz / _UNIT]


def _box_shapes(mi, cx: float, cy: float, cz: float, s: float, material: dict) -> dict:
    """A box with ROUNDED EDGES, from analytic primitives only.

    Mitsuba has no box primitive. Six flat rectangles give a box whose edges are
    perfectly sharp, and a perfectly sharp edge is the single strongest cue that a
    render is untouched programmer output: real objects catch a highlight along every
    edge. Rounding costs nothing perceptually expensive here - the box becomes 6 inset
    faces, 12 edge cylinders and 8 corner spheres, all native primitives that the path
    tracer intersects analytically.

    ``_BEVEL`` is the edge radius as a fraction of the half-size; ``f`` is what is left
    of the half-extent once the rounding is taken off each side.
    """
    T = mi.ScalarTransform4f
    r = max(s * _BEVEL, 1e-4)
    f = max(s - r, 1e-4)
    out: dict = {}
    # 6 faces, inset to f so they meet the rounded edges tangentially
    for nm, tw in (
        ("pz", T().translate([cx, cy, cz + s]).scale(f)),
        ("nz", T().translate([cx, cy, cz - s]).rotate([1, 0, 0], 180).scale(f)),
        ("px", T().translate([cx + s, cy, cz]).rotate([0, 1, 0], 90).scale(f)),
        ("nx", T().translate([cx - s, cy, cz]).rotate([0, 1, 0], -90).scale(f)),
        ("py", T().translate([cx, cy + s, cz]).rotate([1, 0, 0], -90).scale(f)),
        ("ny", T().translate([cx, cy - s, cz]).rotate([1, 0, 0], 90).scale(f)),
    ):
        out[f"f_{nm}"] = {"type": "rectangle", "to_world": tw, **material}
    # 12 edge cylinders, axis-aligned, centerd on the rounding arc's center
    for i, (a, b) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        out[f"ex{i}"] = {"type": "cylinder", "radius": r,
                         "p0": [cx - f, cy + a * f, cz + b * f],
                         "p1": [cx + f, cy + a * f, cz + b * f], **material}
        out[f"ey{i}"] = {"type": "cylinder", "radius": r,
                         "p0": [cx + a * f, cy - f, cz + b * f],
                         "p1": [cx + a * f, cy + f, cz + b * f], **material}
        out[f"ez{i}"] = {"type": "cylinder", "radius": r,
                         "p0": [cx + a * f, cy + b * f, cz - f],
                         "p1": [cx + a * f, cy + b * f, cz + f], **material}
    # 8 corner spheres
    for i, (a, b, c) in enumerate(
        (x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
    ):
        out[f"c{i}"] = {"type": "sphere", "radius": r,
                        "to_world": T().translate([cx + a * f, cy + b * f, cz + c * f]),
                        **material}
    return out


def _object_shapes(mi, obj: GraphObject, canvas: tuple[int, int], material: dict) -> dict:
    """One graph object -> {name: shape_dict}, each carrying ``material`` (a bsdf for the
    color pass, or an area emitter for the id pass). A cube is 6 rectangles that all share
    the same material, so it reads as a single object either way."""
    T = mi.ScalarTransform4f
    cx, cy, cz = _world(*obj.position, canvas)
    s = obj.size / _UNIT
    out: dict = {}
    if obj.shape == "cube":
        out.update(_box_shapes(mi, cx, cy, cz, s, material))
    elif obj.shape in ("cylinder", "cone"):   # cone approximated by an upright cylinder
        out["cyl"] = {"type": "cylinder", "radius": s,
                      "p0": [cx, cy - s, cz], "p1": [cx, cy + s, cz], **material}
    else:  # sphere (default)
        out["sph"] = {"type": "sphere", "radius": s, "to_world": T().translate([cx, cy, cz]),
                      **material}
    return out


def _camera_azimuth(meta: dict) -> float:
    """The camera's horizontal orbit in radians (0 = dead-on)."""
    cam = (meta or {}).get("camera")
    if cam is None or cam == "front":
        return 0.0
    presets = {"three_quarter": 34.0, "elevated": 0.0}
    if isinstance(cam, str):
        return math.radians(presets.get(cam, 0.0))
    return math.radians(float(cam.get("azimuth", 0.0)))


def _camera_distance(meta: dict) -> float:
    """How far the camera sits from the target, in world units."""
    cam = (meta or {}).get("camera")
    if cam is None or cam == "front":
        return _CAM_Z
    presets = {"three_quarter": _CAM_Z * 0.86, "elevated": _CAM_Z * 1.05}
    if isinstance(cam, str):
        return presets.get(cam, _CAM_Z)
    return float(cam.get("distance", _CAM_Z))


def _rig(p: list[float], az: float, dist: float) -> list[float]:
    """Place a rig light: rotate it with the camera, and push it out as the camera
    pulls back.

    Both halves matter. Without the rotation an emitter ends up in front of the lens
    at some angles and its unlit back face blacks out the frame. Without the distance
    scaling a wide shot catches a light rectangle in the corner, which is the visible
    white panel a fixed rig produces the moment someone frames wider."""
    x, y, z = p
    k = dist / _CAM_Z
    x, y, z = x * k, y * k, z * k
    ca, sa = math.cos(az), math.sin(az)
    return [x * ca + z * sa, y, -x * sa + z * ca]


def _area_light(mi, pos, target, size, radiance) -> dict:
    """A rectangular emitter aimed at the scene.

    Area lights rather than point lights, because a point light casts a hard-edged
    shadow with no penumbra, which is the other half of why an untouched render reads
    as synthetic. Emitter area sets penumbra width, so the key is small and crisp and
    the fill is broad and soft.
    """
    T = mi.ScalarTransform4f
    return {
        "type": "rectangle",
        "to_world": T().look_at(origin=pos, target=target, up=[0, 1, 0]).scale(size),
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance}},
        # the emitter geometry itself should not appear in frame
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0, 0, 0]}},
    }


def _material(col: list[float]) -> dict:
    """Rough plastic: a colored diffuse base under a narrow specular coat.

    Flat diffuse has no highlight at all, so every surface reads as matte paper and the
    form has to be carried entirely by shading gradient. The coat gives each face a
    sheen that tracks the light, which is what makes the geometry legible.
    """
    return {"bsdf": {
        "type": "roughplastic",
        "distribution": "ggx",
        "alpha": _ROUGHNESS,
        "int_ior": 1.5,
        "diffuse_reflectance": {"type": "rgb", "value": col},
    }}


def _camera_origin(meta: dict) -> tuple[list[float], list[float]]:
    """Where the camera sits, from ``graph.meta["camera"]``. Returns (origin, target).

    The default is dead-on, and stays the default deliberately: the built-in scenes lay
    objects out in SCREEN coordinates and rely on that projection to keep every object
    unoccluded, so moving the camera for them would silently change what their ground
    truth means. A scene that is genuinely arranged in 3D opts in instead:

        meta["camera"] = "three_quarter"                       # a preset, or
        meta["camera"] = {"azimuth": 35, "elevation": 22, "distance": 6.5}

    Azimuth orbits horizontally, elevation lifts, both in degrees about the origin.
    """
    cam = (meta or {}).get("camera")
    if cam is None or cam == "front":
        return [0.0, 0.0, _CAM_Z], [0.0, 0.0, 0.0]
    presets = {
        "three_quarter": {"azimuth": 32.0, "elevation": 17.0, "distance": _CAM_Z * 0.86},
        "elevated": {"azimuth": 0.0, "elevation": 26.0, "distance": _CAM_Z * 1.05},
    }
    spec = presets.get(cam, {}) if isinstance(cam, str) else dict(cam)
    az = math.radians(float(spec.get("azimuth", 0.0)))
    el = math.radians(float(spec.get("elevation", 0.0)))
    d = float(spec.get("distance", _CAM_Z))
    tgt = [float(v) for v in spec.get("target", (0.0, 0.0, 0.0))]
    return [
        tgt[0] + d * math.cos(el) * math.sin(az),
        tgt[1] + d * math.sin(el),
        tgt[2] + d * math.cos(el) * math.cos(az),
    ], tgt


def _sensor(mi, canvas: tuple[int, int], spp: int, rfilter: str,
            meta: dict | None = None) -> dict:
    T = mi.ScalarTransform4f
    w, h = canvas
    origin, target = _camera_origin(meta or {})
    return {
        "type": "perspective", "fov": _FOV,
        "to_world": T().look_at(origin=origin, target=target, up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": w, "height": h, "rfilter": {"type": rfilter}},
        "sampler": {"type": "independent", "sample_count": spp},
    }


class _Mitsuba3DRenderer:
    name = "mitsuba_3d"
    is_3d = True
    # Pixels are not bit-identical between runs, and cannot be made so without giving
    # up indirect light: measured over 5 trials, max_depth=1 is reproducible 5/5 while
    # max_depth 2, 4 and 8 are not. The cause is the path tracer's recorded loop, whose
    # lane compaction reorders how the sampler is consumed. What remains exactly
    # reproducible is everything that carries meaning - the scene graph, the geometry
    # and the ground truth - so what varies is Monte Carlo noise, which is variance and
    # not bias. Within a run it does not even vary: the runner renders each scene once
    # and reuses that image across every model and condition, so a paired comparison
    # always sees identical pixels. `validate` reads this flag and holds a stochastic
    # renderer to the guarantee it can actually keep.
    stochastic = True

    def render(self, graph: SceneGraph) -> list[Image.Image]:
        mi = _mi()
        spp = int(graph.meta.get("spp", _DEFAULT_SPP))
        az = _camera_azimuth(graph.meta)
        dist = _camera_distance(graph.meta)
        scene: dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 8},
            "sensor": _sensor(mi, graph.canvas, spp,
                              str(graph.meta.get("rfilter", _DEFAULT_RFILTER)),
                              graph.meta),
            # Low constant fill, then a three-point rig of area emitters: a warm key
            # from upper left, a cool broad fill opposite to keep shadows from going
            # dead, and a rim from behind to separate objects from the backdrop.
            "ambient": {"type": "constant", "radiance": 0.17},
            # The rig is defined relative to the camera and orbits with it, so the key
            # stays upper-left of the lens at any angle and no emitter can drift into
            # frame and block the view with its unlit back face.
            # Key well to the SIDE rather than over the lens: a frontal key throws its
            # shadow directly behind the object where the camera cannot see it, which
            # is why a head-on rig looks flat however bright it is. Raking from the
            # side puts the shadow across open floor, and the shadow is what seats the
            # object in the scene.
            "key": _area_light(mi, _rig([-4.6, 3.4, 0.8], az, dist), [0, 0, 0], 2.6,
                               [9.0, 8.5, 7.8]),
            "fill": _area_light(mi, _rig([4.8, 1.4, 2.6], az, dist), [0, 0, 0], 2.6,
                                [1.5, 1.6, 1.9]),
            "rim": _area_light(mi, _rig([1.6, 3.2, -3.6], az, dist), [0, 0, 0], 1.8,
                               [2.2, 2.2, 2.6]),
            "backdrop": self._backdrop(mi, lit=True),
            "ground": self._ground(mi, True, graph.canvas, graph.meta),
        }
        for i, obj in enumerate(graph.objects):
            col = [c / 255 for c in rgb(obj.color)]
            scene.update(_prefixed(_object_shapes(mi, obj, graph.canvas, _material(col)), i))
        img = mi.render(mi.load_dict(scene), seed=0)
        arr = np.array(mi.util.convert_to_bitmap(img))
        return [Image.fromarray(np.ascontiguousarray(arr[..., :3]), "RGB")]

    def render_ids(self, graph: SceneGraph) -> np.ndarray:
        """Object-index map (front-most object per pixel, -1 = background), rendered with
        the SAME camera and geometry as ``render`` so its occlusion is identical. Each
        object is an area emitter with radiance (index+1) in the red channel; one primary
        ray per pixel (spp=1, box filter, max_depth=1) => each pixel is exactly the nearest
        object's index. The backdrop is kept as a radiance-0 emitter so it occludes exactly
        as it does in the color render but decodes to background."""
        mi = _mi()
        scene: dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 1},
            "sensor": _sensor(mi, graph.canvas, 1, "box", graph.meta),
            "backdrop": self._backdrop(mi, lit=False),
            "ground": self._ground(mi, False, graph.canvas, graph.meta),
        }
        for i, obj in enumerate(graph.objects):
            emitter = {"emitter": {"type": "area",
                                   "radiance": {"type": "rgb", "value": [float(i + 1), 0.0, 0.0]}}}
            scene.update(_prefixed(_object_shapes(mi, obj, graph.canvas, emitter), i))
        raw = np.array(mi.render(mi.load_dict(scene), seed=0))
        red = raw[..., 0]
        return np.where(red > 0.5, np.rint(red).astype(np.int32) - 1, -1).astype(np.int32)

    def _backdrop(self, mi, lit: bool) -> dict:
        T = mi.ScalarTransform4f
        # large enough that its edges fall outside the frustum at this distance;
        # a visible backdrop edge reads as a cardboard set, not a studio sweep
        d = {"type": "rectangle", "to_world": T().translate([0, 0, _BACKDROP_Z]).scale(22.0)}
        if lit:
            gray = {"type": "rgb", "value": [0.72, 0.72, 0.76]}
            d["bsdf"] = {"type": "diffuse", "reflectance": gray}
        else:  # id pass: emits nothing (=> background) but still occludes anything behind it
            d["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": [0.0, 0.0, 0.0]}}
        return d

    def _ground(self, mi, lit: bool, canvas: tuple[int, int], meta: dict) -> dict:
        """A matte floor under the objects, for the contact shadow that stops them
        floating in undefined space.

        Its default height is just below the bottom of the canvas, which matters: the
        built-in scenes lay objects out in SCREEN coordinates, so a floor placed at a
        fixed world height would swallow whatever they put low in the frame and destroy
        their ground truth. A scene arranged in real 3D sets its own::

            meta["ground_y"] = -1.05     # world units, where objects should rest

        Present in the id pass too (as a radiance-0 emitter) so both passes occlude
        identically."""
        T = mi.ScalarTransform4f
        default_y = -(canvas[1] / 2.0) / _UNIT - _GROUND_MARGIN
        gy = float((meta or {}).get("ground_y", default_y))
        d = {"type": "rectangle",
             "to_world": T().translate([0, gy, 0])
                             .rotate([1, 0, 0], -90).scale(14.0)}
        if lit:
            d["bsdf"] = {"type": "diffuse",
                         "reflectance": {"type": "rgb", "value": [0.66, 0.65, 0.63]}}
        else:
            d["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": [0.0, 0.0, 0.0]}}
        return d


def _prefixed(shapes: dict, i: int) -> dict:
    """Namespace an object's shape keys so multiple objects (and cube faces) never collide."""
    return {f"o{i}_{k}": v for k, v in shapes.items()}


PLUGIN = _Mitsuba3DRenderer()
