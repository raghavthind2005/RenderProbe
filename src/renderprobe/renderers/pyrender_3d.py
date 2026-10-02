"""Optional photorealistic 3D renderer (pyrender + trimesh) - true perspective and
occlusion via offscreen OpenGL.

HEAVY and environment-dependent (needs an OpenGL / EGL / OSMesa backend), so it is NOT
installed by default and NOT exercised in the test suite:

    pip install 'renderprobe[render3d]'      # then a working GL backend

The import is lazy - this module always loads and registers, so the tooling can list it
and a scene can request it, but the dependency is only needed when you actually
``render()``. For headless depth-shaded output with zero extra deps use ``pil_25d``.

The ground truth is IDENTICAL to the 2D/2.5D renderers: it is derived from the
scene graph, never from the pixels here. This tier only changes realism.
"""
from __future__ import annotations

from PIL import Image

from renderprobe.core.schema import SceneGraph
from renderprobe.renderers._palette import rgb

# Screen pixels -> world units (origin at image center, y up, depth into -z).
_UNIT = 100.0


class _Pyrender3DRenderer:
    name = "pyrender_3d"
    is_3d = True

    def render(self, graph: SceneGraph) -> list[Image.Image]:
        try:
            import numpy as np
            import pyrender
            import trimesh
        except Exception as exc:  # dependency or GL backend missing
            raise RuntimeError(
                "pyrender_3d needs the optional 3D stack - `pip install "
                "'renderprobe[render3d]'` plus a working OpenGL/EGL/OSMesa backend. For "
                "headless depth-shaded rendering with no extra deps, use pil_25d. "
                f"(import failed: {exc})"
            ) from exc

        w, h = graph.canvas
        bg = [c / 255 for c in graph.background]
        scene = pyrender.Scene(bg_color=[*bg, 1.0], ambient_light=[0.3, 0.3, 0.3])

        for obj in graph.objects:
            px, py, pz = obj.position
            wx, wy, wz = (px - w / 2) / _UNIT, (h / 2 - py) / _UNIT, -pz / _UNIT
            s = obj.size / _UNIT
            if obj.shape == "cube":
                mesh = trimesh.creation.box(extents=(2 * s, 2 * s, 2 * s))
            elif obj.shape == "cylinder":
                mesh = trimesh.creation.cylinder(radius=s, height=2 * s)
            elif obj.shape == "cone":
                mesh = trimesh.creation.cone(radius=s, height=2 * s)
            else:
                mesh = trimesh.creation.icosphere(radius=s)
            color = rgb(obj.color)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*[c / 255 for c in color], 1.0],
                metallicFactor=0.1, roughnessFactor=0.6,
            )
            pose = np.eye(4)
            pose[:3, 3] = [wx, wy, wz]
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True), pose=pose)

        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=w / h)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, 0.0, 6.0]
        scene.add(cam, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [-2.0, 3.0, 4.0]
        scene.add(light, pose=light_pose)

        renderer = pyrender.OffscreenRenderer(w, h)
        try:
            color, _ = renderer.render(scene)
        finally:
            renderer.delete()
        return [Image.fromarray(color[..., :3])]


PLUGIN = _Pyrender3DRenderer()
