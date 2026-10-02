"""mitsuba_3d - the optional true-3D CPU photorealism renderer (headless, no OpenGL/GPU).

These run only where Mitsuba is installed (`pip install 'renderprobe[render3d-cpu]'`);
elsewhere they skip. They prove the renderer draws from a scene graph, that its optional
`render_ids` occlusion pass is faithful to the color render (so the occlusion certifier
applies to it), and that a missing dependency degrades to a clear message rather than a
bare ImportError.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("mitsuba", reason="install renderprobe[render3d-cpu] to test mitsuba_3d")

from renderprobe.core.registry import Registry  # noqa: E402
from renderprobe.core.schema import GraphObject, SceneGraph  # noqa: E402
from renderprobe.core.validate import (  # noqa: E402
    _context_stable,
    _object_visibility,
    _occlusion_verdict,
)
from renderprobe.renderers import get_renderer, render_graph  # noqa: E402
from renderprobe.renderers.mitsuba_3d import (  # noqa: E402
    _CAM_Z,
    _UNIT,
    _camera_origin,
    _mi,
)
from tests.fixtures import register

_CANVAS = (160, 120)


def _graph(objs, canvas=_CANVAS) -> SceneGraph:
    return SceneGraph(objects=objs, canvas=canvas)


def test_registered_and_is_3d():
    reg = Registry()
    reg.autodiscover()
    register(reg)
    assert "mitsuba_3d" in reg.list_renderers()
    r = get_renderer("mitsuba_3d")
    assert r.is_3d is True


def test_render_returns_rgb_image():
    g = _graph([GraphObject("a", "sphere", "red", (50.0, 60.0, 0.0), 20.0),
                GraphObject("b", "cube", "blue", (110.0, 60.0, 0.0), 20.0)])
    imgs = get_renderer("mitsuba_3d").render(g)
    assert len(imgs) == 1
    assert imgs[0].size == _CANVAS and imgs[0].mode == "RGB"


def test_render_ids_shape_and_object_indices():
    g = _graph([GraphObject("a", "sphere", "red", (50.0, 60.0, 0.0), 20.0),
                GraphObject("b", "cube", "blue", (110.0, 60.0, 0.0), 20.0),
                GraphObject("c", "sphere", "green", (80.0, 95.0, 0.0), 18.0)])
    ids = get_renderer("mitsuba_3d").render_ids(g)
    assert ids.shape == (_CANVAS[1], _CANVAS[0])            # (H, W)
    assert np.issubdtype(ids.dtype, np.integer)
    assert set(np.unique(ids).tolist()) == {-1, 0, 1, 2}   # background + the three objects


def test_render_ids_is_deterministic():
    g = _graph([GraphObject("a", "sphere", "red", (50.0, 60.0, 0.0), 20.0),
                GraphObject("b", "sphere", "blue", (110.0, 60.0, 0.0), 20.0)])
    r = get_renderer("mitsuba_3d")
    assert np.array_equal(r.render_ids(g), r.render_ids(g))


def test_absolute_projection_is_context_stable():
    """True perspective projects each object independently of the others, so the partial-
    occlusion band is valid for mitsuba_3d (unlike pil_25d's scene-relative scaling)."""
    assert _context_stable(get_renderer("mitsuba_3d"), _CANVAS) is True


def test_render_ids_faithful_to_color_render():
    """Where render_ids marks object i, the color render must actually show that object -
    its dominant RGB channel matches the object's color (allowing for shading)."""
    r = get_renderer("mitsuba_3d")
    g = _graph([GraphObject("a", "sphere", "red", (55.0, 60.0, 0.0), 22.0),
                GraphObject("b", "cube", "blue", (115.0, 60.0, 0.0), 22.0),
                GraphObject("c", "sphere", "green", (85.0, 95.0, 0.0), 20.0)])
    ids = r.render_ids(g)
    col = np.array(render_graph(g, "mitsuba_3d")[0]).astype(float)
    want = {"red": 0, "green": 1, "blue": 2}
    for i, obj in enumerate(g.objects):
        m = ids == i
        assert m.sum() > 0
        assert int(np.argmax(col[m].mean(axis=0))) == want[obj.color]


def test_certifier_catches_occlusion_under_mitsuba():
    """A near object directly over a far one along the view axis hides it -> FAIL. This is
    the whole point: the Step-A certificate works on a real true-3D renderer."""
    r = get_renderer("mitsuba_3d")
    clear = _graph([GraphObject("a", "sphere", "red", (45.0, 60.0, 0.0), 18.0),
                    GraphObject("b", "sphere", "blue", (115.0, 60.0, 0.0), 18.0)])
    occ = _graph([GraphObject("back", "sphere", "red", (80.0, 60.0, 0.0), 16.0),
                  GraphObject("front", "sphere", "blue", (80.0, 60.0, -150.0), 40.0)])
    stable = _context_stable(r, _CANVAS)
    assert _occlusion_verdict(_object_visibility(r, clear), stable)[0] == "pass"
    assert _occlusion_verdict(_object_visibility(r, occ), stable)[0] == "fail"


def test_blocks_certifies_occlusion_free_under_mitsuba():
    """The fronto-parallel framing keeps blocks' count trustable even in true 3D."""
    reg = Registry()
    reg.autodiscover()
    register(reg)
    gen = reg.get_scene("blocks")
    r = get_renderer("mitsuba_3d")
    stable = _context_stable(r, (480, 360))
    for seed in range(2):
        graph = gen.generate({"n_objects": 6}, seed).graph
        status, detail = _occlusion_verdict(_object_visibility(r, graph), stable)
        assert status == "pass", f"seed {seed}: {detail}"


def test_graceful_degradation_without_mitsuba(monkeypatch):
    """A missing optional dependency raises a clear, actionable RuntimeError naming the
    extra to install - never a bare ImportError."""
    monkeypatch.setitem(sys.modules, "mitsuba", None)   # make `import mitsuba` raise
    g = _graph([GraphObject("a", "sphere", "red", (50.0, 60.0, 0.0), 20.0)])
    with pytest.raises(RuntimeError, match="render3d-cpu"):
        get_renderer("mitsuba_3d").render(g)


# ---------------------------------------------------------------------------
# Camera and ground are configurable, and neither may weaken the id pass.
# Both default to values that leave screen-space scenes projecting exactly as before.
# ---------------------------------------------------------------------------

def test_camera_defaults_to_dead_on():
    # Built-in scenes lay objects out in screen coordinates and rely on this.
    assert _camera_origin({}) == ([0.0, 0.0, _CAM_Z], [0.0, 0.0, 0.0])
    assert _camera_origin({"camera": "front"})[0] == [0.0, 0.0, _CAM_Z]


def test_camera_preset_orbits_and_lifts():
    origin, target = _camera_origin({"camera": "three_quarter"})
    assert target == [0.0, 0.0, 0.0]
    assert origin[0] > 0.5      # orbited to the side
    assert origin[1] > 0.5      # and lifted
    assert origin[2] > 0.0      # still in front of the scene


def test_camera_accepts_explicit_angles():
    origin, _ = _camera_origin({"camera": {"azimuth": 90.0, "elevation": 0.0,
                                           "distance": 4.0}})
    assert origin[0] == pytest.approx(4.0, abs=1e-6)   # straight out to the side
    assert origin[2] == pytest.approx(0.0, abs=1e-6)


def test_ground_defaults_below_the_canvas():
    # A floor at a fixed world height would swallow objects a screen-space scene
    # places low in frame; the default must sit clear of the canvas bottom.
    r = get_renderer("mitsuba_3d")
    canvas = (400, 300)
    d = r._ground(_mi(), True, canvas, {})
    lowest_object_y = -(canvas[1] / 2.0) / _UNIT
    gy = d["to_world"].matrix.numpy()[1, 3]
    assert gy < lowest_object_y


def test_scene_can_place_its_own_ground():
    r = get_renderer("mitsuba_3d")
    d = r._ground(_mi(), True, (400, 300), {"ground_y": -1.25})
    assert d["to_world"].matrix.numpy()[1, 3] == pytest.approx(-1.25, abs=1e-6)


def test_render_ids_stays_faithful_under_an_orbited_camera():
    # The certificate is only sound if both passes share a camera; an orbited view
    # must not desynchronise them.
    r = get_renderer("mitsuba_3d")
    objs = [GraphObject(id=f"c{i}", shape="cube", color=c,
                        position=(120 + i * 90, 160, 20 + i * 30), size=20)
            for i, c in enumerate(["red", "blue", "green"])]
    g = SceneGraph(objects=objs, canvas=(400, 300),
                   meta={"spp": 12, "camera": "three_quarter", "ground_y": -1.2})
    ids = r.render_ids(g)
    col = np.array(r.render(g)[0])
    assert set(np.unique(ids).tolist()) == {-1, 0, 1, 2}
    want = {"red": 0, "green": 1, "blue": 2}
    for i, o in enumerate(objs):
        m = ids == i
        assert m.sum() > 0
        assert int(np.argmax(col[m].mean(axis=0))) == want[o.color]
