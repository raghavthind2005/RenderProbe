"""Renderer abstraction - realism as a knob over an exact-GT scene graph.

Proves the core Task-3 thesis: the same SceneGraph renders at different realism levels
(flat 2D, depth-shaded 2.5D) with IDENTICAL ground truth; the optional photorealistic
3D backend degrades gracefully when its heavy deps are absent; and a scientist can drop
in their own renderer.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from renderprobe.core.registry import RegistrationError, Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.core.schema import GraphObject, SceneGraph
from renderprobe.core.validate import all_ok, validate_path
from renderprobe.renderers import get_renderer, render_graph
from tests.fixtures import register

SRC = Path(__file__).resolve().parents[1] / "src" / "renderprobe"


def _reg() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _graph(canvas=(200, 150)) -> SceneGraph:
    return SceneGraph(
        objects=[GraphObject("a", "sphere", "red", (50.0, 50.0, 0.0), 18.0),
                 GraphObject("b", "cube", "blue", (140.0, 90.0, 2.0), 18.0)],
        canvas=canvas,
    )


# ---------------------------------------------------------------------------
# Registry + basic rendering
# ---------------------------------------------------------------------------

def test_renderers_autodiscovered():
    names = _reg().list_renderers()
    assert {"pil_2d", "pil_25d", "pyrender_3d", "mitsuba_3d"} <= set(names)


@pytest.mark.parametrize("name", ["pil_2d", "pil_25d"])
def test_pure_python_renderers_render(name):
    imgs = render_graph(_graph((200, 150)), name)
    assert len(imgs) == 1
    assert isinstance(imgs[0], Image.Image)
    assert imgs[0].size == (200, 150)
    assert imgs[0].mode == "RGB"


def test_pil_renderers_produce_different_pixels():
    g = _graph()
    assert render_graph(g, "pil_2d")[0].tobytes() != render_graph(g, "pil_25d")[0].tobytes()


# ---------------------------------------------------------------------------
# The thesis: ground truth is renderer-independent
# ---------------------------------------------------------------------------

def test_ground_truth_identical_across_renderers():
    reg = _reg()
    g = reg.get_scene("blocks")
    a = g.generate({"n_objects": 8, "renderer": "pil_2d"}, 5)
    b = g.generate({"n_objects": 8, "renderer": "pil_25d"}, 5)
    # same graph seed -> same GT (objects/count), different pixels
    assert a.ground_truth["count"] == b.ground_truth["count"] == 8
    assert a.ground_truth["objects"] == b.ground_truth["objects"]
    assert a.images[0].tobytes() != b.images[0].tobytes()


def test_blocks_count_is_trustable():
    """Every object placed is visible and non-overlapping, so the count is recoverable."""
    reg = _reg()
    g = reg.get_scene("blocks")
    for seed in range(12):
        sc = g.generate({"n_objects": 10}, seed)
        objs = sc.ground_truth["objects"]
        assert sc.ground_truth["count"] == len(objs)
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                d = math.hypot(objs[i]["x"] - objs[j]["x"], objs[i]["y"] - objs[j]["y"])
                assert d >= objs[i]["r"] + objs[j]["r"], "objects overlap -> count untrustable"


# ---------------------------------------------------------------------------
# Optional 3D backend: graceful without deps
# ---------------------------------------------------------------------------

def test_pyrender_3d_registered_and_degrades():
    r = get_renderer("pyrender_3d")
    assert r.is_3d
    try:
        import pyrender  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError, match="render3d"):
            r.render(_graph())


# ---------------------------------------------------------------------------
# End-to-end + validation + extensibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("renderer", ["pil_2d", "pil_25d"])
def test_blocks_runs_end_to_end(renderer):
    reg = _reg()
    cfg = ExperimentConfig(
        scene_generator="blocks",
        scene_params={"n_objects": [4, 6], "renderer": renderer},
        seeds=[0, 1],
        probe="count",
        models=["mock"],
        analyzers=["decomposition"],
    )
    scenes, results, reports = run_experiment_collect(cfg, reg)
    assert scenes and results
    assert all(s.factors["renderer"] == renderer for s in scenes)


def test_renderer_template_validates_clean():
    reports = validate_path(SRC / "renderers" / "_template.py")
    assert all_ok(reports)


def test_external_renderer_loads(tmp_path):
    p = tmp_path / "my_renderer.py"
    p.write_text(textwrap.dedent('''
        from PIL import Image
        from renderprobe.core.schema import SceneGraph
        class R:
            name = "t_ext_renderer"
            is_3d = False
            def render(self, graph):
                return [Image.new("RGB", graph.canvas, (10, 20, 30))]
        PLUGIN = R()
    '''))
    reg = _reg()
    added = reg.load_external(p)
    assert added == ["renderer:t_ext_renderer"]
    assert "t_ext_renderer" in reg.list_renderers()


def test_non_renderer_not_detected_as_renderer(tmp_path):
    # a plain object with a render attr but no is_3d must not sneak in as a renderer
    p = tmp_path / "notr.py"
    p.write_text("PLUGIN = object()\n")
    with pytest.raises(RegistrationError):
        _reg().load_external(p)
