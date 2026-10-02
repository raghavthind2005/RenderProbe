"""Renderer-option passthrough + the render-budget preview.

Ground truth is renderer-independent, so realism is a cost knob. These check that a config
can control an expensive renderer's cost (spp) via `renderer_opts` -> graph.meta, and that
`renderprobe run` previews the render budget (and warns on a true-3D sweep) so an expensive
run is a deliberate choice.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from renderprobe.cli import run_cmd
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, render_plan
from tests.fixtures import register

# The CLI builds its own registry from the installed package, which no longer
# carries the offline scenes, so these end-to-end runs load them the way any
# out-of-tree user would.
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# ---------------------------------------------------------------------------
# D1 - renderer options flow config -> graph.meta
# ---------------------------------------------------------------------------

def test_blocks_forwards_renderer_opts_to_graph_meta():
    reg = Registry()
    reg.autodiscover()
    register(reg)
    gen = reg.get_scene("blocks")
    sc = gen.generate({"n_objects": 3, "renderer_opts": {"spp": 4}}, 0)
    assert sc.graph is not None
    assert sc.graph.meta.get("spp") == 4


def test_blocks_graph_meta_empty_without_opts():
    reg = Registry()
    reg.autodiscover()
    register(reg)
    sc = reg.get_scene("blocks").generate({"n_objects": 3}, 0)
    assert sc.graph.meta == {}


def test_mitsuba_honors_spp_from_meta():
    pytest.importorskip("mitsuba", reason="install renderprobe[render3d-cpu]")
    reg = Registry()
    reg.autodiscover()
    register(reg)
    gen = reg.get_scene("blocks")
    # A low spp still renders a valid image (the cheap-preview path).
    sc = gen.generate({"n_objects": 3, "renderer": "mitsuba_3d", "renderer_opts": {"spp": 1}}, 0)
    assert sc.images and sc.images[0].size == (480, 360)


# ---------------------------------------------------------------------------
# D2 - the render-budget plan
# ---------------------------------------------------------------------------

def _cfg(**kw) -> ExperimentConfig:
    base = dict(scene_generator="blocks", scene_params={"n_objects": 4}, seeds=[0],
                probe="count", models=["mock"], analyzers=[])
    base.update(kw)
    return ExperimentConfig(**base)


def test_render_plan_counts_scenes():
    plan = render_plan(_cfg(scene_params={"n_objects": [4, 6, 8]}, seeds=[0, 1]))
    assert plan["n_scenes"] == 6                      # 3 param sets x 2 seeds
    assert plan["renderers"] == ["pil_2d"]            # default
    assert plan["heavy"] == []


def test_render_plan_flags_heavy_renderer():
    plan = render_plan(_cfg(scene_params={"n_objects": 4, "renderer": "mitsuba_3d"},
                            seeds=[0, 1, 2]))
    assert plan["n_scenes"] == 3
    assert plan["heavy"] == ["mitsuba_3d"]


def test_render_plan_swept_renderer_lists_both():
    plan = render_plan(_cfg(scene_params={"n_objects": 4, "renderer": ["pil_2d", "mitsuba_3d"]},
                            seeds=[0, 1]))
    assert plan["n_scenes"] == 4                      # 2 renderers x 2 seeds
    assert plan["renderers"] == ["mitsuba_3d", "pil_2d"]
    assert plan["heavy"] == ["mitsuba_3d"]


# ---------------------------------------------------------------------------
# D2 - the CLI prints the plan (and the heavy-renderer note)
# ---------------------------------------------------------------------------

def test_run_cmd_prints_render_plan(tmp_path, capsys):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "experiment: t\n"
        "scene: {generator: blocks, params: {n_objects: 3}, seeds: [0, 1]}\n"
        "probe: count\nmodels: [mock]\nanalysis: [accuracy]\n"
    )
    rc = run_cmd(SimpleNamespace(config=str(cfg), plugin_dir=[str(FIXTURE_DIR)]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "render plan:" in out
    assert "2 scene image(s)" in out


def test_run_cmd_warns_on_heavy_renderer(tmp_path, capsys):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "experiment: t\n"
        "scene: {generator: blocks, params: {n_objects: 3, renderer: pyrender_3d}, seeds: [0]}\n"
        "probe: count\nmodels: [mock]\nanalysis: []\n"
    )
    # pyrender has no GL backend here, so generation fails gracefully (0 scenes) - but the
    # budget NOTE must still print before running, which is the point of the preview.
    run_cmd(SimpleNamespace(config=str(cfg), plugin_dir=[str(FIXTURE_DIR)]))
    out = capsys.readouterr().out
    assert "note:" in out and "pyrender_3d" in out
