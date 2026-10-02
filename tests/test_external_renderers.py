"""External-renderer resolution, renderer determinism, and the plugin CLI.

- a scientist's own graph scene can resolve their own --plugin-dir renderer (scoped, no
    global leak) - during both a run and validation.
- `validate` checks a renderer is deterministic (same graph -> identical pixels).
- `validate` confirms a graph scene actually RENDERS under the chosen renderer (pair).
- `renderprobe list` shows the catalog; `renderprobe new` scaffolds from a template.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from renderprobe import renderers
from renderprobe.cli import list_cmd, new_cmd
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.core.validate import all_ok, validate_path
from tests.fixtures import register

SRC = Path(__file__).resolve().parents[1] / "src" / "renderprobe"

_EXT_RENDERER = '''
from PIL import Image, ImageDraw
from renderprobe.core.schema import SceneGraph
class R:
    name = "extflat"
    is_3d = False
    def render(self, graph):
        img = Image.new("RGB", graph.canvas, (255, 255, 255))
        d = ImageDraw.Draw(img)
        for o in graph.objects:
            x, y, _ = o.position; r = o.size
            d.ellipse([x - r, y - r, x + r, y + r], fill=(200, 50, 50))
        return [img]
PLUGIN = R()
'''

_EXT_SCENE = '''
from renderprobe.core.schema import GraphObject, Scene, SceneGraph
from renderprobe.renderers import render_graph
class _S:
    name = "extscene"
    params_schema = {"n": {"type": "int", "min": 3, "max": 3, "default": 3}}
    def generate(self, params, seed):
        objs = [GraphObject("o0","sphere","red",(40.,40.,0.),16.),
                GraphObject("o1","sphere","blue",(120.,40.,0.),16.),
                GraphObject("o2","sphere","green",(80.,100.,0.),16.)]
        g = SceneGraph(objs, (160,140))
        return Scene(id=f"s{seed}", images=render_graph(g, "extflat"),   # their OWN renderer
                     ground_truth={"count":3}, factors={}, meta={}, graph=g)
PLUGIN = _S()
'''


def _checks(reports) -> dict[str, str]:
    return {c.name: c.status for r in reports for c in r.checks}


# ---------------------------------------------------------------------------
# C1 - external-renderer resolution (scoped)
# ---------------------------------------------------------------------------

class _ExtRenderer:
    name = "tmp_ext_renderer"
    is_3d = False
    def render(self, graph):
        return [Image.new("RGB", graph.canvas, (0, 0, 0))]


def test_resolver_scope_resolves_external_and_restores():
    reg = Registry()
    reg.autodiscover()
    register(reg)
    reg.register_renderer(_ExtRenderer())

    with pytest.raises(KeyError):
        renderers.get_renderer("tmp_ext_renderer")          # unknown outside a scope

    with renderers.using_registry(reg):
        assert renderers.get_renderer("tmp_ext_renderer").name == "tmp_ext_renderer"
        assert renderers.get_renderer("pil_2d").name == "pil_2d"   # built-ins still resolve
        assert "tmp_ext_renderer" in renderers.available()

    with pytest.raises(KeyError):
        renderers.get_renderer("tmp_ext_renderer")          # scope restored, no leak


def test_runner_resolves_external_renderer(tmp_path):
    (tmp_path / "extflat.py").write_text(_EXT_RENDERER)
    (tmp_path / "extscene.py").write_text(_EXT_SCENE)
    reg = Registry()
    reg.autodiscover()
    register(reg)
    reg.load_external(tmp_path)

    cfg = ExperimentConfig(scene_generator="extscene", scene_params={"n": 3},
                           seeds=[0, 1], probe="count", models=["mock"], analyzers=[])
    scenes, results, _ = run_experiment_collect(cfg, reg)
    assert scenes, "scene using an external renderer must generate during a run"
    assert all(s.images for s in scenes)
    assert results


def test_validate_dir_scene_uses_external_renderer(tmp_path):
    (tmp_path / "extflat.py").write_text(_EXT_RENDERER)
    (tmp_path / "extscene.py").write_text(_EXT_SCENE)
    reports = validate_path(tmp_path)
    # the scene's determinism check passing proves generate() resolved "extflat"
    scene_report = next(r for r in reports if r.family == "scene")
    names = {c.name: c.status for c in scene_report.checks}
    assert names.get("determinism") == "pass"
    assert all_ok(reports)


# ---------------------------------------------------------------------------
# C2 - renderer determinism check
# ---------------------------------------------------------------------------

def test_renderer_determinism_pass():
    reports = validate_path(SRC / "renderers" / "_template.py")
    assert _checks(reports).get("determinism") == "pass"


def test_nondeterministic_renderer_fails(tmp_path):
    p = tmp_path / "nondet.py"
    p.write_text(textwrap.dedent('''
        from PIL import Image
        from renderprobe.core.schema import SceneGraph
        class R:
            name = "nondet"
            is_3d = False
            _n = 0
            def render(self, graph):
                R._n += 1
                return [Image.new("RGB", graph.canvas, (R._n % 251, 0, 0))]
        PLUGIN = R()
    '''))
    reports = validate_path(p)
    assert _checks(reports).get("determinism") == "fail"
    assert not all_ok(reports)


# ---------------------------------------------------------------------------
# C4 - validate confirms the scene renders under the chosen renderer
# ---------------------------------------------------------------------------

def test_scene_renders_check_present(tmp_path):
    (tmp_path / "extflat.py").write_text(_EXT_RENDERER)
    (tmp_path / "extscene.py").write_text(_EXT_SCENE)
    reports = validate_path(tmp_path)
    assert _checks(reports).get("scene-renders") == "pass"


# ---------------------------------------------------------------------------
# C3 - list / new CLI
# ---------------------------------------------------------------------------

def test_list_cmd_shows_catalog(capsys):
    rc = list_cmd(SimpleNamespace(plugin_dir=None))
    assert rc == 0
    out = capsys.readouterr().out
    for header in ("scenes", "probes", "renderers", "models", "analyzers"):
        assert header in out
    assert "pil_2d" in out and "mitsuba_3d" in out


def test_new_scaffolds_named_and_validates(tmp_path):
    rc = new_cmd(SimpleNamespace(family="renderer", name="my_r", dir=str(tmp_path),
                                 force=False))
    assert rc == 0
    f = tmp_path / "my_r.py"
    assert f.exists()
    assert 'name = "my_r"' in f.read_text()          # pre-renamed
    assert all_ok(validate_path(f))                  # scaffold validates clean


def test_new_rejects_bad_name_and_family(tmp_path):
    assert new_cmd(SimpleNamespace(family="renderer", name="Bad Name",
                                   dir=str(tmp_path), force=False)) == 1
    assert new_cmd(SimpleNamespace(family="nope", name="ok_name",
                                   dir=str(tmp_path), force=False)) == 1


def test_new_refuses_overwrite_without_force(tmp_path):
    args = SimpleNamespace(family="probe", name="dup", dir=str(tmp_path), force=False)
    assert new_cmd(args) == 0
    assert new_cmd(args) == 1                          # exists -> refuse
    args.force = True
    assert new_cmd(args) == 0                          # --force overwrites
