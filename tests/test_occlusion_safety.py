"""Occlusion safety - the honesty guarantee that makes true-3D rendering trustworthy.

A renderer's optional ``render_ids`` reports the front-most object per pixel with the SAME
occlusion as ``render``; ``validate`` uses it to CERTIFY that every ground-truth object
stays visible under the chosen renderer, and FAILs a graph scene whose count/enumeration
ground truth would be corrupted by a hidden object. These tests prove BOTH the PASS path
(nothing hidden) and the FAIL path (an object is occluded) - the PIL renderers occlude
whenever silhouettes overlap, so the whole mechanism is exercised without a 3D backend.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from renderprobe.core.registry import Registry
from renderprobe.core.schema import GraphObject, SceneGraph
from renderprobe.core.validate import (
    FAIL,
    PASS,
    _certify_derivable,
    _object_visibility,
    _occlusion_policy,
    _occlusion_verdict,
    all_ok,
    validate_path,
)
from renderprobe.renderers import get_renderer, render_graph
from renderprobe.renderers._palette import rgb
from tests.fixtures import register

SRC = Path(__file__).resolve().parents[1] / "src" / "renderprobe"
_PIL = ["pil_2d", "pil_25d"]


def _clear_graph() -> SceneGraph:
    """Two well-separated objects - neither occludes the other under any sane renderer."""
    return SceneGraph(
        objects=[GraphObject("a", "sphere", "red", (45.0, 60.0, 0.0), 18.0),
                 GraphObject("b", "cube", "blue", (120.0, 60.0, 0.0), 18.0)],
        canvas=(160, 120),
    )


def _occluded_graph() -> SceneGraph:
    """A big near object exactly over a small far one at the SAME screen position: the
    far object (index 0) is hidden. Under a painter's-order renderer the near one wins."""
    return SceneGraph(
        objects=[GraphObject("back", "sphere", "red", (80.0, 60.0, 5.0), 16.0),
                 GraphObject("front", "sphere", "blue", (80.0, 60.0, 0.0), 26.0)],
        canvas=(160, 120),
    )


def _checks(reports) -> dict[str, str]:
    """name -> status across all reports (used to assert on a specific check)."""
    return {c.name: c.status for r in reports for c in r.checks}


# ---------------------------------------------------------------------------
# render_ids conforms and is faithful to render()
# ---------------------------------------------------------------------------

def test_render_ids_shape_dtype_and_background():
    for name in _PIL:
        ids = get_renderer(name).render_ids(_clear_graph())
        assert ids.shape == (120, 160)                 # (H, W)
        assert np.issubdtype(ids.dtype, np.integer)
        assert set(np.unique(ids)) <= {-1, 0, 1}       # only the two objects + background
        assert np.count_nonzero(ids == -1) > 0         # background is present


def test_render_ids_all_objects_visible_when_separated():
    for name in _PIL:
        ids = get_renderer(name).render_ids(_clear_graph())
        assert np.count_nonzero(ids == 0) > 0
        assert np.count_nonzero(ids == 1) > 0


def test_render_ids_reports_full_occlusion():
    for name in _PIL:
        ids = get_renderer(name).render_ids(_occluded_graph())
        assert np.count_nonzero(ids == 0) == 0, f"{name}: hidden object should have 0 px"
        assert np.count_nonzero(ids == 1) > 0


def test_render_ids_is_deterministic():
    for name in _PIL:
        r = get_renderer(name)
        g = _clear_graph()
        assert np.array_equal(r.render_ids(g), r.render_ids(g))


def test_render_ids_matches_colour_render_pil2d():
    """Faithfulness: pil_2d uses flat fills, so where render_ids marks object i front-most
    the COLOR image must actually show object i's fill. The whole certificate rests on
    render_ids reporting exactly what render() draws - this proves it. (The <100% remainder
    of each region is the 2px black silhouette outline.)"""
    reg = Registry()
    reg.autodiscover()
    register(reg)
    g = reg.get_scene("blocks").generate({"n_objects": 8}, 3).graph
    ids = get_renderer("pil_2d").render_ids(g)
    col = np.asarray(render_graph(g, "pil_2d")[0])
    for i, obj in enumerate(g.objects):
        m = ids == i
        if m.sum() == 0:
            continue
        frac = float(np.all(col[m] == np.array(rgb(obj.color)), axis=1).mean())
        assert frac > 0.6, f"object {i} id-region does not match its fill color ({frac:.2f})"


def test_hidden_object_absent_from_both_maps():
    """A fully occluded object appears in NEITHER the id-map NOR the color image - the
    id-map's occlusion is the render's occlusion, so certifying one certifies the other."""
    g = _occluded_graph()
    ids = get_renderer("pil_2d").render_ids(g)
    col = np.asarray(render_graph(g, "pil_2d")[0])
    assert np.count_nonzero(ids == 0) == 0                                # back: hidden in id-map
    assert int(np.all(col == np.array(rgb("red")), axis=-1).sum()) == 0   # back: absent in pixels


# ---------------------------------------------------------------------------
# The certifier: PASS on clear scenes, FAIL on occlusion
# ---------------------------------------------------------------------------

def test_verdict_pass_on_clear_and_fail_on_occluded():
    for name in _PIL:
        r = get_renderer(name)
        assert _occlusion_verdict(_object_visibility(r, _clear_graph()))[0] == "pass"
        assert _occlusion_verdict(_object_visibility(r, _occluded_graph()))[0] == "fail"


def test_object_visibility_none_without_render_ids():
    class _NoIds:
        name, is_3d = "x", True
    assert _object_visibility(_NoIds(), _clear_graph()) is None


# ---------------------------------------------------------------------------
# The built-in blocks scene stays occlusion-free (its non-overlap guarantee holds)
# ---------------------------------------------------------------------------

def test_blocks_certifies_occlusion_free_under_pil_renderers():
    reg = Registry()
    reg.autodiscover()
    register(reg)
    gen = reg.get_scene("blocks")
    for name in _PIL:
        r = get_renderer(name)
        for seed in range(4):
            graph = gen.generate({"n_objects": 10}, seed).graph
            assert graph is not None, "blocks must expose .graph for certification"
            status, detail = _occlusion_verdict(_object_visibility(r, graph))
            assert status == "pass", f"{name} seed {seed}: {detail}"


# ---------------------------------------------------------------------------
# Renderer-level self-test (validate a renderer)
# ---------------------------------------------------------------------------

def test_renderer_template_passes_occlusion_selftest():
    reports = validate_path(SRC / "renderers" / "_template.py")
    assert all_ok(reports)
    assert _checks(reports).get("occlusion-report") == "pass"


def test_3d_renderer_without_render_ids_warns(tmp_path):
    p = tmp_path / "noids3d.py"
    p.write_text(textwrap.dedent('''
        from PIL import Image
        from renderprobe.core.schema import SceneGraph
        class R:
            name = "t_noids_3d"
            is_3d = True
            def render(self, graph):
                return [Image.new("RGB", graph.canvas, (0, 0, 0))]
        PLUGIN = R()
    '''))
    reports = validate_path(p)
    assert all_ok(reports)                                  # WARN, not FAIL
    assert _checks(reports).get("occlusion-report") == "warn"


def test_2d_renderer_without_render_ids_is_fine(tmp_path):
    p = tmp_path / "noids2d.py"
    p.write_text(textwrap.dedent('''
        from PIL import Image
        from renderprobe.core.schema import SceneGraph
        class R:
            name = "t_noids_2d"
            is_3d = False
            def render(self, graph):
                return [Image.new("RGB", graph.canvas, (0, 0, 0))]
        PLUGIN = R()
    '''))
    reports = validate_path(p)
    assert all_ok(reports)
    assert _checks(reports).get("occlusion-report") == "pass"


# ---------------------------------------------------------------------------
# End-to-end: validate a scientist's graph scene
# ---------------------------------------------------------------------------

_GOOD_SCENE = '''
from renderprobe.core.schema import GraphObject, Scene, SceneGraph
from renderprobe.renderers import render_graph

class _S:
    name = "good_blocks"
    params_schema = {"n_objects": {"type": "int", "min": 3, "max": 3, "default": 3}}
    def generate(self, params, seed):
        objs = [
            GraphObject("o0", "sphere", "red",   (40.0, 40.0, 0.0), 16.0),
            GraphObject("o1", "cube",   "blue",  (120.0, 40.0, 0.0), 16.0),
            GraphObject("o2", "sphere", "green", (78.0, 100.0, 0.0), 16.0),
        ]
        g = SceneGraph(objects=objs, canvas=(160, 140))
        return Scene(id=f"good_{seed}", images=render_graph(g, "pil_2d"),
                     ground_truth={"count": len(objs)}, factors={}, meta={}, graph=g)

PLUGIN = _S()
'''

_BAD_SCENE = '''
from renderprobe.core.schema import GraphObject, Scene, SceneGraph
from renderprobe.renderers import render_graph

class _S:
    name = "bad_blocks"
    params_schema = {"n_objects": {"type": "int", "min": 2, "max": 2, "default": 2}}
    def generate(self, params, seed):
        objs = [
            GraphObject("back",  "sphere", "red",  (80.0, 70.0, 5.0), 16.0),  # far, small
            GraphObject("front", "sphere", "blue", (80.0, 70.0, 0.0), 28.0),  # near, hides back
        ]
        g = SceneGraph(objects=objs, canvas=(160, 140))
        return Scene(id=f"bad_{seed}", images=render_graph(g, "pil_2d"),
                     ground_truth={"count": len(objs)}, factors={}, meta={}, graph=g)

PLUGIN = _S()
'''


def test_good_graph_scene_certifies_occlusion_safe(tmp_path):
    p = tmp_path / "good_scene.py"
    p.write_text(_GOOD_SCENE)
    reports = validate_path(p)
    assert _checks(reports).get("occlusion-safety") == "pass"


def test_occluding_graph_scene_is_caught(tmp_path):
    p = tmp_path / "bad_scene.py"
    p.write_text(_BAD_SCENE)
    reports = validate_path(p)
    assert _checks(reports).get("occlusion-safety") == "fail"
    assert not all_ok(reports)


def test_occlusion_safety_under_selected_renderer(tmp_path):
    """--renderer certifies under a specific tier; the good scene is safe under pil_25d too."""
    p = tmp_path / "good_scene.py"
    p.write_text(_GOOD_SCENE)
    reports = validate_path(p, renderer_name="pil_25d")
    assert _checks(reports).get("occlusion-safety") == "pass"


_PLAIN_SCENE = '''
from PIL import Image
from renderprobe.core.schema import Scene

class _S:
    name = "plain_pixels"
    params_schema = {"n": {"type": "int", "min": 2, "max": 2, "default": 2}}
    def generate(self, params, seed):
        img = Image.new("RGB", (80, 80), (255, 255, 255))
        return Scene(id=f"p{seed}", images=[img],
                     ground_truth={"count": 2}, factors={}, meta={})   # no .graph

PLUGIN = _S()
'''


def test_non_graph_scene_reports_na(tmp_path):
    """A scene that draws pixels directly (sets no .graph) is noted n/a, never FAILed -
    occlusion certification is only for graph-based scenes."""
    p = tmp_path / "plain.py"
    p.write_text(_PLAIN_SCENE)
    reports = validate_path(p)
    checks = _checks(reports)
    assert checks.get("occlusion-safety") == "pass"
    detail = next(c.detail for r in reports for c in r.checks if c.name == "occlusion-safety")
    assert "n/a" in detail


# ---------------------------------------------------------------------------
# Occlusion policy: certifying a subset, and tasks where hiding is the point
# ---------------------------------------------------------------------------

class _FakeScene:
    def __init__(self, gt):
        self.ground_truth = gt


def test_policy_defaults_are_the_strict_ones():
    g = SceneGraph(objects=[], canvas=(10, 10))
    assert _occlusion_policy(g) == (None, "visible")


def test_policy_reads_certify_subset_and_mode():
    g = SceneGraph(objects=[], canvas=(10, 10),
                   meta={"occlusion": {"certify": [1, 2], "policy": "derivable"}})
    assert _occlusion_policy(g) == ([1, 2], "derivable")


def test_hidden_object_fails_by_default():
    # object 1 fully hidden -> the count is not recoverable
    assert _occlusion_verdict([(500, 500), (0, 400)])[0] == FAIL


def test_hidden_object_passes_when_it_is_not_answer_bearing():
    # same scene, but the answer only depends on object 0: clutter may hide
    status, detail = _occlusion_verdict([(500, 500), (0, 400)], certify=[0])
    assert status == PASS
    assert "1 object(s) visible" in detail


def test_certify_index_out_of_range_fails_loudly():
    status, detail = _occlusion_verdict([(500, 500)], certify=[3])
    assert status == FAIL
    assert "index 3" in detail


def test_derivable_without_the_hook_is_refused():
    # A scene must not be able to opt out of the guarantee just by claiming to.
    class _NoHook:
        pass
    g = SceneGraph(objects=[], canvas=(10, 10))
    status, detail = _certify_derivable(_NoHook(), _FakeScene({"count": 5}), g,
                                        [(500, 500), (0, 400)])
    assert status == FAIL
    assert "derive_from_visible" in detail


def test_derivable_passes_when_the_answer_is_re_derived_correctly():
    class _Gen:
        # a stack of 5 where 2 are hidden: support implies the hidden ones
        def derive_from_visible(self, graph, visible):
            return {"count": len(visible) + 2}
    g = SceneGraph(objects=[], canvas=(10, 10))
    status, detail = _certify_derivable(_Gen(), _FakeScene({"count": 5}), g,
                                        [(500, 500), (500, 500), (500, 500),
                                         (0, 400), (0, 400)])
    assert status == PASS
    assert "2 object(s) hidden by design" in detail


def test_derivable_fails_when_the_derivation_disagrees_with_ground_truth():
    class _Gen:
        def derive_from_visible(self, graph, visible):
            return {"count": len(visible)}   # forgets the hidden ones
    g = SceneGraph(objects=[], canvas=(10, 10))
    status, detail = _certify_derivable(_Gen(), _FakeScene({"count": 5}), g,
                                        [(500, 500), (0, 400)])
    assert status == FAIL
    assert "NOT derivable" in detail


def test_derivable_rejects_a_non_dict_derivation():
    class _Gen:
        def derive_from_visible(self, graph, visible):
            return 5
    g = SceneGraph(objects=[], canvas=(10, 10))
    assert _certify_derivable(_Gen(), _FakeScene({"count": 5}), g, [(1, 1)])[0] == FAIL


def test_presence_policy_still_fails_a_fully_hidden_object():
    # "presence" relaxes the partial-occlusion warning, not the real guarantee.
    assert _occlusion_verdict([(500, 500), (0, 400)], presence_only=True)[0] == FAIL


def test_presence_policy_does_not_warn_on_partial_occlusion():
    # A solid object's parts overlap each other by nature; warning there is noise.
    heavy = [(500, 500), (40, 400)]        # second is <50% visible but present
    assert _occlusion_verdict(heavy, partial_reliable=True)[0] == "warn"
    assert _occlusion_verdict(heavy, partial_reliable=True, presence_only=True)[0] == PASS
