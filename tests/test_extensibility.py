"""Extensibility engine - out-of-tree plugin loading + `validate` honesty checks.

Proves a scientist can keep scenes/probes/metrics in their OWN directory, load them
without editing the package, and have `validate` catch the mistakes that would silently
corrupt the perception/reasoning diagnostic (non-reproducible scenes, a constant answer,
a broken scoring function).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from renderprobe.core.registry import RegistrationError, Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.core.validate import FAIL, WARN, all_ok, validate_path
from tests.fixtures import register

SRC = Path(__file__).resolve().parents[1] / "src" / "renderprobe"
# The offline fixtures moved out of the package; they are still validated, because
# the test suite leans on them and a broken fixture would be a silent hole.
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _probe_path(name: str) -> Path:
    shipped = SRC / "probes" / f"{name}.py"
    return shipped if shipped.exists() else FIXTURES / f"probe_{name}.py"


def _reg() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


GOOD_SCENE = '''
    import random as _random
    from PIL import Image
    from renderprobe.core.schema import Scene
    class S:
        name = "t_good_scene"
        params_schema = {"n": {"type": "int", "min": 2, "max": 8, "default": 4}}
        def generate(self, params, seed):
            rng = _random.Random(seed)
            n = int(params["n"])
            img = Image.new("RGB", (64, 64), "white")
            _ = rng.random()
            return Scene(id=f"g{seed}", images=[img],
                         ground_truth={"count": n}, factors={"n": n}, meta={})
    PLUGIN = S()
'''

NONDET_SCENE = '''
    import random as _random
    from PIL import Image
    from renderprobe.core.schema import Scene
    class S:
        name = "t_nondet"
        params_schema = {"n": {"type": "int", "min": 4, "max": 12, "default": 8}}
        def generate(self, params, seed):
            n = _random.randint(4, int(params["n"]))   # BUG: global RNG, ignores seed
            return Scene(id=f"n{seed}", images=[Image.new("RGB", (64, 64), "white")],
                         ground_truth={"count": n}, factors={"n": n}, meta={})
    PLUGIN = S()
'''

CONST_SCENE = '''
    from PIL import Image
    from renderprobe.core.schema import Scene
    class S:
        name = "t_const"
        params_schema = {}
        def generate(self, params, seed):
            return Scene(id=f"c{seed}", images=[Image.new("RGB", (64, 64), "white")],
                         ground_truth={"count": 4}, factors={}, meta={})
    PLUGIN = S()
'''

BAD_SCORE_PROBE = '''
    import re
    from renderprobe.core.schema import Scene
    class P:
        name = "t_bad_probe"
        answer_type = "numeric"
        requires_gt_fields = ["count"]
        def question(self, s): return "how many?"
        def ground_truth(self, s): return int(s.ground_truth["count"])
        def parse(self, raw):
            m = re.search(r"\\d+", raw); return int(m.group()) if m else 0
        def score(self, pred, gt): return pred > gt, 0.0, 0.0   # BUG: gt==gt scores False
        def oracle_prompt(self, s): return None
    PLUGIN = P()
'''

NOT_A_PLUGIN = '''
    PLUGIN = object()
'''

# A scene supplying the `objects` list the synthetic probes below consume. It is written
# into the probe's own tmp_path and validated alongside it, so these tests exercise
# validate's logic rather than depending on whichever scenes happen to ship. They used to
# pair with the built-in `relational` scene, which has since moved into the fixtures.
OBJECTS_SCENE = '''
    import random as _random
    from PIL import Image, ImageDraw
    from renderprobe.core.schema import Scene
    _COLORS = ("red", "blue", "green", "orange", "purple", "yellow")
    class S:
        name = "t_objects"
        params_schema = {"n": {"type": "int", "min": 2, "max": 6, "default": 3}}
        def generate(self, params, seed):
            rng = _random.Random(seed)
            n = int(params["n"])
            img = Image.new("RGB", (120, 120), "white")
            d = ImageDraw.Draw(img)
            # Distinct radii so "the largest" is never a tie, and distinct colors so the
            # answer varies with the seed and the balance check has something to pass.
            radii = rng.sample(range(6, 22), n)
            colors = rng.sample(_COLORS, n)
            objs = []
            for i, (r, c) in enumerate(zip(radii, colors)):
                x, y = 22 + (i % 3) * 38, 30 + (i // 3) * 46
                d.ellipse([x - r, y - r, x + r, y + r], fill=c)
                objs.append({"color": c, "x": x, "y": y, "r": r})
            return Scene(id=f"o{seed}", images=[img],
                         ground_truth={"objects": objs}, factors={"n": n}, meta={})
    PLUGIN = S()
'''


# A counting probe for the scenes above, written beside them for the same reason: these
# tests are about validate's logic, not about which probes happen to ship. It exposes no
# perception_report, which is what makes the two-way-only warning reachable.
COUNT_PROBE = '''
    import re
    from renderprobe.core.schema import Scene
    class P:
        name = "t_count"
        answer_type = "numeric"
        requires_gt_fields = ["count"]
        def question(self, s): return "how many?"
        def ground_truth(self, s): return int(s.ground_truth["count"])
        def parse(self, raw):
            m = re.search(r"\\d+", raw or "")
            return int(m.group()) if m else None
        def score(self, pred, gt):
            ok = pred is not None and int(pred) == int(gt)
            return ok, 1.0 if ok else 0.0, None
        def oracle_prompt(self, s): return None
    PLUGIN = P()
'''


def _with_count_probe(tmp_path, name, body):
    """Write a scene next to a counting probe that consumes it, and return the directory
    so validate pairs the two."""
    _write(tmp_path, "t_count_probe.py", COUNT_PROBE)
    _write(tmp_path, name, body)
    return tmp_path


def _with_count_scene(tmp_path, name, body):
    """The mirror of the above: a probe under test, written beside a scene that feeds
    it."""
    _write(tmp_path, "t_count_scene.py", GOOD_SCENE)
    _write(tmp_path, name, body)
    return tmp_path


def _with_objects_scene(tmp_path, name, body):
    """Write a synthetic probe next to a scene that feeds it, and return the directory
    so validate pairs the two."""
    _write(tmp_path, "t_objects_scene.py", OBJECTS_SCENE)
    _write(tmp_path, name, body)
    return tmp_path


# Probes with a perception_report, exercised against the scene above (an `objects` list
# with color/x/y/r). Only the header differs.
_REPORT_PROBE = '''
    from PIL import ImageDraw
    from renderprobe.core.schema import ReportSpec, Scene
    class P:
        name = "t_report"
        answer_type = "categorical"
        requires_gt_fields = ["objects"]
        def question(self, s): return "color of the largest object?"
        def _big(self, s): return max(s.ground_truth["objects"], key=lambda o: o.get("r", 0))
        def ground_truth(self, s): return self._big(s)["color"]
        def parse(self, raw):
            raw = raw.lower()
            for c in ("red","blue","green","orange","purple","yellow","pink","brown","gray","cyan"):
                if c in raw: return c
            raise ValueError(raw)
        def score(self, pred, gt): return pred == gt, float(pred == gt), None
        def oracle_prompt(self, s): return None
        def perception_report(self, s):
            t = self._big(s); img = s.images[0].copy()
            {ring}
            return ReportSpec(question="color of the circled object?",
                              ground_truth=t["color"], parse=self.parse,
                              score={score}, images={images})
    PLUGIN = P()
'''

_DEFAULT_RING = ("ImageDraw.Draw(img).ellipse("
                 "[t['x']-9,t['y']-9,t['x']+9,t['y']+9], outline=(230,0,230), width=3)")


def _report_probe(ring=_DEFAULT_RING, score="self.score", images="[img]"):
    return _REPORT_PROBE.format(ring=ring, score=score, images=images)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_templates_are_invisible_to_autodiscover():
    r = _reg()
    assert "template_scene" not in r.list_scenes()
    assert "template_probe" not in r.list_probes()
    assert "template_model" not in r.list_models()
    assert "template_metric" not in r.list_analyzers()


def test_load_external_single_file_detects_family():
    r = _reg()
    added = r.load_external(SRC / "scenes" / "_template.py")
    assert added == ["scene:template_scene"]
    assert "template_scene" in r.list_scenes()


def test_load_external_directory(tmp_path):
    _write(tmp_path, "good.py", GOOD_SCENE)
    _write(tmp_path, "probe.py", BAD_SCORE_PROBE)
    r = _reg()
    added = set(r.load_external(tmp_path))
    assert added == {"scene:t_good_scene", "probe:t_bad_probe"}


def test_load_external_skips_underscore_in_dir(tmp_path):
    _write(tmp_path, "good.py", GOOD_SCENE)
    _write(tmp_path, "_ignore.py", CONST_SCENE)
    r = _reg()
    added = r.load_external(tmp_path)
    assert added == ["scene:t_good_scene"]


def test_non_conformant_plugin_raises(tmp_path):
    p = _write(tmp_path, "bad.py", NOT_A_PLUGIN)
    r = _reg()
    with pytest.raises(RegistrationError, match="satisfies no RenderProbe protocol"):
        r.load_external(p)


def test_missing_path_raises():
    r = _reg()
    with pytest.raises(RegistrationError, match="not found"):
        r.load_external("/no/such/path_xyz")


# ---------------------------------------------------------------------------
# validate honesty checks
# ---------------------------------------------------------------------------

def _statuses(reports, check_name, plugin=None):
    """Statuses for a check, across plugins or within one.

    A probe shipping several perception reports names its checks
    `perception-report[cells]` and so on, so the match is on the base name.
    """
    def matches(name):
        return name == check_name or name.startswith(f"{check_name}[")

    return [c.status for pv in reports if plugin is None or pv.name == plugin
            for c in pv.checks if matches(c.name)]


def _fixture_reports(seeds):
    """Validate the whole offline rig in one pass.

    The fixtures are a directory of scenes and probes that pair with each other, so they
    are validated together: a probe on its own has nothing to be checked against now that
    it no longer sits beside the shipped scenes.
    """
    return validate_path(FIXTURES, seeds=seeds)


def test_validate_good_scene_passes(tmp_path):
    p = _with_count_probe(tmp_path, "good_scene.py", GOOD_SCENE)
    reports = validate_path(p, seeds=6)
    assert all_ok(reports)
    assert PASS_in(reports, "gt-roundtrip")


def PASS_in(reports, name):
    return any(c.name == name and c.status == "pass" for pv in reports for c in pv.checks)


def test_validate_flags_nondeterminism(tmp_path):
    p = _write(tmp_path, "nondet_scene.py", NONDET_SCENE)
    reports = validate_path(p, seeds=6)
    assert not all_ok(reports)
    assert FAIL in _statuses(reports, "determinism")


def test_validate_warns_constant_answer(tmp_path):
    p = _with_count_probe(tmp_path, "const_scene.py", CONST_SCENE)
    reports = validate_path(p, seeds=6)
    # a warning, not a hard failure
    assert all_ok(reports)
    assert WARN in _statuses(reports, "answer-balance")


def test_validate_fails_broken_scoring(tmp_path):
    p = _with_count_scene(tmp_path, "bad_probe.py", BAD_SCORE_PROBE)
    reports = validate_path(p, seeds=6)
    assert not all_ok(reports)
    assert FAIL in _statuses(reports, "gt-roundtrip")


def test_validate_all_templates_pass():
    for fam in ("scenes", "probes", "models", "analysis"):
        reports = validate_path(SRC / fam / "_template.py", seeds=6)
        assert all_ok(reports), f"{fam}/_template.py should validate clean"


# ---------------------------------------------------------------------------
# Perception-report validation (the three-way readiness a scientist must verify)
# ---------------------------------------------------------------------------

def test_report_valid_with_handle(tmp_path):
    p = _with_objects_scene(tmp_path, "rep_ok.py", _report_probe())
    reports = validate_path(p, seeds=4)
    assert all_ok(reports)
    detail = " ".join(c.detail for pv in reports for c in pv.checks
                      if c.name == "perception-report")
    assert "VALID" in detail and "handle" in detail


def test_report_warns_when_pointing_without_handle_image(tmp_path):
    # question says "circled" but ships no highlighted image -> localization confound
    p = _with_objects_scene(tmp_path, "rep_nohandle.py",
                            _report_probe(ring="", images="None"))
    reports = validate_path(p, seeds=4)
    assert all_ok(reports)  # a warning, not a hard failure
    assert WARN in _statuses(reports, "perception-report")


def test_report_fails_on_broken_scorer(tmp_path):
    # a report scorer that never returns correct -> caught before wasting API calls
    p = _with_objects_scene(tmp_path, "rep_bad.py",
               _report_probe(score="lambda a, b: (a != b, 0.0, None)"))
    reports = validate_path(p, seeds=4)
    assert not all_ok(reports)
    assert FAIL in _statuses(reports, "perception-report")


def test_every_probe_ships_a_valid_perception_report():
    """Every probe exposes a perception_report, shipped or fixture, and each validates
    clean on that check. Without it the three-way split degrades to two-way."""
    for probe in ("route", "polycube"):
        statuses = _statuses(validate_path(_probe_path(probe), seeds=4),
                             "perception-report")
        assert statuses, f"{probe}: perception-report check did not run"
        assert FAIL not in statuses and WARN not in statuses, \
            f"{probe} perception-report not clean: {statuses}"

    rig = _fixture_reports(seeds=4)
    for probe in ("relational", "count", "spatial_relation", "pathfinding",
                  "identity_under_occlusion"):
        statuses = _statuses(rig, "perception-report", plugin=probe)
        assert statuses, f"{probe}: perception-report check did not run"
        assert FAIL not in statuses and WARN not in statuses, \
            f"{probe} perception-report not clean: {statuses}"


# ---------------------------------------------------------------------------
# Oracle information-completeness certificate
# ---------------------------------------------------------------------------

_ORACLE_PROBE = '''
    import re
    from renderprobe.core.schema import Scene
    class P:
        name = "t_oracle"
        answer_type = "categorical"
        requires_gt_fields = ["objects"]
        def _big(self, s): return max(s.ground_truth["objects"], key=lambda o: o.get("r", 0))
        def question(self, s): return "color of the largest object?"
        def ground_truth(self, s): return self._big(s)["color"]
        def parse(self, raw):
            raw = raw.lower()
            for c in ("red","blue","green","orange","purple","yellow","pink","brown","gray","cyan"):
                if re.search(rf"\\b{c}\\b", raw): return c
            raise ValueError(raw)
        def score(self, pred, gt): return pred == gt, float(pred == gt), None
        def oracle_prompt(self, s):
            facts = ", ".join(f"{o['color']}({o.get('r',0)})" for o in s.ground_truth["objects"])
            return f"objects: {facts}. largest color?"
        __SOLVER__
    PLUGIN = P()
'''


def _oracle_probe(solver):
    return _ORACLE_PROBE.replace("__SOLVER__", solver)


def test_oracle_completeness_certifies_good_solver(tmp_path):
    p = _with_objects_scene(tmp_path, "orc_ok.py",
               _oracle_probe("def oracle_solver(self, s): return self._big(s)['color']"))
    reports = validate_path(p, seeds=6)
    assert all_ok(reports)
    assert PASS_in(reports, "oracle-completeness")


def test_oracle_completeness_fails_wrong_solver(tmp_path):
    # a solver that ignores the facts and always says "red" disagrees with GT
    p = _with_objects_scene(tmp_path, "orc_bad.py",
               _oracle_probe("def oracle_solver(self, s): return 'red'"))
    reports = validate_path(p, seeds=8)
    assert not all_ok(reports)
    assert FAIL in _statuses(reports, "oracle-completeness")


def test_oracle_unproven_warns_without_solver(tmp_path):
    # has an oracle, no solver, not perception-bound -> unproven warning (not a failure)
    p = _with_objects_scene(tmp_path, "orc_none.py", _oracle_probe(""))
    reports = validate_path(p, seeds=6)
    assert WARN in _statuses(reports, "oracle-completeness")


def test_reasoning_probes_certify_their_oracle():
    for probe in ("route", "polycube"):
        statuses = _statuses(validate_path(_probe_path(probe), seeds=6),
                             "oracle-completeness")
        assert statuses, f"{probe}: oracle-completeness did not run"
        assert FAIL not in statuses and WARN not in statuses, \
            f"{probe} oracle not certified: {statuses}"

    rig = _fixture_reports(seeds=6)
    for probe in ("relational", "spatial_relation", "pathfinding"):
        statuses = _statuses(rig, "oracle-completeness", plugin=probe)
        assert statuses, f"{probe}: oracle-completeness did not run"
        assert FAIL not in statuses and WARN not in statuses, \
            f"{probe} oracle not certified: {statuses}"


def test_builtin_oracle_solvers_match_ground_truth():
    """The rigorous core: each reasoning probe's solver re-derives GT from the stated
    facts on every seed, across the PARAMETER SPACE (not just defaults) - certifying the
    oracle is information-complete everywhere it's used. Infeasible configs are skipped."""
    reg = _reg()
    cases = [
        ("relational", "relational",
         [{"n_objects": 6, "hop_depth": 2, "clutter": 0},
          {"n_objects": 10, "hop_depth": 3, "clutter": 1},
          {"n_objects": 16, "hop_depth": 6, "clutter": 2}]),
        ("spatial_config", "spatial_relation",
         [{"n_distractors": 0}, {"n_distractors": 8}, {"n_distractors": 14}]),
        ("pathfinding", "pathfinding",
         [{"n_nodes": 4, "p_edge": 0.3}, {"n_nodes": 12, "p_edge": 0.4},
          {"n_nodes": 20, "p_edge": 0.5}]),
    ]
    checked = 0
    for scene_name, probe_name, param_sets in cases:
        gen, probe = reg.get_scene(scene_name), reg.get_probe(probe_name)
        for params in param_sets:
            for seed in range(8):
                try:
                    sc = gen.generate(params, seed)
                except Exception:
                    continue   # infeasible config (e.g. chain longer than object count)
                assert probe.oracle_solver(sc) == probe.ground_truth(sc), \
                    f"{probe_name} solver != GT at {params} seed {seed}"
                checked += 1
    assert checked > 60, f"expected a broad sweep, only checked {checked}"


def test_perception_bound_probes_declare_states_answer():
    reg = _reg()
    for probe in ("count", "identity_under_occlusion"):
        assert getattr(reg.get_probe(probe), "oracle_states_answer", False), \
            f"{probe} should declare oracle_states_answer"


def test_report_none_surfaces_a_warning(tmp_path):
    # GOOD_SCENE has no per-dot `dots` field, so the count probe's perception_report
    # returns None for it -> validate should surface a two-way-only warning (still not a
    # hard failure).
    p = _with_count_probe(tmp_path, "good_scene.py", GOOD_SCENE)
    reports = validate_path(p, seeds=4)
    assert WARN in _statuses(reports, "perception-report")


# ---------------------------------------------------------------------------
# End-to-end: an out-of-tree scene runs through the whole pipeline
# ---------------------------------------------------------------------------

def test_external_scene_runs_end_to_end(tmp_path):
    _write(tmp_path, "good_scene.py", GOOD_SCENE)
    r = _reg()
    r.load_external(tmp_path)
    cfg = ExperimentConfig(
        scene_generator="t_good_scene",
        scene_params={"n": [4, 6]},
        seeds=[0, 1],
        probe="count",
        models=["mock"],
        analyzers=["decomposition"],
    )
    scenes, results, reports = run_experiment_collect(cfg, r)
    assert scenes and results
    dec = next(rep for rep in reports if rep.name == "decomposition")
    assert "mock" in dec.payload["by_model"]


# ---------------------------------------------------------------------------
# The on-ramp's promise: what `renderprobe new` hands you must unlock everything
#
# This drifted once already. The scene template advertised pairing with a `count`
# probe that had been moved out of the package, and the probe template required
# `objects` that the scene template never emitted, so scaffolding both gave a pair
# that warned on both sides and skipped every check of the answer.
# ---------------------------------------------------------------------------

def _scaffold_pair(tmp_path):
    """Write both templates out the way `renderprobe new` does."""
    scene = tmp_path / "my_task.py"
    probe = tmp_path / "my_task_probe.py"
    scene.write_text((SRC / "scenes" / "_template.py").read_text()
                     .replace('name = "template_scene"', 'name = "my_task"'))
    probe.write_text((SRC / "probes" / "_template.py").read_text()
                     .replace('name = "template_probe"', 'name = "my_task_probe"'))
    return scene, probe


def test_the_two_templates_pair_with_no_edits(tmp_path):
    _scaffold_pair(tmp_path)
    reports = validate_path(tmp_path)
    assert all_ok(reports)
    warned = [pv.name for pv in reports if pv.warned]
    assert not warned, f"scaffolded pair should validate clean, warned: {warned}"


def test_the_scaffolded_pair_supports_every_condition(tmp_path):
    """oracle and report are what the whole decomposition rests on; a scaffold that
    silently omits either leaves the headline feature unavailable."""
    _scaffold_pair(tmp_path)
    reg = Registry()
    reg.autodiscover()
    reg.load_external(str(tmp_path))
    probe = reg.get_probe("my_task_probe")
    scene = reg.get_scene("my_task").generate({"n_shapes": 5}, 0)

    assert probe.oracle_prompt(scene)
    assert probe.oracle_solver(scene) == probe.ground_truth(scene)
    report = probe.perception_report(scene)
    assert report is not None and report.images, "report must point with a drawn handle"
    assert 0.0 < probe.chance_level(scene) <= 1.0


def test_the_scene_template_refuses_a_scene_with_no_clear_answer(tmp_path):
    """Its own lesson, applied to itself: the largest object must be strictly largest,
    or the ground truth is decided by a pixel nobody can resolve."""
    _scaffold_pair(tmp_path)
    reg = Registry()
    reg.autodiscover()
    reg.load_external(str(tmp_path))
    gen = reg.get_scene("my_task")
    for seed in range(12):
        objects = gen.generate({"n_shapes": 6}, seed).ground_truth["objects"]
        radii = sorted(o["r"] for o in objects)
        assert radii[-1] > radii[-2], f"seed {seed}: tie for largest"


def test_the_scene_template_publishes_more_than_its_probe_reads(tmp_path):
    """`count` rides along unused, which is how a second probe asks a different
    question of the same pixels."""
    _scaffold_pair(tmp_path)
    reg = Registry()
    reg.autodiscover()
    reg.load_external(str(tmp_path))
    gt = reg.get_scene("my_task").generate({"n_shapes": 5}, 0).ground_truth
    assert set(reg.get_probe("my_task_probe").requires_gt_fields) <= set(gt)
    assert "count" in gt and gt["count"] == len(gt["objects"])
