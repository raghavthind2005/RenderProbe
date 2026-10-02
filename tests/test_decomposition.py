"""Perception-report condition + three-way decomposition (docs/methodology.md).

Covers: the ReportSpec contract, the runner's `report` condition, relational's
perception_report, the mock's honest report answer, the three demo models landing in
the three buckets end-to-end, and the classifier as a pure function.
"""
from __future__ import annotations

from renderprobe.analysis.decomposition import PLUGIN as DECOMP
from renderprobe.analysis.decomposition import _acc, _classify, _per_scene_counts
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.core.schema import CONDITIONS, ReportSpec, Result
from tests.fixtures import register
from tests.fixtures.probe_relational import PLUGIN as REL_PROBE
from tests.fixtures.scene_relational import PLUGIN as REL_SCENE


def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _scene(seed=3, n_objects=6, hop_depth=2):
    return REL_SCENE.generate({"n_objects": n_objects, "hop_depth": hop_depth}, seed)


# ---------------------------------------------------------------------------
# Contract + probe
# ---------------------------------------------------------------------------

def test_report_is_a_condition():
    assert "report" in CONDITIONS


def test_relational_perception_report_spec():
    s = _scene()
    spec = REL_PROBE.perception_report(s)
    assert isinstance(spec, ReportSpec)
    # GT is the target color == the task answer, and it's a valid color word
    assert spec.ground_truth == s.ground_truth["answer"]
    # target identified by a magenta highlight ring on a report-specific image,
    # NOT by pixel coordinates (which VLMs localize poorly)
    assert "magenta" in spec.question.lower()
    assert "pixel position" not in spec.question.lower()
    assert spec.images is not None and len(spec.images) == 1
    assert spec.images[0].size == s.images[0].size
    # spec reuses the probe's color parse/score
    assert spec.parse("the answer is " + spec.ground_truth) == spec.ground_truth
    assert spec.score(spec.ground_truth, spec.ground_truth)[0] is True


def test_all_probes_now_ship_a_report():
    # Every built-in probe now exposes a perception_report (the 3-way split is no longer
    # relational-only). spatial_relation used to have none; it must now return a ReportSpec.
    from renderprobe.core.schema import ReportSpec
    reg = _registry()
    probe = reg.get_probe("spatial_relation")
    assert hasattr(probe, "perception_report")
    gen = reg.get_scene("spatial_config")
    params = {k: v["default"] for k, v in gen.params_schema.items()}
    spec = probe.perception_report(gen.generate(params, 0))
    assert isinstance(spec, ReportSpec)
    ok, _, _ = spec.score(spec.ground_truth, spec.ground_truth)
    assert ok


# ---------------------------------------------------------------------------
# Runner produces the report condition, scored independently
# ---------------------------------------------------------------------------

def _run(models, seeds=24):
    cfg = ExperimentConfig(
        scene_generator="relational",
        scene_params={"n_objects": [6], "hop_depth": [2]},
        seeds=list(range(seeds)),
        probe="relational",
        models=models,
        analyzers=["decomposition"],
    )
    return run_experiment_collect(cfg, _registry())


def test_runner_emits_report_condition():
    _, results, _ = _run(["mock"])
    conds = {r.condition for r in results}
    assert {"full", "oracle", "report"} <= conds
    assert "blind" not in conds      # opt-in; decomposition never needed it


def test_mock_report_is_honest_and_accurate():
    """Exact mock perceives the color at the queried location almost perfectly."""
    _, results, _ = _run(["mock"])
    rep = [r for r in results if r.condition == "report" and r.model == "mock"]
    assert rep
    acc = sum(r.correct for r in rep) / len(rep)
    assert acc >= 0.9, f"report accuracy too low: {acc}"


# ---------------------------------------------------------------------------
# Three buckets, end-to-end
# ---------------------------------------------------------------------------

def test_exact_mock_no_bottleneck():
    _, _, reps = _run(["mock"])
    cell = _cell(reps, "mock")
    assert cell["bottleneck"] == "none"


def test_arbitration_mock_is_grounding_limited():
    """Perceives fine (report high) + reasons over text (oracle high) but fails the
    image task (full low) -> the failure the two-way framing would mislabel."""
    _, _, reps = _run(["mock_arbitration"])
    cell = _cell(reps, "mock_arbitration")
    assert cell["bottleneck"] == "grounding/integration-limited"
    assert cell["acc_report"] >= 0.85 and cell["acc_full"] < 0.5


def test_noisy_mock_is_encoding_limited():
    """Perception degraded to the point where reporting the primitive goes no better
    than doing the task - which is what encoding-limited means, and what the fixture
    has to actually produce for this branch to be covered at all."""
    _, _, reps = _run(["mock_noisy"])
    cell = _cell(reps, "mock_noisy")
    assert cell["bottleneck"] == "encoding-limited"
    assert cell["acc_report"] <= cell["acc_full"] + 0.10
    assert cell["recovery"] > 0.10       # the oracle still rescues it


def _cell(reports, model):
    return next(r for r in reports if r.name == "decomposition").payload["by_model"][model]


# ---------------------------------------------------------------------------
# Classifier - pure function, all branches
# ---------------------------------------------------------------------------

def test_classify_branches():
    assert _classify(None, 0.9, 0.9)[0] == "insufficient-data"
    assert _classify(0.95, 1.0, 1.0)[0] == "none"                 # solved
    assert _classify(0.2, 0.05, None)[0] == "not-vision-limited"  # image beats text
    assert _classify(0.5, 0.5, None)[0] == "reasoning-limited"    # recovery ~0
    assert _classify(0.2, 0.9, None)[0] == "vision-pathway-limited"  # no report
    assert _classify(0.2, 0.9, 0.95)[0] == "grounding/integration-limited"
    assert _classify(0.2, 0.9, 0.30)[0] == "encoding-limited"


def test_the_split_asks_whether_perception_explains_the_failure():
    """Not whether acc_report clears a fixed bar. Measured live, gemma-3-27b reported
    the primitive on 0.83 of the road maps and solved 0.33 of them: a 0.85 cut called
    that "cannot even report the primitive", which is false about a model getting it
    five times in six. What separates the two cases is whether reporting goes BETTER
    than doing."""
    grounding = "grounding/integration-limited"
    # the four real model-scene pairs this rule was built against
    assert _classify(0.3333, 0.6111, 0.8333)[0] == grounding        # route, gemma
    assert _classify(0.1111, 0.2308, 0.7778)[0] == grounding        # route, llama
    assert _classify(0.1667, 0.3889, 0.0)[0] == "encoding-limited"  # polycube, gemma
    assert _classify(0.1111, 0.2778, 0.0)[0] == "encoding-limited"  # polycube, llama
    # a report far above the task is grounding however far below the old ceiling it is
    assert _classify(0.10, 0.90, 0.40)[0] == grounding
    # and a near-perfect report stays grounding even with the task close behind
    assert _classify(0.80, 0.95, 0.99)[0] == grounding


def test_the_explanation_never_contradicts_the_numbers_it_quotes():
    """The bug this rule replaced shipped the sentence "cannot even report the
    primitive (acc_report=0.83)"."""
    _, why = _classify(0.33, 0.61, 0.83)
    assert "0.83" in why and "cannot" not in why.lower()
    _, why = _classify(0.17, 0.39, 0.0)
    assert "0.00" in why and "0.17" in why


def test_decomposition_gates_on_conditions():
    assert DECOMP.requires_conditions == ["full", "oracle"]
    assert DECOMP.requires_varying_factor is False


# ---------------------------------------------------------------------------
# Verdict stability - a threshold cascade is a step function, so a point estimate
# near a cutoff can flip on one item. These pin that the report says so.
# ---------------------------------------------------------------------------

def _row(scene_id, condition, correct, model="m"):
    return Result(
        scene_id=scene_id, generator="g", factors={}, probe="relational",
        model=model, condition=condition, raw="", pred=None, gt=None,
        correct=correct, score=1.0 if correct else 0.0, error=None,
        confidence=None, error_flag=None,
    )


def _rows(n, k_full, k_oracle, k_report):
    """n scenes; the first k_* of each condition are correct."""
    out = []
    for i in range(n):
        out.append(_row(f"s{i}", "full", i < k_full))
        out.append(_row(f"s{i}", "oracle", i < k_oracle))
        out.append(_row(f"s{i}", "report", i < k_report))
    return out


def test_per_scene_counts_pool_to_the_same_accuracy_as_acc():
    # The bootstrap aggregates per scene while the point estimate uses _acc;
    # this pins the two against drift.
    rs = _rows(20, k_full=4, k_oracle=20, k_report=17)
    per_scene = _per_scene_counts(rs, "m")
    for cond in ("full", "oracle", "report"):
        k = sum(c.get(cond, (0, 0))[0] for c in per_scene)
        n = sum(c.get(cond, (0, 0))[1] for c in per_scene)
        assert k / n == _acc(rs, "m", cond)[0]


def test_decisive_verdict_is_stable():
    # acc_report = 1.00, far above the 0.85 cutoff: nothing to flip.
    row = DECOMP.analyze(_rows(20, k_full=4, k_oracle=20, k_report=20))
    row = row.payload["by_model"]["m"]
    assert row["bottleneck"] == "grounding/integration-limited"
    assert row["bottleneck_stability"] >= 0.95
    assert row["bottleneck_confident"] is True


def test_borderline_verdict_is_flagged_not_asserted():
    # acc_full = 4/20 = 0.20, acc_report = 6/20 = 0.30. The margin between them is
    # exactly the materiality threshold, so the verdict is encoding by a hair and a
    # single item the other way makes it grounding. The payload has to admit that.
    row = DECOMP.analyze(_rows(20, k_full=4, k_oracle=20, k_report=6))
    row = row.payload["by_model"]["m"]
    assert row["bottleneck"] == "encoding-limited"
    assert row["bottleneck_stability"] < 0.95
    assert row["bottleneck_confident"] is False
    assert "grounding/integration-limited" in row["bottleneck_alternatives"]


def test_stability_is_deterministic():
    rs = _rows(20, k_full=4, k_oracle=20, k_report=17)
    a = DECOMP.analyze(rs).payload["by_model"]["m"]["bottleneck_stability"]
    b = DECOMP.analyze(rs).payload["by_model"]["m"]["bottleneck_stability"]
    assert a == b


def test_alternatives_exclude_the_reported_verdict():
    row = DECOMP.analyze(_rows(20, k_full=4, k_oracle=20, k_report=17))
    row = row.payload["by_model"]["m"]
    assert row["bottleneck"] not in row["bottleneck_alternatives"]
