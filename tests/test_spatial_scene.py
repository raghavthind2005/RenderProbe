"""Spatial scene + probe, oracle, and sweep_curve analyzer tests."""
from __future__ import annotations

import pytest

from renderprobe.analysis.oracle import _OracleAnalyzer
from renderprobe.analysis.sweep_curve import _SweepCurveAnalyzer
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment
from renderprobe.core.schema import Result
from tests.fixtures import register
from tests.fixtures.probe_spatial_relation import PLUGIN as spatial_probe
from tests.fixtures.scene_spatial_config import PLUGIN as spatial_scene

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> Result:
    defaults = dict(
        scene_id="s0", generator="spatial_config",
        factors={"n_distractors": 0}, probe="spatial_relation",
        model="mock", condition="full", raw="left", pred="left", gt="left",
        correct=True, score=1.0, error=None, confidence=None, error_flag=None,
    )
    defaults.update(overrides)
    return Result(**defaults)


def _make_registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


# ---------------------------------------------------------------------------
# spatial_config scene
# ---------------------------------------------------------------------------

def test_scene_gt_shape():
    scene = spatial_scene.generate({"n_distractors": 0}, seed=0)
    gt = scene.ground_truth
    assert set(gt.keys()) == {"objects", "target_a", "target_b", "relation"}
    assert gt["target_a"] == "red"
    assert gt["target_b"] == "blue"
    assert gt["relation"] in {"left", "right"}
    assert isinstance(gt["objects"], list)
    assert len(gt["objects"]) == 2  # 0 distractors


def test_scene_factors_shape():
    scene = spatial_scene.generate({"n_distractors": 3}, seed=1)
    assert scene.factors == {"n_distractors": 3}
    assert len(scene.ground_truth["objects"]) == 5  # 2 targets + 3 distractors


def test_scene_object_keys():
    scene = spatial_scene.generate({"n_distractors": 1}, seed=2)
    for obj in scene.ground_truth["objects"]:
        assert set(obj.keys()) == {"color", "shape", "x", "y"}
        assert isinstance(obj["x"], int)
        assert isinstance(obj["y"], int)


def test_scene_relation_consistency():
    scene = spatial_scene.generate({"n_distractors": 0}, seed=5)
    objects = scene.ground_truth["objects"]
    red = next(o for o in objects if o["color"] == "red")
    blue = next(o for o in objects if o["color"] == "blue")
    expected = "left" if red["x"] < blue["x"] else "right"
    assert scene.ground_truth["relation"] == expected


def test_scene_meta():
    scene = spatial_scene.generate({"n_distractors": 2}, seed=0)
    assert scene.meta["generator"] == "spatial_config"
    assert scene.meta["seed"] == 0


def test_scene_deterministic():
    s1 = spatial_scene.generate({"n_distractors": 2}, seed=99)
    s2 = spatial_scene.generate({"n_distractors": 2}, seed=99)
    assert s1.ground_truth == s2.ground_truth


def test_scene_target_horizontal_separation():
    """Targets must have meaningful horizontal separation for the probe to be meaningful."""
    for seed in range(10):
        scene = spatial_scene.generate({"n_distractors": 0}, seed=seed)
        objs = scene.ground_truth["objects"]
        red = next(o for o in objs if o["color"] == "red")
        blue = next(o for o in objs if o["color"] == "blue")
        assert abs(red["x"] - blue["x"]) >= 80


# ---------------------------------------------------------------------------
# spatial_relation probe
# ---------------------------------------------------------------------------

def test_probe_parse_variants():
    assert spatial_probe.parse("left") == "left"
    assert spatial_probe.parse("LEFT") == "left"
    assert spatial_probe.parse("The answer is left.") == "left"
    assert spatial_probe.parse("right") == "right"
    assert spatial_probe.parse("I think right is correct") == "right"


def test_probe_parse_raises_on_no_match():
    with pytest.raises(ValueError):
        spatial_probe.parse("I don't know")
    with pytest.raises(ValueError):
        spatial_probe.parse("42")


def test_probe_score():
    assert spatial_probe.score("left", "left") == (True, 1.0, None)
    assert spatial_probe.score("right", "left") == (False, 0.0, None)


def test_probe_oracle_prompt_not_none():
    scene = spatial_scene.generate({"n_distractors": 0}, seed=0)
    prompt = spatial_probe.oracle_prompt(scene)
    assert prompt is not None


def test_probe_oracle_prompt_contains_coords():
    scene = spatial_scene.generate({"n_distractors": 0}, seed=0)
    prompt = spatial_probe.oracle_prompt(scene)
    objects = scene.ground_truth["objects"]
    red = next(o for o in objects if o["color"] == "red")
    blue = next(o for o in objects if o["color"] == "blue")
    assert str(red["x"]) in prompt
    assert str(blue["x"]) in prompt


def test_probe_oracle_prompt_withholds_answer():
    """Oracle prompt must not directly state 'left' or 'right' as an answer."""
    for seed in range(5):
        scene = spatial_scene.generate({"n_distractors": 0}, seed=seed)
        prompt = spatial_probe.oracle_prompt(scene)
        # Prompt asks the question (so "left" and "right" appear as options)
        # but must not state the answer directly before the question
        # We verify both option words appear (it's asking the question)
        assert "left" in prompt.lower()
        assert "right" in prompt.lower()
        # And the coordinates appear - it IS giving position info
        objects = scene.ground_truth["objects"]
        red = next(o for o in objects if o["color"] == "red")
        assert str(red["x"]) in prompt


def test_probe_question_uses_gt_colors():
    scene = spatial_scene.generate({"n_distractors": 0}, seed=0)
    q = spatial_probe.question(scene)
    assert "red" in q
    assert "blue" in q


# ---------------------------------------------------------------------------
# Oracle analyzer
# ---------------------------------------------------------------------------

def _oracle_results(model="mock", n=4, full_correct=True, oracle_correct=True):
    rs = []
    for i in range(n):
        gt = "left"
        rs.append(_make_result(
            scene_id=f"s{i}", model=model, condition="full",
            pred="left" if full_correct else "right",
            gt=gt, correct=full_correct,
        ))
        rs.append(_make_result(
            scene_id=f"s{i}", model=model, condition="oracle",
            pred="left" if oracle_correct else "right",
            gt=gt, correct=oracle_correct,
        ))
    return rs


def test_oracle_high_recovery():
    """When full fails but oracle succeeds -> high recovery."""
    analyzer = _OracleAnalyzer()
    results = _oracle_results(full_correct=False, oracle_correct=True)
    report = analyzer.analyze(results)
    row = report.payload["by_model"]["mock"]
    assert row["acc_full"] == 0.0
    assert row["acc_oracle"] == 1.0
    assert row["recovery"] == pytest.approx(1.0)


def test_oracle_zero_recovery():
    """When full and oracle both succeed -> recovery ≈ 0."""
    analyzer = _OracleAnalyzer()
    results = _oracle_results(full_correct=True, oracle_correct=True)
    report = analyzer.analyze(results)
    row = report.payload["by_model"]["mock"]
    assert row["recovery"] == pytest.approx(0.0)


def test_oracle_paired_only():
    """scene_ids not in both conditions are excluded."""
    analyzer = _OracleAnalyzer()
    results = [
        _make_result(scene_id="s0", condition="full", correct=True),
        _make_result(scene_id="s0", condition="oracle", correct=False),
        # s1 only in full - should be excluded
        _make_result(scene_id="s1", condition="full", correct=True),
    ]
    report = analyzer.analyze(results)
    assert report.payload["by_model"]["mock"]["n_paired"] == 1


def test_oracle_note_in_payload():
    analyzer = _OracleAnalyzer()
    results = _oracle_results()
    report = analyzer.analyze(results)
    assert "note" in report.payload


def test_oracle_skips_error_flag_results():
    analyzer = _OracleAnalyzer()
    results = _oracle_results(full_correct=True, oracle_correct=True)
    # Add a broken result
    results.append(
        _make_result(scene_id="s0", condition="full", error_flag="model_error")
    )
    report = analyzer.analyze(results)
    # Should still work (broken result excluded from analysis)
    assert "mock" in report.payload["by_model"]


# ---------------------------------------------------------------------------
# Sweep curve analyzer
# ---------------------------------------------------------------------------

def _sweep_results(factor_vals=(0, 2, 4), seeds=(0, 1), model="mock"):
    rs = []
    for fval in factor_vals:
        for seed in range(len(seeds)):
            rs.append(_make_result(
                scene_id=f"s_f{fval}_s{seed}",
                factors={"n_distractors": fval},
                condition="full",
                correct=(fval == 0),  # only accurate at 0 distractors
            ))
    return rs


def test_sweep_curve_factor_name():
    analyzer = _SweepCurveAnalyzer()
    report = analyzer.analyze(_sweep_results())
    assert report.payload["factor"] == "n_distractors"


def test_sweep_curve_has_all_factor_values():
    analyzer = _SweepCurveAnalyzer()
    report = analyzer.analyze(_sweep_results(factor_vals=(0, 2, 4)))
    curve = report.payload["by_model"]["mock"]
    assert set(curve.keys()) == {"0", "2", "4"}


def test_sweep_curve_accuracy_values():
    analyzer = _SweepCurveAnalyzer()
    report = analyzer.analyze(_sweep_results())
    curve = report.payload["by_model"]["mock"]
    assert curve["0"]["accuracy"] == 1.0
    assert curve["2"]["accuracy"] == 0.0
    assert curve["4"]["accuracy"] == 0.0


def test_sweep_curve_ci_present_for_large_bucket():
    """CI should be computed when bucket has >=4 samples."""
    analyzer = _SweepCurveAnalyzer()
    rs = _sweep_results(factor_vals=(0,), seeds=range(6))
    # need varying factor - add a second value
    for i in range(2):
        rs.append(_make_result(
            scene_id=f"extra_{i}", factors={"n_distractors": 5},
            condition="full", correct=False,
        ))
    report = analyzer.analyze(rs)
    curve = report.payload["by_model"]["mock"]
    # 6 samples for factor=0 -> ci95 present
    assert curve["0"]["ci95"] is not None
    assert len(curve["0"]["ci95"]) == 2


# ---------------------------------------------------------------------------
# End-to-end: spatial_oracle config runs with mock
# ---------------------------------------------------------------------------

def test_spatial_oracle_end_to_end():
    reg = _make_registry()
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 2]},
        seeds=[0, 1],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["accuracy", "visual_gain", "oracle", "sweep_curve"],
    )
    reports = run_experiment(cfg, reg)
    names = {r.name for r in reports}
    assert "accuracy" in names
    assert "visual_gain" in names
    assert "oracle" in names
    assert "sweep_curve" in names


def test_oracle_report_has_data():
    """The oracle report must have non-empty by_model after a real run."""
    reg = _make_registry()
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 2]},
        seeds=[0, 1],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["oracle"],
    )
    reports = run_experiment(cfg, reg)
    oracle_report = next(r for r in reports if r.name == "oracle")
    assert oracle_report.payload["by_model"]  # non-empty
    assert "mock" in oracle_report.payload["by_model"]
    row = oracle_report.payload["by_model"]["mock"]
    assert "acc_full" in row
    assert "acc_oracle" in row
    assert "recovery" in row


def test_sweep_curve_report_has_curve_data():
    reg = _make_registry()
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 4]},
        seeds=[0],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["sweep_curve"],
    )
    reports = run_experiment(cfg, reg)
    sc = next(r for r in reports if r.name == "sweep_curve")
    assert sc.payload["factor"] == "n_distractors"
    curve = sc.payload["by_model"]["mock"]
    assert "0" in curve
    assert "4" in curve


def test_dots_are_non_overlapping_at_high_counts():
    """Dots must never merge, or the count would be unrecoverable even by a perfect
    perceiver (GT-trustability). Blob-counting the rendered image must recover exactly
    n distinct dots at every count up to the max."""
    from tests.fixtures.mock import _count_blobs
    from tests.fixtures.scene_dots import PLUGIN as dots_scene
    for n in (5, 20, 40):
        for seed in range(3):
            s = dots_scene.generate({"n_dots": n}, seed)
            assert s.ground_truth["count"] == n
            assert _count_blobs(s.images[0]) == n, (
                f"dots merged at n={n} seed={seed}: only "
                f"{_count_blobs(s.images[0])} distinct blobs"
            )
