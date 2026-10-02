"""Calibration analyzer (ECE) + confidence elicitation.

Known-ECE synthetic pairs check the math; graceful degradation checks the null path;
an end-to-end run checks elicitation -> Result.confidence -> non-null ECE, and that the
digit-free confidence instruction doesn't poison the count probe.
"""
from __future__ import annotations

from renderprobe.analysis.calibration import (
    PLUGIN as CAL,
)
from renderprobe.analysis.calibration import (
    ece_equal_mass,
    ece_equal_width,
    save_reliability_diagram,
)
from renderprobe.core.registry import Registry
from renderprobe.core.runner import (
    ExperimentConfig,
    _parse_confidence,
    run_experiment_collect,
)
from renderprobe.core.schema import Result
from tests.fixtures import register


def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _res(condition="full", correct=True, confidence=None, model="m"):
    return Result(
        scene_id="s", generator="g", factors={}, probe="count", model=model,
        condition=condition, raw="", pred=1, gt=1, correct=correct, score=1.0,
        error=None, confidence=confidence, error_flag=None,
    )


# ---------------------------------------------------------------------------
# ECE math - known values (acceptance)
# ---------------------------------------------------------------------------

def test_ece_all_confident_but_60pct_correct():
    # conf=1.0 everywhere, 6/10 correct -> both schemes give |0.6 - 1.0| = 0.4
    pairs = [(1.0, True)] * 6 + [(1.0, False)] * 4
    assert ece_equal_width(pairs)[0] == 0.4
    assert ece_equal_mass(pairs)[0] == 0.4


def test_ece_low_confidence_30pct_correct():
    # conf=0.0 everywhere, 3/10 correct -> both give 0.3
    pairs = [(0.0, True)] * 3 + [(0.0, False)] * 7
    assert ece_equal_width(pairs)[0] == 0.3
    assert ece_equal_mass(pairs)[0] == 0.3


def test_ece_perfect_calibration_is_zero():
    # each confidence level's accuracy matches it -> equal-width ECE 0
    pairs = (
        [(0.2, True)] * 2 + [(0.2, False)] * 8    # acc 0.2 at conf 0.2
        + [(0.8, True)] * 8 + [(0.8, False)] * 2  # acc 0.8 at conf 0.8
    )
    ece, rel = ece_equal_width(pairs)
    assert ece == 0.0
    assert len(rel) == 2  # two occupied bins


def test_ece_empty_is_none():
    assert ece_equal_width([]) == (None, [])
    assert ece_equal_mass([]) == (None, [])


def test_parse_confidence():
    assert _parse_confidence("left\nConfidence: 85") == 0.85
    assert _parse_confidence("Confidence is 100%") == 1.0
    assert _parse_confidence("Confidence: 250") == 1.0   # clamped
    assert _parse_confidence("no confidence stated") is None
    assert _parse_confidence("") is None


# ---------------------------------------------------------------------------
# Analyzer: reliability + graceful degradation
# ---------------------------------------------------------------------------

def test_analyzer_computes_ece_per_condition():
    results = [_res("full", correct=(i < 6), confidence=1.0) for i in range(10)]
    report = CAL.analyze(results)
    assert report.payload["schema_version"] == "calibration/1"
    cell = report.payload["by_model"]["m"]["full"]
    assert cell["n"] == 10
    assert cell["ece_equal_width"] == 0.4
    assert cell["reliability_equal_width"]


def test_analyzer_graceful_when_no_confidences():
    results = [_res("full", correct=True, confidence=None) for _ in range(5)]
    cell = CAL.analyze(results).payload["by_model"]["m"]["full"]
    assert cell["n"] == 0
    assert cell["ece_equal_width"] is None
    assert cell["ece_equal_mass"] is None
    assert "confidence" in cell["reason"].lower()


def test_reliability_diagram_png_written(tmp_path):
    results = [_res("full", correct=(i % 2 == 0), confidence=0.9) for i in range(10)]
    report = CAL.analyze(results)
    written = save_reliability_diagram(report, tmp_path)
    assert written and all(p.exists() and p.suffix == ".png" for p in written)


# ---------------------------------------------------------------------------
# End-to-end: elicitation populates confidence; off by default
# ---------------------------------------------------------------------------

def test_elicitation_populates_confidence_and_ece():
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 4]},
        seeds=[0, 1, 2, 3],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["calibration"],
        elicit_confidence=True,
    )
    _, results, reports = run_experiment_collect(cfg, _registry())
    assert any(r.confidence is not None for r in results)
    cell = next(r for r in reports if r.name == "calibration").payload["by_model"]["mock"]["full"]
    assert cell["n"] > 0
    assert cell["ece_equal_width"] is not None


def test_no_elicitation_leaves_confidence_none():
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0]},
        seeds=[0, 1],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["calibration"],
        # elicit_confidence defaults False
    )
    _, results, reports = run_experiment_collect(cfg, _registry())
    assert all(r.confidence is None for r in results)
    cell = next(r for r in reports if r.name == "calibration").payload["by_model"]["mock"]["full"]
    assert cell["ece_equal_width"] is None
    assert cell["reason"] is not None


def test_count_probe_unpoisoned_by_confidence_instruction():
    """The digit-free instruction must not make count read a stray integer:
    the parsed count is identical with and without elicitation."""
    reg = _registry()
    base = dict(
        scene_generator="dots", scene_params={"n_dots": [3, 5]}, seeds=[0, 1, 2],
        probe="count", models=["mock"], analyzers=["accuracy"],
    )
    _, plain, _ = run_experiment_collect(ExperimentConfig(**base), reg)
    _, elic, _ = run_experiment_collect(
        ExperimentConfig(**base, elicit_confidence=True), reg
    )

    def full_preds(results):
        return {r.scene_id: r.pred for r in results if r.condition == "full"}

    assert full_preds(plain) == full_preds(elic)  # instruction injected no stray int
    full_elic = [r for r in elic if r.condition == "full"]
    assert all(r.confidence is not None for r in full_elic)
    assert all(r.confidence is None for r in plain if r.condition == "full")
