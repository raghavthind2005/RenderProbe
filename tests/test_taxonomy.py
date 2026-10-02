"""Failure-taxonomy analyzer tests.

Unit tests cover every determinable category per probe; end-to-end tests show the
taxonomy composing with the oracle condition (perception failures vanish under
oracle, reasoning failures would persist); a UI test confirms the category surfaces
in the inspect tab.
"""
from __future__ import annotations

import pytest

from renderprobe.analysis.taxonomy import PLUGIN as TAX
from renderprobe.analysis.taxonomy import classify_result
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment, run_experiment_collect
from renderprobe.core.schema import Result
from renderprobe.ui import app as UI_APP
from renderprobe.ui import data as D
from tests.fixtures import register


def _res(probe="spatial_relation", condition="full", correct=False, pred="right",
         gt="left", raw="right", error_flag=None, model="m"):
    return Result(
        scene_id="s", generator="g", factors={}, probe=probe, model=model,
        condition=condition, raw=raw, pred=pred, gt=gt, correct=correct,
        score=1.0 if correct else 0.0, error=None, confidence=None,
        error_flag=error_flag,
    )


def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


# ---------------------------------------------------------------------------
# classify_result - every category
# ---------------------------------------------------------------------------

def test_shared_categories():
    assert classify_result(_res(correct=True, pred="left", gt="left")) == "correct"
    assert classify_result(_res(error_flag="model_error", raw="")) == "model_error"
    assert classify_result(_res(error_flag="gen_error", raw="")) == "gen_error"
    assert classify_result(_res(error_flag="parse_fail", pred=None, raw="banana")) == "parse_fail"
    assert classify_result(
        _res(error_flag="parse_fail", pred=None, raw="I cannot tell from this image")
    ) == "refusal"


def test_spatial_relation_inversion():
    assert classify_result(
        _res(probe="spatial_relation", pred="right", gt="left", raw="right")
    ) == "relation_inversion"


@pytest.mark.parametrize("pred,gt,expected", [
    (6, 5, "off_by_one_over"),
    (4, 5, "off_by_one_under"),
    (9, 5, "off_by_many_over"),
    (2, 5, "off_by_many_under"),
])
def test_count_categories(pred, gt, expected):
    assert classify_result(
        _res(probe="count", pred=pred, gt=gt, raw=str(pred))
    ) == expected


def test_identity_hallucination():
    assert classify_result(
        _res(probe="identity_under_occlusion", pred="green", gt="red", raw="green")
    ) == "hallucination"


def test_unknown_probe_falls_back_to_incorrect():
    assert classify_result(_res(probe="some_third_party_probe", pred="x", gt="y")) == "incorrect"


# ---------------------------------------------------------------------------
# Analyzer aggregation: counts + proportions per (model, condition)
# ---------------------------------------------------------------------------

def test_analyzer_counts_and_proportions():
    results = [
        _res(probe="count", condition="full", correct=True, pred=5, gt=5, raw="5"),
        _res(probe="count", condition="full", pred=6, gt=5, raw="6"),   # off_by_one_over
        _res(probe="count", condition="full", pred=6, gt=5, raw="6"),   # off_by_one_over
        _res(probe="count", condition="full", pred=1, gt=5, raw="1"),   # off_by_many_under
    ]
    report = TAX.analyze(results)
    assert report.payload["schema_version"] == "taxonomy/1"
    full = report.payload["by_model"]["m"]["full"]
    assert full["n"] == 4
    cats = full["categories"]
    assert cats["off_by_one_over"] == {"count": 2, "proportion": 0.5}
    assert cats["off_by_many_under"] == {"count": 1, "proportion": 0.25}
    assert cats["correct"] == {"count": 1, "proportion": 0.25}


# ---------------------------------------------------------------------------
# Composes with oracle: perception failures vanish under the oracle condition
# ---------------------------------------------------------------------------

def test_taxonomy_composes_with_oracle():
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 4]},
        seeds=list(range(12)),
        probe="spatial_relation",
        models=["mock_noisy"],   # flips ~25% under full; oracle reads exact coords
        analyzers=["taxonomy"],
    )
    report = next(r for r in run_experiment(cfg, _registry()) if r.name == "taxonomy")
    by_cond = report.payload["by_model"]["mock_noisy"]

    full_cats = by_cond["full"]["categories"]
    oracle_cats = by_cond["oracle"]["categories"]
    # full: some inversions, and every wrong parse is an inversion (binary probe)
    assert full_cats.get("relation_inversion", {}).get("count", 0) > 0
    # oracle: text coords -> all correct, no inversions
    assert "relation_inversion" not in oracle_cats
    assert set(oracle_cats) == {"correct"}


def test_taxonomy_runs_on_count_full():
    cfg = ExperimentConfig(
        scene_generator="dots",
        scene_params={"n_dots": [3, 9]},
        seeds=list(range(8)),
        probe="count",
        models=["mock_noisy"],
        analyzers=["taxonomy"],
    )
    report = next(r for r in run_experiment(cfg, _registry()) if r.name == "taxonomy")
    full_cats = report.payload["by_model"]["mock_noisy"]["full"]["categories"]
    # any miscount is an off_by_* bucket
    off_by = [k for k in full_cats if k.startswith("off_by_")]
    assert off_by, f"expected off_by_* categories, got {list(full_cats)}"


# ---------------------------------------------------------------------------
# Inspect tab surfaces the category (acceptance)
# ---------------------------------------------------------------------------

def test_inspect_tab_shows_category():
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [4]},
        # enough seeds that the 25%-flip noisy mock reliably produces a failure case
        # (the mock is deterministic per input, so this is stable, not flaky)
        seeds=list(range(24)),
        probe="spatial_relation",
        models=["mock_noisy"],
        analyzers=["taxonomy"],
    )
    scenes, results, _ = run_experiment_collect(cfg, _registry())

    cases = D.failure_cases(results, "full")
    assert cases, "expected at least one failure case"
    assert all("category" in c for c in cases)
    # classifier is the single source of truth shared with the analyzer
    for c in cases:
        assert c["category"] in {"relation_inversion", "refusal", "parse_fail",
                                 "model_error", "gen_error"}

    images = {s.id: s.images[0] for s in scenes}
    _, detail = UI_APP._show_case(results, images, "full", D.case_label(cases[0]))
    assert "failure category" in detail
    assert cases[0]["category"] in detail
