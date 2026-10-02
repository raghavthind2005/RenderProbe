"""UI data layer, registry catalog, and runner-collect tests.

The gradio app itself is smoke-tested (build only, no launch) and skipped when
gradio isn't installed; all the real presentation logic is in ui.data and is
tested directly here without gradio.
"""
from __future__ import annotations

import importlib.util

import pytest

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment, run_experiment_collect
from renderprobe.core.schema import AnalysisReport, Result
from renderprobe.ui import data as D
from tests.fixtures import register


def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


# ---------------------------------------------------------------------------
# Registry catalog accessors
# ---------------------------------------------------------------------------

def test_registry_catalog_sorted_and_populated():
    r = _registry()
    assert r.list_scenes() == sorted(r.list_scenes())
    assert r.list_models() == sorted(r.list_models())
    assert {"dots", "spatial_config", "tabletop_occlusion"} <= set(r.list_scenes())
    assert {"accuracy", "oracle", "sweep_curve", "visual_gain"} <= set(r.list_analyzers())


# ---------------------------------------------------------------------------
# Grok wiring (config-only, closed model, lazy key)
# ---------------------------------------------------------------------------

def test_grok_registered_as_closed_model():
    r = _registry()
    assert "grok-2-vision" in r.list_models()
    grok = r.get_model("grok-2-vision")
    assert grok.is_open is False


def test_grok_constructs_without_key():
    """Construction must not read XAI_API_KEY (it's read lazily in run())."""
    from renderprobe.models.openai_compatible import OpenAICompatibleAdapter
    a = OpenAICompatibleAdapter(
        name="grok-2-vision", model_id="grok-2-vision-1212",
        api_key_env="XAI_API_KEY", base_url="https://api.x.ai/v1",
    )
    assert a.name == "grok-2-vision"


# ---------------------------------------------------------------------------
# run_experiment_collect
# ---------------------------------------------------------------------------

def test_collect_returns_scenes_results_reports():
    cfg = ExperimentConfig(
        scene_generator="spatial_config",
        scene_params={"n_distractors": [0, 4]},
        seeds=[0, 1],
        probe="spatial_relation",
        models=["mock"],
        analyzers=["accuracy", "oracle", "sweep_curve"],
    )
    scenes, results, reports = run_experiment_collect(cfg, _registry())
    assert scenes and results and reports
    assert all(isinstance(r, Result) for r in results)
    assert all(isinstance(r, AnalysisReport) for r in reports)
    # full + oracle + report (spatial_relation supports oracle AND now a perception
    # report, so the report condition is produced too). No blind: that one is opt-in.
    assert {r.condition for r in results} == {"full", "oracle", "report"}


def test_run_experiment_matches_collect_reports():
    cfg = ExperimentConfig(
        scene_generator="dots", scene_params={"n_dots": [1, 3]}, seeds=[0],
        probe="count", models=["mock"], analyzers=["accuracy"],
    )
    r = _registry()
    reports_a = run_experiment(cfg, r)
    _, _, reports_b = run_experiment_collect(cfg, r)
    assert {x.name for x in reports_a} == {x.name for x in reports_b}


# ---------------------------------------------------------------------------
# ui.data - pure presentation logic
# ---------------------------------------------------------------------------

def test_default_params_reads_schema_defaults():
    gen = _registry().get_scene("spatial_config")
    assert D.default_params(gen) == {"n_distractors": 0}


def test_compatible_probe_auto_pairs_scenes():
    reg = _registry()
    probes = reg.list_probes()
    # showcase scenes are name-matched to their probes
    assert D.compatible_probe(reg, "relational", probes) == "relational"
    assert D.compatible_probe(reg, "pathfinding", probes) == "pathfinding"
    # dots has no same-named probe, but 'count' fits its ground_truth
    assert D.compatible_probe(reg, "dots", probes) == "count"
    # unknown scene falls back to the first probe rather than erroring
    assert D.compatible_probe(reg, "nope", probes) == probes[0]


def test_param_specs_and_values_roundtrip():
    reg = _registry()
    gen = reg.get_scene("relational")
    specs = D.param_specs(gen)
    assert [s["name"] for s in specs] == ["n_objects", "hop_depth", "clutter"]
    assert specs[0]["type"] == "int" and specs[0]["min"] == 4 and specs[0]["max"] == 16
    # slider values map back to a typed params dict (ints stay ints)
    assert D.params_from_values(gen, [8, 3, 1]) == {
        "n_objects": 8, "hop_depth": 3, "clutter": 1,
    }
    # a float param stays float; None (hidden slot) falls back to the default
    pf = reg.get_scene("pathfinding")
    fp = D.params_from_values(pf, [10, None])
    assert fp["n_nodes"] == 10 and isinstance(fp["p_edge"], float)


def test_parse_sweep_values():
    from renderprobe.ui.app import _parse_sweep_values
    reg = _registry()
    rel = D.param_specs(reg.get_scene("relational"))
    assert _parse_sweep_values("4,6,8", rel, "n_objects") == [4, 6, 8]
    assert _parse_sweep_values("4 6 8", rel, "n_objects") == [4, 6, 8]   # spaces ok
    pf = D.param_specs(reg.get_scene("pathfinding"))
    assert _parse_sweep_values("0.3,0.5", pf, "p_edge") == [0.3, 0.5]
    assert _parse_sweep_values("junk", rel, "n_objects") == []           # no crash


def test_preview_path_all_scenes():
    """The live-preview path (dials -> params -> generate -> probe question/GT + image)
    must work for every scene at its default dial values."""
    reg = _registry()
    probes = reg.list_probes()
    for scene in reg.list_scenes():
        gen = reg.get_scene(scene)
        specs = D.param_specs(gen)
        params = D.params_from_values(gen, [s["default"] for s in specs])
        s = gen.generate(params, 0)
        assert s.images and s.images[0] is not None, f"{scene}: no preview image"
        probe = reg.get_probe(D.compatible_probe(reg, scene, probes))
        assert isinstance(probe.question(s), str) and probe.question(s)
        probe.ground_truth(s)   # must not raise


def test_report_by_name():
    reps = [AnalysisReport(name="accuracy", payload={}), AnalysisReport(name="oracle", payload={})]
    assert D.report_by_name(reps, "oracle").name == "oracle"
    assert D.report_by_name(reps, "nope") is None


def _mk_result(**kw) -> Result:
    base = dict(
        scene_id="s0", generator="g", factors={}, probe="p", model="mock",
        condition="full", raw="left", pred="left", gt="left", correct=True,
        score=1.0, error=None, confidence=None, error_flag=None,
    )
    base.update(kw)
    return Result(**base)


def test_results_table_shape():
    rows = D.results_table([_mk_result(), _mk_result(correct=False, pred="right")])
    assert len(rows) == 2
    assert len(rows[0]) == len(D.RESULT_COLUMNS)
    assert rows[0][5] == "yes" and rows[1][5] == "no"


def test_accuracy_and_oracle_rows():
    acc = AnalysisReport(
        name="accuracy",
        payload={"by_model": {"mock": {"n": 4, "accuracy": 0.75, "mean_error": None}}},
    )
    cols, rows = D.accuracy_rows(acc)
    assert cols[0] == "model" and rows[0][:3] == ["mock", 4, 0.75]

    orc = AnalysisReport(name="oracle", payload={
        "by_model": {"mock": {
            "n_paired": 6, "acc_full": 0.5, "acc_oracle": 1.0, "recovery": 0.5,
            "recovery_ci": [0.2, 0.8], "p_value": 0.03, "n_discordant": 4,
            "oracle_prompt_suspect": False,
        }},
        "any_oracle_prompt_suspect": False,
        "note": "framing",
    })
    ocols, orows = D.oracle_rows(orc)
    # first five columns unchanged; stats appended
    assert orows[0][:5] == ["mock", 6, 0.5, 1.0, 0.5]
    assert orows[0][ocols.index("recovery 95% CI")] == "[0.2, 0.8]"
    assert orows[0][ocols.index("McNemar p")] == "0.03"
    assert orows[0][ocols.index("n_discordant")] == 4
    assert orows[0][ocols.index("prompt suspect")] == "no"
    assert D.oracle_note(orc) == "framing"


def test_oracle_rows_tolerates_missing_stats():
    """A payload missing the paired-statistics fields must not crash the table."""
    orc = AnalysisReport(name="oracle", payload={
        "by_model": {"mock": {"n_paired": 6, "acc_full": 0.5, "acc_oracle": 1.0, "recovery": 0.5}},
    })
    _, orows = D.oracle_rows(orc)
    assert orows[0][:5] == ["mock", 6, 0.5, 1.0, 0.5]
    assert "n/a" in orows[0]         # a missing CI is labeled, not blank


def test_oracle_suspect_banner():
    clean = AnalysisReport(name="oracle", payload={"any_oracle_prompt_suspect": False})
    assert D.oracle_suspect_banner(clean) == ""
    flagged = AnalysisReport(name="oracle", payload={"any_oracle_prompt_suspect": True})
    assert "suspect" in D.oracle_suspect_banner(flagged).lower()
    assert D.oracle_suspect_banner(None) == ""


def test_taxonomy_rows():
    rep = AnalysisReport(name="taxonomy", payload={"by_model": {"mock": {
        "full": {"n": 4, "categories": {"correct": {"count": 4, "proportion": 1.0}}},
        "oracle": {"n": 4, "categories": {
            "correct": {"count": 3, "proportion": 0.75},
            "incorrect": {"count": 1, "proportion": 0.25},
        }},
    }}})
    cols, rows = D.taxonomy_rows(rep)
    assert cols == ["model", "condition", "category", "count", "proportion"]
    # full ordered before oracle; both categories present for oracle
    assert rows[0] == ["mock", "full", "correct", 4, 1.0]
    assert ["mock", "oracle", "incorrect", 1, 0.25] in rows


def test_calibration_rows_and_null_reason():
    rep = AnalysisReport(name="calibration", payload={"by_model": {"mock": {
        "full": {"n": 10, "ece_equal_width": 0.13, "ece_equal_mass": 0.13, "reason": None},
        "blind": {"n": 0, "ece_equal_width": None, "ece_equal_mass": None,
                  "reason": "no confidences elicited"},
    }}})
    cols, rows = D.calibration_rows(rep)
    assert cols[3] == "ECE (equal-width)"
    full_row = next(r for r in rows if r[1] == "full")
    assert full_row[3] == "0.13"
    blind_row = next(r for r in rows if r[1] == "blind")
    assert blind_row[3] == "" and "confidence" in blind_row[5].lower()


def test_reliability_series():
    rep = AnalysisReport(name="calibration", payload={"by_model": {"mock": {
        "full": {"reliability_equal_width": [
            {"conf": 0.8, "acc": 0.7, "count": 5},
            {"conf": 0.9, "acc": 0.95, "count": 5},
        ]},
    }}})
    s = D.reliability_series(rep, "full", "equal_width")
    assert s["condition"] == "full"
    assert s["series"]["mock"]["conf"] == [0.8, 0.9]
    assert s["series"]["mock"]["acc"] == [0.7, 0.95]
    # a condition with no bins yields an empty series (no crash)
    assert D.reliability_series(rep, "oracle")["series"] == {}


def test_accuracy_rows_handles_missing_report():
    cols, rows = D.accuracy_rows(None)
    assert rows == [] and "model" in cols


def test_sweep_series_numeric_sort_and_ci_fallback():
    rep = AnalysisReport(name="sweep_curve", payload={
        "factor": "n_distractors",
        "by_model": {"mock": {
            "4": {"accuracy": 0.6, "n": 10, "ci95": [0.4, 0.8]},
            "0": {"accuracy": 0.9, "n": 2, "ci95": None},
        }},
    })
    s = D.sweep_series(rep)
    assert s["factor"] == "n_distractors"
    series = s["series"]["mock"]
    assert series["x"] == [0.0, 4.0]            # numeric sort, not lexical
    assert series["y"] == [0.9, 0.6]
    assert series["lo"][0] == 0.9 and series["hi"][0] == 0.9   # CI=None -> point
    assert series["lo"][1] == 0.4 and series["hi"][1] == 0.8


def test_sweep_series_empty():
    assert D.sweep_series(None)["series"] == {}


def test_failure_cases_and_label():
    rs = [
        _mk_result(correct=True),
        _mk_result(scene_id="s1", correct=False, pred="right", gt="left"),
        _mk_result(scene_id="s2", correct=False, error_flag="parse_fail", pred=None),
        _mk_result(scene_id="s3", condition="blind", correct=False),
    ]
    full_fails = D.failure_cases(rs, "full")
    assert {c["scene_id"] for c in full_fails} == {"s1", "s2"}
    assert D.failure_cases(rs, "blind")[0]["scene_id"] == "s3"
    label = D.case_label(full_fails[0])
    assert "s1" in label and "mock" in label


# ---------------------------------------------------------------------------
# gradio app smoke (build only) - skipped if gradio not installed
# ---------------------------------------------------------------------------

_HAS_GRADIO = importlib.util.find_spec("gradio") is not None


@pytest.mark.skipif(not _HAS_GRADIO, reason="gradio not installed")
def test_build_app_smoke():
    from renderprobe.ui.app import build_app
    app = build_app(_registry())
    assert app is not None


@pytest.mark.skipif(not _HAS_GRADIO, reason="gradio not installed")
def test_render_sweep_no_data_message():
    from renderprobe.ui.app import _render_sweep
    fig, msg = _render_sweep([])
    assert fig is None and "No sweep" in msg


@pytest.mark.skipif(not _HAS_GRADIO, reason="gradio not installed")
def test_render_reliability_no_data_message():
    from renderprobe.ui.app import _render_reliability
    fig, msg = _render_reliability([])
    assert fig is None and "confidence" in msg.lower()


@pytest.mark.skipif(not _HAS_GRADIO, reason="gradio not installed")
def test_run_flow_populates_all_tables():
    """End-to-end through the real analyzers: every Run/Calibration output is
    non-empty on a confidence-elicited relational run, and the plots render."""
    from renderprobe.ui.app import _render_reliability, _render_sweep
    reg = _registry()
    cfg = ExperimentConfig(
        scene_generator="relational",
        scene_params={"n_objects": [4, 6], "hop_depth": 2},
        seeds=[0, 1, 2, 3],
        probe="relational",
        models=["mock", "mock_noisy"],
        analyzers=["accuracy", "visual_gain", "oracle", "taxonomy", "calibration", "sweep_curve"],
        elicit_confidence=True,
    )
    _, results, reports = run_experiment_collect(cfg, reg)
    assert D.accuracy_rows(D.report_by_name(reports, "accuracy"))[1]
    assert len(D.oracle_rows(D.report_by_name(reports, "oracle"))[0]) == 9
    assert D.taxonomy_rows(D.report_by_name(reports, "taxonomy"))[1]
    cal_rows = D.calibration_rows(D.report_by_name(reports, "calibration"))[1]
    assert any(r[3] not in ("", None) for r in cal_rows)   # a real ECE present
    assert _render_sweep(reports)[0] is not None
    assert _render_reliability(reports)[0] is not None


def test_decomposition_table_carries_verdict_confidence():
    # A verdict shown without its confidence is the thing the stability work fixed;
    # the table must not reintroduce it on screen.
    from renderprobe.analysis.decomposition import PLUGIN as DECOMP
    from tests.test_decomposition import _rows

    # acc_full 0.20 against acc_report 0.30: the margin between them sits exactly on
    # the materiality threshold, so one item either way changes the verdict.
    cols, rows = D.decomposition_rows(DECOMP.analyze(_rows(20, 4, 20, 6)))
    assert "confidence" in cols
    cell = rows[0][cols.index("confidence")]
    assert "unsettled" in cell and "grounding/integration-limited" in cell

    cols, rows = D.decomposition_rows(DECOMP.analyze(_rows(20, 4, 20, 20)))
    assert "firm" in rows[0][cols.index("confidence")]


def test_the_package_ships_only_calibrated_scenes():
    """The installed package is the product surface, and the UI reads straight off it.
    The offline fixtures live in tests/ instead, so there is no filter to forget and no
    way for one to reach a reviewer's scene dropdown."""
    from renderprobe.core.registry import Registry

    reg = Registry()
    reg.autodiscover()
    assert set(reg.list_scenes()) == {"route", "polycube"}
    assert set(reg.list_probes()) == {"route", "polycube"}
    # the mock doubles are fixtures too, and are no longer installed as models
    assert not [m for m in reg.list_models() if m.startswith("mock")]


def test_fixtures_are_still_reachable_for_the_offline_suite():
    """Moved, not deleted. They back the occlusion certificate, the decomposition
    cascade and the calibration and taxonomy analyzers."""
    from renderprobe.core.registry import Registry
    from tests.fixtures import MODELS, PROBES, SCENES

    reg = Registry()
    reg.autodiscover()
    register(reg)
    for name in SCENES:
        assert reg.get_scene(name) is not None
    for name in PROBES:
        assert reg.get_probe(name) is not None
    for name in MODELS:
        assert reg.get_model(name) is not None


def test_an_out_of_tree_scene_reaches_the_ui():
    from pathlib import Path

    from renderprobe.core.registry import Registry

    reg = Registry()
    reg.autodiscover()
    reg.load_external(Path(__file__).resolve().parent.parent / "examples")
    assert "gear_train" in reg.list_scenes()
