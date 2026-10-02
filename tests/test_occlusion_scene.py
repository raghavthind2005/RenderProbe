"""tabletop_occlusion scene + identity_under_occlusion probe tests."""
from __future__ import annotations

import pytest

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment
from tests.fixtures import register
from tests.fixtures.probe_identity_under_occlusion import PLUGIN as occ_probe
from tests.fixtures.scene_tabletop_occlusion import PLUGIN as occ_scene


def _make_registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


# ---------------------------------------------------------------------------
# tabletop_occlusion scene
# ---------------------------------------------------------------------------

def test_occ_scene_gt_shape():
    scene = occ_scene.generate({"occluder_frac": 0.5}, seed=0)
    gt = scene.ground_truth
    assert set(gt.keys()) == {"target_color", "target_shape", "occluder_frac"}
    assert gt["target_color"] in {"red", "blue", "green", "orange", "purple"}
    assert gt["target_shape"] in {"circle", "square"}
    assert 0.0 <= gt["occluder_frac"] <= 1.0


def test_occ_scene_factors_shape():
    scene = occ_scene.generate({"occluder_frac": 0.75}, seed=0)
    assert scene.factors == {"occluder_frac": 0.75}


def test_occ_scene_meta():
    scene = occ_scene.generate({"occluder_frac": 0.0}, seed=3)
    assert scene.meta["generator"] == "tabletop_occlusion"
    assert scene.meta["seed"] == 3


def test_occ_scene_occluder_frac_in_gt_matches_factors():
    for frac in [0.0, 0.25, 0.5, 1.0]:
        scene = occ_scene.generate({"occluder_frac": frac}, seed=0)
        assert scene.ground_truth["occluder_frac"] == scene.factors["occluder_frac"]


def test_occ_scene_deterministic():
    s1 = occ_scene.generate({"occluder_frac": 0.5}, seed=7)
    s2 = occ_scene.generate({"occluder_frac": 0.5}, seed=7)
    assert s1.ground_truth == s2.ground_truth


def test_occ_scene_image_produced():
    scene = occ_scene.generate({"occluder_frac": 0.5}, seed=0)
    assert len(scene.images) == 1
    assert scene.images[0].size == (480, 360)


# ---------------------------------------------------------------------------
# identity_under_occlusion probe
# ---------------------------------------------------------------------------

def test_occ_probe_oracle_prompt_supported():
    """identity now supports oracle - it states the target's attributes."""
    scene = occ_scene.generate({"occluder_frac": 0.5}, seed=0)
    prompt = occ_probe.oracle_prompt(scene)
    assert prompt is not None
    assert scene.ground_truth["target_color"] in prompt.lower()
    # ≥3 verbalization variants
    assert set(occ_probe.oracle_prompt_variants(scene)) == {
        "description", "attributes", "json"
    }


def test_occ_probe_parse_valid():
    assert occ_probe.parse("The object is red.") == "red"
    assert occ_probe.parse("BLUE") == "blue"
    assert occ_probe.parse("I think it's green") == "green"


def test_occ_probe_parse_raises_unknown():
    with pytest.raises(ValueError):
        occ_probe.parse("yellow")
    with pytest.raises(ValueError):
        occ_probe.parse("I cannot tell")


def test_occ_probe_score():
    assert occ_probe.score("red", "red") == (True, 1.0, None)
    assert occ_probe.score("blue", "red") == (False, 0.0, None)


def test_occ_probe_gt_field():
    scene = occ_scene.generate({"occluder_frac": 0.0}, seed=0)
    assert occ_probe.ground_truth(scene) == scene.ground_truth["target_color"]


# ---------------------------------------------------------------------------
# End-to-end: oracle skips-with-message, sweep_curve produces occlusion curve
# ---------------------------------------------------------------------------

def test_occlusion_oracle_now_runs():
    """The identity_under_occlusion probe supports oracle, so the diagnostic runs."""
    reg = _make_registry()
    cfg = ExperimentConfig(
        scene_generator="tabletop_occlusion",
        scene_params={"occluder_frac": [0.0, 0.5]},
        seeds=[0, 1],
        probe="identity_under_occlusion",
        models=["mock"],
        analyzers=["oracle"],
    )
    reports = run_experiment(cfg, reg)
    orc = next(r for r in reports if r.name == "oracle")
    row = orc.payload["by_model"]["mock"]
    # oracle states the target color -> mock recovers it perfectly from text
    assert row["acc_oracle"] == 1.0
    assert "recovery" in row


def test_occlusion_sweep_curve_end_to_end():
    reg = _make_registry()
    cfg = ExperimentConfig(
        scene_generator="tabletop_occlusion",
        scene_params={"occluder_frac": [0.0, 0.5, 1.0]},
        seeds=[0, 1],
        probe="identity_under_occlusion",
        models=["mock"],
        analyzers=["accuracy", "sweep_curve"],
    )
    reports = run_experiment(cfg, reg)
    names = {r.name for r in reports}
    assert "accuracy" in names
    assert "sweep_curve" in names
    sc = next(r for r in reports if r.name == "sweep_curve")
    assert sc.payload["factor"] == "occluder_frac"
    curve = sc.payload["by_model"]["mock"]
    assert "0.0" in curve
    assert "0.5" in curve
    assert "1.0" in curve
