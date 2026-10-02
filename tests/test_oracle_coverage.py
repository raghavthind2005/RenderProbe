"""Oracle coverage across probes + loud warning for unsupported probes.

count and identity_under_occlusion are perception-bound: their oracle states the
answer, so recovery reflects the full perception gap. The mock reads the injected
primitive from any verbalization variant. A probe whose oracle_prompt returns None
must produce a STRUCTURED skip report, not silence.
"""
from __future__ import annotations

import pytest

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment
from tests.fixtures import register
from tests.fixtures.mock import PLUGINS as MOCK_PLUGINS
from tests.fixtures.probe_count import PLUGIN as COUNT
from tests.fixtures.probe_identity_under_occlusion import PLUGIN as IDENTITY
from tests.fixtures.scene_dots import PLUGIN as DOTS
from tests.fixtures.scene_tabletop_occlusion import PLUGIN as OCC

_MOCK = next(m for m in MOCK_PLUGINS if m.name == "mock")


def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


# ---------------------------------------------------------------------------
# Probe-level: ≥3 variants, canonical wiring
# ---------------------------------------------------------------------------

def test_count_offers_three_variants():
    scene = DOTS.generate({"n_dots": 4}, seed=0)
    variants = COUNT.oracle_prompt_variants(scene)
    assert set(variants) == {"statement", "count_field", "json"}
    assert COUNT.oracle_prompt(scene) == variants["statement"]


def test_identity_offers_three_variants():
    scene = OCC.generate({"occluder_frac": 0.5}, seed=0)
    variants = IDENTITY.oracle_prompt_variants(scene)
    assert set(variants) == {"description", "attributes", "json"}
    assert IDENTITY.oracle_prompt(scene) == variants["description"]


# ---------------------------------------------------------------------------
# Mock reads the injected primitive from every variant (text-only)
# ---------------------------------------------------------------------------

def test_mock_reads_count_from_every_variant():
    for n in (1, 3, 7, 12):
        scene = DOTS.generate({"n_dots": n}, seed=0)
        for prompt in COUNT.oracle_prompt_variants(scene).values():
            assert COUNT.parse(_MOCK.run([], prompt).text) == n


def test_mock_reads_identity_from_every_variant():
    for seed in range(5):
        scene = OCC.generate({"occluder_frac": 0.5}, seed=seed)
        gt = IDENTITY.ground_truth(scene)
        for prompt in IDENTITY.oracle_prompt_variants(scene).values():
            assert IDENTITY.parse(_MOCK.run([], prompt).text) == gt


# ---------------------------------------------------------------------------
# Scene tiers for the ceiling check
# ---------------------------------------------------------------------------

def test_scene_tiers():
    assert DOTS.generate({"n_dots": 2}, seed=0).meta["tier"] == "trivial"
    assert DOTS.generate({"n_dots": 8}, seed=0).meta["tier"] == "standard"
    assert OCC.generate({"occluder_frac": 0.0}, seed=0).meta["tier"] == "trivial"
    assert OCC.generate({"occluder_frac": 0.6}, seed=0).meta["tier"] == "standard"


# ---------------------------------------------------------------------------
# End-to-end: oracle diagnostic runs on all three probes (acceptance)
# ---------------------------------------------------------------------------

_CASES = [
    ("dots", {"n_dots": [2, 5]}, "count"),
    ("tabletop_occlusion", {"occluder_frac": [0.0, 0.6]}, "identity_under_occlusion"),
    ("spatial_config", {"n_distractors": [0, 4]}, "spatial_relation"),
]


@pytest.mark.parametrize("scene,params,probe", _CASES)
def test_oracle_runs_end_to_end_on_all_probes(scene, params, probe):
    cfg = ExperimentConfig(
        scene_generator=scene,
        scene_params=params,
        seeds=[0, 1, 2],
        probe=probe,
        models=["mock"],
        analyzers=["oracle"],
        oracle_variants=["all"],
    )
    orc = next(r for r in run_experiment(cfg, _registry()) if r.name == "oracle")
    row = orc.payload["by_model"]["mock"]
    assert orc.payload["schema_version"] == "oracle/4"
    assert row["n_paired"] > 0
    assert isinstance(row["recovery"], float)
    # all three variants present, none flagged suspect (valid templates), and the
    # trivial tier reaches ceiling
    assert len(row["by_variant"]) == 3
    for v in row["by_variant"].values():
        assert v["oracle_prompt_suspect"] is False
        assert v["trivial_oracle_acc"] == 1.0
    assert orc.payload["any_oracle_prompt_suspect"] is False


def test_perception_bound_probes_recover_fully():
    """count/identity oracle states the answer -> acc_oracle == 1.0."""
    for scene, params, probe in _CASES[:2]:
        cfg = ExperimentConfig(
            scene_generator=scene, scene_params=params, seeds=[0, 1, 2],
            probe=probe, models=["mock"], analyzers=["oracle"],
        )
        row = next(
            r for r in run_experiment(cfg, _registry()) if r.name == "oracle"
        ).payload["by_model"]["mock"]
        assert row["acc_oracle"] == 1.0


# ---------------------------------------------------------------------------
# Loud warning when a probe lacks oracle support (acceptance)
# ---------------------------------------------------------------------------

class _NoOracleProbe:
    name = "no_oracle_probe"
    answer_type = "numeric"
    requires_gt_fields = ["count"]

    def question(self, scene):
        return "How many dots? Reply with an integer."

    def ground_truth(self, scene):
        return int(scene.ground_truth["count"])

    def parse(self, raw):
        import re
        m = re.search(r"\d+", raw)
        if m is None:
            raise ValueError("no int")
        return int(m.group())

    def score(self, pred, gt):
        return pred == gt, 1.0 if pred == gt else 0.0, float(abs(pred - gt))

    def oracle_prompt(self, scene):
        return None


def test_unsupported_probe_emits_structured_skip_report(capsys):
    reg = _registry()
    reg.register_probe(_NoOracleProbe())
    cfg = ExperimentConfig(
        scene_generator="dots",
        scene_params={"n_dots": [2, 4]},
        seeds=[0],
        probe="no_oracle_probe",
        models=["mock"],
        analyzers=["oracle"],
    )
    reports = run_experiment(cfg, reg)
    # loud: a structured skip report exists (not an empty list / silent drop)
    orc = next(r for r in reports if r.name == "oracle")
    assert orc.payload["status"] == "skipped"
    assert "oracle" in orc.payload["missing_conditions"]
    assert orc.payload["oracle_supported"] is False
    assert "oracle" in capsys.readouterr().out.lower()
