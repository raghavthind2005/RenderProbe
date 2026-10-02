"""Statistical rigor: stats primitives + the paired oracle analyzer.

No network, no API keys. The known-effect assertions use synthetic per-item data
where the ground-truth effect is constructed by hand, so the test is a genuine
check of the test's statistical behavior (not a tautology against the same code).
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout

import pytest

from renderprobe.analysis.oracle import PLUGIN as ORACLE
from renderprobe.analysis.stats import (
    bootstrap_label_stability,
    mcnemar_exact,
    mcnemar_power_n,
    norm_ppf,
    paired_bootstrap_recovery,
    wilson_interval,
)
from renderprobe.cli import power_cmd
from renderprobe.core.schema import Result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairs(both_right: int, both_wrong: int, b: int, c: int) -> list[tuple[bool, bool]]:
    """Build per-item (full_correct, oracle_correct) pairs with exact cell counts.

    b = full-right & oracle-wrong; c = full-wrong & oracle-right (the discordant
    cells that drive McNemar). recovery = (c - b) / n.
    """
    out: list[tuple[bool, bool]] = []
    out += [(True, True)] * both_right
    out += [(False, False)] * both_wrong
    out += [(True, False)] * b
    out += [(False, True)] * c
    return out


def _res(scene_id, condition, correct, model="m", error_flag=None):
    return Result(
        scene_id=scene_id, generator="g", factors={}, probe="spatial_relation",
        model=model, condition=condition, raw="", pred=None, gt=None,
        correct=correct, score=1.0 if correct else 0.0, error=None,
        confidence=None, error_flag=error_flag,
    )


def _results_from_pairs(pairs, model="m"):
    """One full + one oracle Result per pair, valid (no error_flag)."""
    rs = []
    for i, (f, o) in enumerate(pairs):
        rs.append(_res(f"s{i}", "full", f, model=model))
        rs.append(_res(f"s{i}", "oracle", o, model=model))
    return rs


# ---------------------------------------------------------------------------
# norm_ppf (probit)
# ---------------------------------------------------------------------------

def test_norm_ppf_known_quantiles():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-4)
    assert norm_ppf(0.8) == pytest.approx(0.841621, abs=1e-4)


def test_norm_ppf_domain():
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            norm_ppf(bad)


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

def test_wilson_interval_known_value():
    lo, hi = wilson_interval(8, 10, alpha=0.05)
    assert lo == pytest.approx(0.4902, abs=5e-3)
    assert hi == pytest.approx(0.9433, abs=5e-3)


def test_wilson_interval_brackets_phat_and_bounds():
    lo, hi = wilson_interval(3, 10)
    assert 0.0 <= lo < 0.3 < hi <= 1.0


def test_wilson_interval_extremes_stay_in_unit():
    assert wilson_interval(10, 10)[1] <= 1.0
    assert wilson_interval(0, 10)[0] >= 0.0


def test_wilson_interval_zero_n_is_none():
    assert wilson_interval(0, 0) is None


# ---------------------------------------------------------------------------
# McNemar exact test - the core known-effect behavior (acceptance criteria)
# ---------------------------------------------------------------------------

def test_mcnemar_zero_effect_does_not_reject():
    # Balanced discordant cells => no systematic difference => p == 1.0.
    pairs = _pairs(both_right=100, both_wrong=100, b=45, c=45)
    assert len(pairs) == 290
    b = sum(1 for f, o in pairs if f and not o)
    c = sum(1 for f, o in pairs if o and not f)
    assert mcnemar_exact(b, c) == pytest.approx(1.0)


def test_mcnemar_injected_effect_at_n300_rejects():
    # recovery = (c - b)/n = (67 - 22)/300 = 0.15, a real paired difference.
    pairs = _pairs(both_right=155, both_wrong=56, b=22, c=67)
    assert len(pairs) == 300
    recovery = (67 - 22) / 300
    assert recovery == pytest.approx(0.15)
    b = sum(1 for f, o in pairs if f and not o)
    c = sum(1 for f, o in pairs if o and not f)
    assert mcnemar_exact(b, c) < 0.05


def test_mcnemar_no_discordant_pairs_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetry():
    assert mcnemar_exact(5, 20) == mcnemar_exact(20, 5)


# ---------------------------------------------------------------------------
# Paired bootstrap recovery CI
# ---------------------------------------------------------------------------

def test_bootstrap_recovery_all_identical_is_zero_width():
    # Every scene concordant => recovery is exactly 0 in every resample.
    ci = paired_bootstrap_recovery(_pairs(both_right=50, both_wrong=0, b=0, c=0))
    assert ci == [0.0, 0.0]


def test_bootstrap_recovery_brackets_point_estimate():
    pairs = _pairs(both_right=155, both_wrong=56, b=22, c=67)  # recovery = 0.15
    lo, hi = paired_bootstrap_recovery(pairs, seed=0)
    assert lo < 0.15 < hi
    assert lo > 0.0  # a real positive effect at n=300 excludes zero


def test_bootstrap_recovery_deterministic_under_seed():
    pairs = _pairs(both_right=10, both_wrong=5, b=3, c=7)
    assert paired_bootstrap_recovery(pairs, seed=1) == paired_bootstrap_recovery(pairs, seed=1)


def test_bootstrap_recovery_empty_is_none():
    assert paired_bootstrap_recovery([]) is None


# ---------------------------------------------------------------------------
# Power / sample-size
# ---------------------------------------------------------------------------

def test_power_n_matches_hand_computation():
    # effect 0.1, alpha 0.05, power 0.80, discordance 0.30 -> 234 (hand-checked).
    assert mcnemar_power_n(0.1, alpha=0.05, power=0.80, discordance=0.30) == 234


def test_power_n_lower_discordance_needs_more():
    # Lower discordance can't go below the effect; compare two valid rates.
    n_low = mcnemar_power_n(0.1, discordance=0.15)
    n_high = mcnemar_power_n(0.1, discordance=0.5)
    assert n_low < n_high  # smaller discordant pool -> tighter -> fewer needed


def test_power_n_smaller_effect_needs_more():
    assert mcnemar_power_n(0.05) > mcnemar_power_n(0.15)


def test_power_n_rejects_discordance_below_effect():
    with pytest.raises(ValueError):
        mcnemar_power_n(0.4, discordance=0.2)


def test_power_n_rejects_zero_effect():
    with pytest.raises(ValueError):
        mcnemar_power_n(0.0)


# ---------------------------------------------------------------------------
# Oracle analyzer - payload schema (acceptance: required fields present)
# ---------------------------------------------------------------------------

def test_oracle_payload_has_required_fields():
    pairs = _pairs(both_right=10, both_wrong=3, b=2, c=5)
    report = ORACLE.analyze(_results_from_pairs(pairs))
    # oracle/2 was extended to oracle/3 (the per-variant breakdown) and then oracle/4;
    # the earlier fields asserted below remain present and unchanged.
    assert report.payload["schema_version"] == "oracle/4"
    row = report.payload["by_model"]["m"]
    for key in (
        "n_paired", "acc_full", "acc_oracle", "recovery",
        "acc_full_ci", "acc_oracle_ci", "recovery_ci",
        "p_value", "n_discordant",
        "parse_failure_rate_full", "parse_failure_rate_oracle",
        "parse_failure_warning",
    ):
        assert key in row, f"missing {key}"
    assert 0.0 <= row["p_value"] <= 1.0
    assert row["n_discordant"] == 7  # b + c
    assert row["recovery"] == pytest.approx((5 - 2) / 20)


def test_oracle_recovery_matches_paired_accuracies():
    pairs = _pairs(both_right=155, both_wrong=56, b=22, c=67)  # recovery 0.15, n=300
    row = ORACLE.analyze(_results_from_pairs(pairs)).payload["by_model"]["m"]
    assert row["n_paired"] == 300
    assert row["recovery"] == pytest.approx(0.15)
    assert row["p_value"] < 0.05  # the injected effect is detected end-to-end


# ---------------------------------------------------------------------------
# Parse-failure accounting (acceptance: separate category + biased-subsample warn)
# ---------------------------------------------------------------------------

def test_parse_failure_counted_separately_not_as_wrong():
    # 10 scenes; oracle parse-fails on 2. Those must NOT be scored as incorrect -
    # they drop out of the paired set and show up in the rate instead.
    rs = []
    for i in range(10):
        rs.append(_res(f"s{i}", "full", correct=True))
    for i in range(8):
        rs.append(_res(f"s{i}", "oracle", correct=True))
    rs.append(_res("s8", "oracle", correct=False, error_flag="parse_fail"))
    rs.append(_res("s9", "oracle", correct=False, error_flag="parse_fail"))

    row = ORACLE.analyze(rs).payload["by_model"]["m"]
    assert row["n_paired"] == 8  # the 2 parse-failed scenes excluded from pairing
    assert row["acc_oracle"] == 1.0  # not dragged down by the parse failures
    assert row["parse_failure_rate_full"] == 0.0
    assert row["parse_failure_rate_oracle"] == 0.2


def test_parse_failure_warning_fires_when_rates_diverge():
    rs = [_res(f"s{i}", "full", correct=True) for i in range(20)]
    rs += [_res(f"s{i}", "oracle", correct=True) for i in range(18)]
    rs.append(_res("s18", "oracle", correct=False, error_flag="parse_fail"))
    rs.append(_res("s19", "oracle", correct=False, error_flag="parse_fail"))
    row = ORACLE.analyze(rs).payload["by_model"]["m"]
    # oracle 2/20 = 0.10 vs full 0.0 -> 0.10 > 0.05 threshold
    assert row["parse_failure_warning"] is not None
    assert "biased subsample" in row["parse_failure_warning"]


def test_parse_failure_warning_silent_when_rates_match():
    pairs = _pairs(both_right=10, both_wrong=2, b=1, c=1)
    row = ORACLE.analyze(_results_from_pairs(pairs)).payload["by_model"]["m"]
    assert row["parse_failure_warning"] is None


def test_oracle_all_parse_fail_degenerate_row_survives():
    # Every oracle answer fails to parse: n_paired collapses to 0 but the row must
    # not vanish silently - the parse-failure rate is exactly the signal.
    rs = [_res(f"s{i}", "full", correct=True) for i in range(5)]
    rs += [_res(f"s{i}", "oracle", correct=False, error_flag="parse_fail") for i in range(5)]
    row = ORACLE.analyze(rs).payload["by_model"]["m"]
    assert row["n_paired"] == 0
    assert row["acc_oracle"] is None
    assert row["parse_failure_rate_oracle"] == 1.0
    assert row["parse_failure_warning"] is not None


# ---------------------------------------------------------------------------
# CLI power subcommand
# ---------------------------------------------------------------------------

def test_power_cmd_prints_n_and_returns_zero():
    args = argparse.Namespace(effect=0.1, alpha=0.05, power=0.80, discordance=0.30)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = power_cmd(args)
    out = buf.getvalue()
    assert rc == 0
    assert "234" in out
    assert "required paired n" in out


def test_power_cmd_bad_args_returns_one():
    args = argparse.Namespace(effect=0.4, alpha=0.05, power=0.80, discordance=0.2)
    rc = power_cmd(args)
    assert rc == 1


# ---------------------------------------------------------------------------
# Verdict stability under resampling
# ---------------------------------------------------------------------------

def _label(accs):
    """Toy verdict: a step function of one condition, so it can flip on resample."""
    a = accs.get("x")
    return "hi" if a is not None and a >= 0.5 else "lo"


def test_label_stability_empty_is_none():
    assert bootstrap_label_stability([], _label) is None


def test_label_stability_unanimous_input_is_fully_stable():
    scenes = [{"x": (1, 1)} for _ in range(20)]
    out = bootstrap_label_stability(scenes, _label)
    assert out["label"] == "hi"
    assert out["stability"] == 1.0
    assert out["distribution"] == {"hi": 1.0}


def test_label_stability_falls_when_the_point_sits_on_the_cutoff():
    # exactly 50% correct, and the cutoff is 0.5: resampling flips it constantly
    scenes = [{"x": (1, 1)} for _ in range(10)] + [{"x": (0, 1)} for _ in range(10)]
    out = bootstrap_label_stability(scenes, _label)
    assert out["stability"] < 0.95
    assert set(out["distribution"]) == {"hi", "lo"}


def test_label_stability_distribution_sums_to_one():
    scenes = [{"x": (1, 1)} for _ in range(11)] + [{"x": (0, 1)} for _ in range(9)]
    out = bootstrap_label_stability(scenes, _label)
    assert sum(out["distribution"].values()) == pytest.approx(1.0, abs=1e-6)


def test_label_stability_deterministic_under_seed():
    scenes = [{"x": (1, 1)} for _ in range(11)] + [{"x": (0, 1)} for _ in range(9)]
    a = bootstrap_label_stability(scenes, _label, seed=3)
    b = bootstrap_label_stability(scenes, _label, seed=3)
    assert a == b


def test_label_stability_pools_multiple_rows_per_scene():
    # a scene contributing 2 rows (as the oracle does with 2 variants) is pooled,
    # not counted as one observation
    scenes = [{"x": (2, 2)}, {"x": (0, 2)}]
    out = bootstrap_label_stability(scenes, _label)
    assert out["label"] == "hi"   # pooled 2/4 = 0.5 -> "hi" at the cutoff
