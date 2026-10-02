"""Controlled single-variable sweep: accuracy vs. the varying factor, per model.

Emits curve DATA only (plotting lives in the UI). The runner gates this analyzer
with requires_varying_factor=True, so it only runs when >=2 distinct factor
values are present.

Payload shape:
    {
      "factor": "<name of the varying factor>",
      "by_model": {
        model: {
          str(factor_value): {"accuracy": float, "n": int, "ci95": [lo, hi] | None}
        }
      },
    }

Bootstrap CI (500 resamples, 95%) is computed when a bucket has >=4 samples;
otherwise ci95 is None.
"""
from __future__ import annotations

import random as _random
from collections import defaultdict

from renderprobe.core.schema import AnalysisReport, Result


def _bootstrap_ci(
    correct_flags: list[bool],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[float] | None:
    n = len(correct_flags)
    if n < 4:
        return None
    rng = _random.Random(seed)
    boot_means = sorted(
        sum(rng.choices(correct_flags, k=n)) / n for _ in range(n_boot)
    )
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot)]
    return [round(lo, 4), round(hi, 4)]


class _SweepCurveAnalyzer:
    name = "sweep_curve"
    requires_conditions = ["full"]
    requires_varying_factor = True

    def analyze(self, results: list[Result]) -> AnalysisReport:
        full_ok = [r for r in results if r.condition == "full" and r.error_flag is None]

        # Find the factor with >=2 distinct values
        factor_vals: dict[str, set] = defaultdict(set)
        for r in full_ok:
            for k, v in r.factors.items():
                factor_vals[k].add(v)

        varying = {k: v for k, v in factor_vals.items() if len(v) >= 2}
        if not varying:
            return AnalysisReport(
                name="sweep_curve", payload={"error": "no varying factor found"}
            )

        # Use the first varying factor (single-axis sweep)
        factor_name = next(iter(varying))
        factor_domain = sorted(varying[factor_name])

        models = sorted({r.model for r in full_ok})
        by_model: dict[str, dict] = {}
        for model in models:
            model_rs = [r for r in full_ok if r.model == model]
            curve: dict[str, dict] = {}
            for fval in factor_domain:
                bucket = [r for r in model_rs if r.factors.get(factor_name) == fval]
                if not bucket:
                    continue
                flags = [r.correct for r in bucket]
                acc = sum(flags) / len(flags)
                curve[str(fval)] = {
                    "accuracy": round(acc, 4),
                    "n": len(bucket),
                    "ci95": _bootstrap_ci(flags),
                }
            by_model[model] = curve

        return AnalysisReport(
            name="sweep_curve",
            payload={"factor": factor_name, "by_model": by_model},
        )


PLUGIN = _SweepCurveAnalyzer()
