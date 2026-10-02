"""COPY-ME TEMPLATE - a custom analyzer (a metric over results).

WHAT AN ANALYZER IS
    An analyzer reads the flat list of Result rows a run produces and returns an
    AnalysisReport (a named payload dict, plus optional figures). This is how you add
    your OWN metric - orthogonally to scenes, probes and models. You can run a brand-new
    metric over the BUILT-IN scenes, or the built-in metrics over your new scene; they
    do not know about each other.

HOW TO USE
    1. Copy out, drop the underscore, rename the class and ``name``.
    2. Declare which conditions you need (subset of full/blind/oracle/report) and
       whether you need a varying factor (a swept parameter).
    3. renderprobe validate my_analysis/my_metric.py
    4. Add its name under ``analysis:`` in your experiment YAML.

READING RESULTS
    Each Result has: model, condition, probe, scene_id, factors, correct (bool),
    score (float), pred, gt, error, error_flag (None | parse_fail | model_error |
    gen_error). Decide how you treat error_flag rows - this template excludes model/gen
    errors from accuracy (they aren't the model getting the answer wrong).

This template computes accuracy per (model, condition) - deliberately tiny, so the
structure is easy to see.
"""
from __future__ import annotations

from collections import defaultdict

from renderprobe.core.schema import AnalysisReport, Result

SCHEMA_VERSION = "template_metric/1"


class _TemplateAnalyzer:
    name = "template_metric"
    requires_conditions = ["full"]        # subset of {"full","blind","oracle","report"}
    requires_varying_factor = False       # True if your metric needs >=2 factor values

    def analyze(self, results: list[Result]) -> AnalysisReport:
        # bucket[(model, condition)] -> [n_correct, n_total]
        buckets: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for r in results:
            if r.error_flag in ("model_error", "gen_error"):
                continue                  # not a wrong answer - a failed measurement
            key = (r.model, r.condition)
            buckets[key][1] += 1
            if r.correct:
                buckets[key][0] += 1

        by_model: dict[str, dict[str, float]] = defaultdict(dict)
        for (model, cond), (ok, n) in buckets.items():
            by_model[model][cond] = (ok / n) if n else 0.0

        return AnalysisReport(
            name=self.name,
            payload={"schema_version": SCHEMA_VERSION, "accuracy": dict(by_model)},
        )


PLUGIN = _TemplateAnalyzer()
