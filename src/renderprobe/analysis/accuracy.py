"""Accuracy + Mean Relative Error analyzer. No gates - runs on any result set."""
from __future__ import annotations

from collections import defaultdict

from renderprobe.core.schema import AnalysisReport, Result


class _AccuracyAnalyzer:
    name = "accuracy"
    requires_conditions: list[str] = []
    requires_varying_factor = False

    def analyze(self, results: list[Result]) -> AnalysisReport:
        full = [r for r in results if r.condition == "full" and r.error_flag is None]
        if not full:
            return AnalysisReport(
                name=self.name, payload={"error": "no valid full-condition results"}
            )

        by_model: dict[str, list[Result]] = defaultdict(list)
        for r in full:
            by_model[r.model].append(r)

        rows = {}
        for model, rs in by_model.items():
            acc = sum(r.correct for r in rs) / len(rs)
            errors = [r.error for r in rs if r.error is not None]
            mre = (sum(errors) / len(errors)) if errors else None
            rows[model] = {
                "n": len(rs),
                "accuracy": round(acc, 4),
                "mean_error": round(mre, 4) if mre is not None else None,
            }

        return AnalysisReport(name=self.name, payload={"by_model": rows})


PLUGIN = _AccuracyAnalyzer()
