"""Visual-gain analyzer: acc_full - acc_blind per model.

Low gain => the model is ignoring the image (VQA-CP / VLM-RobustBench lineage).
Requires both 'full' and 'blind' conditions.
"""
from __future__ import annotations

from renderprobe.core.schema import AnalysisReport, Result


class _VisualGainAnalyzer:
    name = "visual_gain"
    requires_conditions = ["full", "blind"]
    requires_varying_factor = False

    def analyze(self, results: list[Result]) -> AnalysisReport:
        ok = [r for r in results if r.error_flag is None]
        full = {(r.scene_id, r.model): r for r in ok if r.condition == "full"}
        blind = {(r.scene_id, r.model): r for r in ok if r.condition == "blind"}

        models = {r.model for r in results}
        rows = {}
        for model in sorted(models):
            f_results = [r for (_, m), r in full.items() if m == model]
            b_results = [r for (_, m), r in blind.items() if m == model]
            if not f_results or not b_results:
                continue
            acc_full = sum(r.correct for r in f_results) / len(f_results)
            acc_blind = sum(r.correct for r in b_results) / len(b_results)
            gain = acc_full - acc_blind
            rows[model] = {
                "acc_full": round(acc_full, 4),
                "acc_blind": round(acc_blind, 4),
                "visual_gain": round(gain, 4),
            }

        return AnalysisReport(name=self.name, payload={"by_model": rows})


PLUGIN = _VisualGainAnalyzer()
