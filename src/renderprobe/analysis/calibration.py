"""Calibration analyzer - is the model's stated confidence trustworthy?

Requires elicited confidences (run with ``elicit_confidence: true``). For each
(model, condition) it computes the Expected Calibration Error (ECE) under BOTH binning
schemes - equal-width (fixed [0,1] bins) and equal-mass (equal-count bins) - because the
two disagree when confidences cluster, and reporting only one hides that. Bins are
configurable (default 10). Per-condition reporting means the on-brand question falls out
for free: *is a model better calibrated reasoning from oracle text than from pixels?*

Never imputes: results whose confidence did not parse are excluded, not defaulted. If a
(model, condition) cell has no parseable confidence at all, its ECE is None with a
``reason`` rather than a crash or a fake 0.

The reliability-diagram DATA lives in the payload (bin confidence/accuracy/count);
``save_reliability_diagram`` renders it to PNG with a lazy matplotlib import, so the
analyzer itself stays dependency-light (same split as sweep_curve vs. the UI plot).
"""
from __future__ import annotations

from collections import defaultdict

from renderprobe.core.schema import AnalysisReport, Result

SCHEMA_VERSION = "calibration/1"
_DEFAULT_BINS = 10


def _bin_stats(chunk: list[tuple[float, bool]]) -> dict:
    n = len(chunk)
    conf = sum(c for c, _ in chunk) / n
    acc = sum(1 for _, ok in chunk if ok) / n
    return {"conf": round(conf, 4), "acc": round(acc, 4), "count": n}


def ece_equal_width(
    pairs: list[tuple[float, bool]], n_bins: int = _DEFAULT_BINS
) -> tuple[float | None, list[dict]]:
    """ECE with fixed-width [0,1] bins. Returns (ece, reliability_bins)."""
    if not pairs:
        return None, []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, ok))
    n = len(pairs)
    ece = 0.0
    rel: list[dict] = []
    for chunk in bins:
        if not chunk:
            continue
        s = _bin_stats(chunk)
        ece += (s["count"] / n) * abs(s["acc"] - s["conf"])
        rel.append(s)
    return round(ece, 4), rel


def ece_equal_mass(
    pairs: list[tuple[float, bool]], n_bins: int = _DEFAULT_BINS
) -> tuple[float | None, list[dict]]:
    """ECE with equal-count bins (quantile binning). Returns (ece, reliability_bins)."""
    if not pairs:
        return None, []
    ordered = sorted(pairs, key=lambda p: p[0])
    n = len(ordered)
    ece = 0.0
    rel: list[dict] = []
    for i in range(n_bins):
        chunk = ordered[i * n // n_bins:(i + 1) * n // n_bins]
        if not chunk:
            continue
        s = _bin_stats(chunk)
        ece += (s["count"] / n) * abs(s["acc"] - s["conf"])
        rel.append(s)
    return round(ece, 4), rel


class _CalibrationAnalyzer:
    name = "calibration"
    requires_conditions = ["full"]
    requires_varying_factor = False

    def __init__(self, n_bins: int = _DEFAULT_BINS) -> None:
        self.n_bins = n_bins

    def analyze(self, results: list[Result]) -> AnalysisReport:
        # (model, condition) -> list[(confidence, correct)] for parsed confidences only
        pairs: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seen: dict[str, set] = defaultdict(set)
        for r in results:
            seen[r.model].add(r.condition)
            if r.confidence is not None:
                pairs[r.model][r.condition].append((r.confidence, bool(r.correct)))

        by_model: dict[str, dict] = {}
        for model, conditions in seen.items():
            cond_out: dict[str, dict] = {}
            for condition in sorted(conditions):
                cell = pairs[model].get(condition, [])
                if not cell:
                    cond_out[condition] = {
                        "n": 0,
                        "ece_equal_width": None,
                        "ece_equal_mass": None,
                        "reliability_equal_width": None,
                        "reliability_equal_mass": None,
                        "reason": "no parseable confidences (run with elicit_confidence)",
                    }
                    continue
                ew, rel_w = ece_equal_width(cell, self.n_bins)
                em, rel_m = ece_equal_mass(cell, self.n_bins)
                cond_out[condition] = {
                    "n": len(cell),
                    "ece_equal_width": ew,
                    "ece_equal_mass": em,
                    "reliability_equal_width": rel_w,
                    "reliability_equal_mass": rel_m,
                    "reason": None,
                }
            by_model[model] = cond_out

        return AnalysisReport(
            name="calibration",
            payload={
                "schema_version": SCHEMA_VERSION,
                "n_bins": self.n_bins,
                "by_model": by_model,
                "note": (
                    "ECE per (model, condition) under equal-width and equal-mass "
                    "binning. Compare conditions: lower ECE under oracle than full "
                    "means the model is better calibrated reasoning from text than "
                    "from pixels. Confidences are elicited, never imputed."
                ),
            },
        )


def save_reliability_diagram(report: AnalysisReport, out_dir, binning: str = "equal_width"):
    """Render one reliability-diagram PNG per (model, condition) with data. Returns the
    written paths. Lazy matplotlib import so the analyzer has no plotting dependency."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = f"reliability_{binning}"
    written = []
    for model, conditions in report.payload.get("by_model", {}).items():
        for condition, cell in conditions.items():
            rel = cell.get(key)
            if not rel:
                continue
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)  # perfect line
            ax.plot([b["conf"] for b in rel], [b["acc"] for b in rel], marker="o")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
            ece = cell.get(f"ece_{binning}")
            ax.set_title(f"{model} - {condition} (ECE={ece})")
            fig.tight_layout()
            safe = f"reliability_{model}_{condition}_{binning}".replace("/", "-")
            path = out / f"{safe}.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            written.append(path)
    return written


PLUGIN = _CalibrationAnalyzer()
