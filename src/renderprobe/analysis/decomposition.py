"""Three-way failure decomposition (the corrected thesis - see docs/methodology.md).

The oracle diagnostic's `recovery = acc_oracle - acc_full` cleanly separates
"fails even given the facts as text" (reasoning) from "the visual pathway was the
limit" - but it does NOT, on its own, separate a genuine *encoding* failure (the
vision tower never extracted the fact) from a *grounding/arbitration* failure (it
extracted the fact but didn't use it). Labeling the whole visual-pathway bucket
"perception" over-claims; the literature documents that VLMs frequently encode the
needed information yet fail downstream (arXiv:2604.09364).

This analyzer adds the missing split using the optional `report` condition - a
perception-only sub-task over the image - to classify each model into one bucket:

    encoding-limited            low  acc_report
    grounding/integration-limited   high acc_report, low acc_full, high acc_oracle
    reasoning-limited           recovery ≈ 0 (fails even given text premises)

`recovery` is therefore an UPPER BOUND on the visual-pathway contribution, not an
isolation of the vision encoder. Requires `full` + `oracle`; uses `report` and
`blind` when present (report absent => the encoding/grounding split is left
unresolved, reported honestly rather than guessed).
"""
from __future__ import annotations

from renderprobe.analysis.stats import bootstrap_label_stability
from renderprobe.core.schema import AnalysisReport, Result

SCHEMA_VERSION = "decomposition/2"

# Interpretation thresholds (documented, not tuned to any model).
_HIGH = 0.85        # an accuracy considered near-ceiling / "high"
_MATERIAL = 0.10    # a recovery (or gain) considered materially non-zero
_SOLVED = 0.90      # acc_full at/above this => no material failure to decompose
_STABLE = 0.95      # bootstrap agreement at/above this => the verdict is firm
_N_BOOT = 2000      # resamples behind the stability estimate


def _counts_toward_accuracy(r: Result) -> bool:
    """Whether a valid row feeds the condition accuracies the cascade reads.

    Every condition counts except a non-primary report: a probe may ask several
    perceptual sub-questions of different strictness, and pooling them would make
    acc_report a blend of hard and easy questions that answers neither.
    """
    return r.condition != "report" or bool(r.meta.get("report_primary", True))


def _acc(results: list[Result], model: str, condition: str) -> tuple[float | None, int]:
    rs = [
        r for r in results
        if r.model == model and r.condition == condition and r.error_flag is None
        and _counts_toward_accuracy(r)
    ]
    if not rs:
        return None, 0
    return sum(r.correct for r in rs) / len(rs), len(rs)


def _per_scene_counts(
    results: list[Result], model: str
) -> list[dict[str, tuple[int, int]]]:
    """One entry per scene: condition -> (n_correct, n_rows), valid rows only.

    A scene can contribute several rows to one condition (the oracle runs once per
    verbalization variant), so counts are kept rather than a single bool. Pooling
    these reproduces `_acc` exactly; `test_decomposition` pins that agreement.
    """
    by_scene: dict[str, dict[str, list[int]]] = {}
    for r in results:
        if r.model != model or r.error_flag is not None:
            continue
        if not _counts_toward_accuracy(r):
            continue
        cell = by_scene.setdefault(r.scene_id, {}).setdefault(r.condition, [0, 0])
        cell[0] += 1 if r.correct else 0
        cell[1] += 1
    return [
        {c: (k, n) for c, (k, n) in conds.items()}
        for conds in by_scene.values()
    ]


def _classify(
    acc_full: float | None,
    acc_oracle: float | None,
    acc_report: float | None,
) -> tuple[str, str]:
    """Return (bottleneck, one-line explanation). See docs/methodology.md §3."""
    if acc_full is None or acc_oracle is None:
        return "insufficient-data", "need both full and oracle conditions to decompose"
    recovery = acc_oracle - acc_full
    if acc_full >= _SOLVED:
        return "none", f"solved from the image (acc_full={acc_full:.2f}); nothing to decompose"
    if recovery < -_MATERIAL:
        return (
            "not-vision-limited",
            f"image beats the text oracle (recovery={recovery:+.2f}); the text "
            "linearization loses structure the image supplies",
        )
    if recovery <= _MATERIAL:
        return (
            "reasoning-limited",
            f"handing over the ground-truth premises as text did not help "
            f"(recovery={recovery:+.2f}); the failure survives perfect perception",
        )
    # recovery > _MATERIAL => visual-pathway-limited; split with the report probe.
    if acc_report is None:
        return (
            "vision-pathway-limited",
            f"recovery={recovery:+.2f} points to the visual pathway, but no "
            "perception-report probe was available to split encoding vs. grounding",
        )
    # Does perception EXPLAIN the task failure? A fixed ceiling on acc_report cannot
    # answer that. Measured live, gemma-3-27b reported the primitive on 0.83 of the road
    # maps while solving 0.33 of them: under a 0.85 cut that read as "cannot even report
    # the primitive", which is plainly false about a model getting it five times in six.
    # What separates the two cases is whether reporting goes BETTER than doing - if it
    # does, the fact was available and went unused; if it does not, perception is what
    # the task is waiting on. The ceiling stays as a fast path for a report that is
    # near-perfect while the task sits just under the solved line.
    #
    # The two accuracies answer different questions with different guessing rates, so
    # this is a comparison of like with nearly-like rather than an identity. It is
    # nonetheless the right direction to test, and the margin has been large in every
    # case seen so far: +0.50 and +0.67 where grounding is the answer, -0.17 and -0.11
    # where encoding is.
    if acc_report >= _HIGH or (acc_report - acc_full) > _MATERIAL:
        return (
            "grounding/integration-limited",
            f"the model reports the primitive far more often than it solves the task "
            f"(acc_report={acc_report:.2f} vs acc_full={acc_full:.2f}) - it extracts "
            "the fact and does not use it",
        )
    return (
        "encoding-limited",
        f"reporting the primitive goes no better than doing the task "
        f"(acc_report={acc_report:.2f} vs acc_full={acc_full:.2f}); perception is the "
        "binding constraint",
    )


class _DecompositionAnalyzer:
    name = "decomposition"
    requires_conditions = ["full", "oracle"]   # report + blind used when present
    requires_varying_factor = False

    def analyze(self, results: list[Result]) -> AnalysisReport:
        models = sorted({r.model for r in results})
        by_model: dict[str, dict] = {}
        for model in models:
            acc_full, n_full = _acc(results, model, "full")
            acc_blind, n_blind = _acc(results, model, "blind")
            acc_oracle, n_oracle = _acc(results, model, "oracle")
            acc_report, n_report = _acc(results, model, "report")

            recovery = (
                round(acc_oracle - acc_full, 4)
                if acc_full is not None and acc_oracle is not None else None
            )
            visual_gain = (
                round(acc_full - acc_blind, 4)
                if acc_full is not None and acc_blind is not None else None
            )
            bottleneck, explanation = _classify(acc_full, acc_oracle, acc_report)

            # How often the verdict survives resampling the scenes. The cascade is a
            # step function of the accuracies, so a point estimate sitting near a
            # threshold can flip on a single item; say so rather than imply certainty.
            stab = bootstrap_label_stability(
                _per_scene_counts(results, model),
                lambda a: _classify(a.get("full"), a.get("oracle"), a.get("report"))[0],
                n_boot=_N_BOOT,
            )
            stability = stab["stability"] if stab else None
            alternatives = (
                {k: v for k, v in stab["distribution"].items() if k != bottleneck}
                if stab else {}
            )

            by_model[model] = {
                "acc_full": _r(acc_full),
                "acc_blind": _r(acc_blind),
                "acc_oracle": _r(acc_oracle),
                "acc_report": _r(acc_report),
                "recovery": recovery,
                "visual_gain": visual_gain,
                "bottleneck": bottleneck,
                "explanation": explanation,
                "report_available": acc_report is not None,
                "bottleneck_stability": stability,
                "bottleneck_confident": (
                    None if stability is None else stability >= _STABLE
                ),
                "bottleneck_alternatives": alternatives,
                "n": {"full": n_full, "blind": n_blind,
                      "oracle": n_oracle, "report": n_report},
            }

        return AnalysisReport(
            name=self.name,
            payload={
                "schema_version": SCHEMA_VERSION,
                "thresholds": {
                    "high": _HIGH, "material_recovery": _MATERIAL, "solved": _SOLVED,
                    "stable": _STABLE,
                },
                "by_model": by_model,
                "note": (
                    "Three-way decomposition (docs/methodology.md). recovery = "
                    "acc_oracle - acc_full is an UPPER BOUND on the visual-pathway "
                    "contribution, not an isolation of the vision encoder; the "
                    "perception-report probe (acc_report) splits the visual-pathway "
                    "case into encoding-limited vs. grounding/integration-limited. "
                    "acc_oracle aggregates over any oracle verbalization variants run. "
                    "bottleneck_stability is the share of scene-level bootstrap "
                    "resamples returning the same verdict; below the `stable` "
                    "threshold the label sits near a cutoff and "
                    "bottleneck_alternatives names what it competes with."
                ),
            },
        )


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


PLUGIN = _DecompositionAnalyzer()
