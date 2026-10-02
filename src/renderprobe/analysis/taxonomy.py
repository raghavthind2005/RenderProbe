"""Failure-taxonomy analyzer - WHY each answer was wrong, not just how often.

Rule-based classification against exact ground truth (no LLM judge needed): every
Result is mapped to one category, and categories are counted PER (model, condition)
so the taxonomy composes with the oracle diagnostic. That composition is the point -
e.g. "relation inversions dominate under `full` but vanish under `oracle`" (a
perception failure) reads very differently from "off-by-one counting persists under
`oracle`" (a reasoning failure the ground-truth text didn't fix).

`classify_result` is the single source of truth and is reused by the UI inspect tab,
so a case's category always matches its taxonomy bucket.

Categories
----------
Shared (any probe):  correct, parse_fail, refusal, model_error, gen_error
Per parsed-but-wrong answer, by probe:
  spatial_relation          -> relation_inversion
  count                     -> off_by_one_over / off_by_one_under /
                               off_by_many_over / off_by_many_under
  identity_under_occlusion  -> hallucination
  (unknown/third-party probe -> incorrect)

Honest scope: finer buckets the design sketch imagined - "wrong-object reference" /
"distractor confusion" for spatial, "identity substitution" for occlusion - need
signals we don't collect (a reasoning trace, or a multi-object color palette in the
scene). A bare "left"/"right" or single color can't distinguish them, so we do NOT
invent categories we can't populate; those failures fold into relation_inversion /
hallucination. Extend here (and enrich the scene ground truth) to separate them.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable

from renderprobe.core.schema import AnalysisReport, Result

SCHEMA_VERSION = "taxonomy/1"

_REFUSAL_PAT = re.compile(
    r"\b(cannot|can'?t|can not|unable|not sure|unsure|don'?t know|do not know|"
    r"no idea|unclear|hard to tell|impossible|not enough)\b",
    re.IGNORECASE,
)


def _is_refusal(raw: str | None) -> bool:
    return bool(raw and _REFUSAL_PAT.search(raw))


def _classify_spatial(r: Result) -> str:
    # Binary left/right: a parsed-but-wrong answer is a flipped relation.
    return "relation_inversion"


def _classify_count(r: Result) -> str:
    try:
        delta = int(r.pred) - int(r.gt)
    except (TypeError, ValueError):
        return "incorrect"
    magnitude = "off_by_one" if abs(delta) == 1 else "off_by_many"
    direction = "over" if delta > 0 else "under"
    return f"{magnitude}_{direction}"


def _classify_identity(r: Result) -> str:
    # Only the target is colored in the scene, so a wrong color names an absent
    # object -> hallucination (see the module docstring on why substitution needs
    # richer scene ground truth to detect).
    return "hallucination"


_PROBE_CLASSIFIERS: dict[str, Callable[[Result], str]] = {
    "spatial_relation": _classify_spatial,
    "count": _classify_count,
    "identity_under_occlusion": _classify_identity,
}


def classify_result(r: Result) -> str:
    """Map one Result to its taxonomy category. Pure function of Result fields."""
    if r.correct and r.error_flag is None:
        return "correct"
    if r.error_flag in ("model_error", "gen_error"):
        return r.error_flag
    if r.error_flag == "parse_fail":
        return "refusal" if _is_refusal(r.raw) else "parse_fail"
    # Parsed a well-formed answer that was wrong -> probe-specific bucket.
    classifier = _PROBE_CLASSIFIERS.get(r.probe)
    return classifier(r) if classifier else "incorrect"


class _TaxonomyAnalyzer:
    name = "taxonomy"
    requires_conditions = ["full"]      # runs whenever full exists; also folds in
    requires_varying_factor = False     # blind/oracle when present

    def analyze(self, results: list[Result]) -> AnalysisReport:
        # (model, condition) -> Counter of categories
        buckets: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for r in results:
            buckets[r.model][r.condition][classify_result(r)] += 1

        by_model: dict[str, dict] = {}
        for model, by_condition in buckets.items():
            cond_out: dict[str, dict] = {}
            for condition, counter in by_condition.items():
                n = sum(counter.values())
                cond_out[condition] = {
                    "n": n,
                    "categories": {
                        cat: {"count": c, "proportion": round(c / n, 4)}
                        for cat, c in sorted(counter.items())
                    },
                }
            by_model[model] = cond_out

        return AnalysisReport(
            name="taxonomy",
            payload={
                "schema_version": SCHEMA_VERSION,
                "by_model": by_model,
                "note": (
                    "Rule-based failure categories per (model, condition). Compare a "
                    "category's proportion across conditions to separate perception "
                    "failures (vanish under oracle) from reasoning failures (persist)."
                ),
            },
        )


PLUGIN = _TaxonomyAnalyzer()
