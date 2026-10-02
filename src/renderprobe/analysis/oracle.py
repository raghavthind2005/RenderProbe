"""Oracle diagnostic - the perception-vs-reasoning decomposition (the headline diagnostic).

Compares accuracy under the `full` condition (model must perceive AND reason)
against the `oracle` condition (GT perceptual primitives injected as text, so only
reasoning is tested). Per model:

    recovery = acc_oracle - acc_full

The comparison is PAIRED (same scene, two conditions), so the analysis is paired:
McNemar's exact test on the (full, oracle) correctness pairs, a paired-bootstrap CI
on recovery, and Wilson score intervals on each per-condition accuracy. See
`analysis/stats.py` for the primitives.

VALIDITY GUARDS - a recovery is only trustworthy if the oracle text is a
faithful, competent verbalization of the ground truth. Two guards defend that:

- Verbalization variants: when the probe offers several equivalent oracle templates
  (relational prose, coordinate list, JSON-as-text, ...), the runner tags each
  oracle Result with `meta["oracle_variant"]` and this analyzer reports recovery
  PER VARIANT under `by_variant`. If recovery swings with the template, the oracle
  is measuring verbalization quality, not perception - that must be visible.
- Ceiling check: on the "trivial" scene tier (tagged `meta["tier"]="trivial"`), a
  correct verbalization must let a competent reasoner reach near-ceiling accuracy.
  If trivial-tier oracle accuracy falls below the suspect threshold, the variant is
  flagged `oracle_prompt_suspect: true` - the template, not the model, is at fault.

The canonical variant's numbers are also surfaced at the per-model top level (so the
report stays readable and backward-compatible); the full breakdown is in `by_variant`.

Parse-failure accounting is first-class: a parse failure is NOT a wrong answer, so
it is excluded from the paired accuracy computation and reported as its own per-
condition rate. If those rates differ across conditions by more than a threshold,
the paired set is a biased subsample and recovery can be an artifact - the report
carries an explicit warning in that case.

Only scene_ids valid (no error_flag) in BOTH conditions are compared, so the paired
comparison is always apples-to-apples.
"""
from __future__ import annotations

from collections import defaultdict

from renderprobe.analysis.stats import (
    mcnemar_exact,
    paired_bootstrap_recovery,
    wilson_interval,
)
from renderprobe.core.schema import AnalysisReport, Result

SCHEMA_VERSION = "oracle/4"

# Per-condition parse-failure rates differing by more than this (in proportion,
# i.e. 0.05 = 5 percentage points) mean the paired set is a biased subsample -
# on its own that can manufacture a fake recovery signal, so we warn.
_PARSE_FAIL_WARN_THRESHOLD = 0.05

# On the trivial tier a valid oracle verbalization should reach ~ceiling (>=0.95).
# Below this we flag the template as suspect rather than trust its recovery.
_CEILING_SUSPECT_THRESHOLD = 0.9

# If recovery swings by more than this across equivalent oracle verbalizations, the
# oracle is measuring wording rather than perception - the result is "fragile" and the
# reasoning/perception verdict should not be trusted. Needs >=2 variants to assess.
_FRAGILITY_THRESHOLD = 0.10

_FRAMING_NOTE = (
    "Oracle injects the ground-truth perceptual primitives as text. recovery = "
    "acc_oracle - acc_full is an UPPER BOUND on the visual-pathway contribution "
    "(encoding + grounding), NOT an isolation of the vision encoder: a model that "
    "encodes the fact but fails to use it (a grounding/arbitration failure) also "
    "shows recovery. The perception-report probe and the 'decomposition' analyzer "
    "split that case from a genuine encoding failure. recovery ~ 0 (with a "
    "ceiling-validated prompt) is the clean direction: reasoning-limited. See "
    "docs/methodology.md. Report characterizes the model+probe pairing, not a "
    "context-free property of the model."
)


def _parse_fail_rate(results: list[Result]) -> tuple[float | None, int]:
    """(parse-failure rate, n_attempts) over every attempt in one condition."""
    n = len(results)
    if n == 0:
        return None, 0
    fails = sum(1 for r in results if r.error_flag == "parse_fail")
    return round(fails / n, 4), n


class _OracleAnalyzer:
    name = "oracle"
    requires_conditions = ["full", "oracle"]
    requires_varying_factor = False

    def analyze(self, results: list[Result]) -> AnalysisReport:
        # full is variant-agnostic (one per scene); oracle is grouped by variant.
        full_all: dict[str, list[Result]] = defaultdict(list)
        full_valid: dict[str, dict[str, Result]] = defaultdict(dict)
        oracle_all: dict[str, dict[str, list[Result]]] = defaultdict(lambda: defaultdict(list))
        oracle_valid: dict[str, dict[str, dict[str, Result]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        canonical_variant: dict[str, str] = {}

        for r in results:
            if r.condition == "full":
                full_all[r.model].append(r)
                if r.error_flag is None:
                    full_valid[r.model][r.scene_id] = r
            elif r.condition == "oracle":
                variant = r.meta.get("oracle_variant", "default")
                # First oracle variant seen per model = canonical (runner emits it first).
                canonical_variant.setdefault(r.model, variant)
                oracle_all[r.model][variant].append(r)
                if r.error_flag is None:
                    oracle_valid[r.model][variant][r.scene_id] = r

        models = sorted(set(full_all) & set(oracle_all))
        rows: dict[str, dict] = {}
        any_suspect = False
        for model in models:
            pf_full, n_full = _parse_fail_rate(full_all[model])
            fvalid = full_valid.get(model, {})

            by_variant: dict[str, dict] = {}
            for variant, ovariant_all in oracle_all[model].items():
                vrow = self._variant_row(
                    fvalid,
                    oracle_valid[model].get(variant, {}),
                    ovariant_all,
                    pf_full,
                    n_full,
                )
                by_variant[variant] = vrow
                any_suspect = any_suspect or vrow["oracle_prompt_suspect"]

            canonical = canonical_variant.get(model)
            if canonical not in by_variant:
                canonical = next(iter(by_variant))
            # Top level mirrors the canonical variant (readable + backward-compatible).
            row = dict(by_variant[canonical])
            row["canonical_variant"] = canonical
            row["by_variant"] = by_variant
            # Fragility: how far recovery swings across equivalent verbalizations. A large
            # spread means the oracle measures wording, not perception (empirical companion
            # to the offline oracle_solver completeness certificate). Needs >=2 variants.
            recoveries = [v["recovery"] for v in by_variant.values() if v["recovery"] is not None]
            if len(recoveries) >= 2:
                spread = round(max(recoveries) - min(recoveries), 4)
                row["oracle_variant_fragility"] = spread
                row["oracle_variant_fragile"] = spread > _FRAGILITY_THRESHOLD
            else:
                row["oracle_variant_fragility"] = None   # single variant: not assessable
                row["oracle_variant_fragile"] = False
            rows[model] = row

        return AnalysisReport(
            name="oracle",
            payload={
                "schema_version": SCHEMA_VERSION,
                "by_model": rows,
                "parse_failure_threshold": _PARSE_FAIL_WARN_THRESHOLD,
                "ceiling_suspect_threshold": _CEILING_SUSPECT_THRESHOLD,
                "fragility_threshold": _FRAGILITY_THRESHOLD,
                "any_oracle_prompt_suspect": any_suspect,
                "note": _FRAMING_NOTE,
            },
        )

    def _variant_row(
        self,
        full_valid: dict[str, Result],
        oracle_valid: dict[str, Result],
        oracle_all: list[Result],
        pf_full: float | None,
        n_full: int,
    ) -> dict:
        pf_oracle, n_oracle = _parse_fail_rate(oracle_all)

        # Ceiling check: trivial-tier oracle accuracy over ALL trivial attempts, so a
        # template so bad it induces parse failures is penalised too (not just wrong
        # answers). No trivial scenes in the run => can't check => not suspect.
        trivial = [r for r in oracle_all if r.meta.get("tier") == "trivial"]
        if trivial:
            trivial_acc = round(sum(1 for r in trivial if r.correct) / len(trivial), 4)
            suspect = trivial_acc < _CEILING_SUSPECT_THRESHOLD
        else:
            trivial_acc, suspect = None, False

        row = {
            "parse_failure_rate_full": pf_full,
            "parse_failure_rate_oracle": pf_oracle,
            "n_full_attempts": n_full,
            "n_oracle_attempts": n_oracle,
            "parse_failure_warning": self._parse_warning(pf_full, pf_oracle),
            "trivial_oracle_acc": trivial_acc,
            "oracle_prompt_suspect": suspect,
        }

        paired = set(full_valid) & set(oracle_valid)
        if not paired:
            row.update({
                "n_paired": 0,
                "acc_full": None,
                "acc_oracle": None,
                "recovery": None,
                "acc_full_ci": None,
                "acc_oracle_ci": None,
                "recovery_ci": None,
                "p_value": None,
                "n_discordant": 0,
            })
            return row

        pairs = [
            (full_valid[sid].correct, oracle_valid[sid].correct) for sid in sorted(paired)
        ]
        n = len(pairs)
        n_full_correct = sum(1 for f, _ in pairs if f)
        n_oracle_correct = sum(1 for _, o in pairs if o)
        # Discordant pairs for McNemar: b = full-right & oracle-wrong; c = reverse.
        b = sum(1 for f, o in pairs if f and not o)
        c = sum(1 for f, o in pairs if o and not f)

        row.update({
            "n_paired": n,
            "acc_full": round(n_full_correct / n, 4),
            "acc_oracle": round(n_oracle_correct / n, 4),
            "recovery": round((n_oracle_correct - n_full_correct) / n, 4),
            "acc_full_ci": wilson_interval(n_full_correct, n),
            "acc_oracle_ci": wilson_interval(n_oracle_correct, n),
            "recovery_ci": paired_bootstrap_recovery(pairs),
            "p_value": round(mcnemar_exact(b, c), 6),
            "n_discordant": b + c,
        })
        return row

    @staticmethod
    def _parse_warning(pf_full: float | None, pf_oracle: float | None) -> str | None:
        if pf_full is None or pf_oracle is None:
            return None
        if abs(pf_full - pf_oracle) > _PARSE_FAIL_WARN_THRESHOLD:
            return (
                f"parse-failure rates differ across conditions "
                f"(full={pf_full}, oracle={pf_oracle}, "
                f"threshold={_PARSE_FAIL_WARN_THRESHOLD}); the paired set is a "
                f"biased subsample and the recovery signal may be an artifact."
            )
        return None


PLUGIN = _OracleAnalyzer()
