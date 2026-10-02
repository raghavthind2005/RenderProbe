"""Presentation logic for the UI - pure functions, no gradio dependency.

Everything here turns ``list[Result]`` / ``list[AnalysisReport]`` into plain
table/series structures the gradio layer can render directly. Keeping it
gradio-free means this logic is unit-tested in isolation and the gradio glue in
``app.py`` stays thin.
"""
from __future__ import annotations

from typing import Any

from renderprobe import renderers
from renderprobe.analysis.taxonomy import classify_result
from renderprobe.core import credentials
from renderprobe.core.schema import AnalysisReport, Result

# Report lookup


def report_by_name(reports: list[AnalysisReport], name: str) -> AnalysisReport | None:
    """Return the first report with the given name, or None."""
    for r in reports:
        if r.name == name:
            return r
    return None


def default_scene(registry: Any) -> str | None:
    """Which scene the UI opens on.

    Not simply the first one registered. That ordering is alphabetical, which put
    `polycube` in front: a CPU path trace costing about three seconds on load and again
    on every dial change, and the one scene this repo's own results say sits at chance.
    The first thing somebody saw was a spinner, in front of the example that does not
    work.

    The rule is to prefer a scene that draws instantly, which is a property a scene
    already declares through its renderer. It generalizes: somebody who adds a fast
    scene and a slow one opens on the fast one, without either having to know about
    this function.
    """
    from renderprobe.core.runner import HEAVY_RENDERERS

    names = registry.list_scenes()
    if not names:
        return None
    for name in names:
        try:
            renderer = getattr(registry.get_scene(name), "default_renderer", None)
        except Exception:
            continue
        if renderer not in HEAVY_RENDERERS:
            return name
    return names[0]


# Run tab: default params for a scene (read from its params_schema)


def compatible_probe(
    registry: Any, scene_name: str, probe_names: list[str]
) -> str | None:
    """Pick the probe that fits a scene, so the UI can auto-pair them.

    A probe fits when the scene's ground_truth provides all of the probe's
    ``requires_gt_fields``; a same-named probe is preferred (the convention for
    the showcase scenes). Falls back to the first probe if nothing clearly fits,
    so selecting a scene never leaves the run un-runnable by default.
    """
    fallback = probe_names[0] if probe_names else None
    try:
        gen = registry.get_scene(scene_name)
        with renderers.using_registry(registry):   # resolve --plugin-dir renderers too
            sample = gen.generate(default_params(gen), 0)
        keys = set(sample.ground_truth)
    except Exception:
        return fallback
    fits = [
        p for p in probe_names
        if set(getattr(registry.get_probe(p), "requires_gt_fields", [])) <= keys
    ]
    if not fits:
        return fallback
    return scene_name if scene_name in fits else fits[0]


def default_params(scene_gen: Any) -> dict[str, Any]:
    """Build a default params dict from a scene generator's params_schema.

    Each schema entry is expected to carry a ``default``; entries without one
    are skipped. The UI seeds its params editor with this so a user can run a
    scene immediately and then edit a value into a list to sweep it.
    """
    schema = getattr(scene_gen, "params_schema", {}) or {}
    out: dict[str, Any] = {}
    for key, spec in schema.items():
        if isinstance(spec, dict) and "default" in spec:
            out[key] = spec["default"]
    return out


def param_specs(scene_gen: Any) -> list[dict[str, Any]]:
    """Ordered, UI-ready knob specs from a scene's params_schema - one per adjustable
    parameter, carrying its range so the UI can build labeled sliders (dials) with
    visible bounds instead of a bare text box.

    Each entry: {name, type ("int"|"float"), min, max, step, default}. Order follows
    the schema's insertion order, which the runner and ``params_from_values`` rely on.
    """
    schema = getattr(scene_gen, "params_schema", {}) or {}
    out: list[dict[str, Any]] = []
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        typ = spec.get("type", "float")
        lo = spec.get("min", 0)
        hi = spec.get("max", lo + 10)
        default = spec.get("default", lo)
        if typ == "int":
            step: float = 1
        else:
            span = hi - lo
            step = round(span / 20, 4) if span > 0 else 0.05
            step = step or 0.05
        out.append({"name": name, "type": typ, "min": lo, "max": hi,
                    "step": step, "default": default})
    return out


def params_from_values(scene_gen: Any, values: list[Any]) -> dict[str, Any]:
    """Map ordered slider values back to a typed params dict (inverse of the sliders
    the UI built from ``param_specs``). Values are cast to each param's declared type;
    ``None`` (a hidden/unset slot) falls back to the param's default. Extra values
    beyond the schema's parameters are ignored.
    """
    specs = param_specs(scene_gen)
    params: dict[str, Any] = {}
    for spec, v in zip(specs, values):
        if v is None:
            v = spec["default"]
        params[spec["name"]] = int(round(v)) if spec["type"] == "int" else float(v)
    return params


# Run tab: per-condition result table + analyzer summary tables

RESULT_COLUMNS = [
    "scene_id",
    "model",
    "condition",
    "pred",
    "gt",
    "correct",
    "error_flag",
]


def results_table(results: list[Result]) -> list[list[Any]]:
    """Flatten Results into rows aligned with RESULT_COLUMNS."""
    rows: list[list[Any]] = []
    for r in results:
        rows.append(
            [
                r.scene_id,
                r.model,
                r.condition,
                _short(r.pred),
                _short(r.gt),
                "yes" if r.correct else "no",
                r.error_flag or "",
            ]
        )
    return rows


def accuracy_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    cols = ["model", "n", "accuracy", "mean_error"]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows = [
        [m, d["n"], d["accuracy"], _fmt(d.get("mean_error"))]
        for m, d in report.payload["by_model"].items()
    ]
    return cols, rows


def visual_gain_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    cols = ["model", "acc_full", "acc_blind", "visual_gain"]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows = [
        [m, d["acc_full"], d["acc_blind"], d["visual_gain"]]
        for m, d in report.payload["by_model"].items()
    ]
    return cols, rows


def oracle_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    """Oracle recovery per model, WITH the paired-statistics fields:
    the recovery 95% bootstrap CI, the McNemar exact p-value, the discordant-pair
    count the test is powered by, and the prompt-suspect flag."""
    cols = [
        "model", "n_paired", "acc_full", "acc_oracle", "recovery",
        "recovery 95% CI", "McNemar p", "n_discordant", "prompt suspect",
    ]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows = []
    for m, d in report.payload["by_model"].items():
        rows.append([
            m,
            d.get("n_paired", "n/a"),
            d.get("acc_full", "n/a"),
            d.get("acc_oracle", "n/a"),
            d.get("recovery", "n/a"),
            _fmt_ci(d.get("recovery_ci")),
            _fmt(d.get("p_value")),
            d.get("n_discordant", "n/a"),
            "yes" if d.get("oracle_prompt_suspect") else "no",
        ])
    return cols, rows


def decomposition_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    """Three-way bottleneck per model (docs/methodology.md) - the corrected headline.

    recovery bounds the visual-pathway share; acc_report splits it into
    encoding-limited vs. grounding/integration-limited.
    """
    cols = [
        "model", "bottleneck", "confidence", "acc_full", "acc_report", "acc_oracle",
        "recovery", "why",
    ]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows = []
    for m, d in report.payload["by_model"].items():
        rows.append([
            m,
            d.get("bottleneck", "n/a"),
            _fmt_stability(d),
            _fmt(d.get("acc_full")),
            _fmt(d.get("acc_report")),
            _fmt(d.get("acc_oracle")),
            _fmt(d.get("recovery")),
            d.get("explanation", ""),
        ])
    return cols, rows


def _fmt_stability(d: dict) -> str:
    """Verdict confidence as the share of bootstrap resamples agreeing with it.

    A label read off thresholds can flip on one item, so an unqualified verdict
    overstates what the data supports; the runner-up is named when it is close.
    """
    stab = d.get("bottleneck_stability")
    if stab is None:
        return "\u2014"
    if d.get("bottleneck_confident"):
        return f"firm ({stab:.0%})"
    alts = d.get("bottleneck_alternatives") or {}
    if alts:
        rival = max(alts.items(), key=lambda kv: kv[1])[0]
        return f"unsettled ({stab:.0%}, vs {rival})"
    return f"unsettled ({stab:.0%})"


def oracle_note(report: AnalysisReport | None) -> str:
    if report is None:
        return ""
    return report.payload.get("note", "")


def oracle_suspect_banner(report: AnalysisReport | None) -> str:
    """Loud markdown warning when any oracle prompt is suspect (ceiling
    check): if a *correct* verbalization can't lift a competent reasoner near
    ceiling on the trivial tier, the recovery signal indicts the PROMPT, not the
    model - so we surface it prominently rather than reporting a misleading number.
    Empty string when clean."""
    if report is None:
        return ""
    if not report.payload.get("any_oracle_prompt_suspect"):
        return ""
    return (
        "> **Oracle prompt suspect.** On the trivial tier a correct oracle "
        "verbalization did not reach near-ceiling accuracy, so at least one "
        "recovery number below reflects the *prompt template*, not the model's "
        "reasoning. Treat the flagged rows with caution."
    )


# Run tab: failure taxonomy and calibration / ECE


def taxonomy_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    """Long-format failure taxonomy: one row per (model, condition, category).

    Reading a category's proportion across conditions is the point - a bucket that
    is heavy under `full` but vanishes under `oracle` is a perception failure; one
    that persists under `oracle` is a reasoning failure.
    """
    cols = ["model", "condition", "category", "count", "proportion"]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows: list[list[Any]] = []
    for model, by_cond in report.payload["by_model"].items():
        for condition in _order_conditions(by_cond):
            cats = by_cond[condition].get("categories", {})
            for cat, d in sorted(cats.items()):
                rows.append([model, condition, cat, d["count"], d["proportion"]])
    return cols, rows


def calibration_rows(report: AnalysisReport | None) -> tuple[list[str], list[list[Any]]]:
    """Expected Calibration Error per (model, condition), both binning schemes.

    ECE is null (shown as 'n/a' with the reason) when no confidences were elicited,
    so the table never fabricates a calibration number it cannot compute.
    """
    cols = ["model", "condition", "n", "ECE (equal-width)", "ECE (equal-mass)", "note"]
    if report is None or "by_model" not in report.payload:
        return cols, []
    rows: list[list[Any]] = []
    for model, by_cond in report.payload["by_model"].items():
        for condition in _order_conditions(by_cond):
            d = by_cond[condition]
            rows.append([
                model, condition, d.get("n", 0),
                _fmt(d.get("ece_equal_width")),
                _fmt(d.get("ece_equal_mass")),
                d.get("reason") or "",
            ])
    return cols, rows


def reliability_series(
    report: AnalysisReport | None,
    condition: str = "full",
    binning: str = "equal_width",
) -> dict[str, Any]:
    """Reliability-diagram points per model for one condition: {model: {conf, acc}}.

    Points off the y=x diagonal show mis-calibration (above = under-confident,
    below = over-confident). Returns {} series when no confidences were collected.
    """
    key = f"reliability_{binning}"
    out: dict[str, Any] = {"condition": condition, "series": {}}
    if report is None or "by_model" not in report.payload:
        return out
    for model, by_cond in report.payload["by_model"].items():
        cell = by_cond.get(condition)
        if not cell:
            continue
        bins = cell.get(key) or []
        if not bins:
            continue
        out["series"][model] = {
            "conf": [b["conf"] for b in bins],
            "acc": [b["acc"] for b in bins],
            "count": [b["count"] for b in bins],
        }
    return out


# Sweep tab: accuracy-vs-factor series (with CI band) per model


def sweep_series(report: AnalysisReport | None) -> dict[str, Any]:
    """Convert a sweep_curve payload into plottable series.

    Returns::

        {
          "factor": str,
          "series": {model: {"x": [...], "y": [...], "lo": [...], "hi": [...]}},
        }

    x values are coerced to float when possible so the curve plots on a numeric
    axis; otherwise their original (sorted-string) order is preserved. ``lo``/
    ``hi`` mirror the bootstrap CI (falling back to the point estimate when a
    bucket had too few samples for a CI).
    """
    empty = {"factor": None, "series": {}}
    if report is None or "by_model" not in report.payload:
        return empty

    factor = report.payload.get("factor")
    series: dict[str, Any] = {}
    for model, curve in report.payload["by_model"].items():
        keys = sorted(curve, key=_sort_key)
        xs, ys, los, his = [], [], [], []
        for k in keys:
            cell = curve[k]
            x = _maybe_float(k)
            acc = cell["accuracy"]
            ci = cell.get("ci95")
            xs.append(x)
            ys.append(acc)
            if ci is not None:
                los.append(ci[0])
                his.append(ci[1])
            else:
                los.append(acc)
                his.append(acc)
        series[model] = {"x": xs, "y": ys, "lo": los, "hi": his}
    return {"factor": factor, "series": series}


# Inspect tab: navigable failure cases


def failure_cases(results: list[Result], condition: str = "full") -> list[dict[str, Any]]:
    """Return wrong-or-errored cases under a condition, for the inspect tab.

    Each case carries the scene_id so the UI can pull the matching image, plus
    the model's raw text and the parsed/ground-truth values for display.
    """
    cases: list[dict[str, Any]] = []
    for r in results:
        if r.condition != condition:
            continue
        if r.correct and r.error_flag is None:
            continue
        cases.append(
            {
                "scene_id": r.scene_id,
                "model": r.model,
                "condition": r.condition,
                "pred": r.pred,
                "gt": r.gt,
                "raw": r.raw,
                "error_flag": r.error_flag,
                # Why the adapter call failed, on model_error rows. Credential-scrubbed
                # by the runner. Empty for every other kind of failure.
                "detail": str(r.meta.get("model_error") or ""),
                # Failure-taxonomy bucket, from the same classifier the analyzer uses.
                "category": classify_result(r),
            }
        )
    return cases


def case_label(case: dict[str, Any]) -> str:
    """One-line human label for a failure case (dropdown entries)."""
    reason = case["error_flag"] or "wrong"
    return (
        f"{case['scene_id']}, {case['model']}, {reason} "
        f"(pred={case['pred']!r} gt={case['gt']!r})"
    )


# Helpers


def _short(val: Any, limit: int = 40) -> str:
    s = str(val)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _fmt(val: Any) -> str:
    return "" if val is None else str(val)


def _fmt_ci(ci: Any) -> str:
    """Render a [lo, hi] pair as '[lo, hi]', or 'n/a' when absent."""
    if not ci or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "n/a"
    return f"[{ci[0]}, {ci[1]}]"


_CONDITION_ORDER = {"full": 0, "blind": 1, "oracle": 2}


def _order_conditions(by_cond: dict[str, Any]) -> list[str]:
    """Stable, human-friendly condition order (full, blind, oracle, then any rest)."""
    return sorted(by_cond, key=lambda c: (_CONDITION_ORDER.get(c, 99), c))


def _maybe_float(s: str) -> float | str:
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _sort_key(s: str) -> tuple[int, float | str]:
    """Sort numeric-looking keys numerically, others lexically (numbers first)."""
    v = _maybe_float(s)
    return (0, v) if isinstance(v, float) else (1, s)


# ---------------------------------------------------------------------------
# Run tab: the model roster, and the credentials it needs
#
# Everything below reads adapter attributes through `getattr` with a default, because
# none of them is part of the ModelAdapter Protocol. An adapter that declares nothing
# still shows up - as a model that needs no key, which is exactly what the offline
# doubles in tests/fixtures are.
# ---------------------------------------------------------------------------

# How a roster status reads on screen. Kept here, not on the adapter, so the wording is
# a presentation choice rather than something a plugin author has to phrase correctly.
_STATUS_TEXT = {
    "serves": "serves",
    "down": "not serving",
    "unverified": "unverified",
}


def model_catalog(
    registry: Any, session_keys: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """One display record per registered model, including whether it can run right now.

    ``session_keys`` maps an environment-variable name to a key supplied for this
    browser session. It is consulted through ``credentials.scope``, so a key entered in
    the UI counts as present without being written anywhere.
    """
    out: list[dict[str, Any]] = []
    with credentials.scope(session_keys):
        for name in registry.list_models():
            model = registry.get_model(name)
            env = getattr(model, "api_key_env", None)
            provider = getattr(model, "provider", "") or (env or "no key needed")
            status = getattr(model, "status", "unverified")
            out.append({
                "name": name,
                "provider": provider,
                "api_key_env": env,
                "needs_key": bool(env),
                "key_available": (not env) or credentials.resolve(env) is not None,
                "status": status,
                "status_text": _STATUS_TEXT.get(status, status),
                "status_note": getattr(model, "status_note", "") or "",
                "is_open": bool(getattr(model, "is_open", False)),
                # Local weights are runnable without a key but cost a multi-gigabyte
                # download on first use, so they are never a default.
                "is_local": provider == "local weights",
            })
    return out


def model_choices(
    registry: Any,
    session_keys: dict[str, str] | None = None,
    runnable_only: bool = True,
) -> list[tuple[str, str]]:
    """``(label, value)`` pairs for the model selector.

    ``runnable_only`` hides the models whose key is missing, which is most of them for
    most people: the full roster is eleven entries across six providers, and listing all
    of them makes the one or two somebody can actually run harder to find, not easier.
    The label still carries the provider and the dated roster status, because "this
    endpoint was not serving a fortnight ago" is worth knowing before ticking a box.
    """
    choices: list[tuple[str, str]] = []
    for m in model_catalog(registry, session_keys):
        if runnable_only and not m["key_available"]:
            continue
        label = f"{m['name']} - {m['provider']}"
        if m["needs_key"] or m["is_local"]:
            label += f", {m['status_text']}"
        if m["needs_key"] and not m["key_available"]:
            label += f" (no {m['api_key_env']})"
        choices.append((label, m["name"]))
    return choices


def default_model_selection(
    registry: Any, session_keys: dict[str, str] | None = None
) -> list[str]:
    """Which models the Run tab opens with ticked.

    The rule is: tick something that would actually produce numbers if somebody pressed
    Run without reading anything, and otherwise tick nothing and let the key panel
    explain why. A model known not to be serving is never a default, and neither is a
    local-weights model, whose first run is a multi-gigabyte download.

    This used to be ``["mock"] or models[:1]``. When the offline doubles moved out of
    the package the fallback started selecting whichever model sorted first, which was
    one the roster had recorded as returning 401.
    """
    cat = [m for m in model_catalog(registry, session_keys) if not m["is_local"]]
    free = [m for m in cat if not m["needs_key"]]
    if free:
        return [free[0]["name"]]
    runnable = [m for m in cat if m["key_available"] and m["status"] != "down"]
    serving = [m for m in runnable if m["status"] == "serves"]
    best = serving or runnable
    return [best[0]["name"]] if best else []


def credential_markdown(
    registry: Any, session_keys: dict[str, str] | None = None
) -> str:
    """A compact status line for the key panel.

    This used to be a six-row table, one row per provider, which said the same thing
    three times over: the model labels already carry the provider and whether a key is
    present. What is left is what a table could not say in one glance - which variables
    are filled, where each came from, and how much of the roster that unlocks.
    """
    from renderprobe.models.openai_compatible import ROSTER_CHECKED

    cat = model_catalog(registry, session_keys)
    envs = key_envs(registry)
    have, missing = [], []
    with credentials.scope(session_keys):
        for env in envs:
            info = credentials.describe(env)
            if info["present"]:
                where = "this session" if info["source"] == "session" else "environment"
                have.append(f"`{env}` ({info['length']} chars, {where})")
            else:
                missing.append(f"`{env}`")

    runnable = [m for m in cat if m["key_available"]]
    # Which variables are NOT set is deliberately not listed: the provider dropdown
    # enumerates them and its hint reports the selected one, so repeating the whole
    # list here only costs height.
    if have:
        summary = ", ".join(have)
        if missing:
            summary += f". {len(missing)} other provider(s) not set"
    else:
        summary = f"none. {len(missing)} provider(s) available"
    lines = ["**Keys in use:** " + summary + "."]
    verb = "is" if len(runnable) == 1 else "are"
    lines.append(
        f"**{len(runnable)} of {len(cat)} models** {verb} runnable right now. "
        f"Endpoint statuses were probed on {ROSTER_CHECKED} and are a dated snapshot, "
        "not a live check."
    )
    if not have:
        keyless = [m["name"] for m in cat if not m["needs_key"]]
        tail = f" Models needing no key: {', '.join(keyless)}." if keyless else ""
        lines.append(
            "Nothing is sent anywhere until you press **Run experiment**." + tail
        )
    return "  \n".join(lines)


def provider_hint(env: str, session_keys: dict[str, str] | None = None) -> str:
    """One line about the provider currently selected in the key panel."""
    if not env:
        return ""
    info = provider_info(env)
    tier = "free tier" if info["free"] else "paid"
    with credentials.scope(session_keys):
        state = credentials.describe(env)
    if state["present"]:
        where = ("from this session" if state["source"] == "session"
                 else "from the environment")
        status = f"key set ({state['length']} chars, {where})"
    else:
        status = "no key set"
    names = [m["name"] for m in _models_for_env(env, session_keys)]
    return (f"**{info['provider']}** ({tier}) - {status}. Keys: {info['url']}  \n"
            f"Unlocks: {', '.join(names) if names else 'no models'}")


def _models_for_env(env: str, session_keys: dict[str, str] | None = None) -> list[dict]:
    from renderprobe.models.openai_compatible import PLUGINS

    return [{"name": a.name, "status": a.status}
            for a in PLUGINS if a.api_key_env == env]


def blocked_models(
    registry: Any, selected: list[str], session_keys: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """The selected models that cannot run because no key resolves for them.

    The Run tab calls this BEFORE the runner. A missing key used to surface as a table
    of empty `model_error` rows under a success message, which is the same outcome as a
    model that answered badly and is not distinguishable from one at a glance.
    """
    chosen = set(selected or [])
    return [m for m in model_catalog(registry, session_keys)
            if m["name"] in chosen and m["needs_key"] and not m["key_available"]]


def model_error_summary(results: list[Result]) -> list[str]:
    """Distinct `model_error` reasons from a run, one line per model and reason.

    Deduplicated because one missing key produces an identical failure on every scene
    and condition, and a wall of the same sentence hides how many DIFFERENT things went
    wrong.
    """
    seen: dict[tuple[str, str], int] = {}
    for r in results:
        if r.error_flag != "model_error":
            continue
        reason = str(r.meta.get("model_error") or "no reason recorded")
        key = (r.model, reason)
        seen[key] = seen.get(key, 0) + 1
    return [f"**{model}** ({count} call{'s' if count != 1 else ''}): {reason}"
            for (model, reason), count in seen.items()]


def key_envs(registry: Any) -> list[str]:
    """The credential variables the registered models need, in panel order."""
    from renderprobe.models.openai_compatible import PROVIDERS

    envs: dict[str, int] = {}
    for m in model_catalog(registry):
        if m["needs_key"]:
            envs[m["api_key_env"]] = envs.get(m["api_key_env"], 0) + (
                1 if m["status"] == "serves" else 0
            )
    return sorted(
        envs,
        key=lambda e: (0 if PROVIDERS.get(e, {}).get("free") else 1, -envs[e], e),
    )


def provider_info(env: str) -> dict[str, Any]:
    """Display details for one credential variable: provider name, signup URL, tier."""
    from renderprobe.models.openai_compatible import PROVIDERS

    info = PROVIDERS.get(env, {})
    return {
        "provider": info.get("provider", env),
        "url": info.get("url", ""),
        "free": bool(info.get("free", False)),
    }
