"""Experiment runner: produces Results under conditions, sandboxes all plugin calls."""
from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from renderprobe import renderers
from renderprobe.core import credentials
from renderprobe.core.registry import Registry
from renderprobe.core.schema import AnalysisReport, Result, Scene

# Confidence elicitation: appended to every prompt when the config asks for
# it, so a parsed confidence flows into Result.confidence for the calibration analyzer.
# No digits in this text on purpose: the count probe reads the first integer in the
# prompt, so a literal "0 to 100" here would poison it. Range is spelled in words.
_CONFIDENCE_INSTRUCTION = (
    " After your answer, on a new line, state your confidence as a whole percentage "
    "(from zero to one hundred) in exactly this form: 'Confidence: N'."
)
_CONFIDENCE_RE = re.compile(r"confidence\D{0,10}(\d{1,3})", re.IGNORECASE)


def _without_confidence(raw: str) -> str:
    """The reply with the elicited confidence removed, for the probe's own parser.

    The instruction asks for a trailing "Confidence: N", and N is a number. A probe whose
    parser takes the LAST number - which is the right rule for a model that shows its
    working and ends on the total - then reads the confidence as the answer. Measured
    live: gemma-3-27b replied "12\n\nConfidence: 100" to a question whose answer was 12
    and was scored wrong, which silently zeroed a scene's accuracy, tripped the
    oracle-suspect banner, and pushed calibration error to 1.0.

    Stripping it here rather than in each probe keeps every existing parser correct and
    means a plugin author cannot forget. `Result.raw` still holds the full reply, so what
    the model actually said is never lost.
    """
    return _CONFIDENCE_RE.sub("", raw or "").rstrip()


def _parse_confidence(raw: str) -> float | None:
    """Extract a 0-100 confidence from the model's text -> [0, 1]; None if absent.

    Never imputes: a model that ignores the instruction yields None, and that result
    is simply excluded from calibration (rather than assigned a fabricated value)."""
    m = _CONFIDENCE_RE.search(raw or "")
    if not m:
        return None
    return max(0.0, min(1.0, int(m.group(1)) / 100.0))


@dataclass
class ExperimentConfig:
    scene_generator: str
    scene_params: dict[str, Any]
    seeds: list[int]
    probe: str
    models: list[str]
    analyzers: list[str]
    # Which oracle verbalization variants to run. None = canonical only
    # (backward-compatible, 1× cost). A list selects a subset by name; a probe that
    # exposes no variants always runs its single canonical oracle_prompt.
    oracle_variants: list[str] | None = None
    # Ask the model to report a 0-100 confidence, parsed into Result.confidence for
    # the calibration analyzer. Off by default (unchanged prompts).
    elicit_confidence: bool = False
    # Run the text-only `blind` condition: the same question with the images withheld.
    # Off by default. It measures what a model scores from the prompt alone, which is
    # a prior, not a perceptual ability, and it is only worth its cost where the answer
    # set is small enough for that prior to contaminate the full condition. A scene
    # whose answer balance validate already certifies gains nothing from it, so opt in
    # per experiment rather than paying for it everywhere.
    blind: bool = False


def run_experiment(config: ExperimentConfig, registry: Registry) -> list[AnalysisReport]:
    """Main loop: generate scenes -> run probes under conditions -> analyze.

    Returns the analysis reports only (the CLI's view). Callers that also need
    the raw per-scene Results (e.g. the UI's inspect tab) should use
    :func:`run_experiment_collect`.
    """
    _scenes, _results, reports = run_experiment_collect(config, registry)
    return reports


def run_experiment_collect(
    config: ExperimentConfig, registry: Registry
) -> tuple[list[Scene], list[Result], list[AnalysisReport]]:
    """Same loop as :func:`run_experiment`, but also returns the generated
    scenes and the raw Results so a UI can render individual cases.

    Scenes are returned so the inspect tab can show the actual image alongside
    its Results without regenerating (generation is seeded and cheap, but the
    UI should display exactly what was probed)."""
    scene_gen = registry.get_scene(config.scene_generator)
    probe = registry.get_probe(config.probe)
    models = [registry.get_model(m) for m in config.models]
    analyzers = [registry.get_analyzer(a) for a in config.analyzers]

    # Resolve renderer names (incl. a scientist's --plugin-dir renderers) through the
    # registry while scenes render themselves; the frozen generate() signature can't take
    # the registry, so a graph scene reaches an external renderer via this scope.
    with renderers.using_registry(registry):
        scenes = _generate_scenes(scene_gen, config.scene_params, config.seeds)
    oracle_supported = _probe_supports_oracle(probe, scenes[0] if scenes else None)

    results: list[Result] = []
    for scene in scenes:
        for model in models:
            results.extend(
                _run_scene_model(
                    scene, probe, model, oracle_supported,
                    config.oracle_variants, config.elicit_confidence, config.blind,
                )
            )

    reports = _run_analysis(analyzers, results, oracle_supported)
    return scenes, results, reports


# Render budget (a cheap pre-run summary, so an expensive true-3D sweep is a
# deliberate choice rather than a surprise)

# True-3D renderers cost ~seconds per image (CPU path tracing / GPU) vs microseconds
# for the flat/depth-shaded PIL tiers.
HEAVY_RENDERERS = frozenset({"mitsuba_3d", "pyrender_3d"})


def render_plan(config: ExperimentConfig) -> dict[str, Any]:
    """Summarize the RENDER cost before a run. Each scene's image is rendered ONCE at
    generation and reused across every model and condition, so the render count is just
    the number of scenes - independent of how many models/conditions follow."""
    param_sets = _expand_param_sweep(config.scene_params)
    n_scenes = len(config.seeds) * len(param_sets)
    renderers_used = sorted({str(ps.get("renderer", "pil_2d")) for ps in param_sets})
    heavy = [r for r in renderers_used if r in HEAVY_RENDERERS]
    return {"n_scenes": n_scenes, "renderers": renderers_used, "heavy": heavy}


# Scene generation

def _generate_scenes(
    scene_gen: Any,
    params: dict[str, Any],
    seeds: list[int],
) -> list[Scene]:
    """Expand swept params × seeds into concrete Scene objects."""
    param_sets = _expand_param_sweep(params)
    scenes: list[Scene] = []
    for seed in seeds:
        for param_set in param_sets:
            scene = _safe_generate(scene_gen, param_set, seed)
            if scene is not None:
                scenes.append(scene)
    return scenes


def _safe_generate(scene_gen: Any, params: dict[str, Any], seed: int) -> Scene | None:
    try:
        return scene_gen.generate(params, seed)
    except Exception:
        traceback.print_exc()
        return None


def _expand_param_sweep(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued params into a cartesian product of param dicts.

    Each param whose value is a list is treated as a sweep axis; scalar params
    are held fixed. Multiple list params yield a full cartesian product so callers
    can sweep n_objects × hop_depth (or any N axes) without extra wiring.
    """
    from itertools import product

    sweep_keys = [k for k, v in params.items() if isinstance(v, list)]
    fixed = {k: v for k, v in params.items() if k not in sweep_keys}

    if not sweep_keys:
        return [dict(params)]

    sweep_lists = [params[k] for k in sweep_keys]
    result = []
    for combo in product(*sweep_lists):
        p = dict(fixed)
        for k, v in zip(sweep_keys, combo):
            p[k] = v
        result.append(p)
    return result


# Probe + model loop

def _probe_supports_oracle(probe: Any, sample_scene: Scene | None) -> bool:
    if sample_scene is None:
        return False
    try:
        return probe.oracle_prompt(sample_scene) is not None
    except Exception:
        return False


def _run_scene_model(
    scene: Scene,
    probe: Any,
    model: Any,
    oracle_supported: bool,
    requested_variants: list[str] | None = None,
    elicit_confidence: bool = False,
    blind: bool = False,
) -> list[Result]:
    results = []

    question = _safe_call(lambda: probe.question(scene), fallback="")
    gt = _safe_call(lambda: probe.ground_truth(scene), fallback=None)
    tier = scene.meta.get("tier", "standard")

    # full condition (image + prompt)
    results.append(
        _run_condition(scene, probe, model, "full", question, gt,
                       meta={"tier": tier}, elicit_confidence=elicit_confidence)
    )

    # blind condition (prompt only, no images) - opt-in, see ExperimentConfig.blind
    if blind:
        results.append(
            _run_condition(scene, probe, model, "blind", question, gt, blind=True,
                           meta={"tier": tier}, elicit_confidence=elicit_confidence)
        )

    # oracle condition - once per selected verbalization variant (canonical first).
    # TEXT-ONLY (blind=True): the oracle injects the ground-truth perceptual
    # primitives AS TEXT to isolate REASONING; passing the image too would leave
    # perception in play, conflate the decomposition, and defeat the ceiling check
    # (a real VLM would just read the answer off the image).
    if oracle_supported:
        variants = _oracle_variants(probe, scene, requested_variants)
        for vname, vprompt in variants.items():
            results.append(
                _run_condition(
                    scene, probe, model, "oracle", vprompt, gt, blind=True,
                    meta={"tier": tier, "oracle_variant": vname},
                    elicit_confidence=elicit_confidence,
                )
            )

    # report condition (OVER THE IMAGE): a direct perceptual sub-question whose GT is
    # a primitive, scored by the ReportSpec's own parse/score. It isolates PERCEPTION,
    # splitting encoding failures from grounding/arbitration failures. Optional - a
    # probe opts in via perception_report(scene); atomic probes return None (their
    # perception ~ the task, so a separate report is degenerate). See docs/methodology.md.
    report_fn = getattr(probe, "perception_report", None)
    if callable(report_fn):
        spec = _safe_call(lambda: report_fn(scene), fallback=None)
        # A probe may return one spec or several. Several is how a probe asks both a
        # robust primitive and a tight one; only the primary feeds acc_report.
        specs = [s for s in (spec if isinstance(spec, list) else [spec]) if s is not None]
        for s in specs:
            results.append(
                _run_condition(
                    scene, probe, model, "report", s.question, s.ground_truth,
                    blind=False,
                    meta={"tier": tier,
                          "report_name": getattr(s, "name", "report"),
                          "report_primary": bool(getattr(s, "primary", True))},
                    parse_fn=s.parse, score_fn=s.score,
                    images_override=s.images,
                    # Elicited like every other condition. It was left out while a
                    # trailing "Confidence: N" could be read as the answer by a parser
                    # taking the last number, which is exactly what a report question
                    # ("how many teeth?") uses; the runner now strips it before the
                    # probe sees the text, so the calibration tab no longer has one
                    # permanently blank row.
                    elicit_confidence=elicit_confidence,
                )
            )

    return [r for r in results if r is not None]


def _oracle_variants(
    probe: Any, scene: Scene, requested: list[str] | None
) -> dict[str, str]:
    """Ordered {variant_name: prompt}, canonical first.

    Uses the probe's optional ``oracle_prompt_variants`` when present; otherwise
    falls back to the single canonical ``oracle_prompt`` under the key "default".

    Selection by ``requested``:
      * ``None``        -> canonical variant only (backward-compatible, 1× cost);
      * ``["all"]``     -> every variant the probe offers;
      * ``[names...]``  -> that subset (a fully-mismatched request, e.g. a typo, is
                          ignored rather than silently dropping the oracle condition).
    """
    fn = getattr(probe, "oracle_prompt_variants", None)
    variants: dict[str, str] = {}
    if callable(fn):
        variants = _safe_call(lambda: fn(scene), fallback={}) or {}
    if not variants:
        single = _safe_call(lambda: probe.oracle_prompt(scene), fallback=None)
        variants = {"default": single} if single is not None else {}
    if not variants:
        return {}

    # Canonical-first ordering: the runner emits the canonical variant first so the
    # analyzer can identify it, and its numbers surface at the report's top level.
    canonical = getattr(probe, "canonical_oracle_variant", None)
    if canonical and canonical in variants:
        variants = {canonical: variants[canonical],
                    **{k: v for k, v in variants.items() if k != canonical}}

    if requested is None:                       # canonical only (first entry)
        first_key = next(iter(variants))
        return {first_key: variants[first_key]}
    if "all" in requested:                      # every variant
        return variants
    matched = {k: v for k, v in variants.items() if k in requested}
    return matched or variants                  # ignore fully-mismatched requests


def _run_condition(
    scene: Scene,
    probe: Any,
    model: Any,
    condition: str,
    question: str,
    gt: Any,
    blind: bool = False,
    meta: dict[str, Any] | None = None,
    elicit_confidence: bool = False,
    parse_fn: Any = None,
    score_fn: Any = None,
    images_override: list[Any] | None = None,
) -> Result | None:
    # parse_fn/score_fn override the probe's own parse/score - used by the "report"
    # condition, whose sub-question has its own ground truth and scoring (ReportSpec).
    # images_override lets the report show a modified image (e.g. target highlighted).
    parse_fn = parse_fn or probe.parse
    score_fn = score_fn or probe.score
    if blind:
        images = []
    elif images_override is not None:
        images = images_override
    else:
        images = scene.images
    meta = dict(meta) if meta else {}
    prompt = question + (_CONFIDENCE_INSTRUCTION if elicit_confidence else "")

    response, model_error = _safe_model_run(model, images, prompt)
    if response is None:
        # Why it failed rides on meta so the UI and the inspect view can say it out
        # loud. `Result.error` is a numeric distance, not a message, so it cannot.
        meta = {**meta, "model_error": model_error} if model_error else meta
        return Result(
            scene_id=scene.id,
            generator=scene.meta.get("generator", "unknown"),
            factors=scene.factors,
            probe=probe.name,
            model=model.name,
            condition=condition,
            raw="",
            pred=None,
            gt=gt,
            correct=False,
            score=0.0,
            error=None,
            confidence=None,
            error_flag="model_error",
            meta=meta,
        )

    # The confidence is stripped before the probe sees the text, never after: a parser
    # that takes the last number would otherwise return the confidence as the answer.
    answer_text = _without_confidence(response.text) if elicit_confidence else response.text
    pred, error_flag = _safe_parse(parse_fn, answer_text)
    # A reply cut off at the token limit is not an answer. Flagging it drops the row
    # from every accuracy rather than recording a failure the model may not have made.
    if getattr(response, "truncated", False):
        error_flag = "truncated"
    if error_flag is None:
        correct, score, err = _safe_score(score_fn, pred, gt)
    else:
        correct, score, err = False, 0.0, None

    # Prefer a confidence parsed from the elicited answer; else whatever the adapter
    # exposed (real closed adapters expose none). Never imputed.
    confidence = (
        _parse_confidence(response.text) if elicit_confidence else response.confidence
    )

    return Result(
        scene_id=scene.id,
        generator=scene.meta.get("generator", "unknown"),
        factors=scene.factors,
        probe=probe.name,
        model=model.name,
        condition=condition,
        raw=response.text,
        pred=pred,
        gt=gt,
        correct=correct,
        score=score,
        error=err,
        confidence=confidence,
        error_flag=error_flag,
        meta=meta,
    )


def _safe_model_run(model: Any, images: list, prompt: str) -> tuple[Any, str | None]:
    """Call the model, returning ``(response, None)`` or ``(None, reason)``.

    The reason travels back with the row. It used to be printed to the server's stderr
    and dropped, which meant a UI showed an empty results table and never said the word
    "key" - the single most common way a run fails, rendered invisible.

    Everything here passes through ``credentials.scrub`` first: a provider rejecting a
    request is free to quote that request back, and the request carried the key.
    """
    try:
        return model.run(images, prompt), None
    except Exception as exc:
        print(credentials.scrub(traceback.format_exc()), file=sys.stderr)
        # A missing key is guidance, not a fault: show its message alone. Everything
        # else keeps its exception type, which is what makes an unexpected failure
        # diagnosable from the row.
        if isinstance(exc, credentials.MissingCredential):
            reason = credentials.scrub(str(exc))
        else:
            reason = credentials.scrub(f"{type(exc).__name__}: {exc}".strip())
        return None, reason or type(exc).__name__


def _safe_parse(parse_fn: Any, raw: str) -> tuple[Any, str | None]:
    try:
        return parse_fn(raw), None
    except Exception:
        return None, "parse_fail"


def _safe_score(score_fn: Any, pred: Any, gt: Any) -> tuple[bool, float, float | None]:
    try:
        return score_fn(pred, gt)
    except Exception:
        return False, 0.0, None


def _safe_call(fn, fallback: Any) -> Any:
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        return fallback


# Analysis

def _run_analysis(
    analyzers: list[Any],
    results: list[Result],
    oracle_supported: bool,
) -> list[AnalysisReport]:
    available_conditions = {"full"}
    # "blind" is opt-in, so it is available only when the run actually produced it.
    if any(r.condition == "blind" for r in results):
        available_conditions.add("blind")
    if oracle_supported:
        available_conditions.add("oracle")
    # "report" is opportunistic: present only when a probe produced report results.
    if any(r.condition == "report" for r in results):
        available_conditions.add("report")

    factor_values: dict[str, set] = {}
    for r in results:
        for k, v in r.factors.items():
            factor_values.setdefault(k, set()).add(v)
    has_varying_factor = any(len(v) >= 2 for v in factor_values.values())

    reports: list[AnalysisReport] = []
    for analyzer in analyzers:
        # Gate 1: conditions. Fail LOUDLY - emit a structured skip report (not just a
        # print) so a caller inspecting the reports sees that, e.g., the oracle
        # diagnostic was requested but the probe offers no oracle support.
        missing_conds = sorted(set(analyzer.requires_conditions) - available_conditions)
        if missing_conds:
            reason = (
                f"required conditions {missing_conds} not available"
                + (" (probe has no oracle_prompt support)"
                   if "oracle" in missing_conds else "")
            )
            print(f"[skip] Analyzer '{analyzer.name}' skipped: {reason}.")
            reports.append(AnalysisReport(
                name=analyzer.name,
                payload={
                    "status": "skipped",
                    "reason": reason,
                    "missing_conditions": missing_conds,
                    "oracle_supported": oracle_supported,
                },
            ))
            continue

        # Gate 2: varying factor
        if analyzer.requires_varying_factor and not has_varying_factor:
            print(
                f"[skip] Analyzer '{analyzer.name}' skipped: "
                f"scene set has no varying factor (need >=2 distinct values of some factor)."
            )
            continue

        try:
            report = analyzer.analyze(results)
            reports.append(report)
        except Exception:
            traceback.print_exc()
            print(f"[error] Analyzer '{analyzer.name}' raised an exception - skipped.")

    return reports
