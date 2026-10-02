"""Plugin validation - conformance PLUS the honesty checks that keep the oracle
diagnostic trustworthy.

Registration (registry.py) already proves a plugin satisfies its Protocol. That is
necessary but NOT sufficient: the whole method rests on two things a Protocol cannot
express -

  1. TRUSTABLE ground truth: a perfect perceiver could always answer, and the same
     seed always yields the same scene (reproducibility).
  2. A FAITHFUL, non-degenerate task: the answer varies across seeds, so a blind
     model cannot win by guessing a constant; and an oracle exists if the author
     wants the perception/reasoning split.

``renderprobe validate <path>`` runs these before a scientist spends real API calls,
so a broken scoring function or a scene whose answer is always "red" is caught here
instead of silently corrupting a "perception vs reasoning" verdict.

This module is pure/offline: it never calls a model. Model plugins are checked for
conformance only (a live call needs a key and burns quota).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from renderprobe import renderers
from renderprobe.core.registry import RegistrationError, Registry
from renderprobe.core.schema import (
    ANSWER_TYPES,
    AnalysisReport,
    GraphObject,
    ReportSpec,
    Result,
    SceneGraph,
)

# Question words that POINT to a marked target; if the report uses them but ships no
# highlighted image, the model is forced to localize by coordinate (a known confound).
# Note: bare "circle" is excluded - it is usually a shape *answer*, not a pointer;
# "circled" (the past participle) is the pointer.
_HANDLE_WORDS = ("ring", "circled", "highlight", "marked", "outlined", "boxed", "arrow", "halo")

# One answer dominating more than this fraction of seeds means a blind model can
# score at least that high by always guessing it - the diagnostic's baseline is
# then inflated and recovery/oracle numbers get hard to trust.
_BALANCE_WARN = 0.60
_DEFAULT_SEEDS = 12

# Occlusion-safety certification (graph scenes rendered through a renderer's optional
# render_ids). Occlusion is geometric, so a few seeds sample the density regime.
#   _OCC_MIN_PX     an object with this few visible px is treated as HIDDEN (or, if it has
#                   this few even when rendered ALONE, off-canvas/degenerate). This
#                   full-occlusion FAIL is scale-robust - a fully hidden object has 0 px
#                   regardless of how the renderer scales sizes.
#   _OCC_WARN_FRAC  visible/footprint below this => heavily occluded (WARN). Only sound
#                   when the renderer's per-object projection is CONTEXT-INDEPENDENT (the
#                   footprint measured alone matches the footprint in a full scene). We
#                   self-check that per renderer and skip this band when it doesn't hold
#                   (e.g. pil_25d scales an object by the whole scene's depth range), so a
#                   legitimately depth-shrunk object is never mistaken for an occluded one.
_OCC_SEEDS = 4
_OCC_MIN_PX = 4
_OCC_WARN_FRAC = 0.50

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str          # PASS | WARN | FAIL
    detail: str


@dataclass
class PluginValidation:
    family: str
    name: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == WARN for c in self.checks)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))


# Entry point

def validate_path(
    path: str | Path, seeds: int = _DEFAULT_SEEDS, probe_name: str | None = None,
    renderer_name: str | None = None,
) -> list[PluginValidation]:
    """Load out-of-tree plugin(s) from ``path`` and validate each. The registry is
    seeded with the built-ins first, so a new scene can be paired with a built-in
    probe (and vice versa) for the round-trip checks.

    ``renderer_name`` selects the renderer a graph-based scene is occlusion-certified
    under (default: the scene's own renderer, else pil_2d). Use it to check a scene is
    still count-trustable before committing to a heavier true-3D tier."""
    reg = Registry()
    reg.autodiscover()

    try:
        registered = reg.load_external(path)
    except RegistrationError as exc:
        pv = PluginValidation(family="?", name=str(Path(path).name))
        pv.add("load", FAIL, str(exc))
        return [pv]

    reports: list[PluginValidation] = []
    # Let a scene under test render itself through a --plugin-dir renderer it names
    # (resolved via the registry), just as a real run does.
    with renderers.using_registry(reg):
        for entry in registered:
            family, _, name = entry.partition(":")
            pv = PluginValidation(family=family, name=name)
            try:
                if family == "scene":
                    _validate_scene(reg, name, pv, seeds, probe_name, renderer_name)
                elif family == "probe":
                    _validate_probe(reg, name, pv, seeds)
                elif family == "model":
                    _validate_model(reg, name, pv)
                elif family == "analyzer":
                    _validate_analyzer(reg, name, pv)
                elif family == "renderer":
                    _validate_renderer(reg, name, pv)
            except Exception as exc:  # a plugin bug shouldn't crash the validator
                pv.add("crash", FAIL, f"unexpected error during validation: {exc!r}")
            reports.append(pv)
    return reports


# Scene

def _uses_stochastic_renderer(reg: Registry, gen: Any, params: dict[str, Any]) -> bool:
    """Does this scene draw itself with a renderer that declares stochastic output?"""
    name = params.get("renderer") or getattr(gen, "default_renderer", None)
    if name is None:
        return False
    try:
        return bool(getattr(reg.get_renderer(str(name)), "stochastic", False))
    except KeyError:
        return False


def _graphs_differ(gen: Any, params: dict[str, Any]) -> bool:
    """Is the scene SPECIFICATION unstable across two generations of the same seed?

    For a stochastic renderer this is the guarantee that still has to hold: the
    objects, their positions, colors and sizes must be identical, even when the pixels
    drawn from them carry different noise.
    """
    def spec(scene):
        g = getattr(scene, "graph", None)
        if g is None:
            return None
        return [(o.id, o.shape, o.color, o.position, o.size) for o in g.objects]
    for s in range(3):
        try:
            if spec(gen.generate(params, s)) != spec(gen.generate(params, s)):
                return True
        except Exception:
            return True
    return False


def _default_params(scene_gen: Any) -> dict[str, Any]:
    schema = getattr(scene_gen, "params_schema", {}) or {}
    params: dict[str, Any] = {}
    for name, spec in schema.items():
        if isinstance(spec, dict):
            params[name] = spec.get("default", spec.get("min", 0))
    return params


def _find_probe_for_scene(reg: Registry, gt_keys: set[str], prefer: str) -> Any | None:
    names = reg.list_probes()
    ordered = ([prefer] if prefer in names else []) + [n for n in names if n != prefer]
    for n in ordered:
        p = reg.get_probe(n)
        if set(getattr(p, "requires_gt_fields", [])) <= gt_keys:
            return p
    return None


def _validate_scene(
    reg: Registry, name: str, pv: PluginValidation, seeds: int,
    probe_name: str | None, renderer_name: str | None = None,
) -> None:
    gen = reg.get_scene(name)
    pv.add("protocol", PASS, "registered as a SceneGenerator")

    # params_schema well-formed
    schema = getattr(gen, "params_schema", {})
    if not isinstance(schema, dict):
        pv.add("params_schema", FAIL, "params_schema must be a dict")
        return
    bad = [k for k, v in schema.items() if not isinstance(v, dict)]
    if bad:
        pv.add("params_schema", WARN,
               f"entries not dicts (no range/default): {bad} - the UI can't build dials")
    else:
        pv.add("params_schema", PASS, f"{len(schema)} parameter(s): {list(schema)}")

    params = _default_params(gen)

    # Determinism: same seed -> identical GT and identical pixels. Check several seeds,
    # since a global-RNG bug (the common mistake) only sometimes collides on one seed.
    try:
        a = gen.generate(params, 0)
    except Exception as exc:
        pv.add("generate", FAIL, f"generate() raised: {exc!r}")
        return
    gt_mismatch = pixel_mismatch = False
    for s in (0, 1, 2):
        try:
            x, y = gen.generate(params, s), gen.generate(params, s)
        except Exception as exc:
            pv.add("generate", FAIL, f"generate(seed={s}) raised: {exc!r}")
            return
        if x.ground_truth != y.ground_truth:
            gt_mismatch = True
            break
        if [i.tobytes() for i in x.images] != [i.tobytes() for i in y.images]:
            pixel_mismatch = True
    if gt_mismatch:
        pv.add("determinism", FAIL,
               "same seed produced different ground_truth - not reproducible "
               "(seed every RNG from the `seed` argument; don't use the global random)")
    elif pixel_mismatch and _uses_stochastic_renderer(reg, gen, params):
        # A Monte Carlo renderer's pixels carry noise that differs between runs. The
        # scene specification is what must be reproducible, and it is checked above and
        # below; the noise is variance, not bias, and within a run every model sees the
        # same image because each scene is rendered once and reused.
        if _graphs_differ(gen, params):
            pv.add("determinism", FAIL,
                   "same seed produced a different scene graph - not reproducible "
                   "(seed every RNG from the `seed` argument)")
        else:
            pv.add("determinism", PASS,
                   "same seed -> identical ground truth and scene graph; pixels differ "
                   "only by Monte Carlo noise from a stochastic renderer (each scene is "
                   "rendered once per run and reused, so a paired comparison is unaffected)")
    elif pixel_mismatch:
        pv.add("determinism", FAIL,
               "same seed produced different pixels - the model would see different "
               "images on a re-run (seed every RNG from `seed`)")
    else:
        pv.add("determinism", PASS, "same seed -> identical scene across seeds 0-2")

    # Occlusion safety: for a graph-based scene, certify every object stays visible (not
    # hidden) under the chosen renderer, so a count/enumeration answer is recoverable.
    # Runs before probe-pairing because trustable pixels are a property of scene+renderer,
    # not of the task asked about them.
    _certify_scene_visibility(pv, reg, gen, params, renderer_name, seeds)

    gt_keys = set(a.ground_truth)
    probe = (reg.get_probe(probe_name) if probe_name else
             _find_probe_for_scene(reg, gt_keys, prefer=name))
    if probe is None:
        pv.add("probe-pairing", WARN,
               f"no probe's requires_gt_fields fit ground_truth keys {sorted(gt_keys)} "
               f"- pass --probe or ship a matching probe; skipping answer checks")
        return
    pv.add("probe-pairing", PASS, f"paired with probe '{probe.name}'")

    # GT round-trip: a perfect answer must score correct
    try:
        ans = probe.ground_truth(a)
        ok, score, _ = probe.score(ans, ans)
    except Exception as exc:
        pv.add("gt-roundtrip", FAIL, f"probe.ground_truth/score raised: {exc!r}")
        return
    if not ok:
        pv.add("gt-roundtrip", FAIL,
               f"scoring the ground-truth answer against itself is NOT correct "
               f"(score={score}) - a perfect perceiver would be marked wrong")
    else:
        pv.add("gt-roundtrip", PASS, "the ground-truth answer scores correct")

    # Does the paired probe's perception report work on THIS scene? (three-way readiness)
    _validate_report(pv, probe, a)
    # Is the paired probe's oracle information-complete on THIS scene family?
    _validate_oracle_completeness(pv, probe, gen, params, seeds)

    # Answer balance across seeds (blind-baseline sanity)
    answers = []
    for s in range(seeds):
        try:
            sc = gen.generate(params, s)
            answers.append(probe.ground_truth(sc))
        except Exception:
            continue
    if answers:
        _check_balance(pv, gen, probe, params, answers,
                       getattr(probe, "answer_type", "categorical"))

    # Oracle presence (needed for the perception/reasoning split)
    op = None
    try:
        op = probe.oracle_prompt(a)
    except Exception as exc:
        pv.add("oracle", WARN, f"oracle_prompt raised: {exc!r}")
    if op is None:
        pv.add("oracle", WARN,
               "no oracle for this scene/probe - accuracy works, but the "
               "perception-vs-reasoning decomposition needs an oracle")
    elif not str(op).strip():
        pv.add("oracle", FAIL, "oracle_prompt returned an empty string")
    else:
        pv.add("oracle", PASS, "oracle prompt available")


def _hash(a: Any) -> Any:
    return a if isinstance(a, (str, int, float, bool)) else str(a)


def _param_controls_answer(gen: Any, probe: Any, params: dict[str, Any], base: Any) -> str | None:
    """If the answer is constant across seeds, is it at least controlled by a PARAMETER
    (as with a count fixed by n_dots)? Returns the first param that moves the answer, or
    None if nothing does."""
    schema = getattr(gen, "params_schema", {}) or {}
    for pname, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        for v in (spec.get("min"), spec.get("max")):
            if v is None or v == params.get(pname):
                continue
            try:
                sc = gen.generate({**params, pname: v}, 0)
                if _hash(probe.ground_truth(sc)) != _hash(base):
                    return pname
            except Exception:
                continue
    return None


def _validate_report(pv: PluginValidation, probe: Any, scene: Any) -> None:
    """Check a probe's OPTIONAL perception_report - the sub-task that splits a
    visual-pathway failure into encoding vs grounding. This is what a scientist needs
    to verify before trusting the three-way decomposition on their own scene."""
    report_fn = getattr(probe, "perception_report", None)
    if not callable(report_fn):
        pv.add("perception-report", WARN,
               "no perception_report - the run still works, but recovery can't be split "
               "into encoding vs grounding (two-way only). Add one for the 3-way decomposition.")
        return
    try:
        spec = report_fn(scene)
    except Exception as exc:
        pv.add("perception-report", FAIL, f"perception_report raised: {exc!r}")
        return
    if spec is None or spec == []:
        pv.add("perception-report", WARN,
               "perception_report returned nothing for the sample scene - no report produced "
               "(check it handles this scene's ground_truth fields)")
        return
    # One spec or several. Several is how a probe asks both a robust primitive and a
    # tight one; every spec is checked, and exactly one of them drives acc_report.
    specs = spec if isinstance(spec, list) else [spec]
    bad = [s for s in specs if not isinstance(s, ReportSpec)]
    if bad:
        pv.add("perception-report", FAIL,
               "perception_report must return a ReportSpec, a list of them, or None; "
               f"got {type(bad[0]).__name__} in the result")
        return

    primaries = [s for s in specs if getattr(s, "primary", True)]
    if len(primaries) != 1:
        pv.add("perception-report", FAIL,
               f"{len(specs)} report(s) with {len(primaries)} marked primary - exactly one "
               "must be, since only the primary feeds acc_report and the encoding/grounding "
               "split. Pooling reports of different strictness measures neither.")
        return
    names = [getattr(s, "name", "report") for s in specs]
    if len(set(names)) != len(names):
        pv.add("perception-report", FAIL,
               f"report names must be distinct so results stay separable, got {names}")
        return

    for s in specs:
        label = getattr(s, "name", "report")
        tag = "perception-report" if len(specs) == 1 else f"perception-report[{label}]"

        # The report's answer must score correct against itself under the report's OWN
        # parse/score (the ReportSpec carries its own, distinct from the task's).
        try:
            ok, score, _ = s.score(s.ground_truth, s.ground_truth)
        except Exception as exc:
            pv.add(tag, FAIL, f"ReportSpec.score raised: {exc!r}")
            return
        if not ok:
            pv.add(tag, FAIL,
                   f"the report's ground-truth answer does not score correct against itself "
                   f"(score={score}) - the report scorer is broken")
            return

        role = "primary -> feeds acc_report" if getattr(s, "primary", True) else "diagnostic"
        # Handle check: pointer language ("the circled object") with no highlighted image
        # forces coordinate localization - the confound the ring pattern exists to avoid.
        references_handle = any(w in s.question.lower() for w in _HANDLE_WORDS)
        if s.images is not None:
            pv.add(tag, PASS,
                   f"VALID ({role}); uses a perceivable handle "
                   "(highlighted image, no coordinate localization)")
        elif references_handle:
            pv.add(tag, WARN,
                   "the report points to a marked target but spec.images is None - the model "
                   "must localize by coordinate (a known confound). Ship a highlighted copy in "
                   "spec.images (see the relational probe's ring, or the probe template).")
        else:
            pv.add(tag, PASS, f"VALID ({role}); asked over the unmodified image")

        # Parse round-trip on the report's own parser (best-effort).
        try:
            parsed = s.parse(str(s.ground_truth))
            p_ok, _, _ = s.score(parsed, s.ground_truth)
        except Exception:
            p_ok = False
        if not p_ok:
            pv.add(f"{tag}-parse", WARN,
                   "ReportSpec.parse did not recover the report answer from its string form")


def _validate_oracle_completeness(
    pv: PluginValidation, probe: Any, gen: Any, params: dict[str, Any], seeds: int
) -> None:
    """Certify the oracle is INFORMATION-COMPLETE. A reference reasoner (the probe's
    optional ``oracle_solver``) re-derives the answer from ONLY the facts the oracle
    states - never the stored answer. If it matches GT on every seed, an oracle failure
    by a model is reasoning, not a lossy/under-specified oracle (the exact
    critique this defends against). Perception-bound probes declare ``oracle_states_answer``
    instead - their oracle legitimately hands over the answer."""
    try:
        sample = gen.generate(params, 0)
        if probe.oracle_prompt(sample) is None:
            return   # no oracle - absence is reported by the separate 'oracle' check
    except Exception:
        return
    if getattr(probe, "oracle_states_answer", False):
        pv.add("oracle-completeness", PASS,
               "perception-bound: the oracle states the answer primitive directly, so "
               "completeness is definitional (recovery = the full perception gap)")
        return
    solver = getattr(probe, "oracle_solver", None)
    if not callable(solver):
        pv.add("oracle-completeness", WARN,
               "oracle present but UNPROVEN - add an oracle_solver (re-derive the answer "
               "from the facts your oracle states, never the stored answer) so a "
               "'reasoning-limited' verdict can't be a lossy-oracle artifact")
        return
    n = mism = 0
    example: tuple[Any, Any] | None = None
    for s in range(seeds):
        try:
            sc = gen.generate(params, s)
            solved, gt = solver(sc), probe.ground_truth(sc)
        except Exception as exc:
            pv.add("oracle-completeness", FAIL, f"oracle_solver raised: {exc!r}")
            return
        if solved is None:
            continue
        n += 1
        if solved != gt:
            mism += 1
            example = example or (solved, gt)
    if n == 0:
        pv.add("oracle-completeness", WARN,
               "oracle_solver returned None for every sampled scene - cannot certify")
    elif mism == 0:
        pv.add("oracle-completeness", PASS,
               f"oracle information-complete: a reference reasoner re-derives GT from the "
               f"stated facts on {n}/{n} scenes (an oracle failure is therefore reasoning, "
               f"not missing information)")
    else:
        pv.add("oracle-completeness", FAIL,
               f"oracle_solver disagreed with GT on {mism}/{n} scenes (e.g. solved "
               f"{example[0]!r} but GT is {example[1]!r}) - the oracle facts are "
               f"insufficient OR the ground truth is wrong")


def _check_balance(
    pv: PluginValidation, gen: Any, probe: Any, params: dict[str, Any],
    answers: list[Any], answer_type: str,
) -> None:
    hashable = [_hash(a) for a in answers]
    counts = Counter(hashable)
    top, top_n = counts.most_common(1)[0]
    frac = top_n / len(hashable)
    distinct = len(counts)
    if distinct == 1:
        # Constant across seeds - fine ONLY if a parameter varies it (sweep-controlled,
        # like counting), otherwise the task can never discriminate.
        driver = _param_controls_answer(gen, probe, params, answers[0])
        if driver:
            pv.add("answer-balance", PASS,
                   f"answer is fixed across seeds but controlled by param '{driver}' "
                   f"(sweep it across scenes so the answer moves with it)")
        else:
            pv.add("answer-balance", WARN,
                   f"the answer is ALWAYS {top!r} regardless of seed or parameters - a "
                   f"a model scores 100% without looking; the task cannot discriminate")
    elif frac > _BALANCE_WARN and answer_type in ("categorical", "ordering"):
        pv.add("answer-balance", WARN,
               f"answer {top!r} covers {frac:.0%} of {len(hashable)} seeds - a model "
               f"can score above chance without looking; balance the answers, or run "
               f"the opt-in `blind: true` baseline to measure the prior you are leaving in")
    else:
        pv.add("answer-balance", PASS,
               f"{distinct} distinct answers over {len(hashable)} seeds "
               f"(most common {frac:.0%})")


# Occlusion safety - the honesty guarantee for true-3D rendering
#
# A renderer MAY expose an optional ``render_ids(graph) -> (H, W) int array`` mapping each
# pixel to the index of the FRONT-MOST object (-1 = background), produced with the SAME
# projection/occlusion as ``render``. With it we can CERTIFY, per generated scene, that
# every ground-truth object stays visible under the chosen renderer - so a count /
# enumeration answer is still recoverable from the pixels. Without it a true-3D renderer
# could silently hide objects and corrupt the perception-vs-reasoning verdict.

def _worse(a: str, b: str) -> bool:
    order = {PASS: 0, WARN: 1, FAIL: 2}
    return order[a] > order[b]


def _object_visibility(renderer: Any, graph: SceneGraph) -> list[tuple[int, int]] | None:
    """Per object: (visible px in the full render, footprint px when rendered ALONE).
    None if the renderer exposes no ``render_ids`` (occlusion cannot be measured). The
    solo render gives each object's footprint; ``vis == 0`` with ``foot > 0`` is a fully
    OCCLUDED object (scale-robust), while the visible/footprint RATIO is only meaningful
    when the renderer projects each object context-independently (see ``_context_stable``)."""
    fn = getattr(renderer, "render_ids", None)
    if not callable(fn):
        return None
    full = np.asarray(fn(graph))
    out: list[tuple[int, int]] = []
    for i, obj in enumerate(graph.objects):
        vis = int(np.count_nonzero(full == i))
        # meta must carry over: it holds the camera and floor, so dropping it would
        # measure the footprint under a different projection than the full render and
        # make the two counts incomparable - an object in frame for one and outside it
        # for the other then reads as "renders to ~0 px even unoccluded".
        solo = np.asarray(fn(SceneGraph(objects=[obj], canvas=graph.canvas,
                                        background=graph.background,
                                        meta=dict(graph.meta or {}))))
        foot = int(np.count_nonzero(solo == 0))   # the lone object is index 0 in its solo graph
        out.append((vis, foot))
    return out


def _context_stable(renderer: Any, canvas: tuple[int, int]) -> bool:
    """Does the renderer project a given object to the SAME footprint whether it is alone
    or in a scene with other objects? True for absolute projections (pil_2d, a true-3D
    renderer); False for renderers that scale an object by the whole scene's depth range
    (pil_25d). Only when True is the partial-occlusion (visible/footprint) band sound -
    otherwise a legitimately depth-shrunk object would look 'occluded'. We probe with a
    NON-nearest object (z>0) because scene-relative scaling only bites objects that aren't
    the frontmost, and place the decoy far away so it never actually occludes the probe."""
    fn = getattr(renderer, "render_ids", None)
    if not callable(fn):
        return False
    w, h = canvas
    probe = GraphObject("probe", "sphere", "red", (w * 0.28, h * 0.35, 5.0), 18.0)
    decoy = GraphObject("decoy", "cube", "blue", (w * 0.78, h * 0.72, 0.0), 18.0)
    try:
        alone = int(np.count_nonzero(np.asarray(fn(SceneGraph([probe], canvas))) == 0))
        withd = int(np.count_nonzero(np.asarray(fn(SceneGraph([probe, decoy], canvas))) == 0))
    except Exception:
        return False
    if alone <= _OCC_MIN_PX:
        return False
    return abs(alone - withd) <= max(2, int(0.03 * alone))


def _occlusion_policy(graph: SceneGraph) -> tuple[list[int] | None, str]:
    """Read a graph's optional occlusion policy from ``graph.meta["occlusion"]``.

    Returns ``(certify_indices, policy)``. Defaults are the strict ones: certify
    every object, policy "visible". A scene opts out only by saying so explicitly.

        meta["occlusion"] = {
            "certify": [0, 3, 7],     # objects the ANSWER depends on; default: all
            "policy": "visible",      # or "presence", or "derivable"
        }

    Policies:
      "visible"    (default) every certified object visible; heavy PARTIAL occlusion
                   warns, because a half-hidden object's color or shape may be misread.
      "presence"   full occlusion still FAILS, but partial occlusion is expected and
                   not warned. For a scene built from solid multi-part objects - a
                   polycube piece, say - the parts inevitably overlap each other, and
                   what the answer needs is which parts are present, not a percept of
                   each one. Warning there would be noise about the shape being 3D.
      "derivable"  objects are hidden BY DESIGN; see _certify_derivable.

    `certify` exists because a dense scene may legitimately hide background clutter
    while every object the question turns on stays in view. Certifying the whole
    graph would reject such a scene even though its ground truth is sound.
    """
    meta = getattr(graph, "meta", None) or {}
    spec = meta.get("occlusion") or {}
    idx = spec.get("certify")
    policy = spec.get("policy", "visible")
    if idx is not None:
        idx = [int(i) for i in idx]
    return idx, policy


def _occlusion_verdict(
    vis_foot: list[tuple[int, int]], partial_reliable: bool = False,
    certify: list[int] | None = None, presence_only: bool = False,
) -> tuple[str, str]:
    """Turn per-object (visible, footprint) into a PASS/WARN/FAIL verdict + message.

    Full occlusion / off-canvas is always a FAIL (scale-robust). Heavy PARTIAL occlusion
    is a WARN only when ``partial_reliable`` (the renderer projects context-independently),
    so a legitimately depth-shrunk object is never mistaken for an occluded one."""
    checked = range(len(vis_foot)) if certify is None else certify
    n = len(list(checked))
    offframe, hidden, heavy = [], [], []
    for i in (range(len(vis_foot)) if certify is None else certify):
        if i < 0 or i >= len(vis_foot):
            return FAIL, (f"occlusion policy names object index {i}, but the graph has "
                          f"{len(vis_foot)} object(s)")
        vis, foot = vis_foot[i]
        if foot <= _OCC_MIN_PX:
            offframe.append(i)
        elif vis <= _OCC_MIN_PX:
            hidden.append(i)
        elif (partial_reliable and not presence_only
              and vis / foot < _OCC_WARN_FRAC):
            heavy.append(i)
    if offframe:
        return FAIL, (f"{len(offframe)}/{n} object(s) render to ~0 px even unoccluded "
                      f"(off-canvas or sub-pixel): indices {offframe} - these can never be "
                      f"seen, so the count is wrong regardless of the model")
    if hidden:
        return FAIL, (f"{len(hidden)}/{n} object(s) are (near-)fully OCCLUDED (indices "
                      f"{hidden}) - count/enumeration ground truth is NOT recoverable from "
                      f"the image under this renderer")
    if heavy:
        return WARN, (f"{len(heavy)}/{n} object(s) are heavily occluded (<50% visible, "
                      f"indices {heavy}) - still countable, but a per-object percept "
                      f"(that object's color/shape) may be unreliable")
    return PASS, f"all {n} object(s) visible"


def _certify_derivable(
    gen: Any, scene: Any, graph: SceneGraph, vis_foot: list[tuple[int, int]],
) -> tuple[str, str]:
    """Certify a scene whose task DEPENDS on objects being hidden.

    For an inference-from-occlusion task (how many cubes are in this stack, counting
    the ones you cannot see) the usual guarantee is the wrong one: demanding every
    object be visible would reject exactly the scenes the task is made of. The honest
    replacement is not to trust the scene's claim but to check it: the answer must be
    DERIVABLE from what is visible plus the task's own constraint.

    The scene proves that by exposing

        derive_from_visible(graph, visible_indices) -> dict

    which re-derives whatever ground-truth fields it can from only the visible
    objects. Every field it returns must match the scene's real ground truth. A scene
    that declares the policy without the hook is FAILed rather than taken at its word,
    which is what keeps this an extension of the guarantee and not an escape from it.
    """
    fn = getattr(gen, "derive_from_visible", None)
    if not callable(fn):
        return FAIL, ('occlusion policy "derivable" declared, but the scene exposes no '
                      "derive_from_visible(graph, visible_indices) -> dict, so the claim "
                      "that the answer survives occlusion cannot be checked")
    visible = [i for i, (vis, _f) in enumerate(vis_foot) if vis > _OCC_MIN_PX]
    try:
        derived = fn(graph, visible)
    except Exception as exc:
        return FAIL, f"derive_from_visible raised: {exc!r}"
    if not isinstance(derived, dict) or not derived:
        return FAIL, ("derive_from_visible must return a non-empty dict of ground-truth "
                      f"fields it can re-derive; got {type(derived).__name__}")
    gt = getattr(scene, "ground_truth", {}) or {}
    bad = {k: (v, gt.get(k)) for k, v in derived.items() if k not in gt or gt[k] != v}
    if bad:
        return FAIL, (f"the answer is NOT derivable from the {len(visible)} visible "
                      f"object(s): derive_from_visible disagrees with ground truth on "
                      f"{sorted(bad)} (derived vs actual: {bad})")
    return PASS, (f"{len(vis_foot) - len(visible)} object(s) hidden by design; "
                  f"{sorted(derived)} re-derived correctly from the {len(visible)} "
                  f"visible one(s)")


def _certify_scene_visibility(
    pv: PluginValidation, reg: Registry, gen: Any, params: dict[str, Any],
    renderer_name: str | None, seeds: int,
) -> None:
    """For a graph-based scene, certify that every ground-truth object stays visible under
    the chosen renderer across a sample of seeds. Scenes that draw pixels directly (no
    ``.graph``) don't need this and are noted as n/a. The graph is renderer-independent, so
    we certify it directly under any REGISTERED renderer (built-in or a --plugin-dir one),
    regardless of what the scene used to draw its own images."""
    try:
        sample = gen.generate(params, 0)
    except Exception:
        return  # generate() failures are already reported by the determinism check
    graph = getattr(sample, "graph", None)
    if graph is None:
        pv.add("occlusion-safety", PASS,
               "n/a - scene draws pixels directly (no scene graph); occlusion certification "
               "applies only to graph-based scenes")
        return

    # The scene's own renderer, unless one is named: certifying a 3D scene under the flat
    # default reports objects occluded that the scene never draws that way. Same
    # precedence as _uses_stochastic_renderer above.
    rname = (renderer_name or params.get("renderer")
             or getattr(gen, "default_renderer", None) or "pil_2d")
    try:
        renderer = reg.get_renderer(rname)
    except KeyError:
        pv.add("occlusion-safety", FAIL,
               f"renderer '{rname}' not found; registered: {reg.list_renderers()}")
        return

    # Pair check: does the scene actually render under this renderer (the images a model
    # would see)? An optional renderer whose deps are absent WARNs rather than FAILs.
    try:
        imgs = renderer.render(graph)
        rendered_ok = bool(imgs) and all(isinstance(i, Image.Image) for i in imgs)
    except Exception as exc:
        pv.add("scene-renders", WARN,
               f"scene does not render under '{rname}' - optional deps missing? ({exc})")
        return
    if not rendered_ok:
        pv.add("scene-renders", FAIL,
               f"'{rname}'.render did not return PIL image(s) for this scene")
        return
    pv.add("scene-renders", PASS, f"scene renders under '{rname}' ({imgs[0].size})")

    if not callable(getattr(renderer, "render_ids", None)):
        is_3d = getattr(renderer, "is_3d", False)
        pv.add("occlusion-safety", WARN if is_3d else PASS,
               f"renderer '{rname}' exposes no render_ids(), so occlusion cannot be certified"
               + (" - and it is a 3D renderer, so it MAY hide objects; count/enumeration GT is "
                  "UNVERIFIED under it. Add render_ids (see the renderer template)." if is_3d
                  else " (it is a 2D non-occluding renderer, so there is nothing to certify)."))
        return

    partial_reliable = _context_stable(renderer, graph.canvas)
    certify, policy = _occlusion_policy(graph)
    worst, worst_detail, worst_seed = PASS, "", 0
    n_checked = 0
    for s in range(min(seeds, _OCC_SEEDS)):
        try:
            sc = gen.generate(params, s)
            g = getattr(sc, "graph", None)
            if g is None:
                continue
            vf = _object_visibility(renderer, g)
        except Exception as exc:
            pv.add("occlusion-safety", FAIL, f"render_ids raised on seed {s}: {exc!r}")
            return
        if vf is None:
            continue
        n_checked += 1
        if policy == "derivable":
            status, detail = _certify_derivable(gen, sc, g, vf)
        else:
            status, detail = _occlusion_verdict(vf, partial_reliable, certify,
                                                presence_only=(policy == "presence"))
        if _worse(status, worst):
            worst, worst_detail, worst_seed = status, detail, s
    if worst == PASS:
        if policy == "derivable":
            scope = "the answer stays derivable from the visible objects"
        elif policy == "presence":
            scope = "every object stays at least partly visible"
        elif certify is not None:
            scope = f"the {len(certify)} answer-bearing object(s) stay visible"
        else:
            scope = "every object stays visible"
        pv.add("occlusion-safety", PASS,
               f"under '{rname}': {scope} across {n_checked} seed(s) - ground truth is "
               f"recoverable from the pixels")
    else:
        pv.add("occlusion-safety", worst, f"under '{rname}' (seed {worst_seed}): {worst_detail}")


def _validate_renderer_occlusion(renderer: Any, pv: PluginValidation) -> None:
    """Check the renderer's OPTIONAL render_ids: present, deterministic, correctly shaped,
    and HONEST - it must report all objects visible in a clear scene AND detect a
    deliberately occluded one. This is what lets scenes using the renderer be certified."""
    fn = getattr(renderer, "render_ids", None)
    if not callable(fn):
        pv.add("occlusion-report", WARN if renderer.is_3d else PASS,
               ("no render_ids(): scenes using this renderer cannot be occlusion-certified. "
                "It is a 3D renderer, so it MAY introduce occlusion and silently hide objects "
                "- add render_ids returning an (H, W) object-index map (front-most object per "
                "pixel, -1 = background). See the renderer template." if renderer.is_3d else
                "no render_ids() - fine for a 2D non-occluding renderer (nothing to certify)."))
        return

    canvas = (160, 120)
    clear = SceneGraph(
        objects=[GraphObject("a", "sphere", "red", (45.0, 60.0, 0.0), 18.0),
                 GraphObject("b", "cube", "blue", (120.0, 60.0, 0.0), 18.0)],
        canvas=canvas,
    )
    # A near object (small z) fully over a far one at the SAME screen position -> occlusion.
    occluded = SceneGraph(
        objects=[GraphObject("back", "sphere", "red", (80.0, 60.0, 5.0), 16.0),
                 GraphObject("front", "sphere", "blue", (80.0, 60.0, 0.0), 24.0)],
        canvas=canvas,
    )
    try:
        m1, m2 = np.asarray(fn(clear)), np.asarray(fn(clear))
        clear_v = _object_visibility(renderer, clear)
        occ_v = _object_visibility(renderer, occluded)
    except Exception as exc:
        pv.add("occlusion-report", FAIL, f"render_ids raised: {exc!r}")
        return
    if m1.shape != (canvas[1], canvas[0]):
        pv.add("occlusion-report", FAIL,
               f"render_ids returned shape {m1.shape}, expected (H, W) = {(canvas[1], canvas[0])}")
        return
    if not np.array_equal(m1, m2):
        pv.add("occlusion-report", FAIL,
               "render_ids is not deterministic (same graph produced different id maps)")
        return
    partial_reliable = _context_stable(renderer, canvas)
    clear_status, _ = _occlusion_verdict(clear_v, partial_reliable)
    occ_status, _ = _occlusion_verdict(occ_v, partial_reliable)
    if clear_status == PASS and occ_status == FAIL:
        pv.add("occlusion-report", PASS,
               "render_ids honestly reports visibility (an all-visible scene passes; a "
               "deliberately occluded object is detected) - scenes using this renderer "
               "can be occlusion-certified")
    else:
        pv.add("occlusion-report", WARN,
               f"render_ids present but its occlusion reporting looks off (clear scene -> "
               f"{clear_status}, occluded scene -> {occ_status}); it must map the FRONT-MOST "
               f"object index per pixel with the SAME projection/occlusion as render()")


# Probe

def _find_scene_for_probe(reg: Registry, probe: Any) -> Any | None:
    need = set(getattr(probe, "requires_gt_fields", []))
    names = reg.list_scenes()
    ordered = ([probe.name] if probe.name in names else []) + \
              [n for n in names if n != probe.name]
    for n in ordered:
        gen = reg.get_scene(n)
        try:
            sc = gen.generate(_default_params(gen), 0)
        except Exception:
            continue
        if need <= set(sc.ground_truth):
            return gen
    return None


def _validate_probe(reg: Registry, name: str, pv: PluginValidation, seeds: int) -> None:
    probe = reg.get_probe(name)
    pv.add("protocol", PASS, f"registered as a Probe (answer_type={probe.answer_type})")
    if probe.answer_type not in ANSWER_TYPES:
        pv.add("answer_type", FAIL, f"'{probe.answer_type}' not in {ANSWER_TYPES}")

    gen = _find_scene_for_probe(reg, probe)
    if gen is None:
        pv.add("scene-pairing", WARN,
               f"no built-in scene supplies requires_gt_fields "
               f"{probe.requires_gt_fields}; ship a matching scene to exercise it")
        return
    pv.add("scene-pairing", PASS, f"paired with scene '{gen.name}'")

    scene = gen.generate(_default_params(gen), 0)
    try:
        ans = probe.ground_truth(scene)
        ok, score, _ = probe.score(ans, ans)
        parsed = probe.parse(str(ans))
    except Exception as exc:
        pv.add("gt-roundtrip", FAIL, f"ground_truth/score/parse raised: {exc!r}")
        return
    if not ok:
        pv.add("gt-roundtrip", FAIL,
               f"ground-truth answer does not score correct against itself (score={score})")
    else:
        pv.add("gt-roundtrip", PASS, "ground-truth answer scores correct")
    # parse should recover the answer from its own string form (best-effort)
    p_ok, _, _ = probe.score(parsed, ans) if parsed is not None else (False, 0.0, None)
    if not p_ok:
        pv.add("parse-roundtrip", WARN,
               f"parse(str(gt)) did not recover the answer (got {parsed!r}) - check the "
               f"parser handles a clean model reply")
    else:
        pv.add("parse-roundtrip", PASS, "parse recovers the answer from its string form")

    # Perception report (optional) - the three-way encoding/grounding split.
    _validate_report(pv, probe, scene)
    # Oracle information-completeness (optional) - certifies "reasoning-limited" verdicts.
    _validate_oracle_completeness(pv, probe, gen, _default_params(gen), seeds)


# Model / Analyzer

def _validate_model(reg: Registry, name: str, pv: PluginValidation) -> None:
    model = reg.get_model(name)
    pv.add("protocol", PASS, f"registered as a ModelAdapter (is_open={model.is_open})")
    pv.add("live-call", WARN,
           "conformance only - a real inference call needs an API key and spends "
           "quota, so it is not exercised here")


def _validate_renderer(reg: Registry, name: str, pv: PluginValidation) -> None:
    renderer = reg.get_renderer(name)
    pv.add("protocol", PASS, f"registered as a Renderer (is_3d={renderer.is_3d})")
    graph = SceneGraph(
        objects=[GraphObject("a", "sphere", "red", (40.0, 40.0, 0.0), 16.0),
                 GraphObject("b", "cube", "blue", (110.0, 70.0, 1.0), 16.0)],
        canvas=(160, 120),
    )
    try:
        imgs = renderer.render(graph)
    except Exception as exc:
        # An optional renderer (e.g. pyrender_3d) legitimately needs extra deps - WARN,
        # not FAIL, so a valid-but-unavailable backend doesn't read as broken.
        pv.add("render", WARN, f"render() raised - optional deps missing? ({exc})")
        return
    if not imgs or not all(isinstance(i, Image.Image) for i in imgs):
        pv.add("render", FAIL, "render() must return a non-empty list of PIL Images")
        return
    elif imgs[0].size != (160, 120):
        pv.add("render", FAIL,
               f"rendered image size {imgs[0].size} != the graph canvas (160, 120)")
        return
    else:
        pv.add("render", PASS, f"renders a {imgs[0].size} image from the scene graph")

    # Determinism: the same graph must render to identical pixels, or a re-run shows the
    # model a different image than was scored (seed every RNG deterministically).
    try:
        imgs2 = renderer.render(graph)
        same = [i.tobytes() for i in imgs] == [i.tobytes() for i in imgs2]
    except Exception as exc:
        pv.add("determinism", WARN, f"second render raised: {exc!r}")
    else:
        pv.add("determinism", PASS if same else FAIL,
               "same graph -> identical pixels on re-render" if same else
               "same graph produced DIFFERENT pixels on re-render - seed every RNG from a "
               "fixed seed (a stochastic renderer must fix its sampler seed)")

    # Occlusion-reporting capability + honesty (enables scene-level certification).
    _validate_renderer_occlusion(renderer, pv)


def _validate_analyzer(reg: Registry, name: str, pv: PluginValidation) -> None:
    analyzer = reg.get_analyzer(name)
    conds = list(getattr(analyzer, "requires_conditions", []))
    pv.add("protocol", PASS, f"registered as an Analyzer (requires {conds or 'no'} conditions)")

    # Feed a tiny synthetic result set covering its required conditions; it must
    # return an AnalysisReport without crashing.
    conds = conds or ["full"]
    results = [
        Result(scene_id=f"s{i}", generator="g", factors={"x": i % 2}, probe="p",
               model="m", condition=c, raw="1", pred=1, gt=1, correct=(i % 2 == 0),
               score=float(i % 2 == 0), error=None, confidence=None, error_flag=None)
        for i, c in enumerate(conds * 2)
    ]
    try:
        report = analyzer.analyze(results)
    except Exception as exc:
        pv.add("analyze", FAIL, f"analyze() raised on synthetic results: {exc!r}")
        return
    if not isinstance(report, AnalysisReport):
        pv.add("analyze", FAIL,
               f"analyze() must return AnalysisReport, got {type(report).__name__}")
    else:
        pv.add("analyze", PASS, "returns an AnalysisReport on synthetic input")


# Formatting for the CLI

def format_reports(reports: list[PluginValidation]) -> str:
    mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}
    lines: list[str] = []
    for pv in reports:
        head = f"[{pv.family}] {pv.name}"
        verdict = "FAIL" if not pv.ok else ("OK (warnings)" if pv.warned else "OK")
        lines.append(f"\n{head}  ->  {verdict}")
        for c in pv.checks:
            lines.append(f"  {mark[c.status]:4s}  {c.name:16s}  {c.detail}")
    n_fail = sum(1 for pv in reports if not pv.ok)
    n_warn = sum(1 for pv in reports if pv.ok and pv.warned)
    lines.append(
        f"\n{len(reports)} plugin(s): {len(reports) - n_fail - n_warn} clean, "
        f"{n_warn} with warnings, {n_fail} failing."
    )
    return "\n".join(lines)


def all_ok(reports: list[PluginValidation]) -> bool:
    return all(pv.ok for pv in reports)
