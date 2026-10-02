"""The contract - everything imports from here; this module has zero internal imports.

Frozen once the core contract was validated through the oracle condition.
Changes to data-class fields or Protocol method signatures require explicit
sign-off - both sides of the contract
(scenes/probes/models/analyzers) must be audited before any modification.

Because extensibility (third parties adding their own scenes / probes / metrics)
depends on this contract staying stable, every post-freeze change is additive and
recorded here.

CONTRACT CHANGE LOG
-------------------
- 2026-07: added ``Result.meta`` - an optional, additive,
  backward-compatible metadata bag (defaults to ``{}``) mirroring ``Scene.meta``.
  It lets the runner tag a Result with analysis-only labels the analyzers read
  back (which oracle verbalization variant produced it, which difficulty tier the
  scene was) WITHOUT overloading ``factors`` (reserved for scene variables) or the
  frozen ``condition`` string. No existing field changed; all prior constructions
  (keyword-based) keep working. Reserved keys are documented on the field below;
  third-party plugins may add their own namespaced keys.
- Methodology correction (2026-09): added the ``"report"`` condition and the
  ``ReportSpec`` dataclass (both additive). A probe MAY expose an OPTIONAL
  ``perception_report(scene) -> ReportSpec | None`` method (discovered by the runner
  via ``getattr`` - deliberately NOT added to the ``Probe`` Protocol, so existing
  probes remain conformant without change). It drives a fourth condition that tests
  perception in isolation, splitting encoding failures from grounding/arbitration
  failures. See docs/methodology.md. No existing field or signature changed.
- Several perception reports (2026-09): ``ReportSpec`` gained ``name`` and ``primary``
  (both additive, both defaulted so every existing spec keeps its behavior), and
  ``perception_report(scene)`` MAY now return a list of specs instead of one. The runner
  runs each as a ``"report"`` row and stamps ``meta["report_name"]`` /
  ``meta["report_primary"]``; only the primary feeds ``acc_report`` and the
  encoding/grounding split. This exists so a probe can ask both a robust primitive and a
  tight one: when the two disagree, the gap between them is the model's inability to
  SERIALIZE what it saw, which would otherwise be scored as a failure to see it.
- Truncation is not a wrong answer (2026-09): ``Response.truncated`` (additive,
  defaults False) lets an adapter report that the model hit its output limit mid-reply.
  The runner maps it to ``error_flag="truncated"``, which every analyzer already skips,
  so a cut-off chain of thought stops being scored as an incorrect answer. Discovered
  live: a reasoning model under the oracle condition ran past a 256-token budget on
  every scene, and the parser was reading whichever candidate it was mid-sentence about.
- Session credentials (2026-09): an adapter now resolves its API key through
  ``core.credentials.resolve`` rather than ``os.environ`` directly. Resolution checks a
  ``ContextVar`` scope first and the environment second, so nothing that worked before
  changes, and a UI serving several people can supply a key per session without it ever
  entering the process environment where the next visitor's run would pick it up. Two
  additive public attributes on the shipped adapters support this - ``api_key_env``
  (``None`` = needs no credential) and ``provider`` - plus ``status`` / ``status_note``
  carrying the dated roster probe. None is part of the ``ModelAdapter`` Protocol, so a
  third-party adapter that declares none of them stays conformant and is simply shown
  as needing no key. Also additive: ``Result.meta["model_error"]`` carries the reason an
  adapter call failed, which was previously printed to stderr and discarded.
- Oracle-validity certificate (2026-09): a probe MAY expose an OPTIONAL
  ``oracle_solver(scene) -> answer | None`` - a deterministic reference reasoner that
  re-derives the answer from ONLY the facts the oracle verbalizes (never the stored
  answer). Matching ground truth across seeds proves the oracle is information-complete,
  so a model's oracle failure is reasoning rather than a lossy verbalization.
  Perception-bound probes instead set the class attribute ``oracle_states_answer = True``
  (their oracle legitimately states the answer). Both are discovered via ``getattr`` -
  additive, not part of the ``Probe`` Protocol. ``validate`` runs the certificate.
- Renderer abstraction (2026-09): added ``GraphObject``, ``SceneGraph`` (both additive
  data classes) and the ``Renderer`` Protocol (a fifth plugin family). A scene MAY build
  a ``SceneGraph`` and derive its ground truth from it, then hand the graph to any
  registered renderer (flat 2D, depth-shaded 2.5D, or an optional photorealistic 3D
  backend) - realism becomes a knob orthogonal to the task, and the ground truth is
  identical across renderers. Existing PIL scenes are unaffected (they self-render). See
  renderers/ and scenes/blocks.py. No existing field or signature changed.
- Occlusion-safety certificate (2026-09): added ``Scene.graph`` - an OPTIONAL, additive
  field (defaults to None) holding the renderer-independent ``SceneGraph`` a graph-based
  scene was built from. A Renderer MAY expose an OPTIONAL ``render_ids(graph) ->
  numpy.ndarray`` - an (H, W) int object-index map (front-most object per pixel, -1 =
  background) produced with the SAME projection and occlusion as ``render``. Discovered
  via ``getattr`` (deliberately NOT added to the ``Renderer`` Protocol, so existing
  renderers stay conformant without change). ``validate`` uses the two together to CERTIFY
  that, under a chosen renderer, every ground-truth object stays visible (not occluded) -
  the honesty guarantee that makes true-3D rendering safe: it FAILs a graph scene whose
  count/enumeration ground truth cannot be recovered from the pixels. No existing field or
  signature changed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image

# Data classes

@dataclass
class Scene:
    id: str
    images: list[Image.Image]
    ground_truth: dict[str, Any]
    factors: dict[str, float | int | str]   # {} for static/dataset scenes
    meta: dict[str, Any]
    # Additive (2026-09, occlusion safety): the renderer-independent SceneGraph this scene
    # was built from, or None for scenes that draw pixels directly. When set, a graph-based
    # scene opts in to `renderprobe validate`'s occlusion certification - the check that
    # every ground-truth object stays visible (not hidden) under the chosen renderer, so a
    # count/enumeration answer is still recoverable. See core/validate.py and renderers/.
    graph: "SceneGraph | None" = None


@dataclass
class Response:
    text: str
    confidence: float | None = None         # None if model doesn't expose it
    # The model hit its output limit before finishing. Additive (2026-09), defaulted so
    # every existing adapter and fixture keeps working. A truncated reply is NOT a wrong
    # answer: the model was still working when it was cut off, and a parser reading the
    # last matching token off it recovers whatever it happened to be mid-sentence about.
    # The runner turns this into error_flag="truncated" so the row leaves the accuracies
    # instead of silently counting as a failure the model never made.
    truncated: bool = False


@dataclass
class Result:
    scene_id: str
    generator: str
    factors: dict[str, Any]
    probe: str
    model: str
    condition: str                          # "full" | "blind" | "oracle" | "report"
    raw: str
    pred: Any
    gt: Any
    correct: bool
    score: float
    error: float | None                     # numeric distance; None for categorical
    confidence: float | None
    error_flag: str | None                  # None | "parse_fail" | "model_error" | "gen_error"
    # Additive analysis-metadata bag (see the CONTRACT CHANGE LOG above). Distinct
    # from `factors` (scene variables): `meta` carries labels the runner stamps for
    # analyzers to group/gate on. Reserved keys currently in use:
    #   "oracle_variant": str  - which oracle verbalization template produced this
    #                            result (oracle condition only).
    #   "tier":           str  - scene difficulty tier, e.g. "trivial" | "standard"
    #                            (used by the oracle ceiling / prompt-validity check).
    #   "model_error":    str  - why the adapter call failed, on rows carrying
    #                            error_flag="model_error". Credential-scrubbed. Present
    #                            only on those rows; no analyzer reads it.
    # Third-party plugins may add their own keys; prefer a plugin-name prefix.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    name: str
    payload: dict[str, Any]
    figures: list[Any] = field(default_factory=list)


@dataclass
class GraphObject:
    """One object in a renderer-agnostic scene graph. Positions are in pixels; ``z`` is
    depth (>= 0, larger = farther from camera) so a 3D/2.5D renderer can add perspective
    and occlusion while a flat 2D renderer simply ignores it. The GROUND TRUTH is derived
    from these fields, so it is identical no matter which renderer draws the pixels."""
    id: str
    shape: str                              # "sphere" | "cube" | "cylinder" | "cone"
    color: str                              # palette color name
    position: tuple[float, float, float]    # (x, y, z) - x,y screen px; z depth
    size: float                             # base radius / half-extent in px
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Renderer-independent description of a scene. A scene generator builds this (and
    derives its ground truth from it); a Renderer turns it into pixels. Same graph +
    different renderer = same exact GT, different realism. See renderers/.

    ``meta`` carries renderer hints (camera, lighting, ``spp``) and an optional
    occlusion policy read by ``renderprobe validate``::

        meta["occlusion"] = {
            "certify": [0, 3, 7],   # objects the ANSWER depends on; default: every one
            "policy": "visible",    # or "derivable"
        }

    By default every object must stay visible, since a hidden object usually means the
    ground truth is unrecoverable. ``certify`` narrows that to the answer-bearing
    objects, so a dense scene may hide background clutter without being rejected.
    ``policy: "derivable"`` is for tasks where objects are hidden BY DESIGN (counting a
    stack including the cubes you cannot see); such a scene must also expose
    ``derive_from_visible(graph, visible_indices) -> dict`` re-deriving its ground-truth
    fields from the visible objects alone, and validate checks that against the real
    ground truth rather than accepting the claim. See docs/methodology.md §6.
    """
    objects: list[GraphObject]
    canvas: tuple[int, int]                 # (width, height) in pixels
    background: tuple[int, int, int] = (255, 255, 255)
    meta: dict[str, Any] = field(default_factory=dict)   # camera/lighting hints (renderer-specific)


@dataclass(frozen=True)
class ReportSpec:
    """A perception-report sub-task: a direct perceptual question over the image
    whose answer is a ground-truth primitive requiring NO task reasoning.

    Returned by a probe's optional ``perception_report(scene)``; the runner runs it
    as the ``"report"`` condition (over the image) and scores the response with this
    spec's own ``parse``/``score`` against ``ground_truth`` - which may differ from
    the probe's task parse/score/GT. See docs/methodology.md §3.
    """
    question: str
    ground_truth: Any
    parse: Callable[[str], Any]
    score: Callable[[Any, Any], tuple[bool, float, float | None]]
    # Optional report-specific image(s). When set, the runner shows THESE instead of
    # scene.images for the report condition - e.g. a copy with the target object
    # highlighted, so the perceptual question needs no pixel-coordinate localization
    # (which VLMs are poor at). None => use scene.images unchanged.
    images: list[Image.Image] | None = None
    # Which report this is, when a probe returns several. Stamped onto each Result's
    # meta so they stay separable in analysis.
    name: str = "report"
    # Exactly one report feeds `acc_report` and therefore the encoding/grounding split;
    # the rest are diagnostic. A probe returning several must mark one primary, which
    # is what a single returned spec is by default. Pick the report that most nearly
    # measures "did the model extract enough to solve the task": a looser primitive
    # answers an easier question than the one the cascade is about to ask of it.
    primary: bool = True


# Plugin protocols

@runtime_checkable
class SceneGenerator(Protocol):
    name: str
    params_schema: dict[str, Any]

    def generate(self, params: dict[str, Any], seed: int) -> Scene: ...


@runtime_checkable
class Probe(Protocol):
    name: str
    answer_type: str                        # "categorical"|"numeric"|"bbox"|"ordering"
    requires_gt_fields: list[str]

    def question(self, scene: Scene) -> str: ...
    def ground_truth(self, scene: Scene) -> Any: ...
    def parse(self, raw: str) -> Any: ...
    def score(self, pred: Any, gt: Any) -> tuple[bool, float, float | None]: ...
    def oracle_prompt(self, scene: Scene) -> str | None: ...  # None = no oracle support


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    is_open: bool

    def run(self, images: list[Image.Image], prompt: str) -> Response: ...
    def attention(self, images: list[Image.Image], prompt: str) -> Any: ...


@runtime_checkable
class Analyzer(Protocol):
    name: str
    requires_conditions: list[str]         # subset of {"full", "blind", "oracle", "report"}
    requires_varying_factor: bool          # True => needs >=2 distinct factor values

    def analyze(self, results: list[Result]) -> AnalysisReport: ...


@runtime_checkable
class Renderer(Protocol):
    """Turns a renderer-independent SceneGraph into pixels. The GROUND TRUTH lives with
    the graph, not the renderer, so realism is a pluggable knob that never changes what
    the scene means. A new renderer needs only a PLUGIN entry (see renderers/).

    OPTIONAL occlusion-safety hook (discovered via getattr, NOT part of this Protocol so
    existing renderers stay conformant):

        def render_ids(self, graph: SceneGraph) -> numpy.ndarray: ...

    It returns an (H, W) integer array whose value at each pixel is the INDEX (into
    ``graph.objects``) of the FRONT-MOST object covering that pixel, or -1 for background.
    It MUST use the SAME projection and draw order as ``render`` so that what it reports as
    visible is exactly what ``render`` shows. ``validate`` uses it to certify that every
    ground-truth object stays visible (not occluded) under this renderer - the guarantee
    that keeps count/enumeration ground truth trustworthy once a renderer introduces
    occlusion (true 3D). Omit it only for a renderer that can NEVER occlude."""
    name: str
    is_3d: bool                            # True if it uses depth (perspective/occlusion)

    def render(self, graph: SceneGraph) -> list[Image.Image]: ...


# Valid sentinel values - single source of truth

CONDITIONS = frozenset({"full", "blind", "oracle", "report"})
ANSWER_TYPES = frozenset({"categorical", "numeric", "bbox", "ordering"})
ERROR_FLAGS = frozenset({"parse_fail", "model_error", "gen_error"})
