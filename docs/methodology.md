# Methodology

RenderProbe asks one scene three ways and reads the gaps between the answers. The gaps place a failure in one of three categories: the model could not extract the fact from the image, extracted it but did not use it, or could not reason with it even when handed the
facts in text. This document states what each measurement is, what it supports, and the literature it rests on.

The claim:

> The oracle condition supplies the ground-truth perceptual primitives as text, so the
> model no longer has to extract or ground them from the image. If accuracy recovers, the
> failure lay in the **visual pathway**; if it stays low even with the facts handed over,
> the failure is **reasoning**. A companion **perception-report** probe splits the
> visual-pathway case into *"didn't encode it"* and *"encoded but didn't use it."*

## 1. The conditions,

```
acc_full   = accuracy(image + question)                           perceive AND reason
acc_oracle = accuracy(ground-truth primitives as text + question) reason only
acc_report = accuracy(image + "what is <primitive>?")             perceive only
acc_blind  = accuracy(question alone, no image)                   prior over the answers
recovery   = acc_oracle - acc_full
```

`full` and `oracle` always run. `report` runs whenever the probe offers one. `blind` is
opt-in: it doubles the calls and only earns that where the answer set is small enough for
a prior over it to matter.

Swapping the image for text does not hold reasoning fixed while removing perception. It
changes the modality the premises arrive in, and three distinct causes raise `recovery`:

1. **Encoding failure** - the vision tower never extracted the fact.
2. **Grounding / arbitration failure** - the fact *was* extracted, but the model failed to
   *use* it, overriding a correctly encoded percept with a language prior.
3. **Reasoning failure** - the model fails even when handed perfect premises.

`recovery` separates **{1, 2} from {3}**. It does not separate **1 from 2**. Reporting the
whole `{1, 2}` bucket as "perception" would mislabel a grounding failure, where encoding is
fine, as a failure to see. That is what the `report` condition exists to prevent, and it is
why `recovery` is reported as an **upper bound on the visual-pathway contribution** rather
than an isolation of the vision encoder.

### The buckets

| bottleneck | meaning | rule |
| --- | --- | --- |
| `none` | solved from the image; nothing to decompose | `acc_full >= 0.90` |
| `not-vision-limited` | the image beats the text oracle, so the linearization loses structure the picture supplies | `recovery < -0.10` |
| `reasoning-limited` | fails even given the facts as text | `recovery <= 0.10` |
| `vision-pathway-limited` | the visual pathway is the limit, with no report available to split it | `recovery > 0.10`, no report |
| `grounding/integration-limited` | reports the primitive but still fails the task | `acc_report >= 0.85` or `acc_report - acc_full > 0.10` |
| `encoding-limited` | cannot report the primitive from the image | otherwise |

The thresholds are fixed constants in [`analysis/decomposition.py`](../src/renderprobe/analysis/decomposition.py)
(`0.85` near-ceiling, `0.10` materially non-zero, `0.90` solved). They are documented rather
than tuned to a model.

Each label is a step function of point estimates, so a verdict sitting near a cutoff can
flip on one item. Every label therefore carries `bottleneck_stability`: the share of 2000
scene-level bootstrap resamples that reproduce it. Below 0.95 the verdict prints as
*unsettled*, together with the label it competes with.

Two confounds are guarded outside the cascade. The language-prior shortcut is caught by
**answer balance** (`validate` fails a scene whose answers are skewed across seeds) and
measured directly by the opt-in `blind` baseline. The prompt  **ceiling check**: a correct verbalization must lift a competent reasoner near
ceiling, and if it does not, the oracle template is indicted rather than the model.

## 2. The literature this rests on

That VLM encoders often represent the needed information faithfully while the *integrated*
model fails downstream is well documented, and it is the reason the decomposition needs a
third condition rather than two:

- **"Arbitration Failure, Not Perceptual Blindness"** (arXiv:2604.09364) shows visual
  attributes are linearly decodable from early layers (AUC >> 0.86) *equally
  well whether the model answers right or wrong* (success/failure L2 ratio 0.81-1.20);
  causal activation patching flips 60-84% of outputs by swapping hidden states, while
  text-only patching flips 0-2%. Its conclusion, *"VLMs already see well; the challenge is
  acting on what they see,"* is failure mode (2) isolated empirically, and it is exactly
  the case an oracle-only diagnostic would misattribute to perception.
- **"Position: Reasoning After Perception Means Reasoning Without Vision"**
  (arXiv:2507.16863) formalizes an *information-collapse bound*: projecting vision into
  text-aligned space is a filter governed by *verbalizability* - "information that can be
  easily described in language is preserved, while information that resists linguistic specification is systematically collapsed." A text oracle is a faithful stand-in for
  perception only **when the task-relevant primitives are fully verbalizable**.
- Replacing images with ground-truth captions or scene graphs to test reasoning in
  isolation ("pre-captioned" evaluation, the "reasoning-oracle proxy") is an established
  design, not a fringe one.
- **MARBLE** (arXiv:2506.22992) reports models struggling to convert pieces into arrays,
  which is the operation the `polycube` report asks for directly.
- Three further anchors the scenes are built against: MMVP / "Eyes Wide Shut?"
  (arXiv:2401.06209) on hallucinated, language-prior answers; "Vision language models are
  blind" (arXiv:2407.06581) on low-level perceptual failure; CLEVR (arXiv:1612.06890) on
  controlled synthetic diagnosis.


The information-collapse critique is damning for *natural-image* benchmarks and weak
against this design specifically: the scenes are synthetic with **exactly verbalizable
ground truth** - discrete colors, integer coordinates, adjacency lists with integer
lengths - so the verbalizability filter loses close to nothing and the oracle text really
does carry all the task-relevant primitives. The **arbitration** critique applies in full,
and §3 is the answer to it.

## 3. The perception-report condition

To split *encoding* from *grounding* without model internals, a fourth condition asks a
direct perceptual sub-question **about the same image**, whose answer is a ground-truth
primitive and which requires no task reasoning. It is scored against ground truth.

```
acc_report = accuracy(image + "what is <primitive>?")    perceive only, no reasoning
```

The cascade, for a model whose `acc_full` is not already near ceiling
(see [`analysis/decomposition.py`](../src/renderprobe/analysis/decomposition.py)):

```
recovery = acc_oracle - acc_full
if recovery < -0.10:                    -> not-vision-limited   (image beats the oracle)
elif recovery <= 0.10:                  -> reasoning-limited    (survives perfect premises)
elif no report available:               -> vision-pathway-limited (split unresolved)
elif acc_report >= 0.85
     or acc_report - acc_full > 0.10:   -> grounding/integration-limited (sees it, doesn't use it)
else:                                   -> encoding-limited     (can't see it)
```

The split asks whether **perception explains the task failure**, not whether `acc_report`
clears a fixed bar, and the difference is not academic. On `route`, gemma-3-27b reported
the primitive on 0.83 of the maps and solved 0.33 of them. A rule that simply tested
`acc_report >= 0.85` read that as *encoding-limited* and said so in words: "the model
cannot even report the primitive from the image" - a false statement about a model getting
it five times in six. Comparing the two accuracies instead gets every model-scene pair
measured so far right, where the ceiling test alone got two of them wrong:

| scene | model | `acc_report` | `acc_full` | margin | verdict |
|---|---|---:|---:|---:|---|
| route | gemma-3-27b | 0.83 | 0.33 | +0.50 | grounding/integration-limited |
| route | llama-3.2-11b | 0.78 | 0.11 | +0.67 | grounding/integration-limited |
| polycube | gemma-3-27b | 0.00 | 0.17 | -0.17 | encoding-limited |
| polycube | llama-3.2-11b | 0.00 | 0.11 | -0.11 | encoding-limited |

The ceiling test survives as a fast path, for a report that is near perfect while the task
sits just under the solved line and the margin is therefore small.

**Caveat.** `acc_report` and `acc_full` answer different questions with different guessing
rates, so this compares like with nearly-like rather than being an identity. The margin has
been large in every case seen so far, but a scene whose report question is much easier than
its task, or much harder, will narrow the comparison, and the verdict should be read
alongside `bottleneck_stability` rather than on its own.

This is a black-box analog of the linear-probe result in 2604.09364: if a model can
*report* the primitive from the image but *fails the task*, the fault is downstream of
perception.

## 4. probes report

- **`route`** asks for the number written beside one named road - the length of a road the
  cheapest route actually uses, so getting it wrong is on its own enough to get the task
  wrong. Both towns are named by the letters the image already carries, so nothing has to
  be located by pixel coordinate.
- **`polycube`** asks two questions of different strictness. `cells` is primary: describe
  the target as unit cubes on an integer grid, scored by **sufficiency** - a reading counts
  as correct when that reading, handed to a perfect reasoner, picks the right candidate. So
  `acc_report` is not a proxy for perception but perception measured in the units of the
  task, directly comparable to `acc_full` on the same scenes. `cubes` ("how many unit cubes
  make up the <color> piece") is the robust floor: counting needs almost no output format,
  so `cubes` right while `cells` is wrong localizes the failure to serialization rather
  than sight.

Only the **primary** report feeds `acc_report`. A probe may ask several perceptual
sub-questions, and pooling questions of different strictness would make `acc_report` a
blend that answers neither.

Two rules keep a report trustworthy, and `validate` enforces both: it must round-trip its
own ground truth, and if it points at a particular object it must ship a highlighted image
rather than naming pixel coordinates. The second rule is not cosmetic. An early report
asked for the "color of the object at (x, y)" and scored 0.25 on a model that scored 1.00
on the same perceptual question once the target was marked with a ring: VLMs localize
poorly by coordinate, and a report that depends on it measures localization while claiming
to measure encoding.

**Limit.** A report certifies perception of the **answer-bearing** primitive, not every
intermediate object a chain of reasoning traverses. It is a first-order discriminator, not
a complete perceptual audit. For a task that is perception-bound to begin with, the split
is finer-grained rather than encoding-versus-grounding, since there is no grounding step to
separate.

The offline fixtures in `tests/fixtures/` carry further report designs - a ringed target,
haloed graph nodes, an absolute-position read - which exist to exercise the machinery
without a key rather than to ship as scenes.

## 5. Oracle validity

The decomposition rests on the oracle being a faithful, information-complete verbalization:
a "reasoning-limited" verdict is only honest if the oracle text really did hand the model
everything it needed. Two guards make that checkable rather than assumed.

**Completeness certificate (offline, provable).** A reasoning probe exposes
`oracle_solver(scene)`, a deterministic reference reasoner that re-derives the answer from
**only the facts the oracle states**, never the stored answer field. `route` re-runs
Dijkstra over the stated roads; `polycube` recomputes congruence over the rotation group
from the stated coordinates. `validate` and the test suite check it across seeds, and both
shipped probes certify **12/12**: the facts provably suffice, so a model's oracle failure
is genuinely reasoning rather than missing information. The solver applies the task's *own*
rule, so this is a **completeness** proof and not an independent re-derivation - it catches
missing facts, a corrupted stored answer and question round-trip failures, but it shares
any algorithmic bug with the generator. A perception-bound probe instead declares
`oracle_states_answer = True`, where completeness is definitional and `recovery` measures
the full perception gap.

**Fragility across wordings (empirical).** Each probe offers several equivalent oracle
templates - `route` states the roads as an adjacency block, as prose, and as JSON;
`polycube` states the cells the same three ways - and the `oracle` analyzer runs all of
them and reports `recovery` per variant plus `oracle_variant_fragility`, the spread across
them. A small spread means the result is a property of the task; a spread above threshold
flags the oracle as measuring verbalization quality, and the verdict as untrustworthy. The
canonical variant is what the top-level numbers report, so the summary stays readable.

Together: the facts are *provably sufficient*, and the result is *stable across wordings*.

## 6. Occlusion safety

The decomposition assumes a trustable ground truth, meaning a perfect perceiver could
always answer from the image. For a count or enumeration task that requires every object to
be visible. Flat 2D placement guarantees it, but once realism becomes a knob - 2.5D, and
especially a true-3D renderer with perspective - a nearer object can hide a farther one,
silently making the ground truth unrecoverable and corrupting the verdict. The guarantee is
therefore machine-checked rather than left to the scene author.

- **Segmentation pass.** A renderer may expose `render_ids(graph) -> (H, W) int array`
  giving the front-most object index per pixel (-1 for background), produced with the
  *same projection and draw order* as `render`, so what it reports visible is exactly what
  `render` shows. It is discovered with `getattr` rather than added to the `Renderer`
  protocol, so existing renderers stay conformant. Of the four shipped renderers, `pil_2d`,
  `pil_25d` and `mitsuba_3d` implement it; `pyrender_3d` does not, and a 3D renderer
  without it draws a warning that enumeration ground truth is unverified under it.
- **Certification, before any API call.** For a scene that exposes a `SceneGraph`,
  `validate --renderer <name>` renders the id map across sampled seeds and certifies that
  every object survives. An object with essentially no visible pixels is a FAIL, caught
  before quota is spent rather than after the verdict is polluted. A renderer's
  `render_ids` is itself self-tested: it must pass an all-visible scene and *detect* a
  deliberately occluded one.
- **Scale-robust by construction.** Full occlusion is scale-invariant - a hidden object has
  zero pixels however the renderer sizes things - so that FAIL is always sound. The finer
  partial-occlusion warning needs an object's footprint to be independent of what else is
  in the scene, so `validate` self-checks that property per renderer (an object rendered
  alone against the same object beside a decoy) and applies the partial band only where it
  holds. A renderer that legitimately shrinks distant objects by the scene's depth range,
  such as `pil_25d`, is therefore never mistaken for one that occludes them.
- **Tasks where hiding is the point.** Certifying *every* object is right for a sparse
  scene and wrong for two cases. A dense scene may legitimately hide background clutter
  while everything the question turns on stays in view; and an inference-from-occlusion
  task (*how many cubes are in this stack, including the ones you cannot see*) is **made
  of** hidden objects. A graph declares its policy in `meta["occlusion"]`: `certify`
  narrows certification to the answer-bearing objects, and `policy: "derivable"` marks the
  second case. Derivability is checked rather than believed - such a scene must expose
  `derive_from_visible(graph, visible_indices)` re-deriving its ground-truth fields from
  the visible objects alone, and `validate` compares that against the real ground truth. A
  scene that declares the policy without the hook FAILs, which keeps this an extension of
  the guarantee rather than a way out of it.

This is what lets realism be dialed up honestly: the same certifier that passes the flat
renderers catches a true-3D renderer the moment its perspective starts hiding objects, so a
scene is known to be untrustworthy under a tier before a model is run on it.

## 7. Realism tiers and the render budget

Ground truth is renderer-independent, so realism is a cost and fidelity knob, turned only
when it is needed.

| renderer | cost per image | what it is for |
|---|---|---|
| `pil_2d` | ~1 ms | developing a scene, quick sanity checks (the default) |
| `pil_25d` | ~ms | headless depth-shaded runs, the everyday realism tier |
| `mitsuba_3d` | seconds (about 6 s for a polycube scene) | true-3D CPU path tracing, no OpenGL or GPU |
| `pyrender_3d` | GPU-bound | photoreal where an OpenGL backend exists |

Two levers keep the expensive tier practical:

- **Samples per pixel.** A graph scene forwards `renderer_opts` onto `graph.meta`, which a
  renderer reads as hints; for `mitsuba_3d`, `renderer_opts: {spp: N}` trades noise for
  speed roughly linearly. Use `spp: 4-8` while iterating and `32-64` for a final render.
- **A render-budget preview.** `renderprobe run` prints how many scene images will be
  rendered and warns when a true-3D renderer is in play. Each scene is rendered **once** at
  generation and reused across every model and condition, so the render count is the number
  of scenes, not scenes x models x conditions.

The instrument keeps **no persistent render cache**. For a measurement instrument, a stale
cache silently feeding a model the wrong image is exactly the error the honesty checks
exist to prevent, and within a run nothing is re-rendered anyway. Resuming a long run is
handled outside the package instead: [`tools/run_experiment.py`](../tools/run_experiment.py)
keeps an explicit, discardable cache of generated scenes so an interrupted run re-asks about
the identical picture rather than a fresh sample from a stochastic renderer.

