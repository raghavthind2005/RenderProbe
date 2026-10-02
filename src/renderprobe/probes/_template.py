"""COPY-ME TEMPLATE - a custom probe (a question + how to grade it).

WHAT A PROBE IS
    A probe defines the TASK asked about a scene: the question text, how to read the
    ground-truth answer out of the scene, how to parse a model's raw reply, and how to
    score it. One probe can run over MANY scenes (any scene whose ground_truth carries
    the fields the probe declares) - this is what lets a new metric reuse existing
    scenes, and vice versa.

    This template asks "what color is the largest object?" and reads the scene's
    ``objects`` list. It pairs out of the box with the scene template
    (``renderprobe new scene my_task``): scaffold both and they run together with no
    edits, which is the quickest way to see every condition and every tab populated
    before you write anything of your own.

HOW TO USE
    1. Copy out, drop the underscore, rename the class and ``name``.
    2. Set ``requires_gt_fields`` to the ground_truth keys you read - the runner uses
       this to pair your probe with compatible scenes.
    3. renderprobe validate my_probes/biggest_color.py

WHAT EACH METHOD BUYS YOU
    question/parse/score  the task itself: Accuracy, Taxonomy, Inspect.
    oracle_prompt         the `oracle` condition, and with it `recovery`.
    oracle_solver         the oracle-completeness certificate that makes `recovery`
                          evidence rather than a number.
    perception_report     the `report` condition, and with it the encoding vs
                          grounding split - the headline of the whole tool.
    chance_level          the chance line in tools/calibrate.py, which is what tells
                          you whether the scene can be measured at all.

    Only the first row is required. Everything else is discovered with getattr, so a
    probe that stops after `score` is valid - it just produces less.

THE ORACLE (this is the heart of the diagnostic)
    ``oracle_prompt`` must verbalize the PERCEPTUAL FACTS as text so a model can answer
    WITHOUT the image - it isolates reasoning from perception. Give the facts
    (positions, colors, sizes), not the verdict, UNLESS the answer IS a raw percept
    (as with counting). Return None if your task has no meaningful text form; the
    perception/reasoning decomposition is then unavailable for this probe.
"""
from __future__ import annotations

import re

from PIL import Image, ImageDraw

from renderprobe.core.schema import ReportSpec, Scene

_COLORS = ("red", "blue", "green", "orange", "purple", "yellow", "pink", "brown", "gray", "cyan")

# A high-contrast marker color that is NOT one of the object colors, so the ring can
# never be mistaken for an object. Used by the perception report below.
_RING_RGB = (230, 0, 230)   # magenta


def _ring(img: Image.Image, x: int, y: int, r: int) -> Image.Image:
    """Draw a magenta ring hugging one object - a perceivable HANDLE so the report can
    point at a target without asking the model for pixel coordinates (which VLMs read
    poorly, a confound that would pollute the perception measurement)."""
    out = img.copy()
    ImageDraw.Draw(out).ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3],
                                outline=_RING_RGB, width=3)
    return out


class _TemplateProbe:
    name = "template_probe"
    answer_type = "categorical"                # "categorical" | "numeric" | "bbox" | "ordering"
    requires_gt_fields = ["objects"]           # scenes must provide these ground_truth keys

    def question(self, scene: Scene) -> str:
        return "What color is the largest object in the image? Answer with a single word."

    def ground_truth(self, scene: Scene):
        # Derive the answer from the scene's exact ground truth - never from the pixels.
        objects = scene.ground_truth["objects"]
        biggest = max(objects, key=lambda o: o.get("r", o.get("size", 0)))
        return biggest["color"]

    def parse(self, raw: str):
        # Pull a known color word out of the model's free-text reply.
        low = raw.lower()
        for c in _COLORS:
            if re.search(rf"\b{c}\b", low):
                return c
        raise ValueError(f"no color word in reply: {raw!r}")

    def score(self, pred, gt) -> tuple[bool, float, float | None]:
        correct = (pred == gt)
        return correct, (1.0 if correct else 0.0), None   # (correct?, score, numeric error|None)

    def oracle_prompt(self, scene: Scene) -> str | None:
        # Facts as text: color + size of every object, so "which is largest" is a pure
        # reasoning step. We do NOT state the answer.
        objs = scene.ground_truth["objects"]
        facts = "; ".join(
            f"a {o['color']} object of size {o.get('r', o.get('size', 0))}" for o in objs
        )
        return f"There are these objects: {facts}. {self.question(scene)}"

    # -- OPTIONAL: certify the oracle is INFORMATION-COMPLETE --
    #
    # If you ship an oracle, prove it's fair: an oracle_solver re-derives the answer from
    # ONLY the facts your oracle states - never the stored answer. When it matches ground
    # truth on every seed (validate checks it), a model's oracle failure is genuinely
    # REASONING, not a lossy/under-specified oracle. If your task is perception-bound (the
    # oracle necessarily states the answer, e.g. "what color is X"), skip this and set the
    # class attribute `oracle_states_answer = True` instead.
    def oracle_solver(self, scene: Scene) -> str | None:
        # Here the "facts" are the objects' colors + sizes (what oracle_prompt lists);
        # the reasoning step re-derived is "which is largest". Never read a stored answer.
        return max(scene.ground_truth["objects"],
                   key=lambda o: o.get("r", o.get("size", 0)))["color"]

    # -- OPTIONAL: the perception report (this is what unlocks the 3-way decomposition) --
    #
    # It isolates PERCEPTION of the task's atomic primitive, requiring NO task reasoning.
    # Here the task is "color of the LARGEST object" - the reasoning step is finding the
    # largest. The report removes that step: it RINGS the largest object and just asks its
    # color. Then, in a run:
    #   report high + task low  -> the model sees colors fine but can't pick the largest
    #                              => GROUNDING/reasoning-limited (not perception)
    #   report low              -> it can't even read a pointed-at color => ENCODING-limited
    #
    # Three rules make a report trustworthy (validate checks the first two):
    #   1. Its ground truth must round-trip (score(gt, gt) is correct).
    #   2. If you point at a target, POINT WITH A HANDLE (a drawn ring in spec.images),
    #      never "the object at (x, y)" - coordinate localization is a confound.
    #   3. The question must need no task reasoning - only perception of the primitive.
    # Return None when a scene can't support a report; the run falls back to two-way.
    def perception_report(self, scene: Scene) -> ReportSpec | None:
        if not scene.images:
            return None
        objs = scene.ground_truth["objects"]
        target = max(objs, key=lambda o: o.get("r", o.get("size", 0)))
        r = int(target.get("r", target.get("size", 16)))
        return ReportSpec(
            question=("One object is circled with a bright magenta ring. "
                      "What color is the object inside the ring? Answer with one word."),
            ground_truth=target["color"],
            parse=self.parse,                 # reuse this probe's own parse/score
            score=self.score,
            images=[_ring(scene.images[0], int(target["x"]), int(target["y"]), r)],
        )

    # -- OPTIONAL: what guessing is worth on THIS scene --
    #
    # Only the probe knows how many answers its question admits, and for a scene whose
    # answer set changes with a parameter it is the only honest source.
    # tools/calibrate.py asks for this before you spend a full run; without it the
    # verdict degrades to UNKNOWN, because accuracy alone cannot say whether a score
    # beats guessing.
    #
    # Be GENEROUS to the model here. This one assumes a guesser who has already read
    # every color in the picture and merely picked the wrong one, rather than one
    # guessing from all colors that exist. That makes the bar harder to clear, which is
    # the right direction for a gate you are using to talk yourself out of a run.
    def chance_level(self, scene: Scene) -> float:
        colors = {o["color"] for o in scene.ground_truth["objects"]}
        return 1.0 / len(colors) if colors else 0.0


PLUGIN = _TemplateProbe()
