"""COPY-ME TEMPLATE - a custom scene generator.

WHAT A SCENE IS
    A scene generator turns (params, seed) into a Scene: one or more images plus the
    EXACT ground truth describing what is in them. RenderProbe then asks a probe's
    question about the image and grades the answer.

HOW TO USE THIS FILE
    1. Copy it into your OWN folder and rename it (drop the leading underscore), e.g.
       ``my_scenes/cell_count.py``. The underscore makes RenderProbe skip this file in
       its built-in package; your renamed copy is a real, loadable plugin.
    2. Edit the class below - rename it, set ``name``/``params_schema``, and fill in
       ``generate``.
    3. Check it BEFORE spending API calls:
           renderprobe validate my_scenes/cell_count.py
    4. Run it against a model without editing any RenderProbe source:
           renderprobe run my_exp.yaml --plugin-dir my_scenes

THE HONESTY INVARIANTS (this is what makes the diagnostic trustworthy - keep them)
    * TRUSTABLE GROUND TRUTH. A perfect perceiver must always be able to answer from
      the image alone. If your objects can overlap/occlude so the truth becomes
      unreadable, the "perception" verdict is meaningless. This template places shapes
      so they never overlap - keep that discipline for your own scenes.
    * REPRODUCIBLE. The SAME seed must produce the SAME scene (pixels and ground_truth).
      Seed every source of randomness from ``seed``; never call the global RNG.
    * DISCRIMINATING. The answer must actually vary - across seeds, or across a
      parameter you sweep. If the answer is always the same, a blind model wins by
      guessing and the diagnostic tells you nothing. (``validate`` checks this.)

This template's ground_truth exposes an ``objects`` list, so it pairs out of the box
with the probe template (``renderprobe new probe my_task``): scaffold both and they run
together with no edits. It also carries a ``count`` field it does not itself need, to
show that a scene may publish more than one probe reads - which is what lets a second
probe ask a different question of the same pixels.

Give your ground_truth whatever fields YOUR probe declares in requires_gt_fields.
"""
from __future__ import annotations

import random as _random

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene

_CANVAS = (480, 360)
_MARGIN = 24
_R_MIN, _R_MAX = 12, 30
# How much bigger the largest must be than the runner-up, in pixels. Small enough to
# stay a real question, large enough that the answer is not decided by anti-aliasing.
_R_MARGIN = 4
_PALETTE = {
    "red": (220, 40, 40), "blue": (40, 90, 210), "green": (30, 160, 70),
    "orange": (240, 150, 20), "purple": (150, 60, 200),
}


class _TemplateScene:
    # A unique registry name (lowercase, no spaces). Rename for your scene.
    name = "template_scene"

    # Adjustable knobs. Each entry drives a labeled slider in the UI and a default in
    # the CLI. Shape: {"type": "int"|"float", "min", "max", "default"}.
    params_schema = {
        "n_shapes": {"type": "int", "min": 2, "max": 20, "default": 6},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        # Seed ALL randomness from `seed` so the scene is reproducible.
        rng = _random.Random(seed)
        n = int(params["n_shapes"])

        # Radii vary so "the largest object" means something, and the largest is made
        # STRICTLY largest by a visible margin. A scene whose answer is a near-tie is
        # not a hard scene, it is an unfair one: the ground truth would be decided by a
        # pixel the model cannot be expected to resolve.
        for _attempt in range(100):
            radii = [rng.randint(_R_MIN, _R_MAX) for _ in range(n)]
            biggest = max(radii)
            if sorted(radii)[-2:] == [biggest, biggest]:
                continue                      # tie for largest: no single right answer
            if biggest - sorted(radii)[-2] >= _R_MARGIN:
                break
        else:
            raise RuntimeError(
                f"could not draw {n} shapes with a clearly largest one; widen "
                f"_R_MIN.._R_MAX or lower _R_MARGIN")

        img = Image.new("RGB", _CANVAS, "white")
        draw = ImageDraw.Draw(img)

        # Place non-overlapping shapes (trustable ground truth: nothing is occluded).
        placed: list[dict] = []
        for r in radii:
            for _try in range(200):  # bounded retry keeps generation deterministic & finite
                x = rng.randint(_MARGIN + r, _CANVAS[0] - _MARGIN - r)
                y = rng.randint(_MARGIN + r, _CANVAS[1] - _MARGIN - r)
                clear = all((x - o["x"]) ** 2 + (y - o["y"]) ** 2 >= (r + o["r"] + 8) ** 2
                            for o in placed)
                if clear:
                    placed.append({"x": x, "y": y, "r": r,
                                   "color": rng.choice(list(_PALETTE))})
                    break

        for o in placed:
            draw.ellipse([o["x"] - o["r"], o["y"] - o["r"],
                          o["x"] + o["r"], o["y"] + o["r"]],
                         fill=_PALETTE[o["color"]], outline=(0, 0, 0))

        # Placement can drop a shape, so the largest one actually DRAWN may not be the
        # largest one planned. Re-check rather than trusting the plan: ground truth is
        # derived from what is really in the image, never from what was intended.
        if len(placed) < 2:
            raise RuntimeError("fewer than two shapes fit; lower n_shapes or _R_MAX")
        drawn = sorted(o["r"] for o in placed)
        if len(drawn) > 1 and drawn[-1] == drawn[-2]:
            raise RuntimeError("the drawn shapes tie for largest; re-seed this scene")

        return Scene(
            id=f"{self.name}_n{n}_s{seed}",
            images=[img],
            # `objects` is what the probe template reads. `count` is published as well,
            # unused here, to show a scene may serve more than one probe.
            ground_truth={"objects": placed, "count": len(placed)},
            factors={"n_shapes": n},         # scene variables (enables sweep curves)
            meta={},
        )


# The registry discovers plugins via a module-level PLUGIN (single) or PLUGINS (list).
PLUGIN = _TemplateScene()
