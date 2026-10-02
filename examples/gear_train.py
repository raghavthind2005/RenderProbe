"""Gear train: a worked example of a third-party scene.

This file lives OUTSIDE the renderprobe package and is loaded by path:

    renderprobe validate examples/gear_train.py
    renderprobe run my_config.yaml --plugin-dir examples

It is one file holding both a scene and a probe, because an out-of-tree module is
imported by path and cannot import its neighbors. `PLUGINS` exports both; the registry
sorts them by which protocol each satisfies.

THE TASK. A row of shafts carrying gears, each labeled with its tooth count. Where two
gears sit on the SAME shaft they are bolted together and turn as one. The first shaft is
turned a given number of times and the question asks how many turns the last shaft makes.
The answer is a whole number, so guessing is weak.

WHY IT IS NOT TRIVIAL. In a plain row of meshed gears every gear between the two ends is
an idler: the ratio telescopes and only the first and last tooth counts matter. Put two
gears on one shaft and that stops being true - each stage contributes its own ratio and
they multiply. A model that reaches for the familiar first-over-last shortcut gets a
specific wrong number, which is stored as `simple_train_answer` so a run can tell that
mistake apart from noise. This scene started out as the plain train, scored 6/6 against
the calibration gate, and had to be made harder; docs/on_ramp.md tells that story.

THE THREE CONDITIONS. `question` is the task. `oracle_prompt` restates the gears and
their tooth counts as text, and `oracle_solver` recomputes the answer from only that, so
`renderprobe validate` can prove the oracle is information-complete. `perception_report`
asks for one gear's tooth count, read straight off the picture with no arithmetic.
"""
from __future__ import annotations

import math
import re
from random import Random

from PIL import Image, ImageDraw, ImageFont

from renderprobe.core.schema import ReportSpec, Scene

# Tooth counts are drawn from this set so that first/last ratios land on whole numbers
# for at least one sensible turn count. Divisibility is what keeps the answer exact, and
# an exact answer is what lets `score` be a plain equality instead of a tolerance.
_TEETH = (8, 10, 12, 15, 16, 20, 24, 30, 40)
_LABELS = "ABCDEFGH"
_CANVAS = (860, 340)
_PX_PER_TOOTH = 3.1          # pitch radius in pixels per tooth
_MARGIN = 40
_TRIES = 300

_PAPER = (244, 242, 237)
_INK = (36, 40, 46)
_GEAR_FILL = (214, 218, 223)
_GEAR_EDGE = (120, 127, 136)
_DRIVER = (196, 62, 44)
_ASKED = (32, 96, 132)

_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _font(size: int):
    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def turns_of(shafts: list[tuple[int, int | None]], driver_turns: int) -> float:
    """Turns made by the last shaft when the first is turned `driver_turns` times.

    A shaft is (input gear teeth, output gear teeth); the output is None on a shaft
    carrying a single gear. Each stage is a mesh between one shaft's output and the
    next shaft's input, and contributes driving/driven. The stages MULTIPLY, which is
    what the second gear on a shaft buys and what a plain train does not have.

    Written once and shared by the ground truth, the oracle solver and the tests, so
    there is no second implementation to drift.
    """
    turns = float(driver_turns)
    for i in range(len(shafts) - 1):
        driving = shafts[i][1] or shafts[i][0]
        driven = shafts[i + 1][0]
        turns = turns * driving / driven
    return turns


def simple_train_turns(shafts: list[tuple[int, int | None]], driver_turns: int) -> float:
    """What the first-over-last shortcut gives: right for a plain train, wrong here.

    Kept so a run can name this mistake instead of counting it as noise.
    """
    return driver_turns * shafts[0][0] / shafts[-1][0]


def _radius(t: int) -> float:
    return t * _PX_PER_TOOTH


def _draw(shafts, centers, driver: int, asked: int) -> Image.Image:
    img = Image.new("RGB", _CANVAS, _PAPER)
    d = ImageDraw.Draw(img)
    cy = _CANVAS[1] / 2
    f = _font(19)

    for i, ((t_in, t_out), cx) in enumerate(zip(shafts, centers)):
        edge = _DRIVER if i == driver else _ASKED if i == asked else _GEAR_EDGE
        width = 4 if i in (driver, asked) else 2
        # Largest first, so a smaller gear on the same shaft stays visible on top of it.
        for teeth in sorted([t for t in (t_in, t_out) if t], reverse=True):
            r = _radius(teeth)
            for k in range(teeth):
                a = 2 * math.pi * k / teeth
                d.line([(cx + math.cos(a) * r, cy + math.sin(a) * r),
                        (cx + math.cos(a) * (r + 7), cy + math.sin(a) * (r + 7))],
                       fill=edge, width=3)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_GEAR_FILL, outline=edge,
                      width=width)
        # The shaft itself, drawn through both gears so "same shaft" is visible and not
        # only implied by the two circles sharing a center.
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=_PAPER, outline=edge, width=3)

        label = (f"{_LABELS[i]}: {t_in}" if t_out is None
                 else f"{_LABELS[i]}: in {t_in} / out {t_out}")
        tb = d.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = cy + _radius(max(x for x in (t_in, t_out) if x)) + 18
        d.rectangle((cx - tw / 2 - 7, ty - 4, cx + tw / 2 + 7, ty + th + 8),
                    fill=(255, 254, 251), outline=edge, width=1)
        d.text((cx - tw / 2 - tb[0], ty - tb[1] + 2), label, fill=_INK, font=f)
    return img


class _GearTrainScene:
    name = "gear_train"
    params_schema = {
        # Meshes between shafts. Each one contributes a ratio and they multiply, so this
        # is the reasoning knob: one stage is a single division, three is a chain the
        # first-over-last shortcut gets wrong.
        "n_stages": {"type": "int", "min": 1, "max": 3, "default": 2},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        stages = int(params.get("n_stages", 2))
        if not 1 <= stages <= 3:
            raise ValueError(f"n_stages must be 1..3 (got {stages}); past three the "
                             "train runs off the canvas and the tooth labels collide")
        rng = Random(seed * 2654435761 + stages)
        for _ in range(_TRIES):
            made = self._attempt(stages, rng)
            if made is not None:
                return self._scene(*made, stages, params, seed)
        raise ValueError(f"no whole-number gear train with {stages} stage(s) after "
                         f"{_TRIES} tries")

    def _attempt(self, stages: int, rng: Random):  # noqa: PLR0911
        n = stages + 1
        shafts: list[tuple[int, int | None]] = []
        for i in range(n):
            t_in = rng.choice(_TEETH)
            # The two end shafts carry one gear each: the first is simply driven, the
            # last simply reports. Every shaft between them carries a second gear bolted
            # to the first, and that is what stops the ratios telescoping.
            t_out = None if i in (0, n - 1) else rng.choice(_TEETH)
            if t_out == t_in:
                return None          # a 1:1 pair on one shaft is a drawing with no effect
            shafts.append((t_in, t_out))

        span = sum(_radius(max(x for x in s if x)) for s in shafts) * 2
        if span + 2 * _MARGIN > _CANVAS[0]:
            return None
        if max(_radius(max(x for x in s if x)) for s in shafts) * 2 + 80 > _CANVAS[1]:
            return None

        for driver_turns in rng.sample(range(2, 15), 13):
            value = turns_of(shafts, driver_turns)
            if value != int(value) or not 2 <= value <= 90:
                continue
            naive = simple_train_turns(shafts, driver_turns)
            # If the shortcut happens to land on the same number, the mistake is
            # invisible and the scene is not testing what it claims to. A single-stage
            # train is the exception and not a failure: with no compound shaft the
            # shortcut IS the answer, which is the whole reason the scene does not stop
            # there. It is kept as the trivial tier so the knob can be seen moving.
            if naive == value and stages > 1:
                continue
            return shafts, driver_turns, int(value), naive
        return None

    def _scene(self, shafts, driver_turns, answer, naive, stages, params, seed) -> Scene:
        n = len(shafts)
        centers, x = [], _radius(max(v for v in shafts[0] if v))
        for i in range(n):
            if i:
                x += _radius(shafts[i - 1][1] or shafts[i - 1][0]) + _radius(shafts[i][0])
            centers.append(x)
        width = centers[-1] + _radius(max(v for v in shafts[-1] if v))
        shift = (_CANVAS[0] - width) / 2
        centers = [c + shift for c in centers]
        img = _draw(shafts, centers, 0, n - 1)

        parts = []
        for i, (t_in, t_out) in enumerate(shafts):
            if t_out:
                parts.append(f"shaft {_LABELS[i]} carries a {t_in}-tooth gear driven by "
                             f"the shaft before it, bolted to a {t_out}-tooth gear that "
                             "drives the shaft after it")
            else:
                parts.append(f"shaft {_LABELS[i]} carries a single {t_in}-tooth gear")
        gears_text = "; ".join(parts)
        question = (
            "The picture shows gears on a row of shafts, meshing left to right. Two "
            "gears drawn on the same shaft are bolted together and turn as one. A label "
            "reading \"in X / out Y\" means that shaft is driven through its X-tooth "
            "gear and drives the next shaft through its Y-tooth gear.\n\n"
            f"Shaft {_LABELS[0]} is turned {driver_turns} full times. How many full "
            f"turns does shaft {_LABELS[n - 1]} make? Answer with a single number."
        )
        plausible = sorted({
            int(driver_turns * a / b)
            for a in _TEETH for b in _TEETH if driver_turns * a % b == 0
            and 2 <= driver_turns * a / b <= 90
        })
        return Scene(
            id=f"gear_s{stages}_seed{seed}",
            images=[img],
            ground_truth={
                "question": question,
                "answer": answer,
                "gears_text": gears_text,
                "shafts": [list(s) for s in shafts],
                "driver_turns": driver_turns,
                # The shortcut's answer: right for a plain train, wrong here. Never
                # equal to the real answer, by construction.
                "simple_train_answer": (int(naive) if naive == int(naive) else None),
                "plausible": plausible,
            },
            factors={"n_stages": stages},
            meta={"generator": "gear_train", "seed": seed, "params": dict(params),
                  "tier": "trivial" if stages == 1 else "standard"},
        )


def _last_int(raw: str):
    """Last integer wins: a model that shows its working ends on the total."""
    found = re.findall(r"-?\d+", raw or "")
    return int(found[-1]) if found else None


class _GearTrainProbe:
    name = "gear_train"
    answer_type = "numeric"
    requires_gt_fields = ["question", "answer", "gears_text", "shafts", "driver_turns"]

    def question(self, scene: Scene) -> str:
        return scene.ground_truth["question"]

    def ground_truth(self, scene: Scene) -> int:
        return int(scene.ground_truth["answer"])

    def parse(self, raw: str):
        return _last_int(raw)

    def score(self, pred, gt) -> tuple[bool, float, float | None]:
        if pred is None:
            return False, 0.0, None
        ok = int(pred) == int(gt)
        return ok, 1.0 if ok else 0.0, float(abs(int(pred) - int(gt)))

    def chance_level(self, scene: Scene) -> float:
        """One over the whole-number turn counts any single pair of these gears could
        give. The guesser this assumes has read every label and merely picked one ratio
        instead of chaining them, which is generous to the model and so the honest
        bound to gate on."""
        options = scene.ground_truth.get("plausible") or []
        return 1.0 / len(options) if options else 0.0

    def oracle_prompt(self, scene: Scene) -> str | None:
        gt = scene.ground_truth
        n = len(gt["shafts"])
        return (
            f"Gears sit on a row of shafts: {gt['gears_text']}. Each shaft's gear meshes "
            "with the next shaft's gear, in that order. Two gears on one shaft are "
            f"bolted together and turn as one.\n\nShaft {_LABELS[0]} is turned "
            f"{gt['driver_turns']} full times. How many full turns does shaft "
            f"{_LABELS[n - 1]} make? Answer with a single number."
        )

    def oracle_solver(self, scene: Scene) -> int | None:
        """Re-derive the answer from ONLY what the oracle states, never the stored
        answer. Agreement across seeds is what proves the oracle text is complete."""
        gt = scene.ground_truth
        shafts = [(int(a), int(b) if b is not None else None) for a, b in gt["shafts"]]
        value = turns_of(shafts, int(gt["driver_turns"]))
        return int(value) if value == int(value) else None

    def perception_report(self, scene: Scene) -> ReportSpec:
        """Read one gear's tooth count off the picture. No ratio, no chaining.

        It is the driving gear of the first shaft, which the answer depends on directly,
        so this is the primitive the task consumes rather than one beside it.
        """
        gt = scene.ground_truth
        return ReportSpec(
            name="tooth_count",
            question=(f"In the picture, shaft {_LABELS[0]} carries the gear drawn in "
                      "red. How many teeth does its LARGEST gear have? Answer with a "
                      "single number."),
            ground_truth=max(int(x) for x in gt["shafts"][0] if x is not None),
            parse=_last_int,
            score=lambda p, g: (p is not None and int(p) == int(g),
                                1.0 if (p is not None and int(p) == int(g)) else 0.0,
                                None if p is None else float(abs(int(p) - int(g)))),
        )


PLUGINS = [_GearTrainScene(), _GearTrainProbe()]
