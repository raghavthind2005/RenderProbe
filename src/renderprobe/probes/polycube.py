"""Polycube shape-matching probe: which candidate is the target, rotated?

The task is 3D mental rotation. `scenes/polycube.py` guarantees the question is well
posed - exactly one candidate matches, counting cannot settle it, and every cube of
every piece is visible - so this module only has to ask it, parse a color, and provide
the two extra conditions the decomposition needs:

* ORACLE. The pieces are stated as integer cell coordinates, which is a complete
  description of the shapes: nothing about this task resists being written down, so the
  text oracle loses nothing and a failure under it really is reasoning.
* PERCEPTION REPORT. How many unit cubes make up one named candidate. That is pure
  perception over the image, with no rotation and no comparison, and it is the exact
  operation MARBLE (arXiv:2506.22992) reports models failing at when it notes they
  struggle to convert pieces into arrays.
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene
from renderprobe.scenes._polycube_core import ROTATIONS, _apply, congruent

_COLOR_RE = re.compile(r"\b(red|blue|green|purple|cyan|pink|orange|brown|gray|grey)\b",
                       re.IGNORECASE)


def _cells(obj) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in c) for c in obj]


def _fmt(cells) -> str:
    return "[" + ", ".join(f"({x},{y},{z})" for x, y, z in sorted(cells)) + "]"


class _PolycubeProbe:
    name = "polycube"
    answer_type = "categorical"
    requires_gt_fields = ["question", "answer", "target_cells", "candidates"]

    def question(self, scene: Scene) -> str:
        return (scene.ground_truth["question"]
                + " Answer with the color name only.")

    def ground_truth(self, scene: Scene) -> str:
        return scene.ground_truth["answer"]

    def parse(self, raw: str) -> str | None:
        """Last color word wins: models often restate the options before answering.

        "grey" folds to "gray" so a spelling difference is never scored as a wrong
        answer. That spelling only ever names the target, so it parses to a real but
        always-incorrect prediction rather than to None, which keeps a model that
        misread the question distinguishable from one that failed to answer at all.
        """
        found = _COLOR_RE.findall(raw or "")
        if not found:
            return None
        answer = found[-1].lower()
        return "gray" if answer == "grey" else answer

    def score(self, pred, gt) -> tuple[bool, float, None]:
        ok = pred is not None and pred == gt
        return ok, 1.0 if ok else 0.0, None

    # -- oracle ------------------------------------------------------------
    def oracle_prompt(self, scene: Scene) -> str | None:
        return self.oracle_prompt_variants(scene)["coordinates"]

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        gt = scene.ground_truth
        target = _fmt(_cells(gt["target_cells"]))
        rows = [f"  {c['color']}: {_fmt(_cells(c['cells']))}" for c in gt["candidates"]]
        listing = "\n".join(rows)
        tail = ("Exactly one candidate is the same shape as the target under some "
                "rotation (no mirroring). Which color is it? Answer with the color "
                "name only.")
        return {
            "coordinates": (
                "Each piece is a set of unit cubes given as integer (x, y, z) grid "
                f"coordinates.\n\nTarget: {target}\n\nCandidates:\n{listing}\n\n{tail}"
            ),
            "prose": (
                "A target shape is built from unit cubes at these grid positions: "
                f"{target}. The candidate shapes are: "
                + "; ".join(f"{c['color']} at {_fmt(_cells(c['cells']))}"
                            for c in gt["candidates"])
                + ". " + tail
            ),
            "json": (
                '{"target": ' + str([list(c) for c in _cells(gt["target_cells"])])
                + ', "candidates": {'
                + ", ".join(f'"{c["color"]}": {[list(x) for x in _cells(c["cells"])]}'
                            for c in gt["candidates"])
                + "}}\n\n" + tail
            ),
        }

    def chance_level(self, scene: Scene) -> float:
        """Probability of being right by guessing: one candidate out of however many
        this scene shows. It varies with n_pieces, so a single number across a sweep
        would misstate every scene in it."""
        cands = scene.ground_truth.get("candidates") or []
        return 1.0 / len(cands) if cands else 0.0

    def oracle_solver(self, scene: Scene) -> str | None:
        """Re-derive the answer from ONLY what the oracle states.

        Reads the target and candidate coordinates and recomputes congruence; it never
        looks at the stored answer. If this agrees with ground truth on every seed, the
        oracle text provably contains enough to answer, so a model failing under the
        oracle failed at reasoning and not for want of information.
        """
        gt = scene.ground_truth
        target = _cells(gt["target_cells"])
        hits = [c["color"] for c in gt["candidates"]
                if congruent(_cells(c["cells"]), target)]
        return hits[0] if len(hits) == 1 else None

    # -- perception report -------------------------------------------------
    def perception_report(self, scene: Scene) -> list[ReportSpec]:
        """Two readings of perception on the same image, from tight to robust.

        `cells` is the primary one. It asks for exactly what the ORACLE is handed - the
        target's cubes as grid coordinates - and scores a reading as correct when that
        reading, given to a perfect reasoner, would pick the right candidate. So
        acc_report is not a proxy for perception: it is perception measured in the units
        of the task, and directly comparable to acc_full on the same scenes. Near zero
        indicts encoding outright; high while acc_full is low means the model saw enough
        and still failed, which is grounding.

        `cubes` is the robust floor, kept because the tight question has a confound: a
        model may see a shape perfectly and fail to write it down as a list of triples.
        Counting needs almost no output format, so `cubes` right while `cells` is wrong
        localizes the failure to serialization rather than sight - the distinction
        MARBLE (arXiv:2506.22992) notes but does not separate when it reports models
        struggling to convert pieces into arrays.
        """
        gt = scene.ground_truth
        cands = gt.get("candidates") or []
        if not cands:
            return []
        target = _cells(gt["target_cells"])

        def _sufficient(pred, _gt) -> tuple[bool, float, None]:
            """Would this reading of the target, reasoned over perfectly, answer right?

            The candidates are taken as ground truth, so only the model's reading of the
            TARGET is under test. Congruence, not equality: the model picks its own
            origin, so any rigid re-placement of the right shape passes. The prompt fixes
            the handedness, because a left-handed frame mirrors the shape, and mirroring
            is exactly what this task refuses to call a match.
            """
            if not pred:
                return False, 0.0, None
            hits = [c["color"] for c in cands if congruent(_cells(c["cells"]), pred)]
            ok = len(hits) == 1 and hits[0] == gt["answer"]
            # `correct` is all-or-nothing because the task is, so on its own it cannot
            # tell a percept that missed by one cube from one that was noise. `score`
            # carries that: the best overlap achievable between the reported shape and
            # the true one. At the sample sizes a hand-checked run affords, the
            # difference between 0.8 and 0.1 is most of what there is to learn.
            return ok, _best_overlap(pred, target), None

        pick = cands[scene.meta.get("seed", 0) % len(cands)]
        return [
            ReportSpec(
                name="cells",
                primary=True,
                question=(
                    "The gray piece standing on its own is the TARGET. Describe it as "
                    "unit cubes on an integer grid: put one cube at (0, 0, 0) and give "
                    "every other cube of the target as (x, y, z), where x increases to "
                    "the right, y increases upward, and z increases toward the viewer. "
                    f"The target has {len(target)} cubes. List exactly that many "
                    "coordinate triples and nothing else."
                ),
                ground_truth=[list(c) for c in target],
                parse=_parse_cells,
                score=_sufficient,
            ),
            ReportSpec(
                name="cubes",
                primary=False,
                question=(f"How many unit cubes make up the {pick['color']} piece? "
                          "Answer with a single number."),
                ground_truth=int(pick["n_cubes"]),
                parse=_parse_int,
                score=_score_int,
            ),
        ]


def _best_overlap(pred, target) -> float:
    """Best intersection over union between two cell sets over rigid placements.

    Every proper rotation, and for each one every translation that brings some reported
    cube onto some true cube (any placement that overlaps at all has one of those, and a
    placement that overlaps nowhere scores zero regardless). Rotations are searched
    because the reporter chooses its own axes; reflections are not, since a mirrored
    reading is a different shape in this task and must not be credited as a near miss.
    """
    tgt = set(target)
    best = 0.0
    for rot in ROTATIONS:
        turned = [_apply(rot, c) for c in pred]
        for ax, ay, az in turned:
            for bx, by, bz in tgt:
                dx, dy, dz = bx - ax, by - ay, bz - az
                moved = {(x + dx, y + dy, z + dz) for x, y, z in turned}
                hit = len(moved & tgt)
                best = max(best, hit / len(moved | tgt))
    return round(best, 4)


_TRIPLE_RE = re.compile(
    r"[(\[]\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[)\]]")
# A polycube here has at most 27 cubes (a 3x3x3 cube); anything longer is a model
# looping rather than describing, and parsing it would invent a percept it never had.
_MAX_CELLS = 27


def _parse_cells(raw: str):
    """Integer (x, y, z) triples out of free text, as a set of cells.

    Bracketed triples first, since that is the requested format and it survives any
    surrounding prose. Failing that, whitespace-separated integers are taken three at a
    time: several models answer with a bare grid of numbers, and reading it is fairer
    than scoring them as having produced no percept at all.
    """
    text = raw or ""
    found = [(int(a), int(b), int(c)) for a, b, c in _TRIPLE_RE.findall(text)]
    if not found:
        nums = [int(n) for n in re.findall(r"-?\d+", text)]
        if len(nums) < 3:
            return None
        found = [tuple(nums[i:i + 3]) for i in range(0, len(nums) - 2, 3)]
    cells = sorted(set(found))
    if not cells or len(cells) > _MAX_CELLS:
        return None
    return tuple(cells)


def _parse_int(raw: str):
    m = re.findall(r"-?\d+", raw or "")
    return int(m[-1]) if m else None


def _score_int(pred, gt) -> tuple[bool, float, None]:
    ok = pred is not None and int(pred) == int(gt)
    return ok, 1.0 if ok else 0.0, None


PLUGIN = _PolycubeProbe()
