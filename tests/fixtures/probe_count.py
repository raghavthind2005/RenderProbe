"""Counting probe: asks 'How many dots?' and expects an integer answer.

Counting is a perception-bound task: its ground-truth primitive (the count) IS the
answer, so the oracle condition - which injects that primitive as text - necessarily
hands over the answer. That is not a defect; it is the honest decomposition. Recovery
for this probe measures the FULL perception gap (how much accuracy a model gains once
it no longer has to count from pixels), i.e. it establishes that counting here is
perception-bound rather than reasoning-bound. Compare with spatial_relation, whose
oracle withholds the answer (it gives coordinates, not the left/right verdict).
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene
from tests.fixtures._highlight import ring

_DOT_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]


def _parse_dot_color(raw: str) -> str:
    text = raw.strip().lower()
    if re.search(r"\bgr[ae]y\b", text):
        return "gray"
    for c in _DOT_COLORS:
        if re.search(rf"\b{c}\b", text):
            return c
    raise ValueError(f"No known dot color in: {raw!r}")


def _score_dot_color(pred: str, gt: str) -> tuple[bool, float, None]:
    correct = pred == gt
    return correct, 1.0 if correct else 0.0, None


class _CountProbe:
    name = "count"
    answer_type = "numeric"
    requires_gt_fields = ["count"]
    # Perception-bound: the count IS the answer, so the oracle states it directly and
    # oracle-completeness is definitional (no independent solver - recovery measures the
    # FULL perception gap). See the module docstring and docs/methodology.md.
    oracle_states_answer = True

    def question(self, scene: Scene) -> str:
        return (
            "How many dots are in the image? "
            "Reply with a single integer and nothing else."
        )

    def ground_truth(self, scene: Scene) -> int:
        return int(scene.ground_truth["count"])

    def parse(self, raw: str) -> int:
        """Extract first integer from the model's raw reply."""
        match = re.search(r"\d+", raw.strip())
        if match is None:
            raise ValueError(f"No integer found in reply: {raw!r}")
        return int(match.group())

    def score(self, pred: int, gt: int) -> tuple[bool, float, float]:
        correct = pred == gt
        # normalized score: 1 at exact, drops linearly, floored at 0
        score = max(0.0, 1.0 - abs(pred - gt) / max(gt, 1))
        error = float(abs(pred - gt))
        return correct, score, error

    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Perception report for counting: name the color of ONE ringed dot. Counting
        is perception-bound, but the report still splits the failure - if the model can
        resolve and name an individual element (report high) yet miscounts (full low),
        the bottleneck is ENUMERATION/aggregation, not element encoding; if it can't even
        read one ringed dot, the failure is basic encoding. Needs the additive per-dot
        ground_truth (color + position); returns None for older count scenes without it."""
        dots = scene.ground_truth.get("dots")
        if not scene.images or not dots:
            return None
        target = dots[0]
        r = int(scene.ground_truth.get("dot_radius", 12))
        return ReportSpec(
            question=("One dot is circled with a bright magenta (pink) ring. What color "
                      "is the dot inside that ring? Answer with a single color word."),
            ground_truth=target["color"],
            parse=_parse_dot_color,
            score=_score_dot_color,
            # pad=2 keeps the ring's outer stroke strictly inside the min inter-dot gap
            # (min_dist = 2r+4) even at the tightest packing, so it never clips a neighbor.
            images=[ring(scene.images[0], int(target["x"]), int(target["y"]), r, pad=2)],
        )

    canonical_oracle_variant = "statement"

    def oracle_prompt(self, scene: Scene) -> str | None:
        variants = self.oracle_prompt_variants(scene)
        return variants.get(self.canonical_oracle_variant) if variants else None

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        """≥3 verbalizations of the count primitive, canonical first.

        The count IS the answer for this probe, so every variant states it; recovery
        then reflects the whole perception gap (see module docstring).
        """
        n = int(scene.ground_truth["count"])
        q = "How many dots are in the image? Reply with a single integer and nothing else."
        return {
            "statement": f"The image contains exactly {n} dots. {q}",
            "count_field": f"dot_count = {n}. {q}",
            "json": f'{{"dot_count": {n}}}. {q}',
        }


PLUGIN = _CountProbe()
