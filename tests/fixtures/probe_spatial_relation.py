"""Spatial-relation probe: 'Is the {a} object left or right of the {b} object?'

This probe is the oracle-path carrier. The full condition
requires the model to BOTH perceive object positions AND reason about the
left/right relation. The oracle condition hands it the positions as text, so only
the reasoning step is tested. If accuracy recovers under oracle, perception was
the bottleneck; if it stays low, the failure is reasoning.

Consumes the spatial_config ground_truth shape (see scenes/spatial_config.py).
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene
from tests.fixtures._highlight import ring

_CENTER_X = 240   # half of the spatial_config 480px canvas
_RING_R = 24      # spatial_config object radius


class _SpatialRelationProbe:
    name = "spatial_relation"
    answer_type = "categorical"
    requires_gt_fields = ["objects", "target_a", "target_b", "relation"]

    def question(self, scene: Scene) -> str:
        a = scene.ground_truth["target_a"]
        b = scene.ground_truth["target_b"]
        return (
            f"Is the {a} object to the LEFT or to the RIGHT of the {b} object? "
            f"Answer with a single word: 'left' or 'right'."
        )

    def ground_truth(self, scene: Scene) -> str:
        return scene.ground_truth["relation"]

    def parse(self, raw: str) -> str:
        """Extract 'left' or 'right' case-insensitively; raise on no match -> parse_fail."""
        text = raw.strip().lower()
        if re.search(r"\bleft\b", text):
            return "left"
        if re.search(r"\bright\b", text):
            return "right"
        raise ValueError(f"Cannot parse 'left'/'right' from: {raw!r}")

    def score(self, pred: str, gt: str) -> tuple[bool, float, None]:
        correct = pred == gt
        return correct, 1.0 if correct else 0.0, None

    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Perception report: name which HALF (left/right) the ringed target-A object is
        in - an ABSOLUTE position read, no object-to-object comparison. Splits the two
        skills the full task fuses: reading a position (encoding) vs comparing two
        positions (grounding/reasoning). report high + full low => it perceives positions
        but can't compare them; report low => it can't localize the target at all.

        Known limit: for a target within a few pixels of the center line the half is
        ambiguous to a perfect perceiver too (~1-2% of scenes). The GT stays
        exact; this only adds a small noise floor, so it's a first-order discriminator,
        not a perfect perceptual audit. We can't use 'color of the ringed object' here
        (the task question already names target-A's color, which would leak the answer)."""
        a_color = scene.ground_truth["target_a"]
        a = next((o for o in scene.ground_truth["objects"] if o["color"] == a_color), None)
        if not scene.images or a is None:
            return None
        half = "left" if int(a["x"]) < _CENTER_X else "right"
        return ReportSpec(
            question=("One object is circled with a bright magenta ring. Is the circled "
                      "object in the LEFT or the RIGHT half of the image? "
                      "Answer with a single word: 'left' or 'right'."),
            ground_truth=half,
            parse=self.parse,
            score=self.score,
            images=[ring(scene.images[0], int(a["x"]), int(a["y"]), _RING_R)],
        )

    def oracle_solver(self, scene: Scene) -> str | None:
        """Oracle completeness certificate: re-derive left/right from the two objects'
        x-positions (the facts the oracle states) - NEVER read ground_truth['relation'].
        Matching GT across seeds proves the coordinates suffice, so a model's oracle
        failure here is reasoning, not a lossy verbalization."""
        objs = scene.ground_truth["objects"]
        a = next((o for o in objs if o["color"] == scene.ground_truth["target_a"]), None)
        b = next((o for o in objs if o["color"] == scene.ground_truth["target_b"]), None)
        if a is None or b is None:
            return None
        return "left" if int(a["x"]) < int(b["x"]) else "right"

    # Which of `oracle_prompt_variants` is the canonical verbalization - the one
    # `oracle_prompt` returns and the one whose numbers surface at the top level of
    # the oracle report. The runner emits this variant first.
    canonical_oracle_variant = "relational"

    def oracle_prompt(self, scene: Scene) -> str | None:
        """Canonical oracle verbalization (the "relational" variant).

        States BOTH targets' pixel positions as text and asks the SAME relation
        question. Does NOT reveal the answer - only the perceptual primitives
        (coordinates); the model must still reason about which x is smaller.
        """
        variants = self.oracle_prompt_variants(scene)
        return variants.get(self.canonical_oracle_variant) if variants else None

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        """≥3 semantically-equivalent verbalizations of the same ground truth.

        Robustness across these is a first-class output: if recovery depends on
        which template we pick, the oracle is measuring verbalization quality, not
        perception. All variants inject the SAME (x, y) coordinates - never the
        answer - and place each object's x-coordinate first so any competent
        reasoner can parse them. Returns {} if the targets can't be located.

        The dict is ordered canonical-first; the runner relies on that ordering.
        """
        a_color = scene.ground_truth["target_a"]
        b_color = scene.ground_truth["target_b"]
        objects = scene.ground_truth["objects"]

        a = next((o for o in objects if o["color"] == a_color), None)
        b = next((o for o in objects if o["color"] == b_color), None)
        if a is None or b is None:
            return {}

        question = (
            f"Is the {a_color} object to the LEFT or to the RIGHT of the "
            f"{b_color} object? Answer with a single word: 'left' or 'right'."
        )
        return {
            # 1. Relational prose (canonical).
            "relational": (
                f"The {a_color} object is at pixel position ({a['x']}, {a['y']}) "
                f"and the {b_color} object is at pixel position ({b['x']}, {b['y']}). "
                f"{question}"
            ),
            # 2. Coordinate list.
            "coordinates": (
                f"Object pixel positions (x, y):\n"
                f"  {a_color} = ({a['x']}, {a['y']})\n"
                f"  {b_color} = ({b['x']}, {b['y']})\n"
                f"{question}"
            ),
            # 3. Structured JSON-as-text.
            "json": (
                f'Scene objects as JSON: {{"{a_color}": {{"x": {a["x"]}, '
                f'"y": {a["y"]}}}, "{b_color}": {{"x": {b["x"]}, "y": {b["y"]}}}}}. '
                f"{question}"
            ),
        }


PLUGIN = _SpatialRelationProbe()
