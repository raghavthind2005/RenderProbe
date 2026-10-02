"""Identity-under-occlusion probe: 'What color is the object behind the occluder?'

Like counting, this is a perception-bound task: the ground truth (the target's
color) IS the answer, so the oracle condition - which states the target's
attributes as text - necessarily reveals it. Recovery therefore measures the FULL
perception gap: how much accuracy a model recovers once it no longer has to see
through the occluder. That is the honest decomposition for a perception-bound probe
(contrast spatial_relation, whose oracle withholds the left/right verdict).

Consumes the tabletop_occlusion ground_truth shape.
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene

_KNOWN_COLORS = {"red", "blue", "green", "orange", "purple"}


def _parse_shape(raw: str) -> str:
    text = raw.strip().lower()
    if re.search(r"\bsquare\b", text) or "rectang" in text:
        return "square"
    if re.search(r"\bcircle\b", text) or "round" in text:
        return "circle"
    raise ValueError(f"No known shape in: {raw!r}")


def _score_shape(pred: str, gt: str) -> tuple[bool, float, None]:
    correct = pred == gt
    return correct, 1.0 if correct else 0.0, None


class _IdentityUnderOcclusionProbe:
    name = "identity_under_occlusion"
    answer_type = "categorical"
    requires_gt_fields = ["target_color", "target_shape", "occluder_frac"]
    # Perception-bound: the target color IS the answer, so the oracle states it directly
    # and oracle-completeness is definitional (recovery measures the full perception gap).
    oracle_states_answer = True

    def question(self, scene: Scene) -> str:
        return (
            "What color is the partially hidden object in the image? "
            "Answer with a single color word."
        )

    def ground_truth(self, scene: Scene) -> str:
        return scene.ground_truth["target_color"]

    def parse(self, raw: str) -> str:
        """Extract a known color name from the model's response."""
        text = raw.strip().lower()
        for color in _KNOWN_COLORS:
            if re.search(rf"\b{color}\b", text):
                return color
        raise ValueError(f"No known color found in: {raw!r}")

    def score(self, pred: str, gt: str) -> tuple[bool, float, None]:
        correct = pred == gt
        return correct, 1.0 if correct else 0.0, None

    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Perception report for a perception-bound task: name the object's SHAPE, not its
        color. Form survives partial occlusion better than the color region, so this
        separates 'saw the object at all' from 'couldn't read the occluded color' -
        report high + full low => the failure is specifically the occluded color, not
        object detection; report low => it can't resolve the object.

        The scene has a SINGLE centered object, so no pointer is needed (a drawn ring
        would be a circle and bias a shape question) - the report runs over the plain
        image.

        Caveat: at extreme occluder_frac the shape is unreadable even to a
        perfect perceiver, so a low report there reflects real impossibility, not model
        weakness - the report and the task degrade together (both perception-bound)."""
        if not scene.images:
            return None
        return ReportSpec(
            question=("What is the shape of the partially hidden object - a circle or a "
                      "square? Answer with a single word."),
            ground_truth=scene.ground_truth["target_shape"],
            parse=_parse_shape,
            score=_score_shape,
            images=None,   # single object; no localization handle required
        )

    canonical_oracle_variant = "description"

    def oracle_prompt(self, scene: Scene) -> str | None:
        variants = self.oracle_prompt_variants(scene)
        return variants.get(self.canonical_oracle_variant) if variants else None

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        """≥3 verbalizations of the target's attributes, canonical first.

        The target's color is the answer, so every variant states it; recovery then
        reflects the whole perception gap (see module docstring).
        """
        gt = scene.ground_truth
        color = gt["target_color"]
        shape = gt["target_shape"]
        frac = float(gt["occluder_frac"])
        pct = round(frac * 100)
        q = "What color is the partially hidden object? Answer with a single color word."
        return {
            "description": (
                f"A {shape} of color {color} sits on the table, partially hidden "
                f"behind a gray occluder (~{pct}% covered). {q}"
            ),
            "attributes": (
                f"partially_hidden_object: shape={shape}, color={color}, "
                f"occluded_fraction={frac:.2f}. {q}"
            ),
            "json": (
                f'{{"partially_hidden_object": {{"shape": "{shape}", '
                f'"color": "{color}", "occluded_fraction": {frac:.2f}}}}}. {q}'
            ),
        }


PLUGIN = _IdentityUnderOcclusionProbe()
