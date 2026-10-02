"""Relational-chain probe - multi-hop compositional referential question.

Reads the verified question/answer from the relational scene's ground_truth.
The oracle verbalization injects the full scene graph as text so the model
resolves the chain in text space (perception load removed; reasoning load kept).
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from renderprobe.core.schema import ReportSpec, Scene
from tests.fixtures._relational_core import COLORS, Obj, parse_question, resolve

# Magenta highlight ring for the perception-report target - absent from the object
# palette (so it can't be mistaken for an object color) and high-contrast on every
# fill. The ring hugs the object edge (radius + 2, width 3): its outer stroke stays
# strictly inside the smallest inter-object gap even at the tightest clutter level
# (min_gap=4 => outer edge radius+3.5 < gap), so it always encircles exactly one object.
_HALO_RGB = (230, 0, 230)


def _highlight_target(img: Image.Image, x: int, y: int, radius: int) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    ring = radius + 2
    draw.ellipse([x - ring, y - ring, x + ring, y + ring], outline=_HALO_RGB, width=3)
    return out


class _RelationalProbe:
    name = "relational"
    answer_type = "categorical"
    requires_gt_fields = ["question", "answer", "scene_graph_text", "objects"]

    def question(self, scene: Scene) -> str:
        return scene.ground_truth["question"]

    def ground_truth(self, scene: Scene) -> str:
        return scene.ground_truth["answer"]

    def parse(self, raw: str) -> str:
        """First known color word in response; raise on no match -> parse_fail."""
        text = raw.strip().lower().rstrip(".")
        for color in COLORS:
            if color in text:
                return color
        raise ValueError(f"Cannot parse color from: {raw!r}")

    def score(self, pred: str, gt: str) -> tuple[bool, float, None]:
        correct = pred == gt
        return correct, 1.0 if correct else 0.0, None

    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Perception-only sub-task (docs/methodology.md §3): name the color of the
        chain's TARGET object - but identify it by a magenta highlight ring drawn on a
        copy of the image, NOT by pixel coordinates (VLMs localize poorly by
        coordinate, which would confound acc_report). No chain-tracing needed. High
        accuracy here with a low full-condition accuracy indicts grounding/reasoning,
        not encoding; low accuracy here indicts encoding. GT is the target color
        (== the task answer), scored with this probe's own color parse/score.

        Certifies perception of the answer-bearing object only, not every intermediate
        object the chain traverses - a first-order discriminator (see the note in §3.3).
        """
        if not scene.images:
            return None
        objs = [
            Obj(color=o["color"], shape=o["shape"], size=o["size"], x=o["x"], y=o["y"])
            for o in scene.ground_truth.get("objects", [])
        ]
        chain = parse_question(scene.ground_truth.get("question", ""))
        if chain is None:
            return None
        target = resolve(objs, chain)
        if target is None:
            return None
        # Use the target's RENDERED radius (stored by the scene; depends on size AND
        # clutter) so the ring encircles exactly it, tight enough to never touch a
        # neighbor even at the highest clutter level.
        radius = next(
            (o.get("r", 16) for o in scene.ground_truth.get("objects", [])
             if o["x"] == target.x and o["y"] == target.y),
            16,
        )
        question = (
            "One object in the image is circled with a bright magenta (pink) ring. "
            "What color is the object inside that ring? Answer with a single color word."
        )
        return ReportSpec(
            question=question,
            ground_truth=target.color,
            parse=self.parse,
            score=self.score,
            images=[_highlight_target(scene.images[0], target.x, target.y, radius)],
        )

    def oracle_solver(self, scene: Scene) -> str | None:
        """Oracle information-completeness certificate: re-resolve the referential chain
        over the OBJECTS the oracle verbalizes - NEVER read ground_truth['answer']. When
        this matches the stored answer across seeds (validate/tests check it), the oracle
        text provably conveys enough to solve the task, so an oracle failure is genuine
        REASONING, not missing information. See docs/methodology.md."""
        objs = [
            Obj(color=o["color"], shape=o["shape"], size=o["size"], x=o["x"], y=o["y"])
            for o in scene.ground_truth.get("objects", [])
        ]
        chain = parse_question(scene.ground_truth.get("question", ""))
        if chain is None:
            return None
        target = resolve(objs, chain)
        return target.color if target is not None else None

    canonical_oracle_variant = "scene_graph"

    def oracle_prompt(self, scene: Scene) -> str | None:
        variants = self.oracle_prompt_variants(scene)
        return variants.get(self.canonical_oracle_variant) if variants else None

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        graph_text = scene.ground_truth.get("scene_graph_text", "")
        question   = scene.ground_truth.get("question", "")
        objects    = scene.ground_truth.get("objects", [])
        if not graph_text or not question:
            return {}

        obj_lines = "\n".join(
            f"  {o['size']} {o['color']} {o['shape']} at ({o['x']}, {o['y']})"
            for o in objects
        )
        # Include size: chains may use a size qualifier (~1 in 5), so a size-free
        # verbalization would be an INCOMPLETE view of the ground truth and would
        # depress recovery for reasons of verbalization, not reasoning.
        coord_lines = "\n".join(
            f"  {o['size']} {o['color']} {o['shape']}: ({o['x']}, {o['y']})"
            for o in objects
        )
        return {
            # 1. Scene-graph prose (canonical).
            "scene_graph": f"{graph_text}\n\n{question}",
            # 2. Indented list with sizes.
            "indented_list": f"Objects in the scene:\n{obj_lines}\n\n{question}",
            # 3. Minimal coordinates only.
            "coords_only": (
                "Object positions (x left->right, y top->bottom):\n"
                f"{coord_lines}\n\n{question}"
            ),
        }


PLUGIN = _RelationalProbe()
