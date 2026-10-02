"""Pathfinding probe - same-color-edge reachability question (yes/no).

Reads the verified question/answer from the pathfinding scene's ground_truth.
The oracle verbalization injects the full adjacency list so the model does pure
graph traversal in text (perception isolated; reasoning load retained).
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene


class _PathfindingProbe:
    name = "pathfinding"
    answer_type = "categorical"
    requires_gt_fields = ["question", "answer", "adjacency_text", "nodes", "edges"]

    def question(self, scene: Scene) -> str:
        return scene.ground_truth["question"]

    def ground_truth(self, scene: Scene) -> str:
        return scene.ground_truth["answer"]

    def parse(self, raw: str) -> str:
        """Extract 'yes' or 'no' case-insensitively; raise on no match."""
        text = raw.strip().lower()
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"
        raise ValueError(f"Cannot parse 'yes'/'no' from: {raw!r}")

    def score(self, pred: str, gt: str) -> tuple[bool, float, None]:
        correct = pred == gt
        return correct, 1.0 if correct else 0.0, None

    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Perception report: is there a SINGLE direct edge between the two nodes the
        scene already marks with magenta halos (src/dst)? This isolates edge PERCEPTION
        (seeing one line) from the full task's multi-hop TRAVERSAL. report high + full low
        => it sees the graph but can't traverse it (reasoning); report low => it can't even
        read the edges (encoding). Uses the halos already in the image - no new handle."""
        gt = scene.ground_truth
        if not scene.images or "src" not in gt or "dst" not in gt:
            return None
        pair = sorted((int(gt["src"]), int(gt["dst"])))
        connected = any(sorted(e) == pair for e in gt.get("edges", []))
        return ReportSpec(
            question=("The two nodes marked with magenta halos are the ones in question. "
                      "Is there a SINGLE direct line (one edge) directly connecting those "
                      "two marked nodes? Answer 'yes' or 'no'. Ignore any longer routes."),
            ground_truth="yes" if connected else "no",
            parse=self.parse,
            score=self.score,
            images=[scene.images[0]],   # the base image already carries the src/dst halos
        )

    def oracle_solver(self, scene: Scene) -> str | None:
        """Oracle completeness certificate: re-run the same-color BFS over the nodes and
        edges the oracle verbalizes - NEVER read ground_truth['answer']. Matching GT
        across seeds proves the adjacency list conveys enough to solve the task, so a
        model's oracle failure is reasoning (traversal), not missing structure."""
        from tests.fixtures._pathfinding_core import (
            Graph,
            Node,
            has_same_color_path,
        )
        gt = scene.ground_truth
        if "src" not in gt or "dst" not in gt:
            return None
        nodes = tuple(
            Node(id=int(n["id"]), color=n["color"], x=int(n.get("x", 0)), y=int(n.get("y", 0)))
            for n in gt["nodes"]
        )
        edges = frozenset((min(int(a), int(b)), max(int(a), int(b))) for a, b in gt["edges"])
        graph = Graph(nodes=nodes, edges=edges)
        return "yes" if has_same_color_path(graph, int(gt["src"]), int(gt["dst"])) else "no"

    canonical_oracle_variant = "adjacency_list"

    def oracle_prompt(self, scene: Scene) -> str | None:
        variants = self.oracle_prompt_variants(scene)
        return variants.get(self.canonical_oracle_variant) if variants else None

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        adj_text  = scene.ground_truth.get("adjacency_text", "")
        question  = scene.ground_truth.get("question", "")
        nodes     = scene.ground_truth.get("nodes", [])
        edges     = scene.ground_truth.get("edges", [])
        if not adj_text or not question:
            return {}

        # Variant 2: JSON-style compact
        node_str = ", ".join(f"{n['id']}:{n['color']}" for n in nodes)
        edge_str = ", ".join(f"({a}-{b})" for a, b in edges)

        # Variant 3: prose
        prose_nodes = "; ".join(f"node {n['id']} is {n['color']}" for n in nodes)
        prose_edges = "; ".join(f"nodes {a} and {b} are connected" for a, b in edges)

        return {
            # 1. Structured adjacency list (canonical).
            "adjacency_list": f"{adj_text}\n\n{question}",
            # 2. Compact inline.
            "compact": (
                f"Nodes [id:color]: {node_str}\n"
                f"Edges: {edge_str}\n\n{question}"
            ),
            # 3. Prose description.
            "prose": (
                f"{prose_nodes}. {prose_edges}. {question}"
            ),
        }


PLUGIN = _PathfindingProbe()
