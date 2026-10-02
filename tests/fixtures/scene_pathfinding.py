"""Pathfinding scene - colored graph, same-color-edge reachability question.

Renders a graph of colored nodes connected by edges (PIL 2.5D). The probe asks
whether a path exists between two named nodes moving only along edges that connect
same-colored nodes. Ground truth is BFS-exact; the oracle injects the full
adjacency list so the model reasons over the graph in text space.

CONTRACT (locked):

    ground_truth = {
        "question":        str,     # NL question text
        "answer":          str,     # "yes" | "no"
        "adjacency_text":  str,     # oracle verbalization of the graph
        "nodes": [                  # list for probe / analysis use
            {"id": int, "color": str, "x": int, "y": int},
            ...
        ],
        "edges": [[int, int], ...], # pairs (sorted ascending)
        "src":   int,
        "dst":   int,
    }
    factors = {"n_nodes": int, "p_edge": float}
    meta    = {"generator": "pathfinding", "seed": int, "params": dict,
               "tier": "trivial" | "standard"}

``tier`` is "trivial" when n_nodes <= 5 and p_edge >= 0.6 (dense small graph).
Answer balance: seed % 2 determines yes/no target (enforced by balanced_question).
"""
from __future__ import annotations

import random as _random

from PIL import Image, ImageDraw

from renderprobe.core.schema import Scene
from tests.fixtures._pathfinding_core import (
    Graph,
    balanced_question,
    build_graph,
    graph_to_adjacency_text,
    render_question,
)

_CANVAS = (480, 360)
# Highlight halo for the src/dst nodes - magenta, deliberately NOT one of the six
# node fill colors, so it reads on any node and the mock can isolate it by color.
_HALO_RGB = (230, 0, 230)


def _node_radius(n_nodes: int) -> int:
    """Shrink nodes as the graph grows so they stay non-overlapping and labels stay
    legible - the graph (and its BFS ground truth) is placement-independent, but a
    readable layout keeps the FULL condition solvable in principle (GT trustable)."""
    if n_nodes <= 8:
        return 20
    if n_nodes <= 14:
        return 16
    return 12


def _min_dist(n_nodes: int) -> int:
    return 2 * _node_radius(n_nodes) + (30 if n_nodes <= 8 else 16 if n_nodes <= 14 else 10)

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":    (220,  50,  50),
    "blue":   ( 50,  80, 220),
    "green":  ( 40, 170,  60),
    "orange": (240, 140,  20),
    "purple": (150,  50, 190),
    "yellow": (210, 200,  20),
}


def _render(graph: Graph, src: int, dst: int, r: int) -> Image.Image:
    img = Image.new("RGB", _CANVAS, (248, 248, 248))
    draw = ImageDraw.Draw(img)

    # Draw edges first (behind nodes)
    for a, b in graph.edges:
        na = graph.node_by_id(a)
        nb = graph.node_by_id(b)
        draw.line([(na.x, na.y), (nb.x, nb.y)], fill=(150, 150, 150), width=2)

    # Draw nodes (uniform black outline for all).
    for node in graph.nodes:
        draw.ellipse(
            [node.x - r, node.y - r, node.x + r, node.y + r],
            fill=_COLOR_RGB[node.color], outline=(30, 30, 30), width=2,
        )
        draw.text((node.x - 4, node.y - 6), str(node.id), fill=(255, 255, 255))

    # Highlight src/dst with a magenta HALO drawn OUTSIDE the node, so it stays
    # high-contrast on every fill color (a same-colored ring on an orange/yellow
    # node would be invisible). _HALO_RGB is absent from the node palette, which is
    # also what lets the mock isolate it by color.
    hr = r + 5
    for nid in (src, dst):
        n = graph.node_by_id(nid)
        draw.ellipse(
            [n.x - hr, n.y - hr, n.x + hr, n.y + hr],
            outline=_HALO_RGB, width=4,
        )

    return img


class _PathfindingGenerator:
    name = "pathfinding"
    params_schema = {
        "n_nodes": {"type": "int",   "min": 4,   "max": 20,  "default": 8},
        "p_edge":  {"type": "float", "min": 0.2, "max": 0.7, "default": 0.4},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        n_nodes = int(params["n_nodes"])
        p_edge  = float(params["p_edge"])
        radius = _node_radius(n_nodes)
        min_dist = _min_dist(n_nodes)

        # Re-draw the graph (deterministically, per attempt) until a source/target
        # pair exists whose reachability matches the seed's balance target. Small
        # sparse graphs (n=4, low p_edge) sometimes admit no "yes" pair, so the
        # attempt budget is generous; empirically 40 clears the whole factor grid.
        graph = None
        result = None
        for attempt in range(40):
            g = build_graph(
                _random.Random(seed * 13 + attempt), n_nodes, p_edge, min_dist=min_dist
            )
            result = balanced_question(g, seed, _random.Random(seed * 7 + attempt))
            if result is not None:
                graph = g
                break

        if result is None or graph is None:
            raise RuntimeError(
                f"pathfinding: no balanced (src,dst) pair found "
                f"(seed={seed} n_nodes={n_nodes} p_edge={p_edge}) after retries."
            )

        src, dst, answer_bool = result
        answer_str = "yes" if answer_bool else "no"

        node_dicts = [
            {"id": n.id, "color": n.color, "x": n.x, "y": n.y}
            for n in graph.nodes
        ]
        edge_list = [[a, b] for a, b in sorted(graph.edges)]

        return Scene(
            id=f"pf_n{n_nodes}_p{int(p_edge*10)}_s{seed}",
            images=[_render(graph, src, dst, radius)],
            ground_truth={
                "question":       render_question(src, dst, graph),
                "answer":         answer_str,
                "adjacency_text": graph_to_adjacency_text(graph),
                "nodes":          node_dicts,
                "edges":          edge_list,
                "src":            src,
                "dst":            dst,
            },
            factors={"n_nodes": n_nodes, "p_edge": p_edge},
            meta={
                "generator": "pathfinding",
                "seed":      seed,
                "params":    dict(params),
                "tier":      "trivial" if (n_nodes <= 5 and p_edge >= 0.6) else "standard",
            },
        )


PLUGIN = _PathfindingGenerator()
