"""Pure logic for the pathfinding scene - NO rendering, NO plugin deps.

A scene is a graph of colored nodes connected by edges. The probe question is:
  "Is there a path from <NodeA> to <NodeB> moving only along edges that connect
   nodes of the SAME color?"

Ground truth: BFS restricted to same-color edges.
Oracle verbalization: adjacency list as text -> pure graph reasoning, still hard.

Design:
- n_nodes nodes, each assigned a color from COLORS.
- Edges drawn randomly with target density p_edge, then pruned so no self-loops.
- Answer is boolean (yes/no). Answer balance: enforced by selecting source/target
  pairs that alternate yes/no across seeds (see `balanced_question`).
- Two independent difficulty knobs: n_nodes (perceptual load) and graph density
  (reasoning load - more nodes/edges means longer paths to trace).

The canonical oracle verbalization is the adjacency list plus node colors.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from random import Random

NODE_COLORS = ["red", "blue", "green", "orange", "purple", "yellow"]


@dataclass(frozen=True)
class Node:
    id: int
    color: str
    x: int
    y: int


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: frozenset[tuple[int, int]]   # undirected: (min_id, max_id)

    def neighbors(self, node_id: int) -> list[int]:
        return [
            b if a == node_id else a
            for (a, b) in self.edges
            if node_id in (a, b)
        ]

    def color_of(self, node_id: int) -> str:
        return next(n.color for n in self.nodes if n.id == node_id)

    def node_by_id(self, node_id: int) -> Node:
        return next(n for n in self.nodes if n.id == node_id)


# BFS (same-color paths only) - the ground truth

def has_same_color_path(graph: Graph, src: int, dst: int) -> bool:
    """True iff there is a path from src to dst along same-colored edges.

    An edge (u, v) is traversable iff color(u) == color(v). Because every
    traversable edge joins two equally-colored nodes, ANY valid path is
    necessarily monochromatic end-to-end - so src and dst must share a color
    (early-out below) and the search only ever expands nodes of that color.
    Equivalently: src and dst are connected iff they fall in the same connected
    component of the subgraph induced by same-color edges.
    """
    if src == dst:
        return True
    if graph.color_of(src) != graph.color_of(dst):
        return False
    target_color = graph.color_of(src)
    visited: set[int] = set()
    queue: deque[int] = deque([src])
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        if cur == dst:
            return True
        for nb in graph.neighbors(cur):
            if nb not in visited and graph.color_of(nb) == target_color:
                queue.append(nb)
    return False


# Graph generation

def build_graph(
    rng: Random,
    n_nodes: int,
    p_edge: float,
    canvas: tuple[int, int] = (480, 360),
    margin: int = 50,
    min_dist: int = 70,
) -> Graph:
    """Generate a random graph with non-overlapping node positions."""
    w, h = canvas
    positions: list[tuple[int, int]] = []
    import math
    for _ in range(n_nodes):
        for _ in range(500):
            x = rng.randint(margin, w - margin)
            y = rng.randint(margin, h - margin)
            if all(math.hypot(x - px, y - py) >= min_dist for px, py in positions):
                positions.append((x, y))
                break
        else:
            x = rng.randint(margin, w - margin)
            y = rng.randint(margin, h - margin)
            positions.append((x, y))

    colors = [rng.choice(NODE_COLORS) for _ in range(n_nodes)]
    nodes = tuple(
        Node(id=i, color=colors[i], x=positions[i][0], y=positions[i][1])
        for i in range(n_nodes)
    )

    edges: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p_edge:
                edges.add((i, j))

    return Graph(nodes=nodes, edges=frozenset(edges))


def balanced_question(
    graph: Graph, seed: int, rng: Random
) -> tuple[int, int, bool] | None:
    """Pick (src, dst) such that same-color-path(src, dst) == (seed % 2 == 0).

    Returns (src_id, dst_id, answer) or None if no valid pair found.
    This enforces 50/50 answer balance across seeds.
    """
    want_yes = (seed % 2 == 0)
    node_ids = [n.id for n in graph.nodes]
    rng.shuffle(node_ids)
    for i, src in enumerate(node_ids):
        for dst in node_ids[i + 1:]:
            if src == dst:
                continue
            ans = has_same_color_path(graph, src, dst)
            if ans == want_yes:
                return src, dst, ans
    return None


# Oracle text: adjacency list + node colors

def graph_to_adjacency_text(graph: Graph) -> str:
    lines = ["Nodes (id: color):"]
    for n in sorted(graph.nodes, key=lambda x: x.id):
        lines.append(f"  {n.id}: {n.color}")
    lines.append("Edges (undirected):")
    for a, b in sorted(graph.edges):
        lines.append(f"  {a} - {b}")
    return "\n".join(lines)


def parse_adjacency_text(text: str) -> Graph | None:
    """Inverse of graph_to_adjacency_text; returns None if unparseable."""
    try:
        nodes: list[Node] = []
        node_re = re.compile(r"^\s*(\d+):\s+(\w+)", re.MULTILINE)
        for m in node_re.finditer(text):
            nodes.append(Node(id=int(m.group(1)), color=m.group(2), x=0, y=0))
        if not nodes:
            return None
        edges: set[tuple[int, int]] = set()
        edge_re = re.compile(r"(\d+)\s*[—\-–]\s*(\d+)")
        for m in edge_re.finditer(text):
            a, b = int(m.group(1)), int(m.group(2))
            edges.add((min(a, b), max(a, b)))
        return Graph(nodes=tuple(nodes), edges=frozenset(edges))
    except Exception:
        return None


def render_question(src_id: int, dst_id: int, graph: Graph) -> str:
    src_color = graph.color_of(src_id)
    dst_color = graph.color_of(dst_id)
    return (
        f"Is there a path from node {src_id} (the {src_color} node) to node "
        f"{dst_id} (the {dst_color} node), moving only along edges that connect "
        f"nodes of the SAME color? Answer with a single word: 'yes' or 'no'."
    )


def parse_question(text: str) -> tuple[int, int] | None:
    """Extract (src_id, dst_id) from a rendered question."""
    m = re.search(r"from node (\d+).*?to node (\d+)", text, re.DOTALL)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
