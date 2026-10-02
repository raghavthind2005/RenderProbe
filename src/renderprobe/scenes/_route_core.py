"""Weighted road maps and the shortest route through them: the logic, no pixels.

Kept apart from the scene so the part that decides whether a question is well posed can
be tested without rendering anything, the way `_polycube_core` is.

The task is "how long is the cheapest route from A to B", and the point of the design is
that its two difficulties come apart. How hard the map is to READ is set by how many
towns and roads there are; how hard the route is to WORK OUT is set by the length of the
answer and by whether the cheapest-looking first road is the right one. Neither knob
moves the other, which is what `polycube` lacked and what put it at the floor.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from random import Random

# Town labels. Letters, not names: a label has to survive being drawn inside a small
# circle and being typed back by a model, and "Ashford" fails both.
LABELS = "ABCDEFGHIJKL"
# Road lengths. Integers so the answer is exact, and starting at 2 so no road is free
# and no two-road detour can tie a one-road hop by accident.
MIN_W, MAX_W = 2, 19
# Enumerating every simple route is exponential, and it is only ever used to state how
# likely guessing is. Past this many the estimate is already stable.
_MAX_ROUTES = 20_000


@dataclass(frozen=True)
class Town:
    id: int
    x: float
    y: float

    @property
    def label(self) -> str:
        return LABELS[self.id]


@dataclass(frozen=True)
class RoadMap:
    towns: tuple[Town, ...]
    roads: tuple[tuple[int, int, int], ...]   # (a, b, length), a < b

    def neighbors(self, t: int) -> list[tuple[int, int]]:
        out = []
        for a, b, w in self.roads:
            if a == t:
                out.append((b, w))
            elif b == t:
                out.append((a, w))
        return out

    def length_of(self, a: int, b: int) -> int | None:
        lo, hi = (a, b) if a < b else (b, a)
        for x, y, w in self.roads:
            if (x, y) == (lo, hi):
                return w
        return None

    def town(self, t: int) -> Town:
        return self.towns[t]


def dijkstra(rm: RoadMap, src: int, dst: int) -> tuple[int, list[int]] | None:
    """Cheapest route and its cost, or None when the two towns are not connected.

    Ties are broken by the route that is lexicographically smallest, so the returned
    path is deterministic. A scene only ships when the optimum is unique anyway
    (`optimum_is_unique`), so that tie-break never decides an answer.
    """
    best: dict[int, tuple[int, list[int]]] = {src: (0, [src])}
    seen: set[int] = set()
    queue = [(0, [src], src)]
    while queue:
        cost, path, here = heapq.heappop(queue)
        if here in seen:
            continue
        seen.add(here)
        if here == dst:
            return cost, path
        for nxt, w in rm.neighbors(here):
            if nxt in seen:
                continue
            cand = (cost + w, path + [nxt])
            if nxt not in best or cand < best[nxt]:
                best[nxt] = cand
                heapq.heappush(queue, (cand[0], cand[1], nxt))
    return None


def route_costs(rm: RoadMap, src: int, dst: int) -> list[int]:
    """Distinct costs of every simple route from src to dst, cheapest first.

    This is what `chance_level` is built on: a model that has read the map perfectly but
    picks a route rather than optimising has one of these costs, so the chance of being
    right is one over how many there are. Counting routes instead of costs would
    overstate the difficulty, since several routes can share a cost and any of them
    answers correctly.
    """
    costs: set[int] = set()
    stack = [(src, {src}, 0)]
    seen_routes = 0
    while stack and seen_routes < _MAX_ROUTES:
        here, visited, cost = stack.pop()
        if here == dst:
            costs.add(cost)
            seen_routes += 1
            continue
        for nxt, w in rm.neighbors(here):
            if nxt not in visited:
                stack.append((nxt, visited | {nxt}, cost + w))
    return sorted(costs)


def optimum_is_unique(rm: RoadMap, src: int, dst: int) -> bool:
    """Exactly one route achieves the cheapest cost.

    Needed for the trap check and for any follow-up question about WHICH way the route
    goes. The cost answer alone would survive a tie, but a scene that is ambiguous under
    one reading of the question is not one to ship.
    """
    best = dijkstra(rm, src, dst)
    if best is None:
        return False
    target, _ = best
    hits = 0
    stack = [(src, {src}, 0)]
    while stack:
        here, visited, cost = stack.pop()
        if cost > target:
            continue
        if here == dst:
            if cost == target:
                hits += 1
                if hits > 1:
                    return False
            continue
        for nxt, w in rm.neighbors(here):
            if nxt not in visited:
                stack.append((nxt, visited | {nxt}, cost + w))
    return hits == 1


def greedy_first_hop(rm: RoadMap, src: int) -> int:
    """The town at the end of the shortest road leaving src: where a model that follows
    the cheapest-looking road goes."""
    return min(rm.neighbors(src), key=lambda nw: (nw[1], nw[0]))[0]


def greedy_is_wrong(rm: RoadMap, src: int, dst: int) -> bool:
    """The cheapest road out of src is NOT the first road of the cheapest route.

    This is the whole reasoning content of the task. Without it a model can answer by
    walking greedily and never has to compare whole routes, and the scene would measure
    map-reading with the reasoning step removed.
    """
    best = dijkstra(rm, src, dst)
    if best is None or len(best[1]) < 2:
        return False
    return greedy_first_hop(rm, src) != best[1][1]


def greedy_route_cost(rm: RoadMap, src: int, dst: int) -> int | None:
    """What a purely greedy traveler pays: always take the cheapest road that leads
    somewhere new, give up on a dead end. The wrong answer worth naming, because a model
    that produces exactly this number reasoned lazily rather than misread the map.
    """
    here, visited, cost = src, {src}, 0
    for _ in range(len(rm.towns)):
        if here == dst:
            return cost
        options = [(w, t) for t, w in rm.neighbors(here) if t not in visited]
        if not options:
            return None
        w, nxt = min(options)
        cost += w
        visited.add(nxt)
        here = nxt
    return None


def adjacency_text(rm: RoadMap) -> str:
    """The map as text, which is what the oracle hands over."""
    return "\n".join(
        f"  {LABELS[a]} - {LABELS[b]}: {w}" for a, b, w in sorted(rm.roads)
    )


def _too_close(p: tuple[float, float], placed, gap: float) -> bool:
    return any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < gap * gap for q in placed)


def scatter(n: int, rng: Random, width: float, height: float,
            margin: float, gap: float, tries: int = 4000) -> list[tuple[float, float]]:
    """n points inside the canvas, no two closer than `gap`.

    Rejection sampling rather than a grid: a grid reads as a grid, and a map whose towns
    sit on lattice points invites a model to answer from the layout instead of the
    numbers.
    """
    pts: list[tuple[float, float]] = []
    for _ in range(tries):
        if len(pts) == n:
            break
        p = (rng.uniform(margin, width - margin), rng.uniform(margin, height - margin))
        if not _too_close(p, pts, gap):
            pts.append(p)
    return pts if len(pts) == n else []


def _segments_cross(p1, p2, p3, p4) -> bool:
    def side(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 1e-9) - (v < -1e-9)
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return d1 != d2 and d3 != d4


def proximity_roads(towns: list[tuple[float, float]], rng: Random,
                    max_roads_per_town: int = 4) -> list[tuple[int, int]]:
    """Join nearby towns, skipping any road that would cross one already laid.

    A planar map is the point. Crossing roads are ambiguous to read - at a crossing
    there is no way to tell from the picture whether the two roads meet - which would
    turn a reasoning task into a guess about what the renderer meant.
    """
    n = len(towns)
    pairs = sorted(
        itertools.combinations(range(n), 2),
        key=lambda ab: (towns[ab[0]][0] - towns[ab[1]][0]) ** 2
        + (towns[ab[0]][1] - towns[ab[1]][1]) ** 2,
    )
    roads: list[tuple[int, int]] = []
    degree = [0] * n
    for a, b in pairs:
        if degree[a] >= max_roads_per_town or degree[b] >= max_roads_per_town:
            continue
        if any(_segments_cross(towns[a], towns[b], towns[c], towns[d])
               for c, d in roads if len({a, b, c, d}) == 4):
            continue
        roads.append((a, b))
        degree[a] += 1
        degree[b] += 1
    return roads


def is_connected(n: int, roads) -> bool:
    if n == 0:
        return True
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b, *_ in roads:
        adj[a].append(b)
        adj[b].append(a)
    seen, stack = {0}, [0]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n
