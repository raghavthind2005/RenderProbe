"""Pure logic for the polycube assembly scene - NO rendering, NO plugin deps.

Underscore-prefixed so auto-discovery skips it; `scenes/polycube.py` and
`probes/polycube.py` import from here.

THE TASK. An N x N x N cube is partitioned into K interlocking polycube pieces. One
piece is removed, leaving a cavity. The model is shown the remaining assembly (with
the hole) and the K pieces laid out separately, and must say which piece fills the
cavity.

Two things make a naive version of this task WRONG, and both are closed here rather
than hoped about.

1. AMBIGUOUS ANSWER. If two pieces are congruent under rotation, more than one of them
   genuinely fills the cavity and the "correct" answer is arbitrary. `decompose`
   therefore rejects any partition whose pieces are not pairwise non-congruent, so
   exactly one candidate fits - a property `assert_unique_fit` re-checks by brute force
   over all 24 rotations.

2. A COLOR SHORTCUT. If the assembly is drawn in the pieces' own colors, the missing
   one is found by seeing which color is absent, with no spatial reasoning at all - the
   classic hackable-benchmark failure mode. The assembly is therefore
   drawn in ONE neutral color; color labels the candidates only, so it names the answer
   without carrying any evidence toward it.

3. A COUNTING SHORTCUT. Pieces of a random partition often differ in cell count, and
   measurement shows the answer is then identifiable by counting the cavity's cells
   alone - no spatial reasoning at all - in 17% to 43% of cases depending on n and k.
   `eligible_answers` therefore only offers pieces whose cell count is shared with at
   least one other piece, so counting can narrow the field but can never decide it.

A fourth hazard, whether the cavity is actually VISIBLE, is geometric rather than
combinatorial and is handled by the scene (which owns the camera), not here.
"""
from __future__ import annotations

import math
from random import Random

Cell = tuple[int, int, int]


# ---------------------------------------------------------------------------
# The rotation group of the cube
# ---------------------------------------------------------------------------

def _mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rotation_group() -> list[tuple]:
    """The 24 proper rotations of the cube, by closing the generators.

    Generated rather than written out: a hand-typed table is exactly the kind of thing
    that silently contains 23 entries or a reflection, and `test_polycube_core` pins
    both the count and that every element is a proper rotation (determinant +1).
    """
    ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    gens = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),   # +90 about x
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),   # +90 about y
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),   # +90 about z
    ]
    seen = {ident}
    frontier = [ident]
    while frontier:
        m = frontier.pop()
        for g in gens:
            p = _mul(g, m)
            if p not in seen:
                seen.add(p)
                frontier.append(p)
    return sorted(seen)


ROTATIONS = _rotation_group()


def _apply(r, c: Cell) -> Cell:
    x, y, z = c
    return (r[0][0] * x + r[0][1] * y + r[0][2] * z,
            r[1][0] * x + r[1][1] * y + r[1][2] * z,
            r[2][0] * x + r[2][1] * y + r[2][2] * z)


def normalize(cells) -> tuple[Cell, ...]:
    """Translate a cell set so its bounding box starts at the origin, then sort it.
    Two cell sets are the same shape up to TRANSLATION iff their normal forms match."""
    cs = list(cells)
    mx = min(c[0] for c in cs)
    my = min(c[1] for c in cs)
    mz = min(c[2] for c in cs)
    return tuple(sorted((x - mx, y - my, z - mz) for x, y, z in cs))


def canonical(cells) -> tuple[Cell, ...]:
    """A shape's fingerprint: the smallest normal form over all 24 rotations.
    Two polycubes are congruent (same shape, any orientation) iff these are equal."""
    return min(normalize(_apply(r, c) for c in cells) for r in ROTATIONS)


def congruent(a, b) -> bool:
    return canonical(a) == canonical(b)


# ---------------------------------------------------------------------------
# Connectivity and decomposition
# ---------------------------------------------------------------------------

def is_connected(cells) -> bool:
    """Face-connected (6-neighbor). A piece that falls into two lumps is not a piece."""
    cs = set(cells)
    if not cs:
        return False
    start = next(iter(cs))
    seen = {start}
    stack = [start]
    while stack:
        x, y, z = stack.pop()
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (x + d[0], y + d[1], z + d[2])
            if n in cs and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cs)


def _neighbors(c: Cell):
    x, y, z = c
    yield from ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z),
                (x, y - 1, z), (x, y, z + 1), (x, y, z - 1))


def _grow_partition(n: int, k: int, rng: Random) -> list[list[Cell]] | None:
    """Partition the n^3 grid into k face-connected regions by round-robin growth.

    Every region starts from a seed and only ever gains cells adjacent to itself, so
    connectivity holds by construction rather than being checked afterwards. Growing
    the currently-smallest region keeps the sizes close to n^3/k. Returns None when a
    run strands unreachable cells, which the caller retries with a new seed.
    """
    grid = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
    if not 1 <= k <= len(grid):
        return None
    seeds = rng.sample(grid, k)
    regions: list[list[Cell]] = [[s] for s in seeds]
    owner = {s: i for i, s in enumerate(seeds)}
    remaining = len(grid) - k
    while remaining:
        # smallest region first, so the pieces come out comparable in size
        order = sorted(range(k), key=lambda i: (len(regions[i]), rng.random()))
        grew = False
        for i in order:
            options = [nb for c in regions[i] for nb in _neighbors(c)
                       if nb not in owner and all(0 <= v < n for v in nb)]
            if not options:
                continue
            pick = rng.choice(options)
            owner[pick] = i
            regions[i].append(pick)
            remaining -= 1
            grew = True
            break
        if not grew:
            return None       # stranded cells; caller retries
    return regions


def decompose(n: int, k: int, rng: Random, tries: int = 200,
              require_twin: bool = True) -> list[list[Cell]]:
    """Partition an n^3 cube into k connected, PAIRWISE NON-CONGRUENT pieces.

    Non-congruence is the guarantee that makes the question well posed: with it,
    exactly one piece fills the cavity left by any removed piece. Raises rather than
    returning a flawed decomposition, so a logic failure surfaces as a build error
    instead of a plausible wrong answer.
    """
    for _ in range(tries):
        regions = _grow_partition(n, k, rng)
        if regions is None:
            continue
        if any(not is_connected(r) for r in regions):
            continue                      # defensive: growth should already ensure this
        shapes = [canonical(r) for r in regions]
        if len(set(shapes)) != k:
            continue                      # two pieces the same shape -> ambiguous answer
        if require_twin and not eligible_answers(regions):
            continue                      # every size unique -> counting would decide it
        return regions
    raise ValueError(
        f"could not decompose a {n}x{n}x{n} cube into {k} connected, pairwise "
        f"non-congruent pieces"
        + (" with at least two of equal cell count" if require_twin else "")
        + f" in {tries} tries - try a different k (4 or more works best) or a larger n"
    )


def bounding_box(cells) -> tuple[int, int, int]:
    """A shape's box dimensions, sorted so the measure is rotation-invariant."""
    return tuple(sorted(  # type: ignore[return-value]
        max(c[i] for c in cells) - min(c[i] for c in cells) + 1 for i in range(3)))


def eligible_answers(pieces: list[list[Cell]]) -> list[int]:
    """Indices that make a question no coarse measure can shortcut.

    Two cheaper strategies than matching the shape were measured against random
    decompositions: counting cells identified the answer on its own in 17% to 43% of
    scenes, and comparing bounding boxes did so in 23% to 33%. Both let a model skip
    the rotation entirely on those items.

    A piece is therefore only offered as the answer when at least one OTHER piece
    shares both its cell count and its bounding box. Those measures then narrow the
    field without deciding it, and the shape has to settle it - which also makes the
    surviving distractors genuinely confusable rather than merely present. A workable
    share of pieces stays eligible, so there is still a free choice of answer.
    """
    keys = [(len(p), bounding_box(p)) for p in pieces]
    counts: dict[tuple, int] = {}
    for kk in keys:
        counts[kk] = counts.get(kk, 0) + 1
    # One other piece must match on BOTH measures at once. Requiring them separately
    # is not enough: a piece can share its size with one piece and its box with
    # another, and still be the only one with that combination - which a model
    # applying both filters together would pick out without ever rotating anything.
    return [i for i, kk in enumerate(keys) if counts[kk] >= 2]


def assert_unique_fit(pieces: list[list[Cell]], answer_index: int) -> None:
    """Re-check, by brute force, that exactly one piece fills the cavity.

    `decompose` already guarantees this. Checking it again from the other direction is
    cheap and catches a regression in the fingerprint itself, which is the one piece of
    machinery a bug here would otherwise hide behind.
    """
    cavity = pieces[answer_index]
    fits = [i for i, p in enumerate(pieces) if congruent(p, cavity)]
    if fits != [answer_index]:
        raise AssertionError(
            f"cavity is filled by pieces {fits}, expected exactly [{answer_index}] - "
            f"the question would have no single correct answer"
        )


def tiles_cube(pieces: list[list[Cell]], n: int) -> bool:
    """Do the pieces partition the n^3 cube exactly: every cell once, nothing outside."""
    seen: set[Cell] = set()
    for p in pieces:
        for c in p:
            if c in seen or not all(0 <= v < n for v in c):
                return False
            seen.add(c)
    return len(seen) == n ** 3


# ---------------------------------------------------------------------------
# Visibility: can a perfect perceiver read the shape off one image?
# ---------------------------------------------------------------------------
#
# This decides whether the question is answerable at all, so it is computed rather
# than hoped for. Measurement with the renderer's own id pass showed the naive
# alternative - showing the ASSEMBLY with a piece removed and asking which piece fills
# the hole - is unsound: the cavity is fully visible from a single view in only about
# 30% of scenes, and in the rest the hole's shape simply cannot be read off the image.
# Standalone pieces are the opposite: every piece has some orientation in which all of
# its cubes are visible, in 100% of the scenes measured.

def toward_camera(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit vector from the scene toward the camera, in GRID axes.

    The scene maps grid (x, y, z) to screen (x right, y up, z INTO the frame), and the
    renderer maps screen depth to -z in world space, so the grid z axis runs opposite
    to the world one. That sign is the easy thing to get wrong here.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az))


# Sample points within a cell: its center plus the 8 inset corners. A cell counts as
# visible when ANY of them has a clear line out, which is what the renderer's id pass
# reports - it marks a cell visible if even a sliver of it reaches a pixel. Testing the
# center alone was measured against the id pass and proved far too strict: it called a
# cell hidden in 42 of 50 pieces the renderer drew in full.
_SAMPLES = [(0.0, 0.0, 0.0)] + [
    (sx * 0.38, sy * 0.38, sz * 0.38)
    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]


def hidden_cells(cells, view: tuple[float, float, float], step: float = 0.25,
                 reach: float = 12.0) -> list[Cell]:
    """Cells of one piece with no clear line to the camera.

    Each cell is a unit cube centerd on its integer coordinate. March from several
    points inside the cell toward the camera; the cell is hidden only when EVERY
    sample is blocked by another cell of the same piece. Cross-checked against the
    renderer's id pass, which never hides a cell this reports as visible.
    """
    occ = set(cells)
    out = []
    for c in cells:
        visible = False
        for off in _SAMPLES:
            start = (c[0] + off[0], c[1] + off[1], c[2] + off[2])
            blocked = False
            t = step
            while t <= reach and not blocked:
                p = (start[0] + view[0] * t, start[1] + view[1] * t, start[2] + view[2] * t)
                q = (round(p[0]), round(p[1]), round(p[2]))
                if q != c and q in occ:
                    blocked = True
                t += step
            if not blocked:
                visible = True
                break
        if not visible:
            out.append(c)
    return out


def orient_for_visibility(cells, view, rng: Random | None = None):
    """Return an orientation of `cells` in which every cube is visible, or None.

    Tries the piece as given first, then the remaining rotations. Shapes are unchanged
    by this - a rotated piece is the same piece - so choosing a readable orientation
    costs nothing scientifically and is what makes the image answerable.
    """
    base = normalize(cells)
    if not hidden_cells(base, view):
        return base
    order = list(ROTATIONS)
    if rng is not None:
        rng.shuffle(order)
    for r in order:
        cand = normalize(_apply(r, c) for c in base)
        if not hidden_cells(cand, view):
            return cand
    return None
