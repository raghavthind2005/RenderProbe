"""Deterministic test fixtures standing in for a model.

These are test doubles, not models. They let the runner, the probes and the
analyzers be exercised offline and deterministically, which is what keeps the
suite fast and free of network flake. They are NOT evidence about any real
system: a result produced by a fixture says only that the plumbing works. They
are kept out of the model catalog and out of every experiment config for that
reason.

Handles three answer styles, auto-detected from prompt text:

Categorical (spatial-relation): "exact" mode reads color centroids from the
image; "noisy" mode flips 25% of the time. Oracle prompts with pixel coordinates
are parsed directly - perfect oracle reasoning without an API key.

Numeric (dots counting): blob-counting via PIL.

Relational (compositional referential chain): honest pixel-level perception for
the full condition (blob detection -> perceived scene graph -> resolve chain).
Oracle condition: parse scene-graph text injected in the prompt, then resolve.
Never reads ground_truth: a double that peeked would score like a perfect perceiver,
and every offline number taken from it would mean nothing.

Every answer is a PURE function of (model name, prompt, image bytes): run() reseeds a
fresh RNG from those inputs, so results are reproducible and independent of call order.

Modes:
  exact  - perfect answer (categorical: image analysis; numeric: blob count).
  noisy  - introduces errors (categorical: 25% flip; numeric: gaussian noise;
            relational: flips the resolved/perceived answer with probability
            `flip_rate`, default 0.25 - mock_noisy uses 0.9, which is what it takes
            for the fixture to be genuinely encoding-limited: its perception report has
            to come out no better than the task itself, and at 0.4 it reported the
            primitive on 0.79 of scenes while solving 0.50, which is a model that sees
            the fact and fails to use it).
  arbitration - perceives + reasons-over-text fine but fails from the image
            (relational only) -> a grounding/integration-limited demo.
  fixed  - always returns a constant (tests failure paths, numeric only).
"""
from __future__ import annotations

import hashlib
import random as _random
import re

import numpy as np
from PIL import Image

from renderprobe.core.schema import Response

# Numeric helpers (dots scene)

def _parse_oracle_count(prompt: str) -> int | None:
    """First integer in the prompt = the injected count (count oracle is text-only).
    The plain full/blind counting question contains no digit, so this returns None
    there and the mock falls back to counting blobs (full) or guessing (blind)."""
    m = re.search(r"\d+", prompt)
    return int(m.group()) if m else None


def _count_blobs(img: Image.Image, min_pixels: int = 50) -> int:
    """Count distinct non-white blobs via BFS connected components."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    is_white = np.all(arr > 230, axis=2)
    count = 0
    for y in range(h):
        for x in range(w):
            if visited[y, x] or is_white[y, x]:
                continue
            stack = [(y, x)]
            size = 0
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or is_white[cy, cx]:
                    continue
                visited[cy, cx] = True
                size += 1
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
            if size >= min_pixels:
                count += 1
    return count


# Categorical helpers (spatial-relation scene)

_KNOWN_COLORS = ["red", "blue", "green", "orange", "purple"]


def _is_spatial_prompt(prompt: str) -> bool:
    low = prompt.lower()
    return "left" in low and "right" in low


def _is_color_prompt(prompt: str) -> bool:
    low = prompt.lower()
    return "what color" in low or "color" in low


def _dominant_color(img: Image.Image) -> str:
    """Return the most-present non-white, non-gray color name in the image."""
    arr = np.array(img.convert("RGB")).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    masks = {
        "red":    ((r > 150) & (g < 120) & (b < 120)),
        "blue":   ((b > 150) & (r < 120) & (g < 120)),
        "green":  ((g > 120) & (r < 120) & (b < 120)),
        "orange": ((r > 180) & (g > 80) & (g < 160) & (b < 80)),
        "purple": ((r > 80) & (b > 80) & (g < 80)),
    }
    best = max(masks, key=lambda c: int(np.sum(masks[c])))
    return best


def _parse_oracle_coords(prompt: str) -> str | None:
    """Compute the left/right relation from coordinates stated anywhere in the prompt.

    Format-agnostic so the mock stands in for a *competent oracle reasoner* across
    every verbalization variant (relational prose, coordinate list, JSON-as-text):
    it grabs the first integer following the first mention of "red" and of "blue",
    which is the x-coordinate in all variants (they place x first by design). If
    either coordinate is missing (blind/full prompts, or a garbled template that
    omits positions), returns None so the caller falls back to its normal path.
    """
    red_m = re.search(r"red\D*?(\d+)", prompt, re.IGNORECASE)
    blue_m = re.search(r"blue\D*?(\d+)", prompt, re.IGNORECASE)
    if red_m and blue_m:
        rx, bx = int(red_m.group(1)), int(blue_m.group(1))
        return "left" if rx < bx else "right"
    return None


def _parse_oracle_color(prompt: str) -> str | None:
    """First known color name in the prompt = the injected target color (identity
    oracle is text-only, so any color word present is the ground-truth primitive).
    The plain full/blind question names no color, so this returns None there."""
    m = re.search(rf"\b({'|'.join(_KNOWN_COLORS)})\b", prompt, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _color_centroid_x(img: Image.Image, color: str) -> int | None:
    """Return the mean x-coordinate of pixels that strongly match a named color."""
    arr = np.array(img.convert("RGB")).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    if color == "red":
        mask = (r > 150) & (g < 100) & (b < 100)
    elif color == "blue":
        mask = (b > 150) & (r < 100) & (g < 100)
    else:
        return None
    xs = np.where(mask)[1]
    return int(np.mean(xs)) if len(xs) > 10 else None


def _spatial_answer(
    images: list[Image.Image], prompt: str, rng: _random.Random, noise: bool
) -> str:
    """Return left/right for a spatial-relation prompt."""
    # Oracle path: prompt contains explicit coordinates -> perfect reasoning
    oracle_ans = _parse_oracle_coords(prompt)
    if oracle_ans is not None:
        return oracle_ans

    if not images:
        # Blind: random guess
        return rng.choice(["left", "right"])

    # Full: find red and blue centroids in the image
    rx = _color_centroid_x(images[0], "red")
    bx = _color_centroid_x(images[0], "blue")
    if rx is not None and bx is not None:
        correct = "left" if rx < bx else "right"
        # Noisy mode: flip 25% of the time to simulate spatial reasoning errors
        if noise and rng.random() < 0.25:
            return "right" if correct == "left" else "left"
        return correct

    # Fallback if color detection fails
    return rng.choice(["left", "right"])


# Mock model

class _MockModel:
    def __init__(
        self,
        name: str,
        mode: str = "exact",
        noise_std: float = 1.5,
        fixed_val: int = 0,
        flip_rate: float = 0.25,
    ) -> None:
        self.name = name
        self.is_open = True
        # A deterministic test double, not a model. The catalog lists these
        # separately so they are never mistaken for a measurable system.
        self.is_fixture = True
        self._mode = mode
        self._noise_std = noise_std
        self._fixed_val = fixed_val
        # Probability the noisy mode corrupts a perceived/resolved relational answer.
        # Higher => clearly-degraded perception (a robust encoding-limited example).
        self._flip_rate = flip_rate
        self._rng = _random.Random(42)

    def run(self, images: list[Image.Image], prompt: str) -> Response:
        # Reseed per call from the inputs so the mock is a PURE function of (model,
        # prompt, image) - reproducible and independent of call order. (A single shared
        # RNG that advances across calls made results depend on how many times the mock
        # had been called, i.e. on test/run order.)
        self._rng = _random.Random(self._seed_for(images, prompt))
        text = self._answer(images, prompt)
        # When the prompt asks for a confidence, append one so the
        # elicitation + calibration path is exercisable end-to-end with no API keys.
        if "confidence" in prompt.lower():
            text = f"{text}\nConfidence: {self._confidence_value()}"
        return Response(text=text, confidence=None)

    def _seed_for(self, images: list[Image.Image], prompt: str) -> int:
        h = hashlib.sha256(self.name.encode())
        h.update(prompt.encode())
        for img in images:
            h.update(img.tobytes())
        return int.from_bytes(h.digest()[:8], "big")

    def _confidence_value(self) -> int:
        # Deterministic (seeded) with a little spread so binning is exercised; a
        # confident mode reports higher. Not meant to be well-calibrated.
        base = {"exact": 88, "noisy": 68}.get(self._mode, 55)
        return max(0, min(100, base + self._rng.randint(-8, 8)))

    def _answer(self, images: list[Image.Image], prompt: str) -> str:
        if _is_pathfinding_prompt(prompt):
            return _pathfinding_answer(images, prompt, self._rng, noise=self._mode == "noisy")

        # Must precede the relational + generic-color checks: the report question
        # also mentions "color" and coordinates, but is a perception-only sub-task.
        flip = self._flip_rate if self._mode == "noisy" else 0.0
        if _is_relational_report_prompt(prompt):
            return _relational_report_answer(images, prompt, self._rng, flip=flip)

        if _is_relational_prompt(prompt):
            return _relational_answer(
                images, prompt, self._rng,
                flip=flip,
                arbitration=self._mode == "arbitration",
            )

        if _is_spatial_prompt(prompt):
            noisy = self._mode == "noisy"
            return _spatial_answer(images, prompt, self._rng, noise=noisy)

        if _is_color_prompt(prompt):
            if not images:
                # Oracle (text-only): read the injected color; else blind guess.
                oracle_c = _parse_oracle_color(prompt)
                return oracle_c if oracle_c else self._rng.choice(_KNOWN_COLORS)
            color = _dominant_color(images[0])
            if self._mode == "noisy" and self._rng.random() < 0.3:
                color = self._rng.choice(_KNOWN_COLORS)
            return color

        # Numeric path (dots scene)
        # Oracle (text-only): read the injected count; else count blobs / guess.
        oracle_n = _parse_oracle_count(prompt)
        if oracle_n is not None:
            return str(oracle_n)
        if not images:
            return str(max(0, round(self._rng.gauss(5, 3))))

        base = _count_blobs(images[0])
        if self._mode == "exact":
            val = base
        elif self._mode == "noisy":
            val = max(0, round(base + self._rng.gauss(0, self._noise_std)))
        else:
            val = self._fixed_val
        return str(val)

    def attention(self, images: list[Image.Image], prompt: str) -> None:
        return None


# Relational scene perception (honest: reads pixels, never ground_truth)

# Color -> (R_lo, G_lo, B_lo, R_hi, G_hi, B_hi) broad detection ranges.
_COLOR_RANGES: dict[str, tuple[int, int, int, int, int, int]] = {
    "red":    (160,   0,   0, 255, 110, 110),
    "blue":   (  0,   0, 160, 110, 110, 255),
    "green":  (  0, 110,   0, 110, 255, 110),
    "orange": (170,  70,   0, 255, 165,  80),
    "purple": ( 90,   0,  90, 210,  80, 210),
    "yellow": (170, 160,   0, 255, 255,  80),
}


def _blob_components(mask: "np.ndarray") -> list[list[tuple[int, int]]]:
    """BFS connected components on a 2D boolean mask. Returns list of pixel lists."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    blobs: list[list[tuple[int, int]]] = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        blob: list[tuple[int, int]] = []
        stack = [(sy, sx)]
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            if visited[cy, cx] or not mask[cy, cx]:
                continue
            visited[cy, cx] = True
            blob.append((cy, cx))
            stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
        if blob:
            blobs.append(blob)
    return blobs


def _infer_shape(
    mask: "np.ndarray", blob: list[tuple[int, int]]
) -> str:
    """Heuristic shape inference from blob geometry - intentionally imperfect.

    Checks whether corners of the bounding box are in the blob mask:
      square  -> all 4 corners filled
      triangle -> bottom corners filled but top corners empty
      circle  -> no corners filled
    """
    ys = [p[0] for p in blob]
    xs = [p[1] for p in blob]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)

    def corner_filled(r: int, c: int) -> bool:
        r = max(0, min(mask.shape[0] - 1, r))
        c = max(0, min(mask.shape[1] - 1, c))
        return bool(mask[r, c])

    # Sample corners a few pixels inside the bounding box to avoid outline gaps.
    pad = 3
    tl = corner_filled(min_y + pad, min_x + pad)
    tr = corner_filled(min_y + pad, max_x - pad)
    bl = corner_filled(max_y - pad, min_x + pad)
    br = corner_filled(max_y - pad, max_x - pad)

    top_filled = tl and tr
    bot_filled = bl and br
    if top_filled and bot_filled:
        return "square"
    if bot_filled and not top_filled:
        return "triangle"
    return "circle"


def _perceive_relational_objects(img: "Image.Image") -> list:
    """Detect objects in a relational scene image via color-blob analysis.

    Returns a list of `Obj` from _relational_core. Honesty invariant: only
    pixel data is used, never ground_truth. Color/position/size are reliable;
    shape is a heuristic (may err on ambiguous blobs -> genuine mock failures).
    """
    from tests.fixtures._relational_core import Obj

    arr = np.array(img.convert("RGB"))
    r_ch = arr[:, :, 0].astype(int)
    g_ch = arr[:, :, 1].astype(int)
    b_ch = arr[:, :, 2].astype(int)

    perceived: list[Obj] = []
    for color, (rlo, glo, blo, rhi, ghi, bhi) in _COLOR_RANGES.items():
        mask = (
            (r_ch >= rlo) & (r_ch <= rhi) &
            (g_ch >= glo) & (g_ch <= ghi) &
            (b_ch >= blo) & (b_ch <= bhi)
        )
        for blob in _blob_components(mask):
            if len(blob) < 40:      # discard specks
                continue
            blob_ys = [p[0] for p in blob]
            blob_xs = [p[1] for p in blob]
            cx = int(sum(blob_xs) / len(blob_xs))
            cy = int(sum(blob_ys) / len(blob_ys))
            size = "large" if len(blob) > 900 else "small"
            shape = _infer_shape(mask, blob)
            perceived.append(Obj(color=color, shape=shape, size=size, x=cx, y=cy))
    return perceived


def _is_relational_prompt(prompt: str) -> bool:
    return "start at" in prompt.lower()


# Format-agnostic object reader for the relational oracle: matches an (optional
# size) + color + shape + (x, y) triple regardless of the surrounding punctuation,
# so the mock is a faithful oracle reasoner across ALL verbalization variants
# ("- a small red circle at (100, 50)", "  small red circle: (100, 50)", ...) - the
# analog of the format-agnostic spatial oracle reader. The plain full/blind question
# carries no "(x, y)" pairs, so this yields [] there and the caller falls back.
_REL_OBJ_RE = re.compile(
    r"(?:(small|large)\s+)?"
    r"(red|blue|green|orange|purple|yellow)\s+"
    r"(circle|square|triangle)"
    r"[^\d(]*\((\d+),\s*(\d+)\)",
    re.IGNORECASE,
)


def _parse_relational_objects_tolerant(prompt: str) -> list:
    from tests.fixtures._relational_core import Obj  # noqa: PLC0415

    objs = []
    for m in _REL_OBJ_RE.finditer(prompt):
        size = (m.group(1) or "small").lower()
        objs.append(Obj(
            color=m.group(2).lower(), shape=m.group(3).lower(), size=size,
            x=int(m.group(4)), y=int(m.group(5)),
        ))
    return objs


def _relational_answer(
    images: list["Image.Image"], prompt: str,
    rng: _random.Random, flip: float = 0.0, arbitration: bool = False,
) -> str:
    from tests.fixtures._relational_core import (  # noqa: PLC0415
        COLORS,
        answer_of,
        parse_question,
    )
    chain = parse_question(prompt)

    if not images:
        # Oracle (scene graph in prompt) or blind (question only). NOTE: arbitration
        # mode does NOT degrade this path - it reasons over text fine (oracle high);
        # its failure is confined to the image (full) condition below.
        objs = _parse_relational_objects_tolerant(prompt)
        if objs and chain is not None:
            ans = answer_of(objs, chain)
            if ans is not None:
                return ans
        # Blind: random color guess.
        return rng.choice(COLORS)

    # Arbitration simulation (docs/methodology.md): the model perceives the scene
    # fine (see the honest report answer) and reasons over text fine (oracle above),
    # but on the IMAGE it overrides the percept - a grounding/integration failure, not
    # a perception one. Emitted only by the `mock_arbitration` demo model.
    if arbitration:
        return rng.choice(COLORS)

    # Full: perceive objects from pixels, then resolve.
    perceived = _perceive_relational_objects(images[0])
    if chain is not None and perceived:
        ans = answer_of(perceived, chain)
        if ans is not None:
            if rng.random() < flip:
                return rng.choice(COLORS)
            return ans
    return rng.choice(COLORS)


def _is_relational_report_prompt(prompt: str) -> bool:
    return "magenta" in prompt.lower()


def _relational_report_answer(
    images: list["Image.Image"], prompt: str,
    rng: _random.Random, flip: float = 0.0,
) -> str:
    """Honest perception-report answer: locate the magenta highlight ring, perceive the
    objects from pixels, and name the color of the one INSIDE the ring - no chain
    reasoning, no pixel-coordinate localization, never the GT. This is the
    perception-only measure that separates encoding from grounding."""
    from tests.fixtures._relational_core import COLORS  # noqa: PLC0415

    if not images:
        return rng.choice(COLORS)
    centers = _detect_highlighted_centers(images[0])   # magenta ring center(s)
    perceived = _perceive_relational_objects(images[0])
    if not centers or not perceived:
        return rng.choice(COLORS)
    cx, cy = centers[0]
    nearest = min(perceived, key=lambda o: (o.x - cx) ** 2 + (o.y - cy) ** 2)
    if rng.random() < flip:
        return rng.choice(COLORS)
    return nearest.color


# Pathfinding perception (honest: reads pixels, never ground_truth)


def _detect_nodes_from_image(img: "Image.Image") -> list[tuple[int, str, int, int]]:
    """Find pathfinding graph nodes by color blob detection.

    Returns list of (approx_id, color, cx, cy). IDs assigned by detection order,
    not guaranteed to match ground truth (mock may mis-assign IDs -> genuine errors).
    """
    arr = np.array(img.convert("RGB"))
    r_ch = arr[:, :, 0].astype(int)
    g_ch = arr[:, :, 1].astype(int)
    b_ch = arr[:, :, 2].astype(int)

    nodes: list[tuple[int, str, int, int]] = []
    idx = 0
    for color, (rlo, glo, blo, rhi, ghi, bhi) in _COLOR_RANGES.items():
        mask = (
            (r_ch >= rlo) & (r_ch <= rhi) &
            (g_ch >= glo) & (g_ch <= ghi) &
            (b_ch >= blo) & (b_ch <= bhi)
        )
        for blob in _blob_components(mask):
            if len(blob) < 100:   # nodes are ~π*20²≈1257 px; skip tiny specks
                continue
            blob_ys = [p[0] for p in blob]
            blob_xs = [p[1] for p in blob]
            cx = int(sum(blob_xs) / len(blob_xs))
            cy = int(sum(blob_ys) / len(blob_ys))
            nodes.append((idx, color, cx, cy))
            idx += 1
    return nodes


# Magenta highlight halo drawn around src/dst in scenes/pathfinding.py: (230,0,230).
# Magenta is absent from the node palette, so a simple (high-R, low-G, high-B) mask
# isolates it cleanly on any fill - the mock keys src/dst off this halo, which it can
# perceive, rather than the integer node labels it cannot OCR.
def _detect_highlighted_centers(img: "Image.Image") -> list[tuple[float, float]]:
    arr = np.array(img.convert("RGB")).astype(int)
    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    magenta = (r_ch >= 190) & (g_ch <= 80) & (b_ch >= 190)
    centers: list[tuple[float, float]] = []
    for blob in _blob_components(magenta):
        if len(blob) < 30:   # a 4px halo at r≈25 is ~600 px; skip specks
            continue
        ys = [p[0] for p in blob]
        xs = [p[1] for p in blob]
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centers


def _detect_edges_from_image(
    img: "Image.Image",
    nodes: list[tuple[int, str, int, int]],
) -> list[tuple[int, int]]:
    """Detect edges by sampling along straight lines between node pairs.

    An edge is declared present when enough sampled pixels along the line are
    non-background (the gray lines drawn between node centers). This is a
    heuristic: it may miss edges or find false positives (genuine mock errors).
    """
    arr = np.array(img.convert("RGB")).astype(int)
    edges: list[tuple[int, int]] = []

    for i, (ai, _, ax, ay) in enumerate(nodes):
        for j, (bi, _, bx, by) in enumerate(nodes):
            if j <= i:
                continue
            # Sample 20 points along the line between the two node centers,
            # skipping the first and last few pixels (inside the node circles).
            n_samples = 20
            dark_pixels = 0
            for t in range(4, n_samples - 3):
                frac = t / n_samples
                px = int(ax + frac * (bx - ax))
                py = int(ay + frac * (by - ay))
                py = max(0, min(arr.shape[0] - 1, py))
                px = max(0, min(arr.shape[1] - 1, px))
                r, g, b = arr[py, px, 0], arr[py, px, 1], arr[py, px, 2]
                # Gray edge pixels: R≈G≈B around 150, not white (248) background
                if r < 220 and g < 220 and b < 220 and abs(int(r)-int(g)) < 40:
                    dark_pixels += 1
            if dark_pixels >= 5:
                edges.append((min(ai, bi), max(ai, bi)))
    return edges


def _is_pathfinding_prompt(prompt: str) -> bool:
    return "same color" in prompt.lower()


# Format-agnostic graph reader for the pathfinding oracle: recovers (id -> color)
# and the edge set across every verbalization variant - the structured adjacency
# list ("0: red" / "0 - 1"), the compact inline form ("0:red" / "(0-1)"), and the
# prose form ("node 0 is red" / "nodes 0 and 1 are connected"). Node/edge patterns
# are unioned and de-duplicated (first color seen per id wins), so the mock reasons
# faithfully regardless of which template the probe emitted.
_PF_NODE_RES = (
    re.compile(r"node\s+(\d+)\s+is\s+([a-z]+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*:\s*([a-z]+)"),
)
_PF_EDGE_RES = (
    re.compile(r"nodes\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*[—–-]\s*(\d+)"),
)


def _parse_pathfinding_graph_tolerant(prompt: str):
    from tests.fixtures._pathfinding_core import Graph, Node  # noqa: PLC0415

    colors: dict[int, str] = {}
    for pat in _PF_NODE_RES:
        for m in pat.finditer(prompt):
            nid = int(m.group(1))
            colors.setdefault(nid, m.group(2).lower())
    if not colors:
        return None
    edges: set[tuple[int, int]] = set()
    for pat in _PF_EDGE_RES:
        for m in pat.finditer(prompt):
            a, b = int(m.group(1)), int(m.group(2))
            if a in colors and b in colors and a != b:
                edges.add((min(a, b), max(a, b)))
    nodes = tuple(Node(id=i, color=c, x=0, y=0) for i, c in sorted(colors.items()))
    return Graph(nodes=nodes, edges=frozenset(edges))


def _pathfinding_answer(
    images: list["Image.Image"], prompt: str,
    rng: _random.Random, noise: bool,
) -> str:
    from tests.fixtures._pathfinding_core import (  # noqa: PLC0415
        Graph,
        Node,
        has_same_color_path,
        parse_question,
    )

    ids_pair = parse_question(prompt)

    if not images:
        # Oracle: graph injected in prompt (any variant) -> BFS.
        graph = _parse_pathfinding_graph_tolerant(prompt)
        if graph is not None and ids_pair is not None:
            src, dst = ids_pair
            ans = has_same_color_path(graph, src, dst)
            result = "yes" if ans else "no"
            if noise and rng.random() < 0.15:
                return "no" if result == "yes" else "yes"
            return result
        # Blind: random guess.
        return rng.choice(["yes", "no"])

    # Full: perceive nodes + edges from pixels, then identify src/dst by the GOLD
    # highlight ring (a perceivable handle) - NOT the integer labels, which the
    # mock cannot OCR. Reachability is label-independent, so answering over the
    # mock's own internal node IDs is correct; the difficulty is genuinely
    # perceptual (detect nodes, their colors, edges, and which two are marked).
    raw_nodes = _detect_nodes_from_image(images[0])
    if not raw_nodes:
        return rng.choice(["yes", "no"])

    nodes = tuple(Node(id=ni, color=nc, x=nx, y=ny) for ni, nc, nx, ny in raw_nodes)
    raw_edges = _detect_edges_from_image(images[0], raw_nodes)
    perceived_graph = Graph(nodes=nodes, edges=frozenset(raw_edges))

    centers = _detect_highlighted_centers(images[0])
    if len(centers) < 2:
        return rng.choice(["yes", "no"])   # couldn't locate the two marked nodes

    def _nearest(cx: float, cy: float) -> "Node":
        return min(nodes, key=lambda n: (n.x - cx) ** 2 + (n.y - cy) ** 2)

    src_node = _nearest(*centers[0])
    dst_node = _nearest(*centers[1])
    if src_node.id == dst_node.id:
        return rng.choice(["yes", "no"])   # ambiguous highlight match

    ans = has_same_color_path(perceived_graph, src_node.id, dst_node.id)
    result = "yes" if ans else "no"
    if noise and rng.random() < 0.25:
        return "no" if result == "yes" else "yes"
    return result


PLUGINS = [
    _MockModel(name="mock", mode="exact"),
    _MockModel(name="mock_noisy", mode="noisy", noise_std=1.5, flip_rate=0.9),
    # Simulates a grounding/arbitration failure on the relational scene: perceives
    # fine (report high), reasons over text fine (oracle high), fails from the image
    # (full low). It exists to exercise the grounding-limited branch of the classifier
    # in the test suite. It is not evidence about any real model and must not appear
    # in an experiment config. See docs/methodology.md.
    _MockModel(name="mock_arbitration", mode="arbitration"),
]
