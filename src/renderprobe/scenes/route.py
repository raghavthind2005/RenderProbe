"""Road map scene: how long is the cheapest route from one town to another?

A weighted planar graph drawn as a road map. Towns are lettered circles, roads are lines
carrying their length, and the question asks for the total length of the cheapest route
between two marked towns. The answer is an integer, so guessing is worth almost nothing:
`chance_level` reports one over the number of distinct route costs the map admits, which
is typically under two percent rather than the fifty percent a yes/no question hands out
for free.

Its two difficulties are on separate knobs, which is the whole reason it exists:

    n_towns   how hard the map is to READ. More towns and roads mean more numbers to
              find and keep straight, and nothing else.
    n_hops    how hard the route is to WORK OUT. The cheapest route is built to be
              exactly this many roads long, and no shorter route ties it.

Every map ships with the cheapest road out of the start town leading somewhere other
than the cheapest route (`greedy_is_wrong`). Without that a model can answer by walking
greedily and never compares whole routes, and the scene would be measuring map reading
with the reasoning removed. The cost a greedy traveler would pay is carried in the
ground truth as `greedy_answer`: a model returning exactly that number read the map
correctly and reasoned lazily, which is a different failure from misreading it.

CONTRACT (locked):

    ground_truth = {
        "question":       str,
        "answer":         int,              # cheapest total length
        "adjacency_text": str,              # oracle verbalization
        "towns":  [{"id": int, "label": str, "x": float, "y": float}, ...],
        "roads":  [[a, b, length], ...],    # a < b
        "src":    int, "dst": int,
        "route":  [int, ...],               # the unique cheapest route
        "greedy_answer":  int | None,       # what greedy costs, when it gets there
        "route_costs":    [int, ...],       # distinct costs over all simple routes
    }
    factors = {"n_towns": int, "n_hops": int}
    meta    = {"generator": "route", "seed": int, "params": dict, "tier": str}
"""
from __future__ import annotations

from random import Random

from PIL import Image, ImageDraw, ImageFont

from renderprobe.core.schema import Scene
from renderprobe.scenes._route_core import (
    LABELS,
    MAX_W,
    MIN_W,
    RoadMap,
    Town,
    adjacency_text,
    dijkstra,
    greedy_is_wrong,
    greedy_route_cost,
    is_connected,
    optimum_is_unique,
    proximity_roads,
    route_costs,
    scatter,
)

_CANVAS = (760, 480)
_MARGIN = 54
_TOWN_R = 19
_GAP = 132                 # smallest distance between two town centers
_LABEL_OFF = 15            # how far a road's number sits off the road
_TRIES = 400

# A road map, not a chart: paper ground, dark ink, roads with a casing so they read as
# roads rather than as graph edges. The two marked towns are the only saturated color in
# the frame, so finding them needs no searching and the task stays about the numbers.
_PAPER = (243, 239, 228)
_INK = (38, 42, 48)
_ROAD_CASING = (150, 146, 136)
_ROAD_FILL = (253, 252, 249)
_TOWN_FILL = (252, 251, 248)
_START = (196, 62, 44)
_GOAL = (32, 96, 132)
_PILL_FILL = (255, 254, 250)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _font(size: int):
    """A real typeface if the machine has one, else Pillow's built-in.

    Numbers on the roads are the whole content of the image, so this is not cosmetic: at
    the default bitmap size they are unreadable, and the scene would be unanswerable for
    a reason that has nothing to do with the model.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _mid(a, b, t: float = 0.5):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _perp(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (-dy / n, dx / n)


def _boxes_overlap(p, q, pad: float = 3.0) -> bool:
    return not (p[2] + pad < q[0] or q[2] + pad < p[0]
                or p[3] + pad < q[1] or q[3] + pad < p[1])


def _place_labels(pts, roads, font) -> list[tuple] | None:
    """A readable spot for every road's number, or None if the map cannot take them.

    Tried in order: each side of the road at its midpoint, then further along it. A
    number that lands on a town or on another number makes the map ambiguous, and an
    ambiguous map is not a hard scene, it is a broken one - so the map is refused rather
    than shipped with the collision.
    """
    taken = [(p[0] - _TOWN_R, p[1] - _TOWN_R, p[0] + _TOWN_R, p[1] + _TOWN_R)
             for p in pts]
    out = []
    for a, b, w in roads:
        pa, pb = pts[a], pts[b]
        ux, uy = _perp(pa, pb)
        half_w, half_h = 13 + 3 * (w >= 10), 11
        spot = None
        for t in (0.5, 0.38, 0.62, 0.3, 0.7):
            for side in (1, -1):
                cx, cy = _mid(pa, pb, t)
                cx, cy = cx + ux * _LABEL_OFF * side, cy + uy * _LABEL_OFF * side
                box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
                if box[0] < 2 or box[1] < 2 or box[2] > _CANVAS[0] - 2 \
                        or box[3] > _CANVAS[1] - 2:
                    continue
                if any(_boxes_overlap(box, t2) for t2 in taken):
                    continue
                spot = (cx, cy, box)
                break
            if spot:
                break
        if spot is None:
            return None
        taken.append(spot[2])
        out.append((a, b, w, spot[0], spot[1]))
    return out


def _draw(rm: RoadMap, labels, src: int, dst: int) -> Image.Image:
    img = Image.new("RGB", _CANVAS, _PAPER)
    d = ImageDraw.Draw(img)
    pts = [(t.x, t.y) for t in rm.towns]

    for a, b, _w in rm.roads:
        d.line([pts[a], pts[b]], fill=_ROAD_CASING, width=9)
    for a, b, _w in rm.roads:
        d.line([pts[a], pts[b]], fill=_ROAD_FILL, width=5)

    num_font = _font(19)
    for _a, _b, w, cx, cy in labels:
        text = str(w)
        tb = d.textbbox((0, 0), text, font=num_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 7, 4
        box = (cx - tw / 2 - pad_x, cy - th / 2 - pad_y,
               cx + tw / 2 + pad_x, cy + th / 2 + pad_y)
        d.rounded_rectangle(box, radius=6, fill=_PILL_FILL, outline=_ROAD_CASING, width=1)
        d.text((cx - tw / 2 - tb[0], cy - th / 2 - tb[1]), text, fill=_INK, font=num_font)

    town_font = _font(21)
    for t in rm.towns:
        x, y = t.x, t.y
        ring = _START if t.id == src else _GOAL if t.id == dst else _INK
        width = 5 if t.id in (src, dst) else 2
        d.ellipse((x - _TOWN_R, y - _TOWN_R, x + _TOWN_R, y + _TOWN_R),
                  fill=_TOWN_FILL, outline=ring, width=width)
        tb = d.textbbox((0, 0), t.label, font=town_font)
        d.text((x - (tb[2] - tb[0]) / 2 - tb[0], y - (tb[3] - tb[1]) / 2 - tb[1]),
               t.label, fill=_INK, font=town_font)
    return img


class _RouteScene:
    name = "route"
    params_schema = {
        # How hard the map is to READ. Nothing else depends on it.
        "n_towns": {"type": "int", "min": 6, "max": 10, "default": 8},
        # How hard the route is to WORK OUT: the cheapest route is exactly this many
        # roads long. Independent of n_towns, which is the point.
        "n_hops": {"type": "int", "min": 2, "max": 4, "default": 3},
    }

    def generate(self, params: dict, seed: int) -> Scene:
        n = int(params.get("n_towns", 8))
        hops = int(params.get("n_hops", 3))
        if not 6 <= n <= 10:
            raise ValueError(f"n_towns must be 6..10 (got {n}); "
                             "fewer towns cannot hold a 4-road route, more will not "
                             "space out on this canvas without the numbers colliding")
        rng = Random(seed * 7919 + n * 31 + hops)

        for _ in range(_TRIES):
            made = self._attempt(n, hops, rng)
            if made is not None:
                return self._scene(made, n, hops, params, seed)
        raise ValueError(
            f"no readable map with {n} towns and a {hops}-road cheapest route after "
            f"{_TRIES} tries. Lower n_hops or raise n_towns: a long route needs somewhere "
            "to go, and the numbers still have to fit beside the roads."
        )

    def _attempt(self, n: int, hops: int, rng: Random):
        pts = scatter(n, rng, _CANVAS[0], _CANVAS[1], _MARGIN, _GAP)
        if not pts:
            return None
        pairs = proximity_roads(pts, rng)
        roads = tuple((a, b, rng.randint(MIN_W, MAX_W)) for a, b in pairs)
        if not is_connected(n, roads):
            return None
        rm = RoadMap(tuple(Town(i, x, y) for i, (x, y) in enumerate(pts)), roads)

        candidates = []
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                best = dijkstra(rm, src, dst)
                if best is None or len(best[1]) != hops + 1:
                    continue
                if not optimum_is_unique(rm, src, dst):
                    continue
                if not greedy_is_wrong(rm, src, dst):
                    continue
                greedy = greedy_route_cost(rm, src, dst)
                # The trap only teaches us anything when it produces a DIFFERENT number:
                # if greedy happens to cost the same, a lazy model scores correct and the
                # wrong answer is invisible.
                if greedy is not None and greedy == best[0]:
                    continue
                candidates.append((src, dst, best, greedy))
        if not candidates:
            return None
        src, dst, best, greedy = rng.choice(candidates)
        labels = _place_labels(pts, roads, _font(19))
        if labels is None:
            return None
        return rm, labels, src, dst, best, greedy

    def _scene(self, made, n: int, hops: int, params: dict, seed: int) -> Scene:
        rm, labels, src, dst, (cost, route), greedy = made
        img = _draw(rm, labels, src, dst)
        question = (
            "The picture is a road map. Each circle is a town, labeled with a letter. "
            "Each line is a road, and the number beside a road is its length. "
            f"Traveling only along the roads, what is the total length of the SHORTEST "
            f"route from {LABELS[src]} to {LABELS[dst]}? Answer with a single number."
        )
        return Scene(
            id=f"route_n{n}_h{hops}_s{seed}",
            images=[img],
            ground_truth={
                "question": question,
                "answer": int(cost),
                "adjacency_text": adjacency_text(rm),
                "towns": [{"id": t.id, "label": t.label, "x": t.x, "y": t.y}
                          for t in rm.towns],
                "roads": [[a, b, w] for a, b, w in rm.roads],
                "src": src,
                "dst": dst,
                "route": list(route),
                "greedy_answer": greedy,
                "route_costs": route_costs(rm, src, dst),
            },
            factors={"n_towns": n, "n_hops": hops},
            meta={"generator": "route", "seed": seed, "params": dict(params),
                  "tier": "trivial" if hops <= 2 and n <= 6 else "standard"},
        )


PLUGIN = _RouteScene()
