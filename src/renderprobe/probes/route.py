"""Road map probe: the total length of the cheapest route between two towns.

The answer is an integer, which is what keeps this off the chance floor. A yes/no
question hands a model half the benchmark for free; here `chance_level` is one over the
number of distinct costs the map's routes admit, usually under two percent.

The three conditions separate cleanly on this task, which is why it exists:

* ORACLE. The roads and their lengths as text. That is the whole map, so nothing about
  the task resists being written down and a failure under it is reasoning. `oracle_solver`
  re-runs Dijkstra over only the stated roads, which proves it.
* PERCEPTION REPORT. The length of one named road, read straight off the picture. No
  routing, no comparison, no arithmetic. It is exactly the primitive the task consumes,
  so acc_report near zero indicts encoding and acc_report high with acc_full low means
  the model read the map and failed to use it.
"""
from __future__ import annotations

import re

from renderprobe.core.schema import ReportSpec, Scene
from renderprobe.scenes._route_core import LABELS, RoadMap, Town, dijkstra

_INT_RE = re.compile(r"-?\d+")


def _map_from_gt(gt: dict) -> RoadMap:
    """Rebuild the map from the stated facts alone, never from the stored answer."""
    towns = tuple(Town(t["id"], float(t["x"]), float(t["y"])) for t in gt["towns"])
    roads = tuple((int(a), int(b), int(w)) for a, b, w in gt["roads"])
    return RoadMap(towns, roads)


def _parse_int(raw: str):
    """Last integer wins.

    A model that shows its working ends on the total: "6 + 7 + 11 = 24". Taking the first
    number would score the first leg of the route instead of the answer.
    """
    found = _INT_RE.findall(raw or "")
    return int(found[-1]) if found else None


class _RouteProbe:
    name = "route"
    answer_type = "numeric"
    requires_gt_fields = ["question", "answer", "adjacency_text", "roads", "src", "dst"]

    def question(self, scene: Scene) -> str:
        return scene.ground_truth["question"]

    def ground_truth(self, scene: Scene) -> int:
        return int(scene.ground_truth["answer"])

    def parse(self, raw: str):
        return _parse_int(raw)

    def score(self, pred, gt) -> tuple[bool, float, float | None]:
        """Exact match, with the distance kept.

        `error` carries how far off a wrong answer was, which separates a model that
        routed badly and paid 25 instead of 24 from one that answered 3. The cascade
        only reads `correct`; the distance is there for anyone reading the results.
        """
        if pred is None:
            return False, 0.0, None
        ok = int(pred) == int(gt)
        return ok, 1.0 if ok else 0.0, float(abs(int(pred) - int(gt)))

    def chance_level(self, scene: Scene) -> float:
        """One over the number of distinct costs the map's routes admit.

        The model of a guesser this assumes is generous to the model: someone who has
        read the map perfectly and picks a route without optimising. Guessing an integer
        with no map at all would score far lower, so this is an upper bound on chance and
        the honest one to gate on.
        """
        costs = scene.ground_truth.get("route_costs") or []
        return 1.0 / len(costs) if costs else 0.0

    # -- oracle ------------------------------------------------------------
    def oracle_prompt(self, scene: Scene) -> str | None:
        return self.oracle_prompt_variants(scene)["adjacency"]

    def oracle_prompt_variants(self, scene: Scene) -> dict[str, str]:
        gt = scene.ground_truth
        src, dst = LABELS[gt["src"]], LABELS[gt["dst"]]
        tail = (f"Traveling only along these roads, what is the total length of the "
                f"shortest route from {src} to {dst}? Answer with a single number.")
        rows = gt["adjacency_text"]
        return {
            "adjacency": (
                "Towns are joined by roads. Each line below is a road and its length.\n\n"
                f"{rows}\n\n{tail}"
            ),
            "prose": (
                "There are roads between towns with these lengths: "
                + "; ".join(f"{LABELS[a]} to {LABELS[b]} is {w}"
                            for a, b, w in gt["roads"])
                + ". " + tail
            ),
            "json": (
                '{"roads": '
                + str([[LABELS[a], LABELS[b], w] for a, b, w in gt["roads"]])
                + "}\n\n" + tail
            ),
        }

    def oracle_solver(self, scene: Scene) -> int | None:
        """Re-derive the answer from ONLY the roads the oracle states.

        Never reads the stored answer. Agreeing with ground truth across seeds is what
        proves the oracle is information-complete, so a model failing under it failed at
        reasoning rather than for want of a fact.
        """
        gt = scene.ground_truth
        found = dijkstra(_map_from_gt(gt), int(gt["src"]), int(gt["dst"]))
        return None if found is None else int(found[0])

    # -- perception report -------------------------------------------------
    def perception_report(self, scene: Scene) -> ReportSpec | None:
        """Read one road's length off the picture. No routing, no arithmetic.

        The road is one the cheapest route actually uses, so this is not a primitive off
        to the side of the task: getting it wrong is enough on its own to get the task
        wrong. Both towns are named by their letters, which the image already carries, so
        nothing has to be localized by pixel coordinate.
        """
        gt = scene.ground_truth
        route = gt.get("route") or []
        if len(route) < 2:
            return None
        rm = _map_from_gt(gt)
        # The middle road of the route: the ends touch the two marked towns, which are
        # ringed in color, and a marked town is easier to find than a plain one.
        a, b = route[len(route) // 2 - 1], route[len(route) // 2]
        length = rm.length_of(a, b)
        if length is None:
            return None
        return ReportSpec(
            name="road_length",
            question=(f"On the map, what number is written beside the road that joins "
                      f"town {LABELS[a]} and town {LABELS[b]}? Answer with a single "
                      "number."),
            ground_truth=int(length),
            parse=_parse_int,
            score=lambda p, g: (p is not None and int(p) == int(g),
                                1.0 if (p is not None and int(p) == int(g)) else 0.0,
                                None if p is None else float(abs(int(p) - int(g)))),
        )


PLUGIN = _RouteProbe()
