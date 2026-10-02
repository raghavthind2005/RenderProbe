"""The road map's logic: shortest routes, uniqueness, and the greedy trap.

Correctness lives here, not in the scene, so it can be checked without rendering.
"""
import pytest

from renderprobe.scenes import _route_core as C


def _rm(roads, n=None):
    n = n or (max(max(a, b) for a, b, _ in roads) + 1)
    return C.RoadMap(tuple(C.Town(i, float(i), 0.0) for i in range(n)), tuple(roads))


def test_dijkstra_prefers_the_longer_cheaper_way():
    """The whole task in one map: A-B-C costs 3, the direct A-C costs 10."""
    rm = _rm([(0, 1, 1), (1, 2, 2), (0, 2, 10)])
    cost, path = C.dijkstra(rm, 0, 2)
    assert (cost, path) == (3, [0, 1, 2])


def test_dijkstra_reports_no_route_rather_than_a_wrong_one():
    assert C.dijkstra(_rm([(0, 1, 4), (2, 3, 4)], n=4), 0, 3) is None


def test_a_tie_is_not_a_unique_optimum():
    """Two ways at the same cost. The cost answer survives it, but a question about
    WHICH way does not, so such a map is refused rather than shipped."""
    rm = _rm([(0, 1, 5), (1, 3, 5), (0, 2, 5), (2, 3, 5)], n=4)
    assert C.dijkstra(rm, 0, 3)[0] == 10
    assert C.optimum_is_unique(rm, 0, 3) is False
    # break the tie and it becomes shippable
    rm2 = _rm([(0, 1, 5), (1, 3, 5), (0, 2, 5), (2, 3, 6)], n=4)
    assert C.optimum_is_unique(rm2, 0, 3) is True


def test_the_trap_is_that_the_cheapest_road_out_is_the_wrong_one():
    """Without this a model can walk greedily and never compare whole routes, and the
    scene measures map reading with the reasoning taken out."""
    # cheapest road from A is A-C (1), but that way costs 1+20; A-B-D-... costs less
    rm = _rm([(0, 2, 1), (2, 3, 20), (0, 1, 4), (1, 3, 5)], n=4)
    assert C.greedy_first_hop(rm, 0) == 2
    assert C.dijkstra(rm, 0, 3)[1] == [0, 1, 3]
    assert C.greedy_is_wrong(rm, 0, 3) is True


def test_greedy_names_the_wrong_answer_a_lazy_model_would_give():
    rm = _rm([(0, 2, 1), (2, 3, 20), (0, 1, 4), (1, 3, 5)], n=4)
    assert C.greedy_route_cost(rm, 0, 3) == 21     # 1 + 20, having taken the bait
    assert C.dijkstra(rm, 0, 3)[0] == 9


def test_greedy_gives_up_on_a_dead_end():
    rm = _rm([(0, 1, 1), (0, 2, 9), (2, 3, 1)], n=4)   # greedy walks into B and stops
    assert C.greedy_route_cost(rm, 0, 3) is None


def test_route_costs_counts_costs_not_routes():
    """Several routes can share a cost and any of them answers correctly, so counting
    routes would overstate how hard guessing is."""
    rm = _rm([(0, 1, 5), (1, 3, 5), (0, 2, 5), (2, 3, 5)], n=4)
    assert C.route_costs(rm, 0, 3) == [10]


def test_roads_never_cross():
    """A crossing is ambiguous: the picture cannot say whether the two roads meet."""
    rng = __import__("random").Random(3)
    for _ in range(12):
        pts = C.scatter(9, rng, 760, 480, 54, 120)
        if not pts:
            continue
        roads = C.proximity_roads(pts, rng)
        for i, (a, b) in enumerate(roads):
            for c, d in roads[i + 1:]:
                if len({a, b, c, d}) == 4:
                    assert not C._segments_cross(pts[a], pts[b], pts[c], pts[d])


def test_scattered_towns_keep_their_distance():
    rng = __import__("random").Random(0)
    pts = C.scatter(8, rng, 760, 480, 54, 130)
    assert len(pts) == 8
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            assert ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 >= 130 - 1e-6


def test_adjacency_text_states_every_road_and_no_answer():
    rm = _rm([(0, 1, 7), (1, 2, 3)])
    text = C.adjacency_text(rm)
    assert "A - B: 7" in text and "B - C: 3" in text
    assert str(C.dijkstra(rm, 0, 2)[0]) not in text.replace("7", "").replace("3", "")


def test_connectivity():
    assert C.is_connected(3, [(0, 1, 1), (1, 2, 1)]) is True
    assert C.is_connected(4, [(0, 1, 1), (2, 3, 1)]) is False


@pytest.mark.parametrize("weight", [C.MIN_W, C.MAX_W])
def test_no_road_is_free(weight):
    """A zero-length road would let a detour tie a direct hop for nothing."""
    assert weight >= 2
