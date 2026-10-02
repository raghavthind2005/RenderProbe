"""The road map scene and probe: every map it ships has to be answerable and fair."""
import re

import pytest

from renderprobe.core.registry import Registry
from renderprobe.probes.route import _map_from_gt
from renderprobe.scenes._route_core import greedy_is_wrong, optimum_is_unique

_REG = Registry()
_REG.autodiscover()
GEN = _REG.get_scene("route")
PROBE = _REG.get_probe("route")


def _scene(seed=0, n=8, h=3):
    return GEN.generate({"n_towns": n, "n_hops": h}, seed)


@pytest.mark.parametrize("seed", range(6))
def test_exactly_one_cheapest_route(seed):
    s = _scene(seed)
    gt = s.ground_truth
    assert optimum_is_unique(_map_from_gt(gt), gt["src"], gt["dst"])


@pytest.mark.parametrize("seed", range(6))
def test_the_cheapest_looking_first_road_is_never_the_right_one(seed):
    """The reasoning content of the task. Without it a model walks greedily and never
    has to compare whole routes."""
    s = _scene(seed)
    gt = s.ground_truth
    assert greedy_is_wrong(_map_from_gt(gt), gt["src"], gt["dst"])


@pytest.mark.parametrize("seed", range(6))
def test_the_route_is_as_long_as_asked_for(seed):
    """n_hops is the reasoning knob, so it has to actually hold."""
    assert len(_scene(seed, h=3).ground_truth["route"]) == 4
    assert len(_scene(seed, h=2).ground_truth["route"]) == 3


def test_reading_the_map_harder_does_not_make_the_route_longer():
    """The two knobs are independent, which is what polycube lacked: there, one
    parameter drove both difficulties and no setting put the task in the measurable
    band."""
    for n in (6, 8, 10):
        assert len(_scene(0, n=n, h=3).ground_truth["route"]) == 4


@pytest.mark.parametrize("seed", range(6))
def test_guessing_is_worth_almost_nothing(seed):
    """The complaint that started this scene: a yes/no question gives away half the
    benchmark. Here a guesser who has read the map perfectly still has to pick the
    cheapest of many costs."""
    s = _scene(seed)
    assert PROBE.chance_level(s) < 0.15
    assert len(s.ground_truth["route_costs"]) > 6


@pytest.mark.parametrize("seed", range(6))
def test_the_oracle_re_derives_the_answer_from_the_roads_alone(seed):
    s = _scene(seed)
    assert PROBE.oracle_solver(s) == s.ground_truth["answer"]


def test_the_oracle_never_states_the_answer():
    for seed in range(4):
        s = _scene(seed)
        for name, text in PROBE.oracle_prompt_variants(s).items():
            body = text.split("what is the total length")[0]
            assert "answer is" not in body.lower(), name


def test_the_report_asks_for_a_road_the_route_actually_uses():
    """A primitive off to the side of the task would not bound anything: getting this
    road wrong is on its own enough to get the task wrong."""
    for seed in range(6):
        s = _scene(seed)
        gt = s.ground_truth
        spec = PROBE.perception_report(s)
        assert spec is not None
        route = gt["route"]
        used = {tuple(sorted(pair)) for pair in zip(route, route[1:])}
        letters = re.findall(r"town ([A-L])", spec.question)
        assert len(letters) == 2, spec.question
        a, b = sorted(ord(c) - 65 for c in letters)
        assert (a, b) in used
        assert spec.ground_truth == next(w for x, y, w in gt["roads"] if (x, y) == (a, b))


def test_parse_takes_the_total_not_the_first_leg():
    """A model that shows its working ends on the total: '6 + 7 + 11 = 24'."""
    assert PROBE.parse("6 + 7 + 11 = 24") == 24
    assert PROBE.parse("The shortest route is 17.") == 17
    assert PROBE.parse("no idea") is None


def test_score_keeps_how_far_off_a_wrong_answer_was():
    """Separates a model that routed badly and paid 25 instead of 24 from one that
    answered 3."""
    assert PROBE.score(24, 24) == (True, 1.0, 0.0)
    ok, _, err = PROBE.score(25, 24)
    assert ok is False and err == 1.0
    assert PROBE.score(None, 24)[0] is False


def test_an_unbuildable_map_is_refused_not_faked():
    with pytest.raises(ValueError, match="n_towns"):
        GEN.generate({"n_towns": 3, "n_hops": 2}, 0)


def test_answers_are_spread_across_seeds():
    """A dominant answer would let a model score without reading anything."""
    answers = [_scene(s).ground_truth["answer"] for s in range(12)]
    assert len(set(answers)) >= 7
