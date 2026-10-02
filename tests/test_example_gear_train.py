"""The worked on-ramp example (examples/gear_train.py).

It ships as the thing a newcomer copies, so it is held to the same bar as an in-tree
scene: the arithmetic it teaches has to be right, and the trap it advertises has to
actually be set.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from gear_train import PLUGINS, simple_train_turns, turns_of  # noqa: E402

SCENE, PROBE = PLUGINS


def _scene(seed=0, stages=2):
    return SCENE.generate({"n_stages": stages}, seed)


def test_a_plain_train_telescopes_to_first_over_last():
    """Two single-gear shafts: the textbook case, and the one the shortcut gets right."""
    shafts = [(12, None), (8, None)]
    assert turns_of(shafts, 4) == pytest.approx(6.0)
    assert simple_train_turns(shafts, 4) == pytest.approx(6.0)


def test_a_compound_shaft_breaks_the_shortcut():
    """The whole point of the scene. Bolting a second gear to the middle shaft makes the
    stage ratios multiply, and first-over-last stops being the answer."""
    shafts = [(20, None), (10, 30), (15, None)]
    assert turns_of(shafts, 3) == pytest.approx(3 * 20 / 10 * 30 / 15)
    assert simple_train_turns(shafts, 3) == pytest.approx(3 * 20 / 15)
    assert turns_of(shafts, 3) != simple_train_turns(shafts, 3)


@pytest.mark.parametrize("seed", range(6))
def test_the_shortcut_never_lands_on_the_right_answer(seed):
    """A trap that happens to give the correct number teaches nothing, so such a scene
    is rejected at generation."""
    gt = _scene(seed).ground_truth
    assert gt["simple_train_answer"] != gt["answer"]


@pytest.mark.parametrize("seed", range(6))
def test_the_answer_is_a_whole_number_of_turns(seed):
    """Exactness is what lets score be a plain equality rather than a tolerance."""
    gt = _scene(seed).ground_truth
    assert isinstance(gt["answer"], int)
    shafts = [(a, b) for a, b in gt["shafts"]]
    assert turns_of(shafts, gt["driver_turns"]) == gt["answer"]


@pytest.mark.parametrize("seed", range(6))
def test_the_oracle_re_derives_the_answer_from_its_own_text(seed):
    s = _scene(seed)
    assert PROBE.oracle_solver(s) == s.ground_truth["answer"]


def test_only_the_middle_shafts_are_compound():
    """The end shafts carry one gear each: the first is simply driven and the last
    simply reports, so neither end is ambiguous about which gear meshes."""
    for stages in (1, 2, 3):
        shafts = _scene(0, stages).ground_truth["shafts"]
        assert len(shafts) == stages + 1
        assert shafts[0][1] is None and shafts[-1][1] is None
        assert all(s[1] is not None for s in shafts[1:-1])


def test_guessing_is_weak():
    for seed in range(6):
        assert PROBE.chance_level(_scene(seed)) < 0.15


def test_the_report_asks_for_a_gear_the_answer_depends_on():
    for seed in range(4):
        s = _scene(seed)
        spec = PROBE.perception_report(s)
        assert spec.ground_truth == max(x for x in s.ground_truth["shafts"][0]
                                        if x is not None)
        assert spec.parse("it has 24 teeth") == 24


def test_an_unbuildable_train_is_refused_not_faked():
    with pytest.raises(ValueError, match="n_stages"):
        SCENE.generate({"n_stages": 9}, 0)


def test_it_loads_the_way_a_third_party_would():
    """By path, into a registry that has never heard of it, with no install step."""
    from renderprobe.core.registry import Registry
    reg = Registry()
    reg.autodiscover()
    assert "gear_train" not in reg.list_scenes()
    loaded = reg.load_external(Path(__file__).resolve().parent.parent / "examples")
    assert "scene:gear_train" in loaded and "probe:gear_train" in loaded
    assert "gear_train" in reg.list_scenes()
