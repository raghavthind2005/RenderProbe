"""The calibration gate's arithmetic (tools/calibrate.py).

The tool itself is a development aid and ships outside the package, but the numbers it
prints decide whether a scene gets a full run, so the math is pinned here. A wrong
chance level sends a floor scene to a hundred-call run, or kills a usable one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from calibrate import poisson_binomial_sf, schema_defaults, wilson  # noqa: E402


def test_equal_probabilities_match_the_binomial():
    # P(X >= 2) for 3 fair coins = 4/8
    assert poisson_binomial_sf([0.5, 0.5, 0.5], 2) == pytest.approx(0.5)
    # P(X >= 0) is everything
    assert poisson_binomial_sf([0.3, 0.7], 0) == pytest.approx(1.0)
    # P(X >= n) is the product
    assert poisson_binomial_sf([0.3, 0.7], 2) == pytest.approx(0.21)


def test_unequal_probabilities_are_not_averaged():
    """The reason this is a Poisson-binomial and not a binomial: a 4-candidate scene and
    a 6-candidate one have different floors, and using their mean misstates both."""
    mixed = [0.25, 0.25, 0.2, 0.2, 1 / 6, 1 / 6]
    mean = sum(mixed) / len(mixed)
    exact = poisson_binomial_sf(mixed, 3)
    naive = poisson_binomial_sf([mean] * len(mixed), 3)
    assert exact != pytest.approx(naive)


def test_the_polycube_run_reads_as_chance():
    """The 18-scene run: 3 correct where guessing predicts 3.7. Guessing explains it."""
    chances = [0.25] * 6 + [0.2] * 6 + [1 / 6] * 6
    assert sum(chances) == pytest.approx(3.7, abs=0.05)
    assert poisson_binomial_sf(chances, 3) > 0.5


def test_a_clear_result_separates_from_chance():
    chances = [0.2] * 18
    assert poisson_binomial_sf(chances, 12) < 0.001


def test_wilson_stays_inside_zero_and_one():
    """Where this tool lives - small n, proportions at the extremes - the textbook
    normal interval runs past the ends of the scale."""
    for k, n in ((0, 6), (6, 6), (1, 6), (3, 18)):
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(3, 18)
    assert lo < 3 / 18 < hi


def test_defaults_come_from_the_scene_schema():
    class _Gen:
        params_schema = {"n": {"min": 1, "max": 9, "default": 4},
                         "mode": {"choices": ["a", "b"]}}
    assert schema_defaults(_Gen()) == {"n": 4}
    assert schema_defaults(object()) == {}
