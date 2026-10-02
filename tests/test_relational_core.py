"""Correctness tests for the compositional relational core logic (no rendering).

The keystone is `test_generated_questions_correct_end_to_end`: over hundreds of random
scenes it builds chains, renders them to natural language, parses that text back, and
re-resolves over the true scene graph - asserting the answer is exactly the intended
target. If any of grammar / parser / solver / builder disagree, this fails.
"""
from __future__ import annotations

from random import Random

from tests.fixtures._relational_core import (
    COLORS,
    Chain,
    Obj,
    answer_of,
    build_chain,
    parse_question,
    parse_scene_graph_text,
    render_question,
    resolve,
    scene_graph_to_text,
    unique_descriptor,
)


def _random_scene(rng: Random, n: int) -> list[Obj]:
    """n objects with DISTINCT x and DISTINCT y (tie-free extremes)."""
    xs = rng.sample(range(20, 460, 5), n)
    ys = rng.sample(range(20, 340, 5), n)
    return [
        Obj(color=rng.choice(COLORS), shape=rng.choice(["circle", "square", "triangle"]),
            size=rng.choice(["small", "large"]), x=xs[i], y=ys[i])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# resolve() semantics
# ---------------------------------------------------------------------------

def test_resolve_relations():
    objs = [
        Obj("red", "circle", "small", 100, 100),
        Obj("blue", "square", "large", 200, 100),
        Obj("green", "circle", "small", 300, 100),
    ]
    # anchor the (unique) blue square, then the circle left of it -> red circle
    c = Chain(anchor=("attr", "blue", "square"), hops=(("left of", "circle", ()),))
    assert answer_of(objs, c) == "red"
    # the circle right of the red circle -> green circle
    c2 = Chain(anchor=("attr", "red", "circle"), hops=(("right of", "circle", ()),))
    assert answer_of(objs, c2) == "green"


def test_resolve_above_below():
    objs = [
        Obj("red", "circle", "small", 100, 50),    # higher (smaller y)
        Obj("blue", "circle", "small", 100, 250),  # lower
        Obj("green", "square", "large", 300, 150),
    ]
    c = Chain(anchor=("attr", "green", "square"), hops=(("above", "circle", ()),))
    assert answer_of(objs, c) == "red"
    c2 = Chain(anchor=("attr", "green", "square"), hops=(("below", "circle", ()),))
    assert answer_of(objs, c2) == "blue"


def test_resolve_ambiguous_returns_none():
    objs = [
        Obj("red", "circle", "small", 100, 100),
        Obj("blue", "circle", "small", 150, 100),   # two circles left of the square
        Obj("green", "square", "large", 300, 100),
    ]
    c = Chain(anchor=("attr", "green", "square"), hops=(("left of", "circle", ()),))
    assert resolve(objs, c) is None  # ambiguous hop -> no answer


def test_resolve_nonunique_anchor_returns_none():
    objs = [
        Obj("red", "circle", "small", 100, 100),
        Obj("red", "circle", "small", 200, 200),  # duplicate color+shape
    ]
    assert resolve(objs, Chain(anchor=("attr", "red", "circle"), hops=())) is None


def test_extreme_anchor():
    objs = [
        Obj("red", "circle", "small", 100, 100),
        Obj("blue", "circle", "small", 300, 100),
    ]
    assert resolve(objs, Chain(anchor=("extreme", "leftmost", "circle"), hops=())).color == "red"
    assert resolve(objs, Chain(anchor=("extreme", "rightmost", "circle"), hops=())).color == "blue"


# ---------------------------------------------------------------------------
# unique_descriptor
# ---------------------------------------------------------------------------

def test_unique_descriptor_resolves_back():
    objs = _random_scene(Random(1), 8)
    for o in objs:
        desc = unique_descriptor(objs, o)
        if desc is not None:
            assert resolve(objs, Chain(anchor=desc, hops=())) is o


# ---------------------------------------------------------------------------
# round-trips
# ---------------------------------------------------------------------------

def test_question_render_parse_roundtrip():
    chain = Chain(
        anchor=("extreme", "leftmost", "triangle"),
        hops=(
            ("above", "circle", ()),
            ("right of", "square", (("color", "red"),)),
        ),
    )
    parsed = parse_question(render_question(chain))
    assert parsed == chain


def test_scene_graph_text_roundtrip():
    objs = _random_scene(Random(3), 8)
    assert set(parse_scene_graph_text(scene_graph_to_text(objs))) == set(objs)


# ---------------------------------------------------------------------------
# build_chain - correct by construction, target respected, depth respected
# ---------------------------------------------------------------------------

def test_build_chain_respects_target_and_depth():
    objs = _random_scene(Random(5), 12)
    present = {o.color for o in objs}
    for color in present:
        chain = build_chain(objs, Random(42), depth=2, target_color=color)
        if chain is None:
            continue
        assert len(chain.hops) == 2
        assert answer_of(objs, chain) == color


def test_build_chain_deterministic():
    objs = _random_scene(Random(7), 10)
    a = build_chain(objs, Random(9), depth=2, target_color="red")
    b = build_chain(objs, Random(9), depth=2, target_color="red")
    assert a == b


def test_generated_questions_correct_end_to_end():
    """chain -> NL -> parse -> resolve over the TRUE graph must equal the target."""
    meta = Random(0)
    built = 0
    for trial in range(300):
        objs = _random_scene(Random(trial), n=meta.randint(6, 12))
        depth = meta.choice([1, 2, 3])
        target = meta.choice(COLORS)
        chain = build_chain(objs, Random(trial * 7 + 1), depth, target)
        if chain is None:
            continue
        built += 1
        # correctness over the true graph
        assert answer_of(objs, chain) == target
        assert len(chain.hops) == depth
        # NL round-trips and the parsed question resolves to the same answer
        parsed = parse_question(render_question(chain))
        assert parsed is not None
        assert answer_of(objs, parsed) == target
    assert built > 80, f"only built {built} chains - generator too weak"
