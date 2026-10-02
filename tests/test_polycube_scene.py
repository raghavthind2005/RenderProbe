"""Polycube scene and probe: the guarantees that make the question well posed.

The pure combinatorics live in test_polycube_core. These tests assert the properties
an actual generated scene has to carry - a unique answer, no shortcut that skips the
reasoning, a balanced answer distribution, and an oracle that provably contains enough
to answer.
"""
from __future__ import annotations

from collections import Counter

from renderprobe.probes.polycube import PLUGIN as PROBE
from renderprobe.scenes._polycube_core import (
    bounding_box,
    congruent,
    hidden_cells,
    toward_camera,
)
from renderprobe.scenes.polycube import PLUGIN as SCENE

_VIEW = toward_camera(32.0, 17.0)


def _scene(seed=0, k=5, grid=3):
    # pil_2d keeps these tests fast and free of the optional 3D dependency; the
    # properties under test are all renderer-independent.
    return SCENE.generate({"n_pieces": k, "grid": grid, "renderer": "pil_2d"}, seed)


def _cells(seq):
    return [tuple(c) for c in seq]


# ---------------------------------------------------------------------------
# The question has exactly one right answer
# ---------------------------------------------------------------------------

def test_the_answer_matches_the_target():
    for seed in range(8):
        s = _scene(seed)
        gt = s.ground_truth
        answer = next(c for c in gt["candidates"] if c["color"] == gt["answer"])
        assert congruent(_cells(answer["cells"]), _cells(gt["target_cells"]))


def test_no_other_candidate_matches():
    for seed in range(8):
        s = _scene(seed)
        gt = s.ground_truth
        also = [c["color"] for c in gt["candidates"]
                if c["color"] != gt["answer"]
                and congruent(_cells(c["cells"]), _cells(gt["target_cells"]))]
        assert not also, f"seed {seed}: {also} also match - the question is ambiguous"


def test_the_target_is_genuinely_rotated():
    # If the target were shown in the candidate's own orientation the task would be
    # picture matching, not mental rotation.
    rotated = 0
    for seed in range(8):
        gt = _scene(seed).ground_truth
        answer = next(c for c in gt["candidates"] if c["color"] == gt["answer"])
        if sorted(_cells(answer["cells"])) != sorted(_cells(gt["target_cells"])):
            rotated += 1
    assert rotated == 8


# ---------------------------------------------------------------------------
# No shortcut that skips the reasoning
# ---------------------------------------------------------------------------

def test_counting_cubes_cannot_identify_the_answer():
    for seed in range(10):
        gt = _scene(seed).ground_truth
        sizes = Counter(c["n_cubes"] for c in gt["candidates"])
        answer = next(c for c in gt["candidates"] if c["color"] == gt["answer"])
        assert sizes[answer["n_cubes"]] >= 2, (
            f"seed {seed}: the answer is the only piece with {answer['n_cubes']} cubes, "
            f"so counting alone would settle it")


def test_count_and_bounding_box_together_cannot_identify_the_answer():
    # Each measure alone is already covered. Applying BOTH is the stronger attack: a
    # piece unique in the combination would be separable without any rotation.
    for seed in range(12):
        gt = _scene(seed).ground_truth
        target = _cells(gt["target_cells"])
        key = (len(target), bounding_box(target))
        same = [c for c in gt["candidates"]
                if (c["n_cubes"], bounding_box(_cells(c["cells"]))) == key]
        assert len(same) >= 2, (
            f"seed {seed}: the answer is the only candidate with {key}, so counting "
            f"and measuring together would settle it without rotating anything")


def test_the_target_carries_no_color_information():
    # Drawing the target in its own color would give the answer away outright.
    for seed in range(6):
        s = _scene(seed)
        target_objs = [o for o in s.graph.objects if o.id.startswith("target")]
        assert target_objs
        assert {o.color for o in target_objs} == {"gray"}
        assert s.ground_truth["answer"] != "gray"


def test_answer_colors_are_balanced_across_seeds():
    counts = Counter(_scene(seed).ground_truth["answer"] for seed in range(24))
    assert len(counts) == 6
    assert set(counts.values()) == {4}, f"unbalanced: {counts}"


# ---------------------------------------------------------------------------
# Answerable from the image
# ---------------------------------------------------------------------------

def test_every_piece_is_drawn_fully_visible():
    for seed in range(8):
        gt = _scene(seed).ground_truth
        for c in gt["candidates"]:
            assert hidden_cells(_cells(c["cells"]), _VIEW) == [], \
                f"seed {seed}: {c['color']} has a cube no camera ray reaches"
        assert hidden_cells(_cells(gt["target_cells"]), _VIEW) == []


# ---------------------------------------------------------------------------
# Reproducibility of the specification
# ---------------------------------------------------------------------------

def test_same_seed_gives_the_same_scene_specification():
    for seed in (0, 3, 7):
        a, b = _scene(seed), _scene(seed)
        assert a.ground_truth == b.ground_truth
        assert [(o.id, o.color, o.position, o.size) for o in a.graph.objects] == \
               [(o.id, o.color, o.position, o.size) for o in b.graph.objects]


def test_bulky_pieces_are_never_shown_unreadable():
    # 3 pieces on a 3-cube gives 9-cube lumps, and many of those have no orientation
    # showing every cube. The scene must either find a decomposition it can show or
    # refuse - what it must never do is draw a piece the model cannot read.
    for seed in range(6):
        try:
            s = SCENE.generate({"n_pieces": 3, "grid": 3, "renderer": "pil_2d"}, seed)
        except ValueError:
            continue                      # refused, which is the honest outcome
        for c in s.ground_truth["candidates"]:
            assert hidden_cells(_cells(c["cells"]), _VIEW) == []


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_oracle_is_information_complete():
    # A reference reasoner re-derives the answer from ONLY what the oracle states, so
    # a model's oracle failure is reasoning rather than missing information.
    for seed in range(10):
        s = _scene(seed)
        assert PROBE.oracle_solver(s) == PROBE.ground_truth(s)


def test_oracle_offers_several_verbalizations():
    variants = PROBE.oracle_prompt_variants(_scene(0))
    assert set(variants) == {"coordinates", "prose", "json"}
    for text in variants.values():
        assert "rotation" in text.lower()


def test_oracle_prompt_never_states_the_answer():
    for seed in range(6):
        s = _scene(seed)
        answer = s.ground_truth["answer"]
        # the answer color appears only as one label among the candidates
        for name, text in PROBE.oracle_prompt_variants(s).items():
            assert text.lower().count(answer) == 1, f"{name} mentions {answer} twice"


def test_parse_takes_the_last_color_mentioned():
    assert PROBE.parse("could be red or blue, but I say green") == "green"
    assert PROBE.parse("PURPLE") == "purple"
    assert PROBE.parse("not sure") is None


def test_score_is_exact_match():
    assert PROBE.score("red", "red")[0] is True
    assert PROBE.score("red", "blue")[0] is False
    assert PROBE.score(None, "blue")[0] is False


def test_perception_report_asks_a_pure_perception_question():
    s = _scene(0)
    spec = next(r for r in PROBE.perception_report(s) if r.name == "cubes")
    assert spec is not None
    named = [c for c in s.ground_truth["candidates"] if c["color"] in spec.question]
    assert len(named) == 1
    assert spec.ground_truth == named[0]["n_cubes"]
    assert spec.parse("there are 6 cubes") == 6
    assert spec.score(6, 6)[0] is True


def test_question_names_every_candidate_and_no_others():
    """The answer set has to be explicit in the prompt. It is what makes the balanced
    answer colors a substitute for a text-only baseline, and without it a model can
    answer with the target's own color, which is never one of the choices."""
    for seed in range(6):
        s = _scene(seed)
        q = PROBE.question(s).lower()
        colors = [c["color"] for c in s.ground_truth["candidates"]]
        for color in colors:
            assert color in q, f"seed {seed}: candidate {color} not named"
        # gray names the target, never a candidate, so it must not read as a choice
        assert "gray" not in colors
        assert q.count("gray") == 1


def test_parse_folds_grey_onto_gray():
    """A spelling difference must not decide a score. The color only ever names the
    target, so it stays a real (and always wrong) prediction rather than becoming
    None, which would make a misread question look like no answer at all."""
    assert PROBE.parse("Grey.") == "gray"
    assert PROBE.parse("gray") == "gray"
    assert PROBE.score(PROBE.parse("Grey."), "red")[0] is False


# ---------------------------------------------------------------------------
# Perception reports
# ---------------------------------------------------------------------------

def _cells_report(seed=0):
    s = _scene(seed)
    return s, next(r for r in PROBE.perception_report(s) if r.name == "cells")


def test_exactly_one_report_is_primary():
    """Only the primary feeds acc_report. Pooling a tight question with a loose one
    would make that number a blend of two difficulties and a measure of neither."""
    for seed in range(4):
        specs = PROBE.perception_report(_scene(seed))
        assert [sp.name for sp in specs] == ["cells", "cubes"]
        assert sum(sp.primary for sp in specs) == 1
        assert specs[0].primary is True


def test_cells_report_asks_for_what_the_oracle_is_given():
    """The tight report is only a bound on perception if it asks for the same facts the
    oracle hands over: the target's cubes, and all of them."""
    s, spec = _cells_report()
    assert spec.ground_truth == s.ground_truth["target_cells"]
    assert str(len(spec.ground_truth)) in spec.question
    # the frame has to be pinned, or a left-handed reading mirrors the shape and
    # mirroring is exactly what this task refuses to count as a match
    q = spec.question.lower()
    assert "right" in q and "upward" in q and "toward the viewer" in q


def test_cells_report_credits_any_frame_the_model_chooses():
    """A reported shape is scored by congruence, not equality: the model picks its own
    origin and axes, and getting the orientation right is the reasoning half of the
    task, not the perceptual half."""
    from renderprobe.scenes._polycube_core import ROTATIONS, _apply, normalize
    s, spec = _cells_report()
    target = tuple(tuple(c) for c in spec.ground_truth)
    for rot in ROTATIONS:
        moved = normalize(_apply(rot, c) for c in target)
        shifted = tuple((x + 4, y - 3, z + 2) for x, y, z in moved)
        ok, score, _ = spec.score(shifted, spec.ground_truth)
        assert ok is True and score == 1.0


def test_cells_report_scores_a_wrong_shape_wrong():
    s, spec = _cells_report()
    target = [tuple(c) for c in spec.ground_truth]
    off_by_one = tuple(target[:-1]) + ((9, 9, 9),)
    ok, score, _ = spec.score(off_by_one, spec.ground_truth)
    assert ok is False
    assert 0.0 < score < 1.0, "a near miss should score between noise and exact"
    assert spec.score(None, spec.ground_truth)[0] is False


def test_overlap_separates_a_near_miss_from_noise():
    """`correct` is all-or-nothing because the task is. The graded score is what
    distinguishes a percept that missed by a cube from one that saw nothing."""
    s, spec = _cells_report()
    target = [tuple(c) for c in spec.ground_truth]
    near = spec.score(tuple(target[:-1]), spec.ground_truth)[1]
    noise = spec.score(((0, 0, 0), (7, 7, 7), (9, 0, 9)), spec.ground_truth)[1]
    assert near > noise


def test_cells_parser_reads_the_formats_models_actually_emit():
    from renderprobe.probes.polycube import _parse_cells
    assert _parse_cells("(0,0,0), (0,0,1)") == ((0, 0, 0), (0, 0, 1))
    assert _parse_cells("[0, 0, 0]\n[0, 1, 0]") == ((0, 0, 0), (0, 1, 0))
    assert _parse_cells("cubes at 0 0 0, 0 0 1") == ((0, 0, 0), (0, 0, 1))
    assert _parse_cells("I cannot tell") is None
    # a model looping forever is not a percept; refuse rather than invent one
    assert _parse_cells(" ".join(f"({i},0,0)" for i in range(40))) is None
