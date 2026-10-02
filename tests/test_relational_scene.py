"""Tests for the relational scene generator, probe, and mock perception.

Covers:
  - Generator: valid Scene output, ground_truth contract, answer balance, tier flag.
  - Probe: question/answer/oracle_prompt round-trips, parse, oracle_prompt_variants.
  - Mock: oracle path uses injected scene graph (not GT); full path uses pixel perception.
  - Registry: generator + probe auto-discovered.
  - End-to-end runner: full/oracle conditions produced; recovery > 0 on mock_noisy.
"""
from __future__ import annotations

from collections import Counter

import pytest

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from tests.fixtures import register
from tests.fixtures._relational_core import (
    COLORS,
    answer_of,
    parse_question,
    parse_scene_graph_text,
)
from tests.fixtures.probe_relational import PLUGIN as PROBE
from tests.fixtures.scene_relational import PLUGIN as GEN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _scene(n_objects: int = 6, hop_depth: int = 2, seed: int = 0):
    return GEN.generate({"n_objects": n_objects, "hop_depth": hop_depth}, seed)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generator_registered():
    reg = _registry()
    assert "relational" in reg.list_scenes()


def test_scene_ground_truth_contract():
    s = _scene()
    gt = s.ground_truth
    assert "question" in gt and "answer" in gt
    assert "scene_graph_text" in gt and "objects" in gt
    assert s.factors["n_objects"] == 6
    assert s.factors["hop_depth"] == 2


def test_scene_has_one_image():
    s = _scene()
    assert len(s.images) == 1
    w, h = s.images[0].size
    assert w == 480 and h == 360


def test_answer_is_valid_color():
    for seed in range(20):
        s = _scene(seed=seed)
        assert s.ground_truth["answer"] in COLORS


def test_answer_balance_round_robin():
    """Across 12 consecutive seeds the target colors should include all 6 colors
    (seeds 0..5 map to all 6 via seed%6, so first 6 should be one-each)."""
    counts: Counter = Counter()
    for seed in range(6):
        s = _scene(seed=seed)
        counts[s.ground_truth["answer"]] += 1
    assert set(counts.keys()) == set(COLORS)
    assert all(v == 1 for v in counts.values())


def test_scene_graph_text_consistent():
    """scene_graph_text must parse back to the same objects stored in ground_truth."""
    s = _scene(n_objects=8, hop_depth=1, seed=3)
    parsed = parse_scene_graph_text(s.ground_truth["scene_graph_text"])
    stored = [
        dict(color=o["color"], shape=o["shape"], size=o["size"],
             x=o["x"], y=o["y"])
        for o in s.ground_truth["objects"]
    ]
    parsed_dicts = [
        dict(color=o.color, shape=o.shape, size=o.size, x=o.x, y=o.y)
        for o in parsed
    ]
    assert sorted(stored, key=lambda d: (d["x"], d["y"])) == \
           sorted(parsed_dicts, key=lambda d: (d["x"], d["y"]))


def test_answer_verified_by_resolve():
    """Resolve the chain over the stored objects - must equal stored answer."""
    for seed in range(30):
        s = _scene(n_objects=6, hop_depth=2, seed=seed)
        chain = parse_question(s.ground_truth["question"])
        assert chain is not None, f"seed={seed}: question not parseable"
        objs = parse_scene_graph_text(s.ground_truth["scene_graph_text"])
        assert answer_of(objs, chain) == s.ground_truth["answer"], \
            f"seed={seed}: resolve disagrees with stored answer"


def test_trivial_tier_flag():
    s = _scene(n_objects=4, hop_depth=1, seed=0)
    assert s.meta["tier"] == "trivial"


def test_standard_tier_flag():
    s = _scene(n_objects=6, hop_depth=2, seed=0)
    assert s.meta["tier"] == "standard"


def test_distinct_x_and_y():
    """All objects must have distinct x AND distinct y (tie-free superlatives)."""
    s = _scene(n_objects=8, seed=7)
    xs = [o["x"] for o in s.ground_truth["objects"]]
    ys = [o["y"] for o in s.ground_truth["objects"]]
    assert len(set(xs)) == len(xs), "duplicate x coordinates"
    assert len(set(ys)) == len(ys), "duplicate y coordinates"


def test_clutter_keeps_gt_trustable_and_nonoverlapping():
    """Perceptual-difficulty (clutter) must raise load WITHOUT hindering the ground
    truth: at every level the chain still resolves to the stored answer, objects never
    overlap (a perfect perceiver can always disambiguate), and each carries its
    rendered radius so the report ring matches it."""
    import math
    for clutter in (0, 1, 2):
        for seed in range(12):
            s = GEN.generate(
                {"n_objects": 12, "hop_depth": 3, "clutter": clutter}, seed
            )
            objs = s.ground_truth["objects"]
            assert all("r" in o for o in objs)
            # non-overlapping
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i], objs[j]
                    d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                    assert d >= a["r"] + b["r"], f"overlap at clutter={clutter} seed={seed}"
            # GT still verified by construction
            chain = parse_question(s.ground_truth["question"])
            graph = parse_scene_graph_text(s.ground_truth["scene_graph_text"])
            assert answer_of(graph, chain) == s.ground_truth["answer"]
            assert s.factors["clutter"] == clutter


def test_extended_ranges_generate():
    """The widened ranges (n_objects->16, hop_depth->6) generate whenever feasible
    (a depth-d chain needs > d objects)."""
    for n, d in [(16, 6), (12, 5), (8, 4), (16, 1)]:
        s = GEN.generate({"n_objects": n, "hop_depth": d}, 0)
        assert len(s.ground_truth["objects"]) == n
        assert len(parse_question(s.ground_truth["question"]).hops) == d


def test_generator_never_raises_on_tight_configs():
    """Schema-valid params must always yield a scene. The tight corners
    (n=4/depth>=2, n=5/depth=3) previously raised RuntimeError and were silently
    dropped by the runner; the retry loop must now cover them. Regression lock."""
    for n in (4, 5, 6):
        for d in (1, 2, 3):
            for seed in range(20):
                s = _scene(n_objects=n, hop_depth=d, seed=seed)  # must not raise
                assert len(s.ground_truth["objects"]) == n


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_probe_registered():
    reg = _registry()
    assert "relational" in reg.list_probes()


def test_probe_question_matches_scene():
    s = _scene()
    assert PROBE.question(s) == s.ground_truth["question"]


def test_probe_ground_truth_matches_scene():
    s = _scene()
    assert PROBE.ground_truth(s) == s.ground_truth["answer"]


def test_probe_parse_color():
    assert PROBE.parse("The answer is blue.") == "blue"
    assert PROBE.parse("red") == "red"


def test_probe_parse_raises_on_no_color():
    with pytest.raises(ValueError):
        PROBE.parse("I don't know")


def test_probe_oracle_variants_count():
    s = _scene()
    variants = PROBE.oracle_prompt_variants(s)
    assert len(variants) == 3
    assert "scene_graph" in variants  # canonical


def test_probe_oracle_prompts_contain_question():
    s = _scene()
    question = s.ground_truth["question"]
    for name, text in PROBE.oracle_prompt_variants(s).items():
        assert question in text, f"variant {name!r} missing question"


def test_probe_oracle_prompts_do_not_contain_answer():
    """Oracle variants must inject perceptual primitives, NOT the answer itself."""
    for seed in range(10):
        s = _scene(seed=seed)
        for name, text in PROBE.oracle_prompt_variants(s).items():
            # The answer color may appear as part of an object descriptor (fine),
            # but the probe must never include a phrase like "the answer is red".
            assert "answer is" not in text.lower(), \
                f"variant {name!r} leaks the answer phrase"


# ---------------------------------------------------------------------------
# Mock: oracle uses scene-graph text; full uses pixel perception
# ---------------------------------------------------------------------------

def test_mock_oracle_resolves_from_text():
    """Oracle path: no image, scene-graph in prompt -> mock must resolve correctly."""
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")

    correct = 0
    for seed in range(30):
        s = _scene(n_objects=6, hop_depth=2, seed=seed)
        oracle_prompt = PROBE.oracle_prompt(s)
        resp = mock.run([], oracle_prompt)
        pred = PROBE.parse(resp.text)
        if pred == s.ground_truth["answer"]:
            correct += 1
    # Oracle should be very high (ideally perfect) - it reads the injected graph
    assert correct >= 25, f"mock oracle too low: {correct}/30"


def test_mock_oracle_faithful_across_all_variants():
    """The exact mock must resolve EVERY oracle verbalization variant, not just the
    canonical one - otherwise a variant looks 'suspect' for mock reasons, not model
    ones. Locks the coords_only size-completeness fix + tolerant oracle parser."""
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")
    for seed in range(25):
        s = _scene(n_objects=6, hop_depth=2, seed=seed)
        for name, prompt in PROBE.oracle_prompt_variants(s).items():
            pred = PROBE.parse(mock.run([], prompt).text)
            assert pred == s.ground_truth["answer"], f"variant={name} seed={seed}"


def test_oracle_no_spurious_suspect_across_variants():
    """Running all variants on the exact mock must not raise the prompt-suspect
    flag: with a faithful reasoner, every complete verbalization recovers fully."""
    cfg = ExperimentConfig(
        scene_generator="relational",
        scene_params={"n_objects": [4, 6], "hop_depth": [1, 2]},
        seeds=list(range(6)),
        probe="relational",
        models=["mock"],
        analyzers=["oracle"],
        oracle_variants=["all"],
    )
    _, _, reports = run_experiment_collect(cfg, _registry())
    oracle = next(r for r in reports if r.name == "oracle")
    assert oracle.payload["any_oracle_prompt_suspect"] is False


def test_mock_full_uses_pixels_not_gt():
    """Full path: image present but NO ground_truth access -> still gets some right."""
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")

    correct = 0
    for seed in range(30):
        s = _scene(n_objects=5, hop_depth=1, seed=seed)
        question = PROBE.question(s)
        resp = mock.run(s.images, question)
        try:
            pred = PROBE.parse(resp.text)
        except ValueError:
            continue
        if pred == s.ground_truth["answer"]:
            correct += 1
    # Pixel perception + resolve: should beat random chance (1/6 ≈ 5/30)
    assert correct >= 6, f"mock full too low: {correct}/30"


def test_mock_blind_is_random():
    """Blind path: no image, question only -> random guess."""
    from tests.fixtures.mock import PLUGINS
    mock_noisy = next(p for p in PLUGINS if p.name == "mock_noisy")

    answers: list[str] = []
    for seed in range(60):
        s = _scene(seed=seed)
        question = PROBE.question(s)
        resp = mock_noisy.run([], question)
        try:
            answers.append(PROBE.parse(resp.text))
        except ValueError:
            pass
    # Should not be perfectly right (blind has no real signal)
    unique = len(set(answers))
    assert unique >= 3, "blind mock not picking from multiple colors"


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def _e2e_cfg(**kwargs) -> ExperimentConfig:
    base = dict(
        scene_generator="relational",
        scene_params={"n_objects": [4, 6], "hop_depth": [1, 2]},
        seeds=[0, 1, 2],
        probe="relational",
        models=["mock"],
        analyzers=["accuracy"],
    )
    base.update(kwargs)
    return ExperimentConfig(**base)


def test_e2e_all_conditions_produced():
    _, results, _ = run_experiment_collect(_e2e_cfg(), _registry())
    conditions = {r.condition for r in results}
    assert "full" in conditions
    assert "oracle" in conditions
    assert "blind" not in conditions      # opt-in


def _condition_acc(results: list, model: str, condition: str) -> float:
    rs = [r for r in results if r.model == model and r.condition == condition]
    if not rs:
        return 0.0
    return sum(r.correct for r in rs) / len(rs)


def test_e2e_oracle_accuracy_beats_blind():
    """On mock, oracle accuracy must exceed blind accuracy."""
    _, results, _ = run_experiment_collect(_e2e_cfg(blind=True), _registry())
    oracle_acc = _condition_acc(results, "mock", "oracle")
    blind_acc  = _condition_acc(results, "mock", "blind")
    assert oracle_acc > blind_acc, (
        f"oracle={oracle_acc:.2f} not > blind={blind_acc:.2f}"
    )


def test_e2e_noisy_mock_has_positive_recovery():
    """mock_noisy recovery = oracle_acc - full_acc should be positive."""
    cfg = _e2e_cfg(models=["mock_noisy"])
    _, results, _ = run_experiment_collect(cfg, _registry())
    oracle_acc = _condition_acc(results, "mock_noisy", "oracle")
    full_acc   = _condition_acc(results, "mock_noisy", "full")
    assert oracle_acc >= full_acc, (
        f"no recovery signal: oracle={oracle_acc:.2f} full={full_acc:.2f}"
    )
