"""Tests for the pathfinding scene - core BFS, generator contract, probe, mock, e2e."""
from __future__ import annotations

from random import Random

import pytest

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from tests.fixtures import register
from tests.fixtures._pathfinding_core import (
    Graph,
    Node,
    balanced_question,
    build_graph,
    graph_to_adjacency_text,
    has_same_color_path,
    parse_adjacency_text,
    parse_question,
    render_question,
)
from tests.fixtures.probe_pathfinding import PLUGIN as PROBE
from tests.fixtures.scene_pathfinding import PLUGIN as GEN

# ---------------------------------------------------------------------------
# _pathfinding_core: BFS correctness
# ---------------------------------------------------------------------------

def test_bfs_connected_same_color():
    nodes = (
        Node(0, "red", 0, 0),
        Node(1, "red", 100, 0),
        Node(2, "blue", 200, 0),
    )
    graph = Graph(nodes=nodes, edges=frozenset([(0, 1), (1, 2)]))
    assert has_same_color_path(graph, 0, 1) is True    # both red, direct edge
    assert has_same_color_path(graph, 0, 2) is False   # colors differ -> no path


def test_bfs_no_edge():
    nodes = (Node(0, "red", 0, 0), Node(1, "red", 100, 0))
    graph = Graph(nodes=nodes, edges=frozenset())
    assert has_same_color_path(graph, 0, 1) is False


def test_bfs_self_path():
    nodes = (Node(0, "blue", 0, 0),)
    graph = Graph(nodes=nodes, edges=frozenset())
    assert has_same_color_path(graph, 0, 0) is True


def test_bfs_path_through_color_chain():
    # 0(red)-1(red)-2(red): chain of reds
    nodes = tuple(Node(i, "red", i * 50, 0) for i in range(3))
    graph = Graph(nodes=nodes, edges=frozenset([(0, 1), (1, 2)]))
    assert has_same_color_path(graph, 0, 2) is True


def test_bfs_different_color_blocks():
    # red-blue-red: red is disconnected by the blue node
    nodes = (
        Node(0, "red",  0,   0),
        Node(1, "blue", 100, 0),
        Node(2, "red",  200, 0),
    )
    graph = Graph(nodes=nodes, edges=frozenset([(0, 1), (1, 2)]))
    # src=0 dst=2 both red, but no same-color path (must go through blue)
    assert has_same_color_path(graph, 0, 2) is False


def test_adjacency_text_roundtrip():
    rng = Random(42)
    graph = build_graph(rng, 6, 0.4)
    text = graph_to_adjacency_text(graph)
    recovered = parse_adjacency_text(text)
    assert recovered is not None
    original_colors = {n.id: n.color for n in graph.nodes}
    recovered_colors = {n.id: n.color for n in recovered.nodes}
    assert original_colors == recovered_colors
    assert recovered.edges == graph.edges


def test_question_render_parse_roundtrip():
    nodes = (Node(0, "red", 0, 0), Node(3, "blue", 100, 0))
    graph = Graph(nodes=nodes, edges=frozenset())
    q = render_question(0, 3, graph)
    pair = parse_question(q)
    assert pair == (0, 3)


def test_balanced_question_alternates():
    """Even seeds -> yes, odd seeds -> no (balance invariant)."""
    for seed in range(20):
        rng = Random(seed)
        graph = build_graph(Random(seed * 13), 8, 0.5)
        result = balanced_question(graph, seed, rng)
        if result is None:
            continue
        _, _, ans = result
        expected = (seed % 2 == 0)
        assert ans == expected, f"seed={seed}: expected {'yes' if expected else 'no'}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _scene(n_nodes: int = 7, p_edge: float = 0.4, seed: int = 0):
    return GEN.generate({"n_nodes": n_nodes, "p_edge": p_edge}, seed)


def test_generator_registered():
    assert "pathfinding" in _registry().list_scenes()


def test_scene_ground_truth_contract():
    s = _scene()
    gt = s.ground_truth
    for key in ("question", "answer", "adjacency_text", "nodes", "edges", "src", "dst"):
        assert key in gt, f"missing key: {key}"
    assert gt["answer"] in ("yes", "no")


def test_scene_has_one_image():
    s = _scene()
    assert len(s.images) == 1
    w, h = s.images[0].size
    assert w == 480 and h == 360


def test_answer_is_bfs_exact():
    """The stored answer must match BFS over the stored adjacency."""
    for seed in range(30):
        s = _scene(n_nodes=8, p_edge=0.4, seed=seed)
        gt = s.ground_truth
        graph = parse_adjacency_text(gt["adjacency_text"])
        assert graph is not None
        bfs_ans = has_same_color_path(graph, gt["src"], gt["dst"])
        bfs_str = "yes" if bfs_ans else "no"
        assert bfs_str == gt["answer"], f"seed={seed}: BFS={bfs_str} stored={gt['answer']}"


def test_answer_balance_near_5050():
    """Over 20 seeds, yes and no should each appear (and be close to 50/50)."""
    yes = sum(
        1 for seed in range(20)
        if _scene(seed=seed).ground_truth["answer"] == "yes"
    )
    assert 6 <= yes <= 14, f"imbalanced: {yes}/20 yes answers"


def test_trivial_tier_flag():
    s = _scene(n_nodes=4, p_edge=0.7, seed=0)
    assert s.meta["tier"] == "trivial"


def test_standard_tier_flag():
    s = _scene(n_nodes=8, p_edge=0.4, seed=0)
    assert s.meta["tier"] == "standard"


def test_generator_never_raises_on_grid():
    """Schema-valid params must always yield a balanced scene. Small sparse graphs
    (n=4, low p_edge) previously exhausted the retry budget; the bumped budget must
    cover the whole grid. Regression lock."""
    for n in (4, 5, 8, 11):
        for p in (0.2, 0.3, 0.5, 0.7):
            for seed in range(15):
                s = _scene(n_nodes=n, p_edge=p, seed=seed)  # must not raise
                assert s.ground_truth["answer"] in ("yes", "no")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_probe_registered():
    assert "pathfinding" in _registry().list_probes()


def test_probe_parse_yes_no():
    assert PROBE.parse("Yes.") == "yes"
    assert PROBE.parse("No, there is no path.") == "no"


def test_probe_parse_raises_on_unknown():
    with pytest.raises(ValueError):
        PROBE.parse("maybe")


def test_probe_oracle_variants_count():
    s = _scene()
    variants = PROBE.oracle_prompt_variants(s)
    assert len(variants) == 3
    assert "adjacency_list" in variants


def test_probe_oracle_prompts_contain_question():
    s = _scene()
    q = s.ground_truth["question"]
    for name, text in PROBE.oracle_prompt_variants(s).items():
        assert q in text, f"variant {name!r} missing the question"


def test_probe_oracle_not_leak_answer():
    """None of the oracle variants should contain the literal answer word
    preceding the question (i.e. it should not say 'yes' or 'no' before asking)."""
    for seed in range(10):
        s = _scene(seed=seed)
        answer = s.ground_truth["answer"]
        question = s.ground_truth["question"]
        for name, text in PROBE.oracle_prompt_variants(s).items():
            pre_question = text[: text.index(question)]
            assert answer not in pre_question.lower().split(), \
                f"variant {name!r} seed={seed} leaks answer '{answer}' before question"


# ---------------------------------------------------------------------------
# Mock: oracle uses adjacency text; full uses pixel perception
# ---------------------------------------------------------------------------

def test_mock_oracle_bfs_from_text():
    """Oracle path: adjacency list injected -> mock must compute BFS correctly."""
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")

    correct = 0
    for seed in range(30):
        s = _scene(n_nodes=7, p_edge=0.4, seed=seed)
        oracle_p = PROBE.oracle_prompt(s)
        resp = mock.run([], oracle_p)
        try:
            pred = PROBE.parse(resp.text)
        except ValueError:
            continue
        if pred == s.ground_truth["answer"]:
            correct += 1
    assert correct >= 24, f"mock oracle too low: {correct}/30"


def test_mock_oracle_faithful_across_all_variants():
    """The exact mock must compute BFS correctly for EVERY oracle verbalization
    (structured adjacency list, compact inline, prose) - locks the tolerant graph
    parser so no variant looks spuriously 'suspect'."""
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")
    correct = 0
    total = 0
    for seed in range(25):
        s = _scene(n_nodes=7, p_edge=0.4, seed=seed)
        for name, prompt in PROBE.oracle_prompt_variants(s).items():
            pred = PROBE.parse(mock.run([], prompt).text)
            total += 1
            correct += int(pred == s.ground_truth["answer"])
    assert correct == total, f"mock not faithful across variants: {correct}/{total}"


def test_mock_full_perception_beats_chance():
    """Full condition must identify src/dst via the GOLD highlight ring and trace
    colors/edges from pixels - not collapse to chance because it can't OCR labels.
    Regression lock for the gold-ring src/dst fix (label-based mapping gave ~0.54).
    """
    from tests.fixtures.mock import PLUGINS
    mock = next(p for p in PLUGINS if p.name == "mock")

    correct = 0
    n = 0
    for seed in range(40):
        s = _scene(n_nodes=7, p_edge=0.4, seed=seed)
        resp = mock.run(s.images, PROBE.question(s))
        try:
            pred = PROBE.parse(resp.text)
        except ValueError:
            continue
        n += 1
        correct += int(pred == s.ground_truth["answer"])
    # Honest pixel perception runs ~0.85; must clear chance (0.5) with real margin.
    assert correct / n >= 0.70, f"full perception near chance: {correct}/{n}"


def test_mock_blind_is_random():
    from tests.fixtures.mock import PLUGINS
    mock_noisy = next(p for p in PLUGINS if p.name == "mock_noisy")
    answers: list[str] = []
    for seed in range(40):
        s = _scene(seed=seed)
        resp = mock_noisy.run([], PROBE.question(s))
        try:
            answers.append(PROBE.parse(resp.text))
        except ValueError:
            pass
    # Both yes and no should appear
    assert "yes" in answers and "no" in answers


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def _e2e_cfg(**kwargs) -> ExperimentConfig:
    base = dict(
        scene_generator="pathfinding",
        scene_params={"n_nodes": [5, 8], "p_edge": [0.4]},
        seeds=[0, 1, 2, 3],
        probe="pathfinding",
        models=["mock"],
        analyzers=["accuracy"],
    )
    base.update(kwargs)
    return ExperimentConfig(**base)


def _condition_acc(results: list, model: str, condition: str) -> float:
    rs = [r for r in results if r.model == model and r.condition == condition]
    return sum(r.correct for r in rs) / len(rs) if rs else 0.0


def test_e2e_all_conditions_produced():
    _, results, _ = run_experiment_collect(_e2e_cfg(), _registry())
    conditions = {r.condition for r in results}
    assert {"full", "oracle"} <= conditions
    assert "blind" not in conditions      # opt-in


def test_e2e_oracle_beats_blind():
    _, results, _ = run_experiment_collect(_e2e_cfg(blind=True), _registry())
    oracle_acc = _condition_acc(results, "mock", "oracle")
    blind_acc  = _condition_acc(results, "mock", "blind")
    assert oracle_acc > blind_acc, (
        f"oracle={oracle_acc:.2f} not > blind={blind_acc:.2f}"
    )
