"""Runner tests: conditions produced, sandboxing, skip-with-message."""

from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment
from renderprobe.core.schema import AnalysisReport
from tests.fixtures import register


def _make_registry() -> Registry:
    r = Registry()
    r.autodiscover()
    register(r)
    return r


def _smoke_config(analyzers=None) -> ExperimentConfig:
    return ExperimentConfig(
        scene_generator="dots",
        scene_params={"n_dots": [1, 3]},
        seeds=[0],
        probe="count",
        models=["mock"],
        analyzers=analyzers or ["accuracy", "visual_gain"],
        blind=True,        # visual_gain is defined against the text-only baseline
    )


def test_runner_produces_results():
    reg = _make_registry()
    reports = run_experiment(_smoke_config(), reg)
    assert len(reports) == 2  # accuracy + visual_gain


def test_blind_condition_runs_only_when_asked():
    """The text-only baseline costs a call per scene and measures a prior rather than
    a perceptual ability, so it is opt-in: absent by default, present when requested."""
    from renderprobe.core.runner import (
        _run_scene_model,
    )
    reg = _make_registry()
    scene_gen = reg.get_scene("dots")
    probe = reg.get_probe("count")
    model = reg.get_model("mock")

    scene = scene_gen.generate({"n_dots": 3}, seed=0)
    # oracle_supported=False forces the oracle branch off regardless of the probe.
    off = {r.condition for r in
           _run_scene_model(scene, probe, model, oracle_supported=False)}
    assert "full" in off and "blind" not in off

    on = {r.condition for r in
          _run_scene_model(scene, probe, model, oracle_supported=False, blind=True)}
    assert on == off | {"blind"}


def test_runner_no_oracle_when_probe_unsupported():
    """A probe whose oracle_prompt returns None is correctly detected as unsupported."""
    from renderprobe.core.runner import _probe_supports_oracle

    class _NoOracleProbe:
        name = "no_oracle"
        answer_type = "numeric"
        requires_gt_fields = ["count"]
        def question(self, scene): return "q"
        def ground_truth(self, scene): return 0
        def parse(self, raw): return 0
        def score(self, pred, gt): return True, 1.0, 0.0
        def oracle_prompt(self, scene): return None

    scene = _make_registry().get_scene("dots").generate({"n_dots": 2}, seed=0)
    assert _probe_supports_oracle(_NoOracleProbe(), scene) is False


def test_sweep_analyzer_skipped_without_varying_factor(capsys):
    """A sweep-type analyzer must be skipped with a clear message when there's no varying factor."""

    class _SweepAnalyzer:
        name = "fake_sweep"
        requires_conditions = ["full"]
        requires_varying_factor = True
        def analyze(self, results): return AnalysisReport(name="fake_sweep", payload={})

    reg = _make_registry()
    reg.register_analyzer(_SweepAnalyzer())

    # Single n_dots value -> no varying factor
    cfg = ExperimentConfig(
        scene_generator="dots",
        scene_params={"n_dots": 3},  # scalar, not a list
        seeds=[0],
        probe="count",
        models=["mock"],
        analyzers=["fake_sweep"],
    )
    reports = run_experiment(cfg, reg)
    captured = capsys.readouterr()
    assert "fake_sweep" in captured.out
    assert "skipped" in captured.out
    assert reports == []


def test_bad_model_produces_model_error_flag():

    class _BrokenModel:
        name = "broken"
        is_open = False
        def run(self, images, prompt): raise RuntimeError("boom")
        def attention(self, images, prompt): return None

    reg = _make_registry()
    reg.register_model(_BrokenModel())

    cfg = ExperimentConfig(
        scene_generator="dots",
        scene_params={"n_dots": 2},
        seeds=[0],
        probe="count",
        models=["broken"],
        analyzers=["accuracy"],
    )
    reports = run_experiment(cfg, reg)
    # accuracy report should have 0 valid results or still return a report
    assert any(r.name == "accuracy" for r in reports)


def test_accuracy_report_structure():
    reg = _make_registry()
    reports = run_experiment(_smoke_config(["accuracy"]), reg)
    assert len(reports) == 1
    acc = reports[0]
    assert acc.name == "accuracy"
    assert "by_model" in acc.payload
    assert "mock" in acc.payload["by_model"]
    row = acc.payload["by_model"]["mock"]
    assert 0.0 <= row["accuracy"] <= 1.0


def test_visual_gain_report_structure():
    reg = _make_registry()
    reports = run_experiment(_smoke_config(["visual_gain"]), reg)
    assert len(reports) == 1
    vg = reports[0]
    assert vg.name == "visual_gain"
    assert "by_model" in vg.payload
    row = vg.payload["by_model"]["mock"]
    assert "visual_gain" in row
    assert "acc_full" in row
    assert "acc_blind" in row


def test_a_truncated_reply_is_not_scored_as_a_wrong_answer():
    """A model cut off at its token limit was still working. Scoring the fragment
    records a failure it may never have made, and a parser that takes the last match
    off it recovers whatever it happened to be mid-sentence about."""
    from renderprobe.core.runner import _run_condition
    from renderprobe.core.schema import Response, Scene

    class _Truncating:
        name = "truncating"

        def run(self, images, prompt):
            return Response(text="Let me compare. red... blue...", truncated=True)

        def attention(self, images, prompt):
            return None

    class _Probe:
        name = "p"

        def question(self, scene):
            return "q"

    scene = Scene(id="s", images=[], ground_truth={}, factors={}, meta={})
    r = _run_condition(scene, _Probe(), _Truncating(), "full", "q", "green",
                       parse_fn=lambda t: t.split()[-1], score_fn=lambda p, g: (p == g, 0.0, None))
    assert r.error_flag == "truncated"
    assert r.correct is False
    # and it leaves the accuracies, rather than counting as an answer the model gave
    from renderprobe.analysis.decomposition import _acc
    assert _acc([r], "truncating", "full") == (None, 0)


def test_an_elicited_confidence_is_never_read_as_the_answer():
    """The instruction asks for a trailing "Confidence: N", and N is a number. A probe
    whose parser takes the last number - the right rule for a model that shows its
    working - would otherwise return the confidence. Measured live: gemma-3-27b replied
    "12\n\nConfidence: 100" to a question whose answer was 12 and was scored wrong."""
    import re

    from renderprobe.core.runner import _run_condition, _without_confidence
    from renderprobe.core.schema import Response, Scene

    class _Confident:
        name = "confident"

        def run(self, images, prompt):
            return Response(text="12\n\nConfidence: 100")

        def attention(self, images, prompt):
            return None

    class _Probe:
        name = "p"

        def question(self, scene):
            return "how many turns?"

    def last_int(raw):
        found = re.findall(r"-?\d+", raw or "")
        return int(found[-1]) if found else None

    scene = Scene(id="s", images=[], ground_truth={}, factors={}, meta={})
    r = _run_condition(scene, _Probe(), _Confident(), "full", "q", 12,
                       parse_fn=last_int,
                       score_fn=lambda p, g: (p == g, 1.0 if p == g else 0.0, None),
                       elicit_confidence=True)
    assert r.pred == 12, "the confidence was parsed as the answer"
    assert r.correct is True
    assert r.confidence == 1.0
    assert "Confidence: 100" in r.raw, "the full reply must still be inspectable"

    # a reply with no confidence passes through untouched
    assert _without_confidence("6 + 7 + 11 = 24") == "6 + 7 + 11 = 24"
