"""Registry: discovery + validation tests."""
import pytest

from renderprobe.core.registry import RegistrationError, Registry
from renderprobe.core.schema import AnalysisReport, Response
from tests.fixtures import register

# ---------------------------------------------------------------------------
# Minimal valid plugins for testing
# ---------------------------------------------------------------------------

class _GoodScene:
    name = "good_scene"
    params_schema = {"n": {"type": "int"}}
    def generate(self, params, seed): return None  # type: ignore[return-value]

class _GoodProbe:
    name = "good_probe"
    answer_type = "numeric"
    requires_gt_fields = ["count"]
    def question(self, scene): return "q"
    def ground_truth(self, scene): return 0
    def parse(self, raw): return int(raw)
    def score(self, pred, gt): return (pred == gt, 1.0, None)
    def oracle_prompt(self, scene): return None

class _GoodModel:
    name = "good_model"
    is_open = False
    def run(self, images, prompt): return Response(text="1")
    def attention(self, images, prompt): return None

class _GoodAnalyzer:
    name = "good_analyzer"
    requires_conditions = ["full"]
    requires_varying_factor = False
    def analyze(self, results): return AnalysisReport(name="good_analyzer", payload={})


def test_register_scene_ok():
    r = Registry()
    r.register_scene(_GoodScene())
    assert r.get_scene("good_scene") is not None


def test_register_probe_ok():
    r = Registry()
    r.register_probe(_GoodProbe())
    assert r.get_probe("good_probe") is not None


def test_register_model_ok():
    r = Registry()
    r.register_model(_GoodModel())
    assert r.get_model("good_model") is not None


def test_register_analyzer_ok():
    r = Registry()
    r.register_analyzer(_GoodAnalyzer())
    assert r.get_analyzer("good_analyzer") is not None


def test_bad_answer_type_raises():
    class _BadProbe(_GoodProbe):
        answer_type = "invalid_type"
    r = Registry()
    with pytest.raises(RegistrationError, match="answer_type"):
        r.register_probe(_BadProbe())


def test_bad_condition_raises():
    class _BadAnalyzer(_GoodAnalyzer):
        requires_conditions = ["full", "nonexistent"]
    r = Registry()
    with pytest.raises(RegistrationError, match="condition"):
        r.register_analyzer(_BadAnalyzer())


def test_missing_protocol_method_raises():
    class _MissingGenerate:
        name = "bad"
        params_schema = {}
        # no generate()
    r = Registry()
    with pytest.raises(RegistrationError):
        r.register_scene(_MissingGenerate())


def test_get_unknown_raises():
    r = Registry()
    with pytest.raises(KeyError):
        r.get_scene("nonexistent")


def test_autodiscover_registers_plugins():
    r = Registry()
    r.autodiscover()
    register(r)
    # dots + count + mock + accuracy + visual_gain should all be discovered
    assert r.get_scene("dots") is not None
    assert r.get_probe("count") is not None
    assert r.get_model("mock") is not None
    # PLUGINS-list discovery: both mock variants from one module
    assert r.get_model("mock_noisy") is not None
    assert r.get_analyzer("accuracy") is not None
    assert r.get_analyzer("visual_gain") is not None


def test_autodiscover_handles_plugins_list():
    """A module exposing PLUGINS=[...] registers every entry."""
    r = Registry()
    r.autodiscover()
    register(r)
    # mock.py exposes two models via PLUGINS
    assert r.get_model("mock").name == "mock"
    assert r.get_model("mock_noisy").name == "mock_noisy"
