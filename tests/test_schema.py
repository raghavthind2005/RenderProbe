"""Schema round-trip tests."""
from PIL import Image

from renderprobe.core.schema import (
    AnalysisReport,
    Analyzer,
    ModelAdapter,
    Probe,
    Response,
    Result,
    Scene,
    SceneGenerator,
)


def _make_scene() -> Scene:
    img = Image.new("RGB", (10, 10), "white")
    return Scene(
        id="test_scene",
        images=[img],
        ground_truth={"count": 3},
        factors={"n_dots": 3},
        meta={"generator": "dots", "seed": 0, "params": {}},
    )


def _make_result(**overrides) -> Result:
    defaults = dict(
        scene_id="test_scene", generator="dots", factors={"n_dots": 3},
        probe="count", model="mock", condition="full",
        raw="3", pred=3, gt=3, correct=True, score=1.0,
        error=0.0, confidence=None, error_flag=None,
    )
    defaults.update(overrides)
    return Result(**defaults)


def test_scene_fields():
    scene = _make_scene()
    assert scene.id == "test_scene"
    assert scene.factors == {"n_dots": 3}
    assert scene.ground_truth == {"count": 3}
    assert len(scene.images) == 1


def test_result_fields():
    r = _make_result()
    assert r.condition == "full"
    assert r.correct is True
    assert r.error_flag is None


def test_response_defaults():
    r = Response(text="5")
    assert r.confidence is None


def test_analysis_report():
    rep = AnalysisReport(name="accuracy", payload={"by_model": {}})
    assert rep.figures == []


def test_scene_generator_protocol():
    class BadGen:
        name = "bad"
        params_schema = {}
        # missing generate()

    assert not isinstance(BadGen(), SceneGenerator)


def test_probe_protocol():
    class BadProbe:
        name = "bad"

    assert not isinstance(BadProbe(), Probe)


def test_model_adapter_protocol():
    class BadAdapter:
        name = "bad"

    assert not isinstance(BadAdapter(), ModelAdapter)


def test_analyzer_protocol():
    class BadAnalyzer:
        name = "bad"

    assert not isinstance(BadAnalyzer(), Analyzer)
