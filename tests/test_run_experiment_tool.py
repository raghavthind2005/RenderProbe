"""The resume cache (tools/run_experiment.py).

A run killed part way has to cost nothing to finish, so the cache's identity rule is
what matters: replay a reply only when the model, the prompt AND the pixels are the
same one it answered.
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from run_experiment import (  # noqa: E402
    _Cache,
    _key,
    _scene_key,
    _SceneCache,
    wrap_scenes,
)

from renderprobe.core.registry import Registry  # noqa: E402
from renderprobe.core.schema import Response  # noqa: E402


def _img(color):
    return Image.new("RGB", (8, 8), color)


def test_the_same_call_has_the_same_key():
    a = _key("m", [_img("red")], "how long?")
    b = _key("m", [_img("red")], "how long?")
    assert a == b


def test_a_redrawn_scene_is_asked_again():
    """The images are hashed, not named. A scene whose rendering changed must not
    replay a reply given to a picture that no longer exists."""
    assert _key("m", [_img("red")], "q") != _key("m", [_img("blue")], "q")


def test_model_and_prompt_both_matter():
    assert _key("m1", [], "q") != _key("m2", [], "q")
    assert _key("m", [], "q1") != _key("m", [], "q2")


def test_a_reply_survives_the_process(tmp_path):
    path = tmp_path / "responses.jsonl"
    first = _Cache(path)
    first.put("k1", "m", "prompt", Response(text="24", truncated=False))
    assert (first.hits, first.misses) == (0, 1)

    # a new process opens the same file and finds the reply already there
    second = _Cache(path)
    hit = second.get("k1")
    assert hit is not None and hit.text == "24"
    assert second.get("never-asked") is None
    assert (second.hits, second.misses) == (1, 0)


def test_truncation_survives_the_round_trip(tmp_path):
    """A truncated reply is dropped from the accuracies, so replaying it as a clean one
    would silently turn a skipped row into a wrong answer."""
    path = tmp_path / "responses.jsonl"
    _Cache(path).put("k", "m", "p", Response(text="Let me compare", truncated=True))
    assert _Cache(path).get("k").truncated is True


def test_a_half_written_last_line_does_not_break_resume(tmp_path):
    """Exactly what a hard kill leaves behind."""
    path = tmp_path / "responses.jsonl"
    c = _Cache(path)
    c.put("good", "m", "p", Response(text="ok"))
    c.fh.close()
    with path.open("a") as fh:
        fh.write('{"key": "trunc", "text":')
    resumed = _Cache(path)
    assert resumed.get("good").text == "ok"
    assert resumed.get("trunc") is None


# ---------------------------------------------------------------------------
# The scene cache
#
# The reply cache hashes pixels, which is the right identity rule and also the reason
# resumption used to fail on exactly the runs worth resuming: mitsuba_3d is stochastic,
# so the same scene at the same seed came out with different pixels every run, every
# reply key missed, and a killed polycube run paid for itself twice.
# ---------------------------------------------------------------------------


def _polycube_registry():
    reg = Registry()
    reg.autodiscover()
    return reg


def test_a_stochastic_scene_is_not_reproducible_without_the_cache():
    """The premise the scene cache rests on.

    Mitsuba is stochastic INTERMITTENTLY, not on every render. At the scene's own
    settings, 10 pairs at one seed gave 3 differing pairs across 6 distinct images, so a
    single pair proves nothing either way: an earlier version of this test compared two
    renders and passed about seven times in ten.

    This renders a batch at a low sample count instead. Fewer samples means more
    sampling noise and so more variation - 12 of 12 distinct rather than 6 - and it runs
    in about 7 seconds where the scene's own settings take 45. The mechanism under test
    is the same one; only the rate differs, and the rate that matters for a real run is
    the 3-in-10 above.

    If this ever fails the renderer has become deterministic, and the scene cache is no
    longer load-bearing for correctness - only for the seconds a re-render costs.
    """
    reg = _polycube_registry()
    gen = reg.get_scene("polycube")
    cheap = {"n_pieces": 4, "renderer_opts": {"spp": 4}}
    scenes = [gen.generate(cheap, 0) for _ in range(12)]

    assert len({str(s.ground_truth) for s in scenes}) == 1, "the answer must never move"
    assert len({_key("m", s.images, "q") for s in scenes}) > 1


def test_the_scene_cache_replays_the_identical_pixels(tmp_path):
    reg = _polycube_registry()
    cache = _SceneCache(tmp_path / "scenes")
    wrap_scenes(reg, "polycube", cache)
    gen = reg.get_scene("polycube")

    first = gen.generate({"n_pieces": 4}, 0)
    second = gen.generate({"n_pieces": 4}, 0)

    assert (cache.misses, cache.hits) == (1, 1)
    assert _key("m", first.images, "q") == _key("m", second.images, "q")
    assert first.ground_truth == second.ground_truth


def test_a_resumed_run_replays_scenes_from_disk(tmp_path):
    """The actual resume path: a fresh process, a fresh registry, the same cache dir."""
    reg = _polycube_registry()
    wrap_scenes(reg, "polycube", _SceneCache(tmp_path / "scenes"))
    before = reg.get_scene("polycube").generate({"n_pieces": 4}, 0)

    reg2 = _polycube_registry()                       # as if the process had restarted
    cache2 = _SceneCache(tmp_path / "scenes")
    wrap_scenes(reg2, "polycube", cache2)
    after = reg2.get_scene("polycube").generate({"n_pieces": 4}, 0)

    assert cache2.hits == 1 and cache2.misses == 0
    assert _key("m", before.images, "q") == _key("m", after.images, "q")


def test_the_scene_graph_survives_the_round_trip(tmp_path):
    """A Scene replayed without its graph would validate differently from the one that
    was actually probed, so the cache stores the whole Scene rather than just pixels."""
    reg = _polycube_registry()
    wrap_scenes(reg, "polycube", _SceneCache(tmp_path / "scenes"))
    gen = reg.get_scene("polycube")
    first = gen.generate({"n_pieces": 4}, 0)
    second = gen.generate({"n_pieces": 4}, 0)
    assert (first.graph is None) == (second.graph is None)
    if first.graph is not None:
        assert len(first.graph.objects) == len(second.graph.objects)


def test_editing_the_scene_invalidates_its_cached_renders():
    """The bounded guarantee the docstring claims: the generator's own source is part of
    the key, so an edited scene is re-rendered rather than replayed."""
    a = _scene_key("polycube", {"n_pieces": 4}, 0, "sourcehash-aaaa")
    b = _scene_key("polycube", {"n_pieces": 4}, 0, "sourcehash-bbbb")
    assert a != b


def test_params_and_seed_are_part_of_the_scene_identity():
    base = _scene_key("polycube", {"n_pieces": 4}, 0, "fp")
    assert base != _scene_key("polycube", {"n_pieces": 5}, 0, "fp")
    assert base != _scene_key("polycube", {"n_pieces": 4}, 1, "fp")
    assert base != _scene_key("route", {"n_pieces": 4}, 0, "fp")
    assert base == _scene_key("polycube", {"n_pieces": 4}, 0, "fp")


def test_an_unreadable_cache_entry_degrades_to_a_miss(tmp_path):
    cache = _SceneCache(tmp_path / "scenes")
    (tmp_path / "scenes" / "deadbeef.pkl").write_bytes(b"not a pickle")
    assert cache.get("deadbeef") is None
    assert cache.hits == 0
