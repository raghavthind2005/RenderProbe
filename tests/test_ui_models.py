"""The Run tab's model roster and API-key panel.

Covers what somebody sees before they spend anything: which models are offered, which
are ticked by default, and what happens when the key for a selected model is missing.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from renderprobe.core import credentials
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.models.openai_compatible import PLUGINS as OAI_PLUGINS
from renderprobe.models.openai_compatible import OpenAICompatibleAdapter
from renderprobe.ui import data as D

_ALL_KEY_ENVS = [
    "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
]


@pytest.fixture
def reg(monkeypatch):
    """A registry seen with no credentials anywhere, so tests do not depend on a shell."""
    for env in _ALL_KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    r = Registry()
    r.autodiscover()
    return r


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

def test_every_registered_model_reports_whether_it_needs_a_key(reg):
    cat = {m["name"]: m for m in D.model_catalog(reg)}
    assert cat["gpt-4o"]["needs_key"] is True
    assert cat["gpt-4o"]["api_key_env"] == "OPENAI_API_KEY"
    assert cat["gpt-4o"]["key_available"] is False
    # Local weights need no credential, only a download.
    assert cat["qwen2.5-vl-7b"]["needs_key"] is False
    assert cat["qwen2.5-vl-7b"]["key_available"] is True


def test_an_adapter_declaring_no_metadata_shows_as_needing_no_key(reg):
    """None of these attributes is in the ModelAdapter Protocol, so a third-party
    adapter that declares none of them must still appear, and must not be reported as
    blocked on a key it never asked for."""
    class _Bare:
        name = "bare-adapter"
        is_open = True

        def run(self, images, prompt): ...
        def attention(self, images, prompt): return None

    reg.register_model(_Bare())
    entry = next(m for m in D.model_catalog(reg) if m["name"] == "bare-adapter")
    assert entry["needs_key"] is False
    assert entry["key_available"] is True
    assert entry["provider"] == "no key needed"
    assert D.blocked_models(reg, ["bare-adapter"]) == []


def test_a_session_key_makes_its_models_available_without_touching_the_environment(reg):
    session = {"NVIDIA_API_KEY": "nvapi-" + "x" * 30}
    cat = {m["name"]: m for m in D.model_catalog(reg, session)}
    assert cat["llama-3.2-11b-vision-nv"]["key_available"] is True
    assert cat["gpt-4o"]["key_available"] is False          # a different provider
    # and the scope is gone again afterwards
    assert credentials.resolve("NVIDIA_API_KEY") is None


# ---------------------------------------------------------------------------
# What is ticked when the page opens
# ---------------------------------------------------------------------------

def test_nothing_is_ticked_when_no_key_resolves(reg):
    """The regression this replaced: the old default fell through to models[:1], which
    sorted to a model the roster had recorded as returning 401. Pressing Run then gave
    a success message over an empty table."""
    assert D.default_model_selection(reg) == []


def test_the_default_is_a_model_that_can_actually_run(reg):
    picked = D.default_model_selection(reg, {"NVIDIA_API_KEY": "nvapi-" + "x" * 30})
    assert picked == ["llama-3.2-11b-vision-nv"]
    entry = next(m for m in D.model_catalog(reg) if m["name"] == picked[0])
    assert entry["status"] == "serves"


def test_the_default_is_never_a_model_the_roster_records_as_down(reg):
    down = {a.name for a in OAI_PLUGINS if a.status == "down"}
    assert down, "roster should record at least one dead endpoint"
    for env in _ALL_KEY_ENVS:
        picked = D.default_model_selection(reg, {env: "k" * 32})
        assert not (set(picked) & down), f"{env} defaulted to a dead endpoint: {picked}"


def test_the_default_is_never_a_local_weights_model(reg):
    """It needs no key, so it would otherwise win on every path - but its first run is
    a multi-gigabyte download, which is not something to start by accident."""
    assert "qwen2.5-vl-7b" not in D.default_model_selection(reg)
    assert "qwen2.5-vl-7b" not in D.default_model_selection(reg, {"GROQ_API_KEY": "k" * 32})


# ---------------------------------------------------------------------------
# Refusing before spending
# ---------------------------------------------------------------------------

def test_blocked_models_names_the_variable_to_set(reg):
    blocked = D.blocked_models(reg, ["gpt-4o", "qwen2.5-vl-7b"])
    assert [b["name"] for b in blocked] == ["gpt-4o"]
    assert blocked[0]["api_key_env"] == "OPENAI_API_KEY"


def test_a_session_key_unblocks_its_model(reg):
    assert D.blocked_models(reg, ["gpt-4o"], {"OPENAI_API_KEY": "sk-" + "x" * 32}) == []


# ---------------------------------------------------------------------------
# What the panel displays, and what it must not
# ---------------------------------------------------------------------------

def test_the_key_panel_shows_the_variable_and_never_the_key(reg):
    secret = "nvapi-super-secret-value-9876543210"
    md = D.credential_markdown(reg, {"NVIDIA_API_KEY": secret})
    assert "NVIDIA_API_KEY" in md
    assert "this session" in md
    assert str(len(secret)) in md            # length is shown, as a paste receipt
    assert secret not in md
    assert secret[:12] not in md             # not even a prefix


def test_the_provider_hint_shows_status_and_never_the_key(reg):
    secret = "nvapi-super-secret-value-9876543210"
    hint = D.provider_hint("NVIDIA_API_KEY", {"NVIDIA_API_KEY": secret})
    assert "NVIDIA NIM" in hint and "free tier" in hint
    assert "https://build.nvidia.com" in hint
    assert "llama-3.2-11b-vision-nv" in hint          # what the key unlocks
    assert secret not in hint and secret[:12] not in hint


def test_the_key_panel_says_how_much_of_the_roster_is_reachable(reg):
    assert "1 of 11 models** is runnable" in D.credential_markdown(reg)
    with_key = D.credential_markdown(reg, {"NVIDIA_API_KEY": "k" * 32})
    assert "3 of 11 models** are runnable" in with_key


def test_the_key_panel_dates_its_roster_snapshot(reg):
    from renderprobe.models.openai_compatible import ROSTER_CHECKED
    assert ROSTER_CHECKED in D.credential_markdown(reg)


def test_free_tier_providers_that_serve_come_first_in_the_dropdown(reg):
    """Somebody arriving with no keys should meet the two providers that answered the
    last probe before the four that cost money or were never reached."""
    assert D.key_envs(reg)[:2] == ["NVIDIA_API_KEY", "OPENROUTER_API_KEY"]


def test_the_selector_lists_only_what_can_be_run(reg):
    """Eleven entries across six providers made the one or two somebody can actually
    run harder to find. The unreachable ones are behind `show all`."""
    runnable = [v for _, v in D.model_choices(reg)]
    assert runnable == ["qwen2.5-vl-7b"]                  # needs no key
    assert "gpt-4o" not in runnable

    with_key = [v for _, v in D.model_choices(reg, {"OPENAI_API_KEY": "sk-" + "x" * 32})]
    assert "gpt-4o" in with_key and "gemma-3-27b-or" not in with_key


def test_show_all_reveals_the_rest_and_says_what_each_one_needs(reg):
    labels = dict((v, k) for k, v in D.model_choices(reg, runnable_only=False))
    assert len(labels) == 11
    assert "(no OPENAI_API_KEY)" in labels["gpt-4o"]
    assert "OpenAI" in labels["gpt-4o"]


def test_model_labels_carry_the_roster_status(reg):
    labels = dict((v, k) for k, v in D.model_choices(reg, runnable_only=False))
    assert "serves" in labels["gemma-3-27b-or"]
    assert "not serving" in labels["llama-3.2-90b-vision-nv"]


# ---------------------------------------------------------------------------
# A failed call explains itself
# ---------------------------------------------------------------------------

def test_a_missing_key_reaches_the_result_row_instead_of_only_stderr(reg):
    cfg = ExperimentConfig(
        scene_generator="route", scene_params={"n_towns": 6, "n_hops": 2}, seeds=[0],
        probe="route", models=["gpt-4o"], analyzers=["accuracy"],
    )
    _, results, _ = run_experiment_collect(cfg, reg)
    assert results and all(r.error_flag == "model_error" for r in results)
    why = results[0].meta["model_error"]
    assert "OPENAI_API_KEY" in why
    assert "MissingCredential" not in why      # guidance, not a stack trace type


def test_the_error_summary_collapses_one_cause_into_one_line(reg):
    cfg = ExperimentConfig(
        scene_generator="route", scene_params={"n_towns": 6, "n_hops": 2},
        seeds=[0, 1], probe="route", models=["gpt-4o"], analyzers=["accuracy"],
    )
    _, results, _ = run_experiment_collect(cfg, reg)
    lines = D.model_error_summary(results)
    assert len(lines) == 1, "one missing key should not print six identical sentences"
    assert "gpt-4o" in lines[0] and "OPENAI_API_KEY" in lines[0]
    assert f"({len(results)} calls)" in lines[0]


def test_a_provider_error_quoting_the_key_is_scrubbed_before_it_reaches_the_row(reg,
                                                                               monkeypatch):
    """A 401 body can echo the request, and the request carried the credential."""
    secret = "sk-live-do-not-display-2468013579"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    mod = MagicMock()
    mod.OpenAI.side_effect = RuntimeError(f"401 rejected: Bearer {secret}")
    cfg = ExperimentConfig(
        scene_generator="route", scene_params={"n_towns": 6, "n_hops": 2}, seeds=[0],
        probe="route", models=["gpt-4o"], analyzers=["accuracy"],
    )
    with patch.dict(sys.modules, {"openai": mod}):
        _, results, _ = run_experiment_collect(cfg, reg)
    why = results[0].meta["model_error"]
    assert secret not in why
    assert credentials.MASK in why
    assert "401 rejected" in why


# ---------------------------------------------------------------------------
# The key actually gets used
# ---------------------------------------------------------------------------

def test_a_scoped_key_is_what_the_client_is_built_with(reg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-lose")
    adapter = OpenAICompatibleAdapter(
        name="gpt-4o", model_id="gpt-4o", api_key_env="OPENAI_API_KEY")
    choice = MagicMock()
    choice.message.content = "12"
    reply = MagicMock()
    reply.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = reply
    mod = MagicMock()
    mod.OpenAI.return_value = client

    with patch.dict(sys.modules, {"openai": mod}):
        with credentials.scope({"OPENAI_API_KEY": "session-key-should-win"}):
            adapter.run([Image.new("RGB", (32, 32), "white")], "how many?")

    assert mod.OpenAI.call_args.kwargs["api_key"] == "session-key-should-win"


# ---------------------------------------------------------------------------
# End to end through the actual wired callbacks
# ---------------------------------------------------------------------------

_STATUS = 3          # index of the status markdown in _run's output tuple


def _callback(app, name):
    return next(f.fn for f in app.fns.values()
                if getattr(f.fn, "__name__", "") == name)


@pytest.fixture
def app(reg):
    pytest.importorskip("gradio")
    from renderprobe.ui.app import build_app
    return build_app(registry=reg)


@pytest.fixture
def run_cb(app):
    fn = _callback(app, "_run")

    def call(models, keys=None):
        return fn("route", "route", models, ["accuracy"], 6, 2, None,
                  "0", False, False, "(none)", "", keys or {})
    return call


def test_a_run_with_no_key_is_refused_before_anything_is_spent(run_cb):
    out = run_cb(["gpt-4o"])
    status = out[_STATUS]
    assert "Nothing was run" in status
    assert "OPENAI_API_KEY" in status
    assert "https://platform.openai.com/api-keys" in status
    assert out[0] == [] and out[4] == []          # no results, no table


def test_the_status_never_reports_success_over_an_empty_table(run_cb):
    """The bug this closes: a missing key produced 'Ran 3 probes over 1 scenes' above
    three blank model_error rows, with the word 'key' appearing nowhere in the browser."""
    assert "Ran " not in run_cb(["gpt-4o"])[_STATUS]


def test_a_key_saved_for_one_provider_does_not_unblock_another(run_cb):
    nvidia = {"NVIDIA_API_KEY": "nvapi-" + "x" * 30}
    assert "Nothing was run" not in run_cb(["llama-3.2-11b-vision-nv"], nvidia)[_STATUS]
    status = run_cb(["gpt-4o"], nvidia)[_STATUS]
    assert "Nothing was run" in status and "OPENAI_API_KEY" in status


def test_a_session_key_flows_from_the_panel_into_the_adapter(run_cb):
    choice = MagicMock()
    choice.message.content = "24"
    reply = MagicMock()
    reply.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = reply
    mod = MagicMock()
    mod.OpenAI.return_value = client

    with patch.dict(sys.modules, {"openai": mod}):
        out = run_cb(["gpt-4o"], {"OPENAI_API_KEY": "sk-from-the-ui-box-0123456789"})

    assert "Ran " in out[_STATUS]
    assert mod.OpenAI.call_args.kwargs["api_key"] == "sk-from-the-ui-box-0123456789"
    # and it is gone again the moment the run returned
    assert credentials.resolve("OPENAI_API_KEY") is None


def test_a_failed_call_says_why_in_the_status_line(run_cb):
    mod = MagicMock()
    mod.OpenAI.side_effect = RuntimeError("upstream 503, try later")
    with patch.dict(sys.modules, {"openai": mod}):
        status = run_cb(["gpt-4o"], {"OPENAI_API_KEY": "sk-" + "x" * 30})[_STATUS]
    assert "upstream 503" in status
    assert "excluded from every accuracy" in status


# --- the key panel's own callbacks ----------------------------------------

def test_saving_a_key_updates_the_panel_and_clears_the_box(app, reg):
    save = _callback(app, "_save_key")
    secret = "nvapi-" + "x" * 30
    keys, box, hint, status, models = save("NVIDIA_API_KEY", secret, {}, False, [])

    assert keys == {"NVIDIA_API_KEY": secret}
    assert box["value"] == "", "the credential must not be left sitting in the page"
    assert secret not in hint and secret not in status
    assert "key set" in hint
    unlocked = [v for _, v in models["choices"]]
    assert "llama-3.2-11b-vision-nv" in unlocked


def test_saving_an_empty_key_falls_back_to_the_environment(app):
    save = _callback(app, "_save_key")
    keys, *_ = save("NVIDIA_API_KEY", "   ", {"NVIDIA_API_KEY": "old" * 10}, False, [])
    assert "NVIDIA_API_KEY" not in keys


def test_saving_a_second_key_keeps_the_first(app):
    save = _callback(app, "_save_key")
    keys, *_ = save("NVIDIA_API_KEY", "nvapi-" + "x" * 30, {}, False, [])
    keys, *_ = save("OPENAI_API_KEY", "sk-" + "y" * 30, keys, False, [])
    assert set(keys) == {"NVIDIA_API_KEY", "OPENAI_API_KEY"}


def test_saving_a_key_keeps_a_selection_that_is_still_runnable(app):
    save = _callback(app, "_save_key")
    _, _, _, _, models = save("NVIDIA_API_KEY", "nvapi-" + "x" * 30, {}, False,
                              ["qwen2.5-vl-7b"])
    assert models["value"] == ["qwen2.5-vl-7b"]


def test_show_all_widens_the_selector_without_a_key(app):
    update = _callback(app, "_model_update")
    assert len([v for _, v in update({}, False, [])["choices"]]) == 1
    assert len([v for _, v in update({}, True, [])["choices"]]) == 11


def test_switching_provider_shows_that_provider(app):
    hint = _callback(app, "_on_provider")("OPENROUTER_API_KEY", {})
    assert "OpenRouter" in hint and "no key set" in hint
    assert "gemma-3-27b-or" in hint


# ---------------------------------------------------------------------------
# Which scene the UI opens on
# ---------------------------------------------------------------------------

def test_the_ui_does_not_open_on_a_slow_scene(reg):
    """It used to open on whichever scene sorted first, which was `polycube`: a CPU path
    trace of several seconds on load and on every dial change, and the one scene this
    repo's results say sits at chance. The first thing anybody saw was a spinner."""
    from renderprobe.core.runner import HEAVY_RENDERERS

    opening = D.default_scene(reg)
    assert opening == "route"
    assert opening != sorted(reg.list_scenes())[0], "alphabetical order is not the rule"
    renderer = getattr(reg.get_scene(opening), "default_renderer", None)
    assert renderer not in HEAVY_RENDERERS


def test_a_slow_scene_is_still_chosen_when_it_is_the_only_one(reg, monkeypatch):
    """The rule prefers a fast scene; it must not return nothing when none is fast."""
    monkeypatch.setattr(reg, "list_scenes", lambda: ["polycube"])
    assert D.default_scene(reg) == "polycube"


def test_no_scenes_means_no_opening_scene(reg, monkeypatch):
    monkeypatch.setattr(reg, "list_scenes", lambda: [])
    assert D.default_scene(reg) is None


def test_the_opening_scene_drives_the_dropdown_and_its_probe(reg):
    pytest.importorskip("gradio")
    from renderprobe.ui.app import build_app

    app = build_app(registry=reg)
    dropdowns = {c["props"].get("label"): c["props"].get("value")
                 for c in app.get_config_file()["components"]
                 if c["type"] == "dropdown"}
    assert dropdowns["Scene"] == "route"
    assert dropdowns["Probe (auto-paired to the scene)"] == "route"


def test_a_preview_already_drawn_is_not_drawn_again(reg):
    """A true-3D preview costs seconds; dragging a dial back to a value already seen
    should not pay for it twice."""
    pytest.importorskip("gradio")
    import time

    from renderprobe.ui.app import build_app

    app = build_app(registry=reg)
    preview = next(f.fn for f in app.fns.values()
                   if getattr(f.fn, "__name__", "") == "_preview")

    t = time.perf_counter()
    first_img, first_md = preview("polycube", 5, None, None, "0")
    cold = time.perf_counter() - t

    t = time.perf_counter()
    again_img, again_md = preview("polycube", 5, None, None, "0")
    warm = time.perf_counter() - t

    assert again_img is first_img and again_md == first_md
    assert warm < cold / 10, f"cold {cold:.2f}s, warm {warm:.2f}s"
