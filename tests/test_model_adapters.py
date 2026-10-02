"""Real model adapter tests - no live network, no API keys."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from renderprobe.core.registry import Registry
from renderprobe.models.hf_vlm import HFVLMAdapter
from renderprobe.models.openai_compatible import PLUGINS as OAI_PLUGINS
from renderprobe.models.openai_compatible import OpenAICompatibleAdapter


def _blank_image() -> Image.Image:
    return Image.new("RGB", (64, 64), "white")


# ---------------------------------------------------------------------------
# Registry: adapters discover cleanly without keys or heavy deps installed
# ---------------------------------------------------------------------------

def test_autodiscover_registers_openai_adapters():
    r = Registry()
    r.autodiscover()
    # The three PLUGINS entries in openai_compatible.py should all register
    assert r.get_model("gpt-4o") is not None
    assert r.get_model("gpt-4o-mini") is not None
    assert r.get_model("gemini-2.0-flash") is not None


def test_autodiscover_registers_hf_adapter():
    r = Registry()
    r.autodiscover()
    assert r.get_model("qwen2.5-vl-7b") is not None


def test_adapter_construction_needs_no_key():
    """OpenAICompatibleAdapter construction must not read env or import openai."""
    adapter = OpenAICompatibleAdapter(
        name="test", model_id="gpt-4o", api_key_env="NONEXISTENT_KEY"
    )
    assert adapter.name == "test"
    assert adapter.is_open is False


def test_hf_adapter_construction_needs_no_torch():
    """HFVLMAdapter construction must not import torch/transformers."""
    adapter = HFVLMAdapter(name="test-hf", model_id="some/model")
    assert adapter.name == "test-hf"
    assert adapter.is_open is True


# ---------------------------------------------------------------------------
# OpenAICompatibleAdapter.run() - mock openai client
# ---------------------------------------------------------------------------

def _mock_openai_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_openai_mock(response_text: str) -> tuple[MagicMock, MagicMock]:
    """Return (mock_openai_module, mock_client) wired for a single response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(response_text)
    mock_openai_mod = MagicMock()
    mock_openai_mod.OpenAI.return_value = mock_client
    return mock_openai_mod, mock_client


def test_openai_adapter_run_with_images(monkeypatch):
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAICompatibleAdapter(
        name="gpt-4o", model_id="gpt-4o", api_key_env="OPENAI_API_KEY"
    )
    mock_openai_mod, mock_client = _make_openai_mock("left")

    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        result = adapter.run([_blank_image()], "Is the red object left or right?")

    assert result.text == "left"
    assert result.confidence is None
    call_args = mock_client.chat.completions.create.call_args
    content = call_args.kwargs["messages"][0]["content"]
    types = [part["type"] for part in content]
    assert "image_url" in types
    assert "text" in types


def test_openai_adapter_run_blind(monkeypatch):
    """No images -> no image_url parts in the message."""
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAICompatibleAdapter(
        name="gpt-4o", model_id="gpt-4o", api_key_env="OPENAI_API_KEY"
    )
    mock_openai_mod, mock_client = _make_openai_mock("right")

    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        result = adapter.run([], "Is the red object left or right?")

    assert result.text == "right"
    call_args = mock_client.chat.completions.create.call_args
    content = call_args.kwargs["messages"][0]["content"]
    types = [part["type"] for part in content]
    assert "image_url" not in types


def test_openai_adapter_missing_key_raises(monkeypatch):
    import sys

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAICompatibleAdapter(
        name="gpt-4o", model_id="gpt-4o", api_key_env="OPENAI_API_KEY"
    )
    mock_openai_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            adapter.run([_blank_image()], "test prompt")


def test_openai_adapter_base_url_forwarded(monkeypatch):
    import sys

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = OpenAICompatibleAdapter(
        name="gemini-2.0-flash",
        model_id="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    mock_openai_mod, _ = _make_openai_mock("left")

    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        adapter.run([_blank_image()], "prompt")

    mock_openai_mod.OpenAI.assert_called_once_with(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=0,
    )


def test_openai_adapter_attention_returns_none():
    adapter = OpenAICompatibleAdapter(
        name="gpt-4o", model_id="gpt-4o", api_key_env="OPENAI_API_KEY"
    )
    assert adapter.attention([_blank_image()], "prompt") is None


# ---------------------------------------------------------------------------
# HFVLMAdapter.run() - mock transformers + torch
# ---------------------------------------------------------------------------

def test_hf_adapter_run_mocked(monkeypatch):
    adapter = HFVLMAdapter(name="qwen2.5-vl-7b", model_id="Qwen/Qwen2.5-VL-7B-Instruct")

    import sys
    # Mock out torch and transformers so they don't need to be installed
    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = lambda s: None
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.float16 = "float16"
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float32 = "float32"

    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, key: MagicMock(shape=[1, 5])

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = mock_output

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt text"
    mock_processor.batch_decode.return_value = ["left"]

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, k: MagicMock(shape=[1, 4])
    mock_inputs.to.return_value = mock_inputs

    mock_processor.return_value = mock_inputs

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForVision2Seq.from_pretrained.return_value = mock_model
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor

    from renderprobe.models import hf_vlm
    hf_vlm._MODEL_CACHE.clear()
    hf_vlm._PROCESSOR_CACHE.clear()

    with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
        result = adapter.run([_blank_image()], "Is the red object left or right?")

    assert isinstance(result.text, str)


def test_hf_adapter_attention_returns_none():
    adapter = HFVLMAdapter(name="test-hf", model_id="some/model")
    assert adapter.attention([_blank_image()], "prompt") is None


# ---------------------------------------------------------------------------
# PLUGINS list shape
# ---------------------------------------------------------------------------

def test_openai_plugins_list_has_required_models():
    names = [p.name for p in OAI_PLUGINS]
    assert "gpt-4o" in names
    assert "gemini-2.0-flash" in names


def test_all_openai_plugins_are_closed():
    for p in OAI_PLUGINS:
        assert p.is_open is False, f"{p.name} should be closed (is_open=False)"


def test_inline_images_stay_under_the_provider_payload_cap():
    """A 820x430 render is ~700 KB as base64 PNG, past what NVIDIA accepts inline, and
    the request hangs rather than erroring. Every image goes out under the cap."""
    from PIL import Image

    from renderprobe.models.openai_compatible import _MAX_B64, _encode_image
    for size in ((820, 430), (1600, 1200), (3000, 3000)):
        img = Image.effect_noise(size, 64).convert("RGB")
        url = _encode_image(img)
        assert url.startswith("data:image/jpeg;base64,")
        assert len(url) <= _MAX_B64 + 32, f"{size} encoded to {len(url)} chars"


def test_every_model_gets_the_same_encoding():
    """A comparison across models must not also be a comparison across codecs, so the
    encoder takes no provider argument and has no per-provider branch."""
    import inspect

    from PIL import Image

    from renderprobe.models.openai_compatible import _encode_image
    assert list(inspect.signature(_encode_image).parameters) == ["img"]
    img = Image.effect_noise((820, 430), 64).convert("RGB")
    assert _encode_image(img) == _encode_image(img.copy())
