"""OpenAI-compatible vision model adapter (OpenAI, Gemini via base_url, etc.).

Design (lazy import so discovery needs no heavy deps):
- Construction takes config ONLY. No key read, no client built at construction.
- The key is resolved at call time through ``core.credentials``, which reads the
  active session scope first and the environment second. A UI can therefore supply
  a key per session without it ever entering ``os.environ``.
- `import openai` is lazy - inside run() only - so autodiscovery works with
  no `openai` package installed and no key in env.
- A missing key or package raises at call time; the runner sandboxes it into
  a `model_error` Result rather than crashing the run.
- Images are base64-encoded PNG data URLs sent as image_url message parts.
- A new model needs only a PLUGINS entry, no new code.
"""
from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from renderprobe.core import credentials
from renderprobe.core.schema import Response

# Largest inline image payload to send, in base64 characters. NVIDIA's OpenAI-compatible
# endpoint rejects inline images past ~180 KB and wants an asset upload instead; a
# 820x430 render is ~640-800 KB as base64 PNG, which in practice hangs the request
# rather than returning an error. The cap sits below that limit with room to spare.
_MAX_B64 = 170_000
# JPEG quality ladder. Every model gets the SAME encoding for the same image, whatever
# its provider allows, because a comparison across models must not also be a comparison
# across codecs. The first rung is visually indistinguishable from the PNG on these
# renders; the lower ones exist so an unusually large scene degrades predictably instead
# of failing. Lossy at all is a real choice on a perception benchmark, and the honest
# reason for it is that no mainstream vision endpoint accepts these images losslessly
# inline.
_JPEG_QUALITY = (92, 85, 75, 60)


def _encode_image(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    for quality in _JPEG_QUALITY:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64) <= _MAX_B64:
            return f"data:image/jpeg;base64,{b64}"
    # Still too big at the lowest quality: shrink until it fits, halving each time.
    # Reaching here means a scene far larger than anything this repo renders.
    scaled = rgb
    for _ in range(4):
        scaled = scaled.resize((max(1, scaled.width // 2), max(1, scaled.height // 2)))
        buf = io.BytesIO()
        scaled.save(buf, format="JPEG", quality=_JPEG_QUALITY[0])
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64) <= _MAX_B64:
            return f"data:image/jpeg;base64,{b64}"
    return f"data:image/jpeg;base64,{b64}"


# Provider directory, keyed by the environment variable that holds the key. One place
# for the UI to learn what a variable is for and where somebody gets one, so adding a
# provider stays a data edit. `free` marks a tier that costs nothing to try, which is
# what a first-time reader of this repo needs to know first.
PROVIDERS: dict[str, dict[str, Any]] = {
    "NVIDIA_API_KEY": {
        "provider": "NVIDIA NIM",
        "url": "https://build.nvidia.com",
        "free": True,
    },
    "OPENROUTER_API_KEY": {
        "provider": "OpenRouter",
        "url": "https://openrouter.ai/keys",
        "free": True,
    },
    "GROQ_API_KEY": {
        "provider": "Groq",
        "url": "https://console.groq.com/keys",
        "free": True,
    },
    "GEMINI_API_KEY": {
        "provider": "Google AI Studio",
        "url": "https://aistudio.google.com/apikey",
        "free": True,
    },
    "OPENAI_API_KEY": {
        "provider": "OpenAI",
        "url": "https://platform.openai.com/api-keys",
        "free": False,
    },
    "XAI_API_KEY": {
        "provider": "xAI",
        "url": "https://console.x.ai",
        "free": False,
    },
}

# When the roster below was last probed with one real call per entry. Displayed next to
# the statuses so nobody reads a stale snapshot as a live check.
ROSTER_CHECKED = "2026-09-20"


class OpenAICompatibleAdapter:
    """Single adapter class; concrete models differ only in construction args."""

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key_env: str,
        base_url: str | None = None,
        is_open: bool = False,
        # Enough for a model that reasons aloud to REACH its conclusion. Measured on
        # the polycube oracle, where gemma-3-27b writes out its comparison: it runs
        # 1100-1300 tokens and stops on its own, while 256 and 1024 both cut it off
        # mid-comparison. Models that answer in one word still emit one word; this is a
        # ceiling, not a target.
        max_tokens: int = 2048,
        sdk_max_retries: int = 0,
        # What the last roster probe found: "serves", "down", or "unverified". This is a
        # dated snapshot, not a live check - see ROSTER_CHECKED. It is carried as data
        # rather than a comment so the UI can warn before a run instead of after it.
        status: str = "unverified",
        status_note: str = "",
    ) -> None:
        self.name = name
        self.is_open = is_open
        # Public: the UI reads these to group models by provider and to tell somebody
        # which variable to fill in. Neither is or contains a credential.
        self.api_key_env = api_key_env
        self.provider = PROVIDERS.get(api_key_env, {}).get("provider", api_key_env)
        self.status = status
        self.status_note = status_note
        self._model_id = model_id
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._sdk_max_retries = sdk_max_retries

    def run(self, images: list[Image.Image], prompt: str) -> Response:
        import openai  # lazy import - fails cleanly if not installed

        api_key = credentials.resolve(self.api_key_env)
        if not api_key:
            where = PROVIDERS.get(self.api_key_env, {}).get("url", "")
            raise credentials.MissingCredential(
                f"No API key for {self.name}. Export {self.api_key_env}, or enter it in "
                f"the API keys panel on the UI's Run tab."
                + (f" Keys for {self.provider}: {where}" if where else "")
            )

        client = openai.OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            # 0 = fail fast (correct for daily-cap providers like OpenRouter/Gemini where
            # retrying is futile and burns quota). Set sdk_max_retries>0 for per-minute
            # throttling providers (e.g. Groq) where the SDK's retry-after logic helps.
            max_retries=self._sdk_max_retries,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(img)},
            })

        response = client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=self._max_tokens,
        )
        if not response.choices:
            raise RuntimeError(
                f"Empty choices in response from {self._model_id}"
                " - provider may have blocked or dropped the request."
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        return Response(text=text, confidence=None,
                        truncated=getattr(choice, "finish_reason", None) == "length")

    def attention(self, images: list[Image.Image], prompt: str) -> None:
        return None  # closed API: no attention weights


# Concrete model entries - add a model here, no new code needed. Keys are resolved at
# call time (see run()), never at import, so this list is safe to read with no keys set.
#
# `status` records what the last probe found, one real call per entry, on the date in
# ROSTER_CHECKED. Endpoints move under you: a "serves" here is evidence that the id was
# right on that date, not a promise about today. The UI shows both the status and the
# date, so a stale snapshot cannot be mistaken for a live check.

PLUGINS: list[OpenAICompatibleAdapter] = [
    OpenAICompatibleAdapter(
        name="gpt-4o",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        status="unverified",
        status_note="never probed from this repo; no OpenAI key was available",
    ),
    OpenAICompatibleAdapter(
        name="gpt-4o-mini",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        status="unverified",
        status_note="never probed from this repo; no OpenAI key was available",
    ),
    OpenAICompatibleAdapter(
        name="gemini-2.0-flash",
        model_id="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        status="unverified",
        status_note="401 on the key available at the time, so the endpoint itself was "
                    "never exercised; bring your own key and it may well work",
    ),
    # xAI Grok exposes an OpenAI-compatible endpoint; grok-2-vision is its VLM.
    # NOTE: DeepSeek is deliberately absent - its hosted API (deepseek-chat /
    # deepseek-reasoner) is text-only; DeepSeek-VL2 ships as open weights, so it
    # would be driven through hf_vlm.py, not this adapter.
    OpenAICompatibleAdapter(
        name="grok-2-vision",
        model_id="grok-2-vision-1212",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        status="unverified",
        status_note="never probed from this repo; no xAI key was available",
    ),
    # OpenRouter aggregates many providers behind one OpenAI-compatible endpoint
    # and lists several vision models with a ":free" suffix (zero cost, daily
    # rate-limited). NOTE: the free-tier roster rotates - verify the model_id
    # still exists at openrouter.ai/models before running; swapping it is a
    # one-line edit here, no code change. (Verified 2026-06-30 against the live
    # openrouter.ai/api/v1/models catalog; the original qwen2.5-vl-72b/llama-3.2
    # ":free" slugs had since been retired in favor of these.)
    OpenAICompatibleAdapter(
        name="nemotron-vl-or",
        model_id="nvidia/nemotron-nano-12b-v2-vl:free",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        status="down",
        status_note="404, no endpoints for that slug",
    ),
    OpenAICompatibleAdapter(
        name="gemma-4-26b-or",
        model_id="google/gemma-4-26b-a4b-it:free",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        status="unverified",
        status_note="429 from the upstream provider, so the model was never reached; "
                    "the free tier is heavily contended",
    ),
    OpenAICompatibleAdapter(
        name="gemma-3-27b-or",
        model_id="google/gemma-3-27b-it",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        status="serves",
        status_note="answered in under a second; the model behind this repo's route run",
    ),
    # Groq runs open VLMs at low latency with a generous free tier (console.groq.com).
    OpenAICompatibleAdapter(
        name="llama-4-scout-groq",
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        sdk_max_retries=2,  # Groq has per-minute throttling (short waits); retrying is fine
        status="down",
        status_note="404, the model id moved again since 2026-07-06",
    ),
    # NVIDIA NIM (build.nvidia.com) - OpenAI-compatible, generous free-tier credits.
    # NOTE (verified 2026-09-02): NVIDIA's /v1/models CATALOG lists many more VLMs
    # (phi-3-vision, gemma-3, kosmos-2, ...), but this account's serverless
    # `integrate.api.nvidia.com` only actually SERVES the two llama-3.2 vision models
    # - every other id returns 404 "Function not found for account". A stronger
    # contrast model therefore means either the 90b below or a different provider key.
    OpenAICompatibleAdapter(
        name="llama-3.2-11b-vision-nv",
        model_id="meta/llama-3.2-11b-vision-instruct",
        api_key_env="NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        sdk_max_retries=1,
        status="serves",
        status_note="answered in under a second",
    ),
    OpenAICompatibleAdapter(
        name="llama-3.2-90b-vision-nv",
        model_id="meta/llama-3.2-90b-vision-instruct",
        api_key_env="NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        sdk_max_retries=1,
        status="down",
        status_note="in NVIDIA's catalog but times out at 180 s even on a text-only "
                    "request, so it is not an image-size problem",
    ),
]
