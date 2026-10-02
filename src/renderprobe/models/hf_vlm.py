"""HuggingFace vision-language model adapter.

Design (lazy import so discovery needs no heavy deps):
- `transformers` and `torch` are lazy-imported inside run() only.
- Construction is config-only: model_id string, device preference, dtype.
- is_open=True - these models can in principle provide attention, but
  attention() returns None; saliency maps are not implemented.
- A new open model needs only a PLUGINS entry with a different model_id.

Supported model families (tested model IDs listed in PLUGINS):
  - Qwen/Qwen2.5-VL-*  (via AutoModelForVision2Seq + AutoProcessor)
  - Any model following the same transformers vision-language API
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from renderprobe.core.schema import Response

# Cache loaded models/processors across calls to avoid re-loading on every run.
_MODEL_CACHE: dict[str, Any] = {}
_PROCESSOR_CACHE: dict[str, Any] = {}


class HFVLMAdapter:
    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.name = name
        self.is_open = True
        # Public metadata the UI groups and gates on, mirroring the closed adapters.
        # `api_key_env = None` is the signal that this model needs no credential: the
        # weights run locally, so the UI offers it whether or not any key is set.
        self.api_key_env: str | None = None
        self.provider = "local weights"
        self.status = "unverified"
        self.status_note = ("runs locally through transformers; needs the model "
                            "downloaded and enough memory, not a key")
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens

    def _load(self) -> tuple[Any, Any]:
        """Lazy-load and cache the model + processor."""
        if self._model_id not in _MODEL_CACHE:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            dtype_map = {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self._dtype, "auto")

            processor = AutoProcessor.from_pretrained(self._model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                device_map=self._device,
            )
            model.eval()
            _PROCESSOR_CACHE[self._model_id] = processor
            _MODEL_CACHE[self._model_id] = model

        return _MODEL_CACHE[self._model_id], _PROCESSOR_CACHE[self._model_id]

    def run(self, images: list[Image.Image], prompt: str) -> Response:
        import torch

        model, processor = self._load()

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self._max_new_tokens)

        # Decode only the generated tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        return Response(text=text, confidence=None)

    def attention(self, images: list[Image.Image], prompt: str) -> None:
        return None  # future: implement saliency maps here


PLUGINS: list[HFVLMAdapter] = [
    HFVLMAdapter(
        name="qwen2.5-vl-7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        device="auto",
        dtype="bfloat16",
    ),
]
