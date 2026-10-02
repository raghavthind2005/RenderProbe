"""COPY-ME TEMPLATE - a custom model adapter.

WHAT A MODEL ADAPTER IS
    It wires one VLM into RenderProbe: given a list of PIL images and a prompt string,
    return the model's text reply as a Response. That is the only thing RenderProbe
    needs - how you produce the reply (a hosted API, local weights, a router) is up to
    you.

HOW TO USE
    1. Copy out, drop the underscore, rename the class and ``name``.
    2. Replace the body of ``run`` with your real inference call.
    3. renderprobe validate my_models/my_vlm.py      (conformance only - see note)
    4. renderprobe run my_exp.yaml --plugin-dir my_models

KEYS & COST
    Resolve credentials at CALL time through ``core.credentials`` (as below), never
    hard-code them and never read them at construction - that keeps autodiscovery and
    validation free of secrets and side effects. ``credentials.resolve`` checks the
    session scope the UI binds before falling back to the environment, so declaring
    ``api_key_env`` below is what lets somebody run your adapter from the UI's key
    panel instead of having to restart the server with a new variable exported.

    Because a real call costs quota, ``validate`` checks a model adapter for
    CONFORMANCE ONLY; it never actually calls it.

    For OpenAI-compatible HTTP APIs you usually don't need this file at all - just add
    an entry to ``models/openai_compatible.py``. Write a full adapter when your model
    isn't OpenAI-compatible (local weights, a bespoke SDK, a custom router).
"""
from __future__ import annotations

from PIL import Image

from renderprobe.core import credentials
from renderprobe.core.schema import Response


class _TemplateModel:
    name = "template_model"
    is_open = True   # True for open weights / self-hosted; False for a closed hosted API

    # OPTIONAL display metadata. None of it is part of the ModelAdapter Protocol, so an
    # adapter that omits all of it stays conformant - the UI then shows it as a model
    # needing no key. Declaring `api_key_env` is what puts a key box for your provider
    # on the UI's Run tab and what lets the UI refuse a run before spending a call.
    api_key_env: str | None = "MY_PROVIDER_API_KEY"
    provider = "My Provider"            # grouping label for the key panel
    status = "unverified"               # "serves" | "down" | "unverified"
    status_note = ""                    # one line of context for the status

    def run(self, images: list[Image.Image], prompt: str) -> Response:
        # --- replace everything below with your real inference call ---
        _api_key = credentials.resolve("MY_PROVIDER_API_KEY")   # read at call time
        if not _api_key:
            # Raising here is fine: the runner sandboxes it into a `model_error`
            # Result instead of crashing the whole experiment.
            raise credentials.MissingCredential(
                "Export MY_PROVIDER_API_KEY, or enter it in the UI's API keys panel."
            )

        # e.g. reply = my_client.generate(images=images, prompt=prompt)
        reply = "TODO: call your model and return its text here"
        return Response(text=reply, confidence=None)   # confidence optional

    def attention(self, images: list[Image.Image], prompt: str):
        # Optional: return attention maps if your model exposes them, else None.
        return None


PLUGIN = _TemplateModel()
