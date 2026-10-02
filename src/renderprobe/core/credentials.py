"""Where a model adapter's API key comes from.

A key is resolved at CALL time, from two places, in this order:

1. the active **credential scope** - a mapping the caller binds around one call tree,
   held in a :class:`contextvars.ContextVar`;
2. the process **environment**.

The scope exists for the UI. One ``renderprobe ui`` process can serve more than one
person, so writing a visitor's key into ``os.environ`` would hand it to every other
visitor's run and leave it there after the request returned. A ``ContextVar`` binds to
the calling context instead: a key supplied for one run is unreachable from another,
and it is unbound when the scope exits whether or not the run raised. That is what
makes a key box in the UI safe to expose on a machine somebody else can reach.

The environment stays the default and the fallback, so nothing that worked before
changes: the CLI, the tools and a bare ``adapter.run()`` all still read the same
variables they always did.

Nothing here logs, prints or persists a key, and nothing returns one to a caller that
could not already read it. :func:`describe` is what a UI displays. :func:`scrub` is
what every other piece of text passes through before it reaches a screen, a log or a
``Result`` row, so a provider error quoting our own request back at us cannot put the
credential on somebody's monitor.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_SCOPE: ContextVar[Mapping[str, str]] = ContextVar("renderprobe_credentials", default={})

# What a credential is replaced with anywhere text is shown.
MASK = "<redacted>"

# Under this length a string is not scrubbed. A real key is far longer, and masking
# every occurrence of a 3-character value would mangle the very text scrubbing exists
# to keep readable.
_MIN_SECRET_LEN = 8

# Environment variables treated as credentials by `scrub`. Adapters name their own
# variable explicitly, but an error body can quote a key this process holds under a
# name no adapter declared, so scrubbing works by shape as well as by declaration.
_SECRET_NAME = re.compile(r"(?:API_KEY|_TOKEN|_SECRET|PASSWORD)$")


class MissingCredential(EnvironmentError):
    """No key could be resolved for a model the caller asked to run.

    Subclasses ``EnvironmentError`` so anything that already caught that keeps working.
    It exists as its own type so the runner can show its message on its own: this is
    the one adapter failure that is a piece of user guidance rather than a fault, and
    prefixing it with an exception class name buries the instruction.
    """


@contextmanager
def scope(values: Mapping[str, str] | None) -> Iterator[None]:
    """Bind credentials for the duration of the block.

    ``values`` maps an environment-variable NAME to a key, for example
    ``{"NVIDIA_API_KEY": "nvapi-..."}``. Blank and missing entries are dropped, so a UI
    can pass its whole panel in without first filtering out the boxes nobody filled.

    Bound names shadow the environment; names absent from the mapping still fall
    through to it. That is what lets somebody paste one provider's key into the UI and
    keep another's in their shell. Nested scopes merge, inner winning.
    """
    clean = {k: v.strip() for k, v in (values or {}).items() if isinstance(v, str) and v.strip()}
    token = _SCOPE.set({**_SCOPE.get(), **clean})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def resolve(env_name: str) -> str | None:
    """The credential for ``env_name``: the active scope first, then the environment."""
    scoped = _SCOPE.get().get(env_name)
    if scoped and scoped.strip():
        return scoped.strip()
    from_env = os.environ.get(env_name) or ""
    return from_env.strip() or None


def source(env_name: str) -> str | None:
    """Where ``env_name`` resolves from: ``"session"``, ``"environment"``, or None."""
    scoped = _SCOPE.get().get(env_name)
    if scoped and scoped.strip():
        return "session"
    if (os.environ.get(env_name) or "").strip():
        return "environment"
    return None


def describe(env_name: str) -> dict[str, Any]:
    """A displayable status record for one credential.

    Carries where the key came from and how long it is, never any part of the key
    itself. Length is enough to confirm a paste landed and short enough to be useless
    to anyone reading over a shoulder.
    """
    value = resolve(env_name)
    return {
        "env": env_name,
        "source": source(env_name),
        "present": value is not None,
        "length": len(value) if value else 0,
    }


def _candidates() -> list[str]:
    """Every credential value this process can currently see. Never displayed."""
    out = [v for v in _SCOPE.get().values() if isinstance(v, str)]
    out += [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and v]
    # Longest first, so a key that contains a shorter one is masked whole rather than
    # being left with a readable tail.
    return sorted({v.strip() for v in out if len(v.strip()) >= _MIN_SECRET_LEN},
                  key=len, reverse=True)


def scrub(text: str) -> str:
    """``text`` with every credential this process can see replaced by ``MASK``.

    A provider is free to quote the request it rejected, and the request carried the
    key. Anything on its way to a screen, a log line or a stored ``Result`` goes
    through here first.
    """
    if not text:
        return text
    for secret in _candidates():
        if secret in text:
            text = text.replace(secret, MASK)
    return text
