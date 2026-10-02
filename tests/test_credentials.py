"""Credential resolution: the session scope, the environment fallback, and scrubbing.

The UI tells people their key is held for their session only, is not copied into the
server's environment, and cannot be seen by anyone else using the same instance. These
tests are what that promise rests on.
"""
from __future__ import annotations

import os
import threading

import pytest

from renderprobe.core import credentials

_ENV = "RENDERPROBE_TEST_API_KEY"
_SECRET = "sk-test-0123456789abcdefghij"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    yield


def test_resolves_from_the_environment_when_no_scope_is_active(monkeypatch):
    monkeypatch.setenv(_ENV, _SECRET)
    assert credentials.resolve(_ENV) == _SECRET
    assert credentials.source(_ENV) == "environment"


def test_a_scope_shadows_the_environment(monkeypatch):
    monkeypatch.setenv(_ENV, _SECRET)
    with credentials.scope({_ENV: "session-key-value"}):
        assert credentials.resolve(_ENV) == "session-key-value"
        assert credentials.source(_ENV) == "session"
    assert credentials.resolve(_ENV) == _SECRET


def test_names_absent_from_the_scope_still_fall_through_to_the_environment(monkeypatch):
    """Somebody may paste one provider's key in the UI and keep another in their shell."""
    monkeypatch.setenv(_ENV, _SECRET)
    with credentials.scope({"OTHER_API_KEY": "x" * 20}):
        assert credentials.resolve(_ENV) == _SECRET


def test_blank_entries_are_dropped_so_an_empty_box_means_use_the_environment(monkeypatch):
    monkeypatch.setenv(_ENV, _SECRET)
    with credentials.scope({_ENV: "   ", "UNSET_API_KEY": ""}):
        assert credentials.resolve(_ENV) == _SECRET
        assert credentials.resolve("UNSET_API_KEY") is None


def test_the_scope_unbinds_even_when_the_run_raises():
    with pytest.raises(ValueError):
        with credentials.scope({_ENV: _SECRET}):
            raise ValueError("the run blew up")
    assert credentials.resolve(_ENV) is None


def test_nested_scopes_merge_with_the_inner_one_winning():
    with credentials.scope({"A_API_KEY": "outer-aaaaaaaa", "B_API_KEY": "outer-bbbbbbbb"}):
        with credentials.scope({"A_API_KEY": "inner-aaaaaaaa"}):
            assert credentials.resolve("A_API_KEY") == "inner-aaaaaaaa"
            assert credentials.resolve("B_API_KEY") == "outer-bbbbbbbb"
        assert credentials.resolve("A_API_KEY") == "outer-aaaaaaaa"


def test_a_scope_is_invisible_to_another_thread():
    """The whole reason this is a ContextVar rather than os.environ.

    One `renderprobe ui` process can serve several people. A key bound for one request
    must not be resolvable from another, which is exactly what writing it into the
    environment would do.
    """
    bound = threading.Event()
    checked = threading.Event()
    seen: list[str | None] = []

    def holder():
        with credentials.scope({_ENV: _SECRET}):
            bound.set()
            checked.wait(timeout=5)

    def observer():
        bound.wait(timeout=5)
        seen.append(credentials.resolve(_ENV))
        checked.set()

    t1, t2 = threading.Thread(target=holder), threading.Thread(target=observer)
    t1.start(), t2.start()
    t1.join(timeout=5), t2.join(timeout=5)
    assert seen == [None]


def test_a_scope_never_reaches_the_process_environment():
    with credentials.scope({_ENV: _SECRET}):
        assert _ENV not in os.environ
    assert _ENV not in os.environ


# ---------------------------------------------------------------------------
# describe / scrub: what may be displayed, and what must never be
# ---------------------------------------------------------------------------

def test_describe_reports_presence_and_length_but_never_the_key():
    with credentials.scope({_ENV: _SECRET}):
        info = credentials.describe(_ENV)
    assert info == {"env": _ENV, "source": "session", "present": True,
                    "length": len(_SECRET)}
    assert _SECRET not in repr(info)


def test_scrub_masks_a_session_key_quoted_back_by_a_provider():
    text = f"401 Unauthorized: the key {_SECRET} is not valid for this endpoint"
    with credentials.scope({_ENV: _SECRET}):
        cleaned = credentials.scrub(text)
    assert _SECRET not in cleaned
    assert credentials.MASK in cleaned
    assert "401 Unauthorized" in cleaned          # the useful part survives


def test_scrub_masks_a_key_held_in_the_environment(monkeypatch):
    monkeypatch.setenv(_ENV, _SECRET)
    assert _SECRET not in credentials.scrub(f"request failed, sent {_SECRET}")


def test_scrub_leaves_short_strings_alone(monkeypatch):
    """A 3-character value would otherwise mangle every word it happens to appear in."""
    monkeypatch.setenv("TINY_API_KEY", "abc")
    assert credentials.scrub("abcdef is a fine word") == "abcdef is a fine word"


def test_missing_credential_is_an_environment_error():
    """Subclassing keeps anything that already caught EnvironmentError working."""
    assert issubclass(credentials.MissingCredential, EnvironmentError)
