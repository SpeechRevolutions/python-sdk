"""Shared fixtures — no network is touched by any test in this suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, json_body=None, text="", headers=None, content=None):
        self.status_code = status_code
        self._json = json_body
        self.text = text if text else ("" if json_body is None else "{}")
        self.headers = headers or {}
        self.content = content if content is not None else self.text.encode()

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class RecordingSession(requests.Session):
    """A Session that replays a scripted list of outcomes and records calls.

    Each entry in `script` is either an Exception (raised) or a FakeResponse.
    The last entry repeats once exhausted, so a test asserting "no retry" fails
    loudly on an extra call rather than hanging.
    """

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):  # type: ignore[override]
        self.calls.append((method, url, kwargs))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def attempts(self) -> int:
        return len(self.calls)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch, request):
    """Collapse retry backoff so the suite stays fast.

    This makes the functional tests quick, but it also means they say NOTHING
    about how long a retry ladder actually takes in production. Anything
    asserting real elapsed time must opt out with `@pytest.mark.real_sleep`
    — see tests/test_timing_behaviour.py, which pins the costs this would
    otherwise hide.
    """
    if "real_sleep" in request.keywords:
        return

    import asyncio as _asyncio

    import speechrevolutions.async_client as async_mod
    import speechrevolutions.client as client_mod

    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)

    # The async client sleeps via asyncio, which the sync patch above does not
    # touch — without this its retry ladders run at full wall-clock cost.
    async def _instant(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(async_mod.asyncio, "sleep", _instant)
    monkeypatch.setattr(_asyncio, "sleep", _instant)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("SPEECHREVOLUTIONS_API_KEY", "test-key-123")
    return "test-key-123"
