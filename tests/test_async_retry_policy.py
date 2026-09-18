"""The async client must enforce the same create-safety rule as the sync one.

Before this suite existed the async client had no retry loop at all, so it
silently ignored `max_retries` and diverged from the documented behaviour.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from speechrevolutions import AsyncSpeechRevolutions
from speechrevolutions.exceptions import APIError, TimeoutError as STTTimeoutError

pytestmark = pytest.mark.asyncio

OK_BODY = {"job_id": "j1", "upload_url": "u", "download_url": "d"}


def make_client(script, max_retries=3):
    """script: list of (status, body) tuples or Exceptions, replayed in order."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        item = script[i]
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)
    client = AsyncSpeechRevolutions(
        api_key="k",
        client=httpx.AsyncClient(transport=transport),
        max_retries=max_retries,
    )
    return client, calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(*_):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


# --- create calls: no retry on ambiguity ----------------------------------

@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_create_does_not_retry_on_5xx(status):
    c, calls = make_client([(status, {"detail": "boom"})])
    with pytest.raises(APIError):
        await c.create_upload_job(1024)
    assert calls["n"] == 1


async def test_create_does_not_retry_on_read_timeout():
    c, calls = make_client([httpx.ReadTimeout("read timed out")])
    with pytest.raises(STTTimeoutError):
        await c.create_upload_job(1024)
    assert calls["n"] == 1


async def test_create_does_not_retry_on_remote_protocol_error():
    c, calls = make_client([httpx.RemoteProtocolError("server disconnected")])
    with pytest.raises(APIError):
        await c.create_upload_job(1024)
    assert calls["n"] == 1


# --- create calls: retry when it provably never landed --------------------

async def test_create_retries_on_connect_timeout():
    c, calls = make_client([httpx.ConnectTimeout("connect timed out"), (200, OK_BODY)])
    job = await c.create_upload_job(1024)
    assert job.job_id == "j1"
    assert calls["n"] == 2


async def test_create_retries_on_connect_error():
    c, calls = make_client([httpx.ConnectError("refused"), (200, OK_BODY)])
    job = await c.create_upload_job(1024)
    assert job.job_id == "j1"
    assert calls["n"] == 2


async def test_create_retries_on_429():
    c, calls = make_client([(429, {"detail": "slow down"}), (200, OK_BODY)])
    job = await c.create_upload_job(1024)
    assert job.job_id == "j1"
    assert calls["n"] == 2


# --- non-create calls retry freely ----------------------------------------

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_safe_path_retries_on_transient_status(status):
    c, calls = make_client([(status, {}), (200, {})])
    await c.cancel_job("j1")
    assert calls["n"] == 2


async def test_safe_path_retries_on_read_timeout():
    c, calls = make_client([httpx.ReadTimeout("t"), (200, {})])
    await c.cancel_job("j1")
    assert calls["n"] == 2


async def test_max_retries_is_honoured():
    err = httpx.ConnectError("refused")
    c, calls = make_client([err] * 6, max_retries=2)
    with pytest.raises(APIError):
        await c.create_upload_job(1024)
    assert calls["n"] == 3  # 1 initial + 2 retries


async def test_max_retries_zero_means_one_attempt():
    c, calls = make_client([(500, {}), (200, {})], max_retries=0)
    with pytest.raises(APIError):
        await c.cancel_job("j1")
    assert calls["n"] == 1
