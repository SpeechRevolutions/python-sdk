"""Retry ladders, measured in real wall-clock seconds.

Every other suite runs under an autouse fixture that no-ops `sleep`, which keeps
them fast but means they cannot see how long anything actually takes. These opt
out with `@pytest.mark.real_sleep` and measure, so a latency regression fails
here rather than in a customer's p99.

The headline finding these pin: **an SSE endpoint that refuses the connection
costs ~30 seconds before the SDK falls back to polling.** That is
`SSE_MAX_RECONNECTS` (10) x `SSE_RECONNECT_DELAY` (3.0s), and it applies
identically to the sync and async clients.

It matters because the refusal case is not hypothetical: a proxy, load balancer
or corporate egress that does not pass `text/event-stream` answers every stream
request with a non-2xx forever. Retrying ten times on a 3-second timer is the
right shape for a DROPPED connection and the wrong shape for a REFUSED one —
the endpoint is not going to start working three seconds later. For a customer
behind such a proxy this adds ~30s to every single job, which is longer than
the median job takes to transcribe (~24s in production).

These tests assert the CURRENT behaviour, not the desired behaviour. If the
retry policy is changed to distinguish "refused" from "dropped", change the
expectations here deliberately.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mock_api import MockAPI
from speechrevolutions import AsyncSpeechRevolutions, SpeechRevolutions
from speechrevolutions._config import SSE_MAX_RECONNECTS, SSE_RECONNECT_DELAY

pytestmark = pytest.mark.real_sleep

# 10 reconnects x 3.0s. Allow generous slack so the test is not flaky on a
# loaded machine, but tight enough to catch an order-of-magnitude change.
EXPECTED_FALLBACK = SSE_MAX_RECONNECTS * SSE_RECONNECT_DELAY
LOWER, UPPER = EXPECTED_FALLBACK * 0.8, EXPECTED_FALLBACK * 1.6


@pytest.mark.slow
def test_sse_refusal_costs_the_full_reconnect_ladder_sync():
    with MockAPI(progress_steps=1) as api:
        api.stream_status = 503
        start = time.monotonic()
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            result = c.transcribe(b"x" * 64)
        elapsed = time.monotonic() - start

    assert result.text, "it does eventually succeed, via polling"
    attempts = len([p for p in api.paths("GET") if p.endswith("/stream")])
    assert attempts == SSE_MAX_RECONNECTS + 1, (
        f"expected {SSE_MAX_RECONNECTS + 1} stream attempts before giving up, got {attempts}"
    )
    assert LOWER < elapsed < UPPER, (
        f"SSE refusal fallback took {elapsed:.1f}s; expected ~{EXPECTED_FALLBACK:.0f}s. "
        "If this dropped, the retry policy changed — update this test deliberately."
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_sse_refusal_costs_the_full_reconnect_ladder_async():
    with MockAPI(progress_steps=1) as api:
        api.stream_status = 503
        start = time.monotonic()
        async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            result = await c.transcribe(b"x" * 64)
        elapsed = time.monotonic() - start

    assert result.text
    assert LOWER < elapsed < UPPER, (
        f"async SSE refusal fallback took {elapsed:.1f}s; expected ~{EXPECTED_FALLBACK:.0f}s"
    )


@pytest.mark.slow
def test_sync_and_async_pay_the_same_fallback_cost():
    """The two clients must not diverge on timing any more than on behaviour."""
    def sync_cost() -> float:
        with MockAPI(progress_steps=1) as api:
            api.stream_status = 503
            start = time.monotonic()
            with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
                c.transcribe(b"x" * 64)
            return time.monotonic() - start

    async def _async() -> float:
        with MockAPI(progress_steps=1) as api:
            api.stream_status = 503
            start = time.monotonic()
            async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
                await c.transcribe(b"x" * 64)
            return time.monotonic() - start

    s, a = sync_cost(), asyncio.run(_async())
    assert abs(s - a) < EXPECTED_FALLBACK * 0.4, (
        f"sync took {s:.1f}s but async took {a:.1f}s — the clients have diverged"
    )


def test_a_healthy_stream_costs_nothing_extra():
    """The ladder must only engage on failure, never on the happy path."""
    with MockAPI(progress_steps=2) as api:
        start = time.monotonic()
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.transcribe(b"x" * 64)
        elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"a healthy job took {elapsed:.1f}s; no retry should have fired"
    assert len([p for p in api.paths("GET") if p.endswith("/stream")]) == 1
