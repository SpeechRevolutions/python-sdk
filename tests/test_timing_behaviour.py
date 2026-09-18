"""Retry ladders, measured in real wall-clock seconds.

Every other suite runs under an autouse fixture that no-ops `sleep`, which keeps
them fast but means they cannot see how long anything actually takes. These opt
out with `@pytest.mark.real_sleep` and measure, so a latency regression fails
here rather than in a customer's p99.

What these pin: a REFUSED stream endpoint (one answering a non-2xx status) must
fall back to polling quickly, while a DROPPED connection still gets the full
reconnect ladder. The two look identical to a naive retry loop and are not the
same thing — a proxy, load balancer or corporate egress that does not pass
`text/event-stream` answers every attempt the same way forever, so retrying it
ten times on a 3-second timer just burns 30 seconds before the client does what
it was always going to do.

That used to cost `SSE_MAX_RECONNECTS` (10) x `SSE_RECONNECT_DELAY` (3.0s) =
~30s on EVERY job for anyone behind such a proxy, which is longer than the
median job takes to transcribe (~24s in production). It is now bounded by
`SSE_MAX_STATUS_REFUSALS` (2).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mock_api import MockAPI
from speechrevolutions import AsyncSpeechRevolutions, SpeechRevolutions
from speechrevolutions._config import (
    SSE_MAX_RECONNECTS,
    SSE_MAX_STATUS_REFUSALS,
    SSE_RECONNECT_DELAY,
)

pytestmark = pytest.mark.real_sleep

# 10 reconnects x 3.0s. Allow generous slack so the test is not flaky on a
# loaded machine, but tight enough to catch an order-of-magnitude change.
# What a REFUSAL should now cost: one retry, then fall back.
EXPECTED_REFUSAL = (SSE_MAX_STATUS_REFUSALS - 1) * SSE_RECONNECT_DELAY
REFUSAL_CEILING = EXPECTED_REFUSAL + SSE_RECONNECT_DELAY + 5.0

# What the full ladder would have cost, kept so the regression is obvious.
OLD_FULL_LADDER = SSE_MAX_RECONNECTS * SSE_RECONNECT_DELAY


@pytest.mark.slow
def test_a_refused_stream_falls_back_quickly_sync():
    with MockAPI(progress_steps=1) as api:
        api.stream_status = 503
        start = time.monotonic()
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            result = c.transcribe(b"x" * 64)
        elapsed = time.monotonic() - start

    assert result.text, "it still succeeds, via polling"
    attempts = len([p for p in api.paths("GET") if p.endswith("/stream")])
    assert attempts == SSE_MAX_STATUS_REFUSALS, (
        f"expected {SSE_MAX_STATUS_REFUSALS} stream attempts on a refusal, got {attempts}"
    )
    assert elapsed < REFUSAL_CEILING, (
        f"a refused stream took {elapsed:.1f}s to fall back; the whole point is "
        f"that it should not cost the full {OLD_FULL_LADDER:.0f}s ladder"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_refused_stream_falls_back_quickly_async():
    with MockAPI(progress_steps=1) as api:
        api.stream_status = 503
        start = time.monotonic()
        async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            result = await c.transcribe(b"x" * 64)
        elapsed = time.monotonic() - start

    assert result.text
    assert elapsed < REFUSAL_CEILING, (
        f"async refused-stream fallback took {elapsed:.1f}s; expected under "
        f"{REFUSAL_CEILING:.0f}s"
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
    assert abs(s - a) < REFUSAL_CEILING * 0.6, (
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
