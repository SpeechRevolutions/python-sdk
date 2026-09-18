"""Live progress (SSE) and the polling fallback behind it.

Neither had any coverage. Both matter more than they look: SSE is the feature
neither AssemblyAI nor Deepgram exposes for pre-recorded audio, and polling is
what every caller silently falls back to when a proxy or load balancer refuses
to hold a streaming connection open.
"""

from __future__ import annotations

import time

import pytest

from mock_api import MockAPI
from speechrevolutions import SpeechRevolutions
from speechrevolutions.exceptions import JobFailedError, TimeoutError as STTTimeoutError


@pytest.fixture
def api():
    with MockAPI(progress_steps=3) as a:
        yield a


@pytest.fixture
def client(api):
    with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        yield c


# ---------------------------------------------------------------------------
# SSE progress
# ---------------------------------------------------------------------------

def test_progress_callback_fires_for_every_event(api, client):
    events = []
    client.transcribe(b"x" * 64, on_progress=events.append)
    assert len(events) == 3
    assert [e.completed for e in events] == [1, 2, 3]
    assert all(e.total == 3 for e in events)


def test_progress_events_carry_percent_and_step(api, client):
    events = []
    client.transcribe(b"x" * 64, on_progress=events.append)
    assert [round(e.percent) for e in events] == [33, 67, 100]
    # Verified live: steps are preprocess, then chunk:N per chunk, then
    # aggregation — not a single generic label.
    assert [e.step for e in events] == ["preprocess", "chunk:0", "aggregation"]


def test_progress_events_carry_elapsed_time(api, client):
    events = []
    client.transcribe(b"x" * 64, on_progress=events.append)
    assert all(e.elapsed_seconds >= 0 for e in events)
    assert events[-1].elapsed_seconds >= events[0].elapsed_seconds


def test_progress_fires_live_not_replayed_at_the_end(api, client):
    # The callback must run as events arrive. If it were replayed after the
    # stream closed, every timestamp would be effectively identical.
    stamps: list[float] = []
    client.transcribe(b"x" * 64, on_progress=lambda e: stamps.append(time.monotonic()))
    assert len(stamps) == 3
    assert stamps[-1] - stamps[0] > 0


def test_no_callback_still_completes(api, client):
    assert client.transcribe(b"x" * 64).text


# ---------------------------------------------------------------------------
# Reconnect, resuming from Last-Event-ID
# ---------------------------------------------------------------------------

def test_stream_drop_reconnects_and_completes(api, client):
    # Cut the stream after the first event. The SDK must reconnect, resume from
    # Last-Event-ID, and still deliver the transcript.
    api.drop_stream_after = 1
    events = []
    result = client.transcribe(b"x" * 64, on_progress=events.append)
    assert result.text
    streams = [p for p in api.paths("GET") if p.endswith("/stream")]
    assert len(streams) > 1, "expected at least one reconnect"


def test_reconnect_does_not_replay_events_already_seen(api, client):
    api.drop_stream_after = 1
    events = []
    client.transcribe(b"x" * 64, on_progress=events.append)
    # Resuming from Last-Event-ID means each step is reported once.
    assert [e.completed for e in events] == sorted({e.completed for e in events})


# ---------------------------------------------------------------------------
# Falling back to polling
# ---------------------------------------------------------------------------

def test_sse_unavailable_falls_back_to_polling(api, client):
    # A 503 on the stream endpoint is what a proxy that refuses SSE looks like.
    api.stream_status = 503
    result = client.transcribe(b"x" * 64)
    assert result.text == "Good morning everyone. Thanks for joining."


def test_polling_path_checks_for_failure(api, client):
    api.stream_status = 503
    api.fail_job_at = "preprocess"
    with pytest.raises(JobFailedError):
        client.transcribe(b"x" * 64)
    assert api.count("POST", "/api/v1/jobs/check-failed") >= 1


def test_polling_retries_until_the_result_exists(api, client, monkeypatch):
    import speechrevolutions.client as mod
    monkeypatch.setattr(mod, "POLL_INTERVAL", 0.05)
    api.stream_status = 503
    api.result_available_after = 0.12     # 404 for the first couple of polls
    result = client.transcribe(b"x" * 64)
    assert result.text


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

def test_timeout_raises_rather_than_hanging(api):
    api.stream_status = 503
    api.result_available_after = 30       # never ready inside the window
    with SpeechRevolutions(api_key="test-key", base_url=api.base_url, timeout=0.3) as c:
        with pytest.raises(STTTimeoutError):
            c.transcribe(b"x" * 64)


# ---------------------------------------------------------------------------
# wait_for_result, used directly
# ---------------------------------------------------------------------------

def test_wait_for_result_after_submit(api, client):
    job_id = client.submit(b"x" * 64)
    status = client.get_job_status(job_id)
    content, url = client.wait_for_result(job_id, status.download_url)
    assert b"Good" in content
    assert url


def test_failure_over_sse_reports_step_and_reason(api, client):
    api.fail_job_at = "gpu_segmentation"
    with pytest.raises(JobFailedError) as exc:
        client.transcribe(b"x" * 64)
    msg = str(exc.value)
    assert "gpu_segmentation" in msg and "simulated failure" in msg
