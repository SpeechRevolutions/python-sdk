"""Retry policy, and the duplicate-job hazard it exists to prevent.

The rule under test: a request that CREATES a job is retried only when the
request provably never reached the server. Every other request retries freely.

A regression here is expensive and silent — the customer gets two transcripts
and two charges for one file — so these assert attempt COUNTS, not just the
final outcome.
"""

from __future__ import annotations

import pytest
import requests

from conftest import FakeResponse, RecordingSession
from speechrevolutions import SpeechRevolutions
from speechrevolutions._config import creates_job
from speechrevolutions.exceptions import APIError, TimeoutError as STTTimeoutError

CREATE_PATH = "/api/v1/upload"
MULTIPART_CREATE_PATH = "/api/v1/upload/multipart/create"
SAFE_PATH = "/api/v1/jobs/cancel"

OK = FakeResponse(200, {"job_id": "j1", "upload_url": "u", "download_url": "d"})


def client(script, **kw):
    session = RecordingSession(script)
    return SpeechRevolutions(api_key="k", session=session, max_retries=3, **kw), session


# --------------------------------------------------------------------------
# creates_job() — the single source of truth for the policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,expected",
    [
        ("/api/v1/upload", True),
        ("/api/v1/upload/", True),
        ("/api/v1/upload?x=1", True),
        ("/api/v1/upload/multipart/create", True),
        # Everything below acts on a job that already exists.
        ("/api/v1/upload/complete", False),
        ("/api/v1/upload/progress", False),
        ("/api/v1/upload/multipart/complete", False),
        ("/api/v1/upload/multipart/abort", False),
        ("/api/v1/jobs/cancel", False),
        ("/api/v1/jobs/check-failed", False),
        ("/api/v1/jobs", False),
        ("/api/v1/jobs/abc123", False),
    ],
)
def test_creates_job_classification(path, expected):
    assert creates_job(path) is expected


# --------------------------------------------------------------------------
# Create calls: must NOT retry on an ambiguous failure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_create_does_not_retry_on_5xx(status):
    """A 5xx means the server saw the request. It may have created the job
    before failing, so retrying risks a duplicate."""
    c, s = client([FakeResponse(status, text="boom")])
    with pytest.raises(APIError):
        c.create_upload_job(1024)
    assert s.attempts == 1


def test_create_does_not_retry_on_read_timeout():
    """The request was sent; we just never heard back. Ambiguous."""
    c, s = client([requests.exceptions.ReadTimeout("read timed out")])
    with pytest.raises(STTTimeoutError):
        c.create_upload_job(1024)
    assert s.attempts == 1


def test_create_does_not_retry_on_midflight_connection_error():
    """A reset after the connection was established is ambiguous."""
    c, s = client([requests.exceptions.ConnectionError("connection reset by peer")])
    with pytest.raises(APIError):
        c.create_upload_job(1024)
    assert s.attempts == 1


def test_multipart_create_does_not_retry_on_5xx():
    c, s = client([FakeResponse(503, text="unavailable")])
    with pytest.raises(APIError):
        c._api_request("POST", MULTIPART_CREATE_PATH, json={})
    assert s.attempts == 1


# --------------------------------------------------------------------------
# Create calls: SHOULD retry when the request provably never landed
# --------------------------------------------------------------------------

def test_create_retries_on_connect_timeout():
    """No connection was ever established, so no job can exist."""
    c, s = client([requests.exceptions.ConnectTimeout("connect timed out"), OK])
    job = c.create_upload_job(1024)
    assert job.job_id == "j1"
    assert s.attempts == 2


def test_create_retries_on_429():
    """The server refused it outright — it did no work."""
    c, s = client([FakeResponse(429, text="slow down", headers={"Retry-After": "1"}), OK])
    job = c.create_upload_job(1024)
    assert job.job_id == "j1"
    assert s.attempts == 2


def test_create_gives_up_after_max_retries_on_connect_timeout():
    err = requests.exceptions.ConnectTimeout("connect timed out")
    c, s = client([err, err, err, err, err])
    with pytest.raises(STTTimeoutError):
        c.create_upload_job(1024)
    assert s.attempts == 4  # 1 initial + 3 retries


# --------------------------------------------------------------------------
# Non-create calls keep the old, permissive behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_safe_path_retries_on_all_transient_statuses(status):
    c, s = client([FakeResponse(status, text="x"), FakeResponse(200, {})])
    c.cancel_job("j1")
    assert s.attempts == 2


def test_safe_path_retries_on_read_timeout():
    c, s = client([requests.exceptions.ReadTimeout("t"), FakeResponse(200, {})])
    c.cancel_job("j1")
    assert s.attempts == 2


def test_safe_path_retries_on_connection_error():
    c, s = client([requests.exceptions.ConnectionError("reset"), FakeResponse(200, {})])
    c.cancel_job("j1")
    assert s.attempts == 2


def test_complete_upload_is_retryable():
    """complete acts on a job_id the caller already holds — replay is harmless."""
    c, s = client([FakeResponse(500, text="x"), FakeResponse(200, {})])
    c.complete_upload("j1")
    assert s.attempts == 2


# --------------------------------------------------------------------------
# The scenario this whole policy exists for
# --------------------------------------------------------------------------

def test_lost_response_creates_exactly_one_job():
    """Regression guard for the duplicate-charge bug.

    The server creates the job and then the response is lost to a gateway 502.
    The SDK must surface the error rather than silently creating a second job.
    """
    c, s = client([FakeResponse(502, text="bad gateway"), OK])
    with pytest.raises(APIError) as exc:
        c.create_upload_job(1024)
    assert exc.value.status_code == 502
    assert s.attempts == 1, "a retry here would have created a duplicate job"
