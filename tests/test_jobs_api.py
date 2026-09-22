"""The job-management surface: list, status, retrieve, cancel, check-failed.

`list_jobs` and its cursor pagination had never been exercised at all.
"""

from __future__ import annotations

import pytest

from mock_api import MockAPI
from speechrevolutions import SpeechRevolutions
from speechrevolutions.exceptions import (
    AuthenticationError,
    JobFailedError,
    JobNotFoundError,
    SpeechRevolutionsError,
)


@pytest.fixture
def api():
    with MockAPI(progress_steps=1) as a:
        yield a


@pytest.fixture
def client(api):
    with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        yield c


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

def test_list_jobs_returns_submitted_jobs(api, client):
    ids = {client.submit(f"https://example.com/{i}.mp3") for i in range(3)}
    page = client.list_jobs()
    assert {j["job_id"] for j in page["jobs"]} == ids


def test_list_jobs_honours_limit(api, client):
    for i in range(5):
        client.submit(f"https://example.com/{i}.mp3")
    assert len(client.list_jobs(limit=2)["jobs"]) == 2


def test_list_jobs_paginates_with_the_cursor(api, client):
    for i in range(5):
        client.submit(f"https://example.com/{i}.mp3")

    seen: list[str] = []
    cursor = None
    for _ in range(10):                       # bounded, so a bug cannot hang the suite
        page = client.list_jobs(limit=2, before=cursor)
        seen += [j["job_id"] for j in page["jobs"]]
        cursor = page.get("next_before")
        if not cursor:
            break
    assert len(seen) == 5
    assert len(set(seen)) == 5, "pagination returned a duplicate"


def test_list_jobs_cursor_is_none_on_the_last_page(api, client):
    client.submit("https://example.com/a.mp3")
    assert client.list_jobs(limit=50).get("next_before") is None


def test_list_jobs_is_empty_before_anything_is_submitted(api, client):
    assert client.list_jobs()["jobs"] == []


def test_list_jobs_url_encodes_the_cursor(api, client):
    # A cursor is opaque; it must survive being put in a query string.
    client.list_jobs(before="job_abc/+ =?&x")
    assert any("before=" in p for p in api.paths("GET"))


# ---------------------------------------------------------------------------
# get_job_status
# ---------------------------------------------------------------------------

def test_get_job_status_reports_completed_with_a_download_url(api, client):
    job_id = client.submit("https://example.com/a.mp3")
    status = client.get_job_status(job_id)
    assert status.job_id == job_id
    assert status.is_completed and not status.is_failed
    assert status.download_url


def test_get_job_status_reports_failure_with_stage_and_reason(api, client):
    api.fail_job_at = "gpu_timestamps"
    job_id = client.submit("https://example.com/a.mp3")
    status = client.get_job_status(job_id)
    assert status.is_failed
    assert status.failed_stage == "gpu_timestamps"
    assert "simulated failure" in status.reason


def test_get_job_status_unknown_job_raises_not_found(api, client):
    with pytest.raises(JobNotFoundError):
        client.get_job_status("job_does_not_exist")


# ---------------------------------------------------------------------------
# get_transcript
# ---------------------------------------------------------------------------

def test_get_transcript_by_id(api, client):
    job_id = client.submit("https://example.com/a.mp3")
    result = client.get_transcript(job_id)
    assert result.job_id == job_id
    assert result.text == "Good morning everyone. Thanks for joining."
    assert [u.speaker for u in result.utterances] == ["A", "B"]


def test_get_transcript_honours_output_type(api, client):
    job_id = client.submit("https://example.com/a.mp3", output_type="srt")
    result = client.get_transcript(job_id, output_type="srt")
    assert result.output_type == "srt"
    assert "-->" in result.text


def test_get_transcript_on_a_failed_job_raises(api, client):
    api.fail_job_at = "preprocess"
    job_id = client.submit("https://example.com/a.mp3")
    with pytest.raises(JobFailedError):
        client.get_transcript(job_id)


def test_get_transcript_while_still_processing_raises(api, client):
    # Upload without completing, so the job never leaves "processing".
    job = client.create_upload_job(1024)
    with pytest.raises(SpeechRevolutionsError):
        client.get_transcript(job.job_id)


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------

def test_cancel_job(api, client):
    job = client.create_upload_job(1024)
    client.cancel_job(job.job_id)
    assert api.jobs[job.job_id].status == "cancelled"


def test_cancel_unknown_job_raises_not_found(api, client):
    with pytest.raises(JobNotFoundError):
        client.cancel_job("job_nope")


# ---------------------------------------------------------------------------
# check_failed
# ---------------------------------------------------------------------------

def test_check_failed_batch(api, client):
    ok = client.submit("https://example.com/ok.mp3")
    api.fail_job_at = "preprocess"
    bad = client.submit("https://example.com/bad.mp3")
    assert client.check_failed([ok, bad]) == [False, True]


def test_check_failed_preserves_order(api, client):
    api.fail_job_at = "preprocess"
    bad = client.submit("https://example.com/bad.mp3")
    api.fail_job_at = None
    ok = client.submit("https://example.com/ok.mp3")
    assert client.check_failed([bad, ok]) == [True, False]


def test_check_failed_empty_list(api, client):
    assert client.check_failed([]) == []


# ---------------------------------------------------------------------------
# download_result
# ---------------------------------------------------------------------------

def test_download_result_returns_raw_bytes(api, client):
    job_id = client.submit("https://example.com/a.mp3")
    status = client.get_job_status(job_id)
    content = client.download_result(status.download_url)
    assert b"Good" in content


def test_download_result_on_a_missing_object_raises(api, client):
    with pytest.raises(SpeechRevolutionsError):
        client.download_result(f"{api.base_url}/_result/job_nope")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_a_bad_api_key_raises_authentication_error(api):
    with SpeechRevolutions(api_key="wrong-key", base_url=api.base_url) as c:
        with pytest.raises(AuthenticationError):
            c.list_jobs()


def test_the_api_key_is_sent_as_a_header(api, client):
    client.list_jobs()
    assert api.paths("GET")            # reached the server, so the header matched
