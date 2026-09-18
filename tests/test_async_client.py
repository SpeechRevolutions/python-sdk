"""The async client, driven end to end against the mock API.

It publishes the same surface as the sync client and had no functional coverage
at all. These mirror the sync suites deliberately: where the two clients are
supposed to behave identically, the tests say so, and a divergence fails here
rather than in a customer's async service.
"""

from __future__ import annotations

import json

import pytest

from mock_api import MockAPI
from speechrevolutions import AsyncSpeechRevolutions
from speechrevolutions.exceptions import (
    AuthenticationError,
    JobFailedError,
    JobNotFoundError,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def api():
    with MockAPI(progress_steps=3) as a:
        yield a


# ---------------------------------------------------------------------------
# Upload paths
# ---------------------------------------------------------------------------

async def test_transcribe_bytes(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        result = await c.transcribe(b"x" * 512)
    assert result.text == "Good morning everyone. Thanks for joining."


async def test_transcribe_url_is_fetched_server_side(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        result = await c.transcribe("https://example.com/a.mp3")
    assert result.text
    assert not api.paths("PUT")


async def test_transcribe_local_file(api, tmp_path):
    p = tmp_path / "a.mp3"
    p.write_bytes(b"audio")
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        assert (await c.transcribe(str(p))).text


async def test_multipart_disabled_falls_back_to_single_shot(api):
    api.multipart_enabled = False
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        assert (await c.transcribe(b"x" * 512)).text
    assert api.count("POST", "/api/v1/upload") == 1
    assert api.count("POST", "/api/v1/upload/complete") == 1


async def test_options_reach_the_server(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        await c.transcribe(b"x" * 64, output_type="vtt", speaker_labels=False,
                           custom_vocabulary=["Kubernetes"])
    opts = api.only_job().options
    assert opts["output_type"] == "vtt"
    assert opts["speaker_labels"] is False
    assert opts["custom_vocabulary"] == ["Kubernetes"]


# ---------------------------------------------------------------------------
# Progress and polling
# ---------------------------------------------------------------------------

async def test_progress_callback_fires_for_every_event(api):
    events = []
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        await c.transcribe(b"x" * 64, on_progress=events.append)
    assert [e.completed for e in events] == [1, 2, 3]
    assert all(e.total == 3 for e in events)


async def test_sse_unavailable_falls_back_to_polling(api):
    api.stream_status = 503
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        assert (await c.transcribe(b"x" * 64)).text


async def test_stream_drop_reconnects(api):
    api.drop_stream_after = 1
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        assert (await c.transcribe(b"x" * 64)).text
    streams = [p for p in api.paths("GET") if p.endswith("/stream")]
    assert len(streams) > 1


async def test_failure_raises_job_failed_error(api):
    api.fail_job_at = "gpu_timestamps"
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        with pytest.raises(JobFailedError):
            await c.transcribe(b"x" * 64)


# ---------------------------------------------------------------------------
# Jobs API
# ---------------------------------------------------------------------------

async def test_submit_and_retrieve(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        job_id = await c.submit("https://example.com/a.mp3")
        status = await c.get_job_status(job_id)
        assert status.is_completed
        result = await c.get_transcript(job_id)
    assert result.job_id == job_id
    assert result.text


async def test_list_jobs_paginates(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        for i in range(5):
            await c.submit(f"https://example.com/{i}.mp3")
        seen, cursor = [], None
        for _ in range(10):
            page = await c.list_jobs(limit=2, before=cursor)
            seen += [j["job_id"] for j in page["jobs"]]
            cursor = page.get("next_before")
            if not cursor:
                break
    assert len(seen) == len(set(seen)) == 5


async def test_cancel_and_check_failed(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        job = await c.create_upload_job(1024)
        await c.cancel_job(job.job_id)
        assert api.jobs[job.job_id].status == "cancelled"

        ok = await c.submit("https://example.com/ok.mp3")
        api.fail_job_at = "preprocess"
        bad = await c.submit("https://example.com/bad.mp3")
        assert await c.check_failed([ok, bad]) == [False, True]


async def test_unknown_job_raises_not_found(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        with pytest.raises(JobNotFoundError):
            await c.get_job_status("job_nope")


async def test_bad_key_raises_authentication_error(api):
    async with AsyncSpeechRevolutions(api_key="wrong", base_url=api.base_url) as c:
        with pytest.raises(AuthenticationError):
            await c.list_jobs()


async def test_download_result(api):
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        job_id = await c.submit("https://example.com/a.mp3")
        status = await c.get_job_status(job_id)
        content = await c.download_result(status.download_url)
    assert b"Good" in content


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

async def test_callback_url_is_plumbed_through_and_delivered(api):
    from test_webhooks import Receiver

    with Receiver() as rcv:
        async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            job_id = await c.submit("https://example.com/a.mp3", callback_url=rcv.url)
        assert api.only_job().callback_url == rcv.url
        event = json.loads(rcv.wait(1)[0]["raw"])
    assert event == {"job_id": job_id, "status": "completed"}


# ---------------------------------------------------------------------------
# Parity with the sync client
# ---------------------------------------------------------------------------

async def test_same_transcript_as_the_sync_client(api):
    from speechrevolutions import SpeechRevolutions

    with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as s:
        sync_result = s.transcribe(b"x" * 64)
    async with AsyncSpeechRevolutions(api_key="test-key", base_url=api.base_url) as a:
        async_result = await a.transcribe(b"x" * 64)

    assert sync_result.text == async_result.text
    assert sync_result.to_dict()["words"] == async_result.to_dict()["words"]
    assert [u.speaker for u in sync_result.utterances] == \
           [u.speaker for u in async_result.utterances]
