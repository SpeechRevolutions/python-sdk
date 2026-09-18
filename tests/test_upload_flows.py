"""Every way audio gets into the platform, end to end against the mock API.

None of this was covered before: the tests that existed exercised the retry
policy and the transcript parser, but nothing ever drove a whole upload.
"""

from __future__ import annotations

import io

import pytest

from mock_api import MockAPI
from speechrevolutions import SpeechRevolutions
from speechrevolutions.exceptions import JobFailedError


@pytest.fixture
def api():
    with MockAPI(progress_steps=2) as a:
        yield a


@pytest.fixture
def client(api):
    with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
        yield c


# ---------------------------------------------------------------------------
# Multipart (the default path)
# ---------------------------------------------------------------------------

def test_transcribe_bytes_uses_multipart(api, client):
    result = client.transcribe(b"x" * 1024)
    assert result.text == "Good morning everyone. Thanks for joining."
    assert api.count("POST", "/api/v1/upload/multipart/create") == 1
    assert api.count("POST", "/api/v1/upload/multipart/complete") == 1
    assert api.count("POST", "/api/v1/upload") == 0


def test_multipart_splits_into_parts_and_sends_every_one(api, client):
    api.part_size = 1024
    client.transcribe(b"y" * 4096)          # exactly 4 parts
    job = api.only_job()
    assert sorted(job.parts) == [1, 2, 3, 4]


def test_multipart_uploads_the_whole_file(api, client):
    api.part_size = 300
    client.transcribe(b"z" * 1000)
    job = api.only_job()
    assert len(job.parts) == 4             # 300+300+300+100


# ---------------------------------------------------------------------------
# Single-shot, and the fallback into it
# ---------------------------------------------------------------------------

def test_multipart_disabled_falls_back_to_single_shot(api, client):
    # The real server 404s the multipart route when it is disabled; the SDK is
    # expected to notice and take the presigned-PUT path instead of failing.
    api.multipart_enabled = False
    result = client.transcribe(b"x" * 512)
    assert result.text
    assert api.count("POST", "/api/v1/upload/multipart/create") == 1   # tried
    assert api.count("POST", "/api/v1/upload") == 1                    # fell back
    assert api.count("POST", "/api/v1/upload/complete") == 1


def test_single_shot_uploads_the_exact_bytes(api, client):
    api.multipart_enabled = False
    payload = b"the quick brown fox" * 40
    client.transcribe(payload)
    assert api.only_job().uploaded == payload


def test_multipart_false_skips_the_multipart_attempt(api):
    with SpeechRevolutions(api_key="test-key", base_url=api.base_url,
                           multipart=False) as c:
        c.transcribe(b"x" * 512)
    assert api.count("POST", "/api/v1/upload/multipart/create") == 0
    assert api.count("POST", "/api/v1/upload") == 1


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------

def test_transcribe_local_file(api, client, tmp_path):
    p = tmp_path / "meeting.mp3"
    p.write_bytes(b"audio-bytes-here")
    result = client.transcribe(str(p))
    assert result.text
    assert api.only_job().parts or api.only_job().uploaded


def test_transcribe_file_object(api, client):
    result = client.transcribe(io.BytesIO(b"audio-from-a-file-object"))
    assert result.text


def test_transcribe_url_is_fetched_server_side(api, client):
    # A URL is handed to the platform; nothing is uploaded from the client.
    result = client.transcribe("https://example.com/audio.mp3")
    assert result.text
    assert api.count("POST", "/api/v1/upload") == 1
    assert api.count("POST", "/api/v1/upload/multipart/create") == 0
    assert not api.paths("PUT")


def test_transcribe_file_and_transcribe_url_helpers(api, client, tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"abc")
    assert client.transcribe_file(str(p)).text
    assert client.transcribe_url("https://example.com/b.mp3").text


def test_empty_audio_is_rejected_before_any_request(api, client):
    with pytest.raises(Exception):
        client.transcribe(b"")


# ---------------------------------------------------------------------------
# Options reach the server
# ---------------------------------------------------------------------------

def test_options_are_sent_on_the_create_call(api, client):
    client.transcribe(
        b"x" * 64,
        output_type="srt",
        speaker_labels=False,
        word_timestamps=True,
        nltk=False,
        custom_vocabulary=["Kubernetes", "Postgres"],
    )
    opts = api.only_job().options
    assert opts["output_type"] == "srt"
    assert opts["speaker_labels"] is False
    assert opts["word_timestamps"] is True
    assert opts["nltk"] is False
    assert opts["custom_vocabulary"] == ["Kubernetes", "Postgres"]


def test_diarize_is_an_alias_for_speaker_labels(api, client):
    client.transcribe(b"x" * 64, diarize=True)
    assert api.only_job().options["speaker_labels"] is True


def test_output_type_srt_returns_subtitle_text(api, client):
    result = client.transcribe(b"x" * 64, output_type="srt")
    assert result.output_type == "srt"
    assert "00:00:00,000 --> 00:00:01,440" in result.text


# ---------------------------------------------------------------------------
# submit(): enqueue without waiting
# ---------------------------------------------------------------------------

def test_submit_returns_a_job_id_without_waiting(api, client):
    job_id = client.submit(b"x" * 64)
    assert job_id.startswith("job_")
    # submit must NOT open the progress stream — that is the whole point.
    assert not [p for p in api.paths("GET") if p.endswith("/stream")]


def test_submit_url(api, client):
    job_id = client.submit("https://example.com/audio.mp3")
    assert api.jobs[job_id].status == "completed"


# ---------------------------------------------------------------------------
# Upload progress heartbeat
# ---------------------------------------------------------------------------

def test_upload_progress_callback_reports_bytes(api, client):
    api.multipart_enabled = False
    seen: list[tuple[int, int]] = []
    client.transcribe(b"x" * 4096,
                      on_upload_progress=lambda e: seen.append((e.completed, e.total)))
    assert seen, "no upload progress events fired"
    assert seen[-1][0] == seen[-1][1] == 4096


# ---------------------------------------------------------------------------
# Failure surfaces as a typed error
# ---------------------------------------------------------------------------

def test_job_failure_raises_job_failed_error(api, client):
    api.fail_job_at = "gpu_timestamps"
    with pytest.raises(JobFailedError) as exc:
        client.transcribe(b"x" * 64)
    assert "gpu_timestamps" in str(exc.value)
