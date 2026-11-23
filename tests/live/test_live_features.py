"""Live tests for the 1.1 features: add-on options, delete_job and real-time streaming (including a
forced connection drop and resume) against the REAL API.

    SR_LIVE=1 SPEECHREVOLUTIONS_API_KEY=stt_... pytest tests/live/test_live_features.py -v

The clips in assets/ are synthetic (macOS `say`, two voices) with known scripts beside them
(call_script.txt, noname_script.txt), so the assertions can be about content, not just shape.
Each test bills a few seconds to a minute of audio.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from speechrevolutions import (
    APIError,
    AsyncSpeechRevolutions,
    JobNotFoundError,
    SpeechRevolutions,
    SpeechRevolutionsError,
    load_pcm16,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SR_LIVE") != "1",
        reason="live API tests are opt-in: set SR_LIVE=1 (creates real, billable jobs)",
    ),
    pytest.mark.real_sleep,
]

ASSETS = Path(__file__).parent / "assets"
CALL = ASSETS / "call.mp3"  # bank support call: names, card, phone, email, "Northwind Bank"
NONAME = ASSETS / "noname.mp3"  # two people discuss quarterly numbers; no names said
PROF = ASSETS / "prof.mp3"  # one sentence with three strong swear words


@pytest.fixture(scope="module")
def client() -> SpeechRevolutions:
    with SpeechRevolutions() as c:
        yield c


def _norm(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").replace(".", " ").split())


# ------------------------------------------------------------------ batch options


def test_speaker_labels_off_by_default(client):
    t = client.transcribe(NONAME)
    assert "quarterly" in _norm(t.text)
    assert all(w.speaker is None for w in t.words), "speaker_labels is an add-on and should be off by default"


def test_speaker_labels_on_finds_two_speakers(client):
    t = client.transcribe(NONAME, speaker_labels=True)
    assert len({w.speaker for w in t.words}) == 2
    assert len(t.utterances) >= 4


def test_summary_and_paragraphs(client):
    t = client.transcribe(NONAME, speaker_labels=True, summarize=True, paragraphs=True)
    assert t.summary is not None and len(t.summary.text) > 40
    assert "revenue" in t.summary.text.lower() or "quarter" in t.summary.text.lower()
    assert t.summary.bullets
    assert len(t.paragraphs) >= 2 and all(p.text and p.sentences for p in t.paragraphs)
    assert {p.speaker for p in t.paragraphs} == {w.speaker for w in t.words}
    assert t.to_dict()["summary"]["text"] == t.summary.text


def test_profanity_filter(client):
    plain = client.transcribe(PROF)
    filtered = client.transcribe(PROF, filter_profanity=True)
    assert "shit" in plain.text.lower()
    assert "shit" not in filtered.text.lower() and "fucking" not in filtered.text.lower()
    assert "s***" in filtered.text


def test_custom_vocabulary_fixes_the_brand(client):
    t = client.transcribe(CALL, custom_vocabulary=["Northwind"])
    assert "northwind bank" in _norm(t.text)
    assert "north northwind" not in _norm(t.text)


def test_extra_options_pass_through(client):
    t = client.transcribe(PROF, extra={"filter_profanity": True})
    assert "s***" in t.text


def test_srt_output_with_speakers(client):
    t = client.transcribe(NONAME, output_type="srt", speaker_labels=True)
    assert t.text.startswith("1\n") and "-->" in t.text and "SPEAKER_" in t.text


def test_delete_job_removes_the_transcript(client):
    job_id = client.submit(PROF)
    for _ in range(60):
        if client.get_job_status(job_id).is_completed:
            break
        import time

        time.sleep(2)
    assert client.get_transcript(job_id).text
    try:
        client.delete_job(job_id)
    except APIError as exc:
        if "temporarily unavailable" in str(exc):
            pytest.xfail("prod user_cluster lacks s3:DeleteObject (2026-11-23); see cluster_change_requests.md")
        raise
    with pytest.raises((JobNotFoundError, SpeechRevolutionsError, APIError)):
        client.get_transcript(job_id)


# ------------------------------------------------------------------ streaming


def test_stream_file_sync():
    events = []
    with SpeechRevolutions() as c:
        result = c.stream_file(NONAME, language="en", speed=1.5, on_event=lambda e: events.append(e.type))
    assert result.complete and result.job_id
    text = _norm(result.text)
    assert "quarterly" in text and "planning meeting" in text
    assert "session.created" in events and "final" in events and events[-1] == "transcription.done"
    assert "transcription.partial" in events


def test_stream_async_events_and_turns():
    async def run():
        async with AsyncSpeechRevolutions() as c:
            seen = []
            session = await c.stream(language="en", on_event=lambda e: seen.append(e))
            pcm = load_pcm16(str(CALL))
            from speechrevolutions.streaming import stream_pcm

            result = await stream_pcm(session, pcm, speed=2.0)  # front-loads 8 s, then real time
            return result, seen

    result, seen = asyncio.run(run())
    types = [e.type for e in seen]
    assert result.complete
    assert "northwind" in _norm(result.text) and "rachel" in _norm(result.text)
    assert "speech.started" in types and "turn.ended" in types
    finals = [e for e in seen if e.type == "final"]
    assert all(e.start is not None and e.end is not None and e.end >= e.start for e in finals)
    assert [f.index for f in result.finals] == sorted(f.index for f in result.finals)


def test_stream_auto_language_detection():
    with SpeechRevolutions() as c:
        events = []
        result = c.stream_file(NONAME, speed=1.5, on_event=events.append)  # no language: detected
    assert result.complete and "quarterly" in _norm(result.text)
    detected = [e.language for e in events if e.type == "language.detected"]
    assert detected and detected[0] == "en"


def test_stream_vocabulary():
    with SpeechRevolutions() as c:
        result = c.stream_file(CALL, language="en", speed=2.0, vocabulary=["Northwind"])
    assert "northwind bank" in _norm(result.text)


def test_stream_resumes_after_the_connection_is_cut():
    """Cut the TCP connection mid-stream. The session must reconnect with ?job_id=, replay what the
    server hadn't acknowledged, and deliver the whole transcript under the same job id."""

    async def run():
        async with AsyncSpeechRevolutions() as c:
            created = []
            session = await c.stream(language="en", on_event=lambda e: created.append(e.type))
            job_id = session.job_id
            pcm = load_pcm16(str(NONAME))
            step = 80 * 32
            cut_at = len(pcm) // 2
            for i in range(0, len(pcm), step):
                await session.send_audio(pcm[i: i + step])
                if i <= cut_at < i + step:
                    session._ws.transport.abort()  # noqa: SLF001 -- simulate a network drop
                await asyncio.sleep(0.08 / 1.4)
            result = await session.finish(timeout=120)
            return job_id, result, created

    job_id, result, events = asyncio.run(run())
    assert "session.resumed" in events, "the session should have resumed after the cut"
    assert result.complete and result.job_id == job_id
    text = _norm(result.text)
    assert "quarterly" in text and "deck with the numbers" in text  # start and end both present


def test_stream_bad_key_is_refused():
    from speechrevolutions import AuthenticationError, StreamError

    with SpeechRevolutions(api_key="stt_" + "0" * 64) as c:
        with pytest.raises((AuthenticationError, StreamError)):
            c.stream(language="en")
