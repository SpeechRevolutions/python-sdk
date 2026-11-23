"""Streaming session against a local fake server: normal flow, resume after a drop, failover replay,
fatal errors and the sync wrapper."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.server import serve  # noqa: E402

from speechrevolutions import AsyncSpeechRevolutions, SpeechRevolutions, StreamError  # noqa: E402
from speechrevolutions.streaming import BYTES_PER_MS  # noqa: E402

MS = BYTES_PER_MS

# Pacing, resume backoff and the fake server all run on real time.
pytestmark = pytest.mark.real_sleep


def pcm(ms: int, value: int = 1) -> bytes:
    """`ms` milliseconds of PCM16 where every sample is `value` (so replays can be checked byte-wise)."""
    return value.to_bytes(2, "little", signed=True) * (ms * MS // 2)


class FakeServer:
    """A scripted /api/v1/stream. `script(conn_no, ws, path)` drives each connection."""

    def __init__(self, script):
        self.script = script
        self.connections = []  # (path, headers, [messages received])
        self.server = None

    async def __aenter__(self):
        async def handler(ws):
            got = []
            self.connections.append((ws.request.path, dict(ws.request.headers), got))
            await self.script(len(self.connections), ws, ws.request.path, got)

        self.server = await serve(handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


async def recv(ws, got):
    m = json.loads(await ws.recv())
    got.append(m)
    return m


def audio_bytes(msgs):
    return b"".join(base64.b64decode(m["audio"]) for m in msgs if m.get("type") == "input_audio_buffer.append")


async def finish_normally(ws, got, job="job-1"):
    """Read until the final commit, ack everything, then send one final and done."""
    while True:
        m = await recv(ws, got)
        if m.get("type") == "input_audio_buffer.commit" and m.get("final"):
            break
    await ws.send(json.dumps({"type": "final", "index": 0, "text": "hello world", "start": 0.0, "end": 1.0}))
    await ws.send(json.dumps({"type": "transcription.done", "text": "hello world"}))
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_normal_session_sends_update_audio_and_final_commit():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "job-1"}))
        await finish_normally(ws, got)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        session = await client.stream(language="en", vocabulary=["Zephyr"], turn_detection={"threshold": 0.5})
        await session.send_audio(pcm(400))
        result = await session.finish(timeout=5)

    path, headers, got = srv.connections[0]
    assert path == "/api/v1/stream"
    assert headers["x-api-key"] == "k" and headers["user-agent"].startswith("speechrevolutions-python/")
    upd = got[0]
    assert upd["type"] == "session.update" and upd["language"] == "en" and upd["vocabulary"] == ["Zephyr"]
    assert upd["turn_detection"] == {"threshold": 0.5}
    assert upd["audio_format"] == {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    assert got[1] == {"type": "input_audio_buffer.commit"}
    assert audio_bytes(got) == pcm(400)
    assert result.complete and result.text == "hello world" and result.job_id == "job-1"
    assert [f.text for f in result.finals] == ["hello world"]


@pytest.mark.asyncio
async def test_auto_language_omits_language_field():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "j"}))
        await finish_normally(ws, got)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        session = await client.stream(languages=["en", "ja"])
        await session.finish(timeout=5)
    upd = srv.connections[0][2][0]
    assert "language" not in upd and upd["languages"] == ["en", "ja"]


@pytest.mark.asyncio
async def test_revised_final_replaces_text():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "j"}))
        while True:
            m = await recv(ws, got)
            if m.get("final"):
                break
        await ws.send(json.dumps({"type": "final", "index": 0, "text": "helo", "start": 0, "end": 1}))
        await ws.send(json.dumps({"type": "final.revised", "index": 0, "text": "hello"}))
        await ws.send(json.dumps({"type": "transcription.done", "text": "hello"}))
        await asyncio.sleep(0.05)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        seen = []
        session = await client.stream(language="en", on_event=lambda e: seen.append(e.type))
        result = await session.finish(timeout=5)
    assert result.finals[0].text == "hello" and result.finals[0].start == 0
    assert seen == ["session.created", "final", "final.revised", "transcription.done"]


@pytest.mark.asyncio
async def test_resume_after_drop_replays_unacknowledged_audio():
    """The connection drops after 600 ms were sent and 200 ms acked: the client reconnects with
    ?job_id=, rewinds to the server's resume point and replays exactly the audio after it."""
    first_done = asyncio.Event()

    async def script(n, ws, path, got):
        if n == 1:
            await ws.send(json.dumps({"type": "session.created", "job_id": "job-r"}))
            while len(audio_bytes(got)) < 600 * MS:
                await recv(ws, got)
            await ws.send(json.dumps({"type": "audio.committed", "through_ms": 200}))
            await asyncio.sleep(0.05)
            first_done.set()
            await ws.close(code=1012)  # "going away": resume at once
            return
        await ws.send(json.dumps({"type": "session.resumed", "job_id": "job-r", "resume_from_ms": 200}))
        await finish_normally(ws, got, job="job-r")

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        session = await client.stream(language="en")
        await session.send_audio(pcm(600, 7))
        await first_done.wait()
        await asyncio.sleep(0.2)
        await session.send_audio(pcm(200, 9))  # live audio after the resume
        result = await session.finish(timeout=10)

    assert len(srv.connections) == 2
    path2, _, got2 = srv.connections[1]
    assert path2 == "/api/v1/stream?job_id=job-r"
    assert got2[0] == {"type": "input_audio_buffer.rewind", "from_ms": 200}
    assert not any(m.get("type") == "session.update" for m in got2)  # the server keeps the session's settings
    assert audio_bytes(got2) == pcm(400, 7) + pcm(200, 9)
    assert result.complete and result.job_id == "job-r"


@pytest.mark.asyncio
async def test_failover_resend_rewinds_on_the_same_connection():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "j"}))
        while len(audio_bytes(got)) < 480 * MS:
            await recv(ws, got)
        await ws.send(json.dumps({"type": "audio.committed", "through_ms": 160}))
        await ws.send(json.dumps({"type": "resend", "from_ms": 160}))
        await finish_normally(ws, got)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        session = await client.stream(language="en")
        await session.send_audio(pcm(480, 3))
        await asyncio.sleep(0.3)
        result = await session.finish(timeout=10)

    got = srv.connections[0][2]
    rw = [i for i, m in enumerate(got) if m.get("type") == "input_audio_buffer.rewind"]
    assert len(rw) == 1 and got[rw[0]]["from_ms"] == 160
    assert audio_bytes(got[rw[0]:]) == pcm(320, 3)  # everything after the ack, once more
    assert result.complete


@pytest.mark.asyncio
async def test_fatal_error_at_connect_raises():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "error", "code": "insufficient_balance", "message": "top up"}))
        await asyncio.sleep(0.05)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        with pytest.raises(StreamError) as ei:
            await client.stream(language="en")
    assert ei.value.code == "insufficient_balance"


@pytest.mark.asyncio
async def test_fatal_error_mid_session_ends_finish_with_error():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "j"}))
        await recv(ws, got)
        await ws.send(json.dumps({"type": "error", "code": "use_batch", "message": "sent too fast"}))
        await asyncio.sleep(0.2)

    async with FakeServer(script) as srv:
        client = AsyncSpeechRevolutions(api_key="k", base_url=srv.base_url)
        session = await client.stream(language="en")
        await session.send_audio(pcm(100))
        with pytest.raises(StreamError) as ei:
            await session.finish(timeout=5)
    assert ei.value.code == "use_batch"


def test_sync_session_and_events():
    async def script(n, ws, path, got):
        await ws.send(json.dumps({"type": "session.created", "job_id": "job-s"}))
        await ws.send(json.dumps({"type": "transcription.partial", "index": 0, "text": "hel", "stable": 2}))
        await finish_normally(ws, got, job="job-s")

    async def run():
        async with FakeServer(script) as srv:
            def sync_part():
                client = SpeechRevolutions(api_key="k", base_url=srv.base_url)
                session = client.stream(language="en")
                session.send_audio(pcm(240))
                result = session.finish(timeout=5)
                events = [e.type for e in session]
                return session.job_id, result, events

            return await asyncio.to_thread(sync_part)

    job_id, result, events = asyncio.run(run())
    assert job_id == "job-s" and result.text == "hello world"
    assert events == ["session.created", "transcription.partial", "final", "transcription.done"]


def test_send_audio_rejects_odd_length():
    from speechrevolutions.streaming import AsyncLiveSession

    s = AsyncLiveSession(url="ws://x", api_key="k", session_update={})
    with pytest.raises(ValueError):
        asyncio.run(s.send_audio(b"\x00\x01\x02"))
