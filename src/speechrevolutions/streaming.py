"""Real-time streaming transcription over WebSocket (``wss://.../api/v1/stream``).

One asyncio implementation, :class:`AsyncLiveSession`; :class:`LiveSession` runs it on a background
event loop for synchronous code.

What the session does for you, per the streaming protocol:

- **Buffers** every chunk you send until the server acknowledges it (``audio.committed``). That is at
  most ~30 s of audio, because utterances are capped at 25 s.
- **Resumes** a dropped connection: it reconnects with ``?job_id=`` for up to 60 s (at once when the
  server closes with 1012, "going away for a deploy"), then sends ``rewind`` and replays the buffer.
- **Replays on failover:** when the server moves the session to another GPU (``resend``), it rewinds and
  replays the buffer the same way. Replays are paced at up to 3x real time, as the server requires.

Audio is PCM16 little-endian, mono, 16 kHz. :func:`load_pcm16` converts any file (ffmpeg for anything
that isn't already a 16 kHz mono 16-bit WAV).
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from speechrevolutions._config import USER_AGENT
from speechrevolutions.exceptions import AuthenticationError, RateLimitError, SpeechRevolutionsError

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # PCM16 mono: 32 bytes per millisecond
CHUNK_MS = 80
REPLAY_SPEED = 3.0  # the server accepts a replayed backlog at up to 3x real time
RESUME_WINDOW_S = 60.0  # the server keeps a dropped session this long
PING_INTERVAL_S = 10.0

# Server errors that end the session for good: resuming can't help.
FATAL_ERRORS = frozenset({
    "invalid_api_key", "insufficient_balance", "session_expired", "no_capacity", "idle_timeout",
    "max_duration", "use_batch", "frame_too_large", "bad_audio_format",
})


class StreamError(SpeechRevolutionsError):
    """The live session ended with an error. ``code`` is the server's error code when it sent one."""

    def __init__(self, message: str, *, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class StreamEvent:
    """One server event. ``type`` is the protocol name; the other fields are set when it carries them.

    Types: ``session.created``, ``session.resumed``, ``transcription.partial`` (the open utterance so far;
    replaces the previous partial with the same ``index``; ``text[:stable]`` is settled), ``final``,
    ``final.revised`` (a better version of an earlier final), ``speech.started``, ``turn.ended``,
    ``language.detected``, ``language.unsupported``, ``audio.committed``, ``transcription.done``,
    ``error``. Unknown types are passed through with ``raw`` set.
    """

    type: str
    index: Optional[int] = None
    text: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    stable: Optional[int] = None
    audio_s: Optional[float] = None
    language: Optional[str] = None
    probability: Optional[float] = None
    forced: Optional[bool] = None
    job_id: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_message(cls, m: dict) -> "StreamEvent":
        err = m.get("error")
        return cls(
            type=str(m.get("type", "")),
            index=m.get("index"),
            text=m.get("text"),
            start=m.get("start"),
            end=m.get("end"),
            stable=m.get("stable"),
            audio_s=m.get("audio_s"),
            language=m.get("language"),
            probability=m.get("probability"),
            forced=m.get("forced"),
            job_id=m.get("job_id"),
            code=m.get("code") or (err if isinstance(err, str) and " " not in err else None),
            message=m.get("message") or (err if isinstance(err, str) else None),
            raw=m,
        )


@dataclass
class Final:
    index: int
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class StreamResult:
    """What a finished session produced."""

    job_id: Optional[str]
    #: All finals joined, as the server sent it in ``transcription.done``.
    text: str
    #: Each utterance's latest text (``final.revised`` applied), in order.
    finals: list[Final]
    #: False when the session ended before ``transcription.done`` (``text`` is then the finals so far).
    complete: bool = True


def load_pcm16(path: str) -> bytes:
    """Any audio file as PCM16 mono 16 kHz bytes: read directly if it already is that WAV, else via ffmpeg."""
    try:
        with wave.open(path, "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE):
                return w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        pass
    if shutil.which("ffmpeg") is None:
        raise SpeechRevolutionsError(
            "ffmpeg is needed to convert this file to PCM16 mono 16 kHz (or pass such a WAV)"
        )
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        raise SpeechRevolutionsError(
            f"ffmpeg could not decode {path}: {r.stderr.decode(errors='replace').strip()[:300]}"
        )
    return r.stdout


def _session_update(
    language: Optional[str],
    languages: Optional[list[str]],
    vocabulary: Optional[list[str]],
    turn_detection: Optional[dict],
    extra: Optional[dict],
) -> dict:
    m: dict[str, Any] = {
        "type": "session.update",
        "audio_format": {"encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE, "channels": 1},
    }
    if language and language != "auto":
        m["language"] = language
    if languages:
        m["languages"] = list(languages)
    if vocabulary:
        m["vocabulary"] = list(vocabulary)
    if turn_detection:
        m["turn_detection"] = dict(turn_detection)
    if extra:
        m.update(extra)
    return m


def stream_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/api/v1/stream"


class AsyncLiveSession:
    """A live transcription session. Create it with ``await client.stream(...)``.

    Send audio with :meth:`send_audio` as it arrives, read events by iterating the session (or pass
    ``on_event``), and call :meth:`finish` at the end of the audio to get the :class:`StreamResult`.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        session_update: dict,
        on_event: Optional[Callable[[StreamEvent], Any]] = None,
        resume_window_s: float = RESUME_WINDOW_S,
        connect_timeout: float = 15.0,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._update = session_update
        self._on_event = on_event
        self._resume_window_s = resume_window_s
        self._connect_timeout = connect_timeout

        self.job_id: Optional[str] = None
        self._ws: Any = None
        # Audio not yet acknowledged, as (start_ms, bytes) chunks on the job timeline.
        self._buffer: list[tuple[int, bytes]] = []
        self._audio_ms = 0  # total audio handed to send_audio
        self._next_ms = 0  # the next millisecond to put on the wire
        self._replay_until = 0  # sending below this is a replay, paced at REPLAY_SPEED
        self._committed_ms = 0
        self._finishing = False
        self._final_sent = False
        self._done: Optional[StreamEvent] = None
        self._fatal: Optional[StreamError] = None
        self._finals: dict[int, Final] = {}
        self._events: asyncio.Queue = asyncio.Queue()
        self._queue_events = True  # off under LiveSession, which keeps its own queue
        self._wake = asyncio.Event()
        self._closed = asyncio.Event()
        self._reader: Optional[asyncio.Task] = None
        self._sender: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------ public

    async def connect(self) -> "AsyncLiveSession":
        """Open the session. Returns once the server has sent ``session.created``."""
        await self._open(resume=False)
        self._sender = asyncio.ensure_future(self._send_loop())
        return self

    async def send_audio(self, pcm: bytes) -> None:
        """Queue PCM16 mono 16 kHz audio. Send it at the pace it is produced (real time for a microphone)."""
        if self._fatal:
            raise self._fatal
        if self._finishing:
            raise StreamError("send_audio after finish()")
        if len(pcm) % 2:
            raise ValueError("PCM16 audio must be an even number of bytes")
        if not pcm:
            return
        self._buffer.append((self._audio_ms, bytes(pcm)))
        self._audio_ms += len(pcm) // BYTES_PER_MS
        self._wake.set()

    async def end_turn(self) -> None:
        """End the current turn now (push-to-talk release). The open utterance becomes a final."""
        await self._send_json({"type": "turn.end"})

    async def finish(self, timeout: float = 120.0) -> StreamResult:
        """End the audio, wait for ``transcription.done``, and return the result."""
        self._finishing = True
        self._wake.set()
        try:
            await asyncio.wait_for(self._closed.wait(), timeout)
        except asyncio.TimeoutError:
            await self.close()
            return self._result(complete=False)
        await self._teardown()
        if self._fatal and self._done is None:
            raise self._fatal
        return self._result(complete=self._done is not None)

    async def _teardown(self) -> None:
        """Stop the background tasks and close the socket cleanly (the server is waiting on the close)."""
        for task in (self._sender, self._reader):
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def close(self) -> None:
        """Drop the connection without waiting for the transcript."""
        self._closed.set()
        await self._teardown()
        await self._events.put(None)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
        while True:
            ev = await self._events.get()
            if ev is None:
                return
            yield ev

    async def __aenter__(self) -> "AsyncLiveSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        if not self._closed.is_set():
            await self.close()

    # ------------------------------------------------------------ connection

    async def _open(self, *, resume: bool) -> None:
        import websockets  # optional dependency: pip install "speechrevolutions[streaming]"

        url = self._url + (f"?job_id={self.job_id}" if resume and self.job_id else "")
        headers = {"X-API-Key": self._api_key, "User-Agent": USER_AGENT}
        try:
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                max_size=None,
                open_timeout=self._connect_timeout,
                ping_interval=PING_INTERVAL_S,
                ping_timeout=PING_INTERVAL_S * 2,
            )
        except websockets.InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise AuthenticationError(f"streaming refused (HTTP {status}): check the API key") from exc
            if status == 429:
                raise RateLimitError("streaming refused: too many sessions or connects (HTTP 429)") from exc
            raise StreamError(f"streaming refused (HTTP {status})") from exc
        self._ws = ws
        if not resume:
            await ws.send(json.dumps(self._update))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))  # start of audio
        # Wait for session.created / session.resumed before handing back.
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), self._connect_timeout))
            ev = StreamEvent.from_message(m)
            if ev.type == "error":
                try:
                    await ws.close()
                except Exception:
                    pass
                raise self._error(ev)
            await self._handle(ev)
            if ev.type in ("session.created", "session.resumed"):
                break
        self._reader = asyncio.ensure_future(self._read_loop())

    def _error(self, ev: StreamEvent) -> StreamError:
        code = ev.code or ""
        msg = ev.message or code or "stream error"
        if code == "invalid_api_key":
            return StreamError(f"invalid API key: {msg}", code=code)
        return StreamError(msg, code=code or None)

    async def _read_loop(self) -> None:
        import websockets

        ws = self._ws
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev = StreamEvent.from_message(json.loads(raw))
                await self._handle(ev)
                if ev.type == "transcription.done":
                    self._closed.set()
                    await self._events.put(None)
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return
                if ev.type == "error" and (ev.code in FATAL_ERRORS or not self.job_id):
                    self._fatal = self._error(ev)
                    self._closed.set()
                    await self._events.put(None)
                    return
            close_code = ws.close_code
        except websockets.ConnectionClosed as exc:
            close_code = exc.rcvd.code if exc.rcvd else None
        except asyncio.CancelledError:
            return
        if self._closed.is_set():
            return
        await self._resume(immediately=close_code == 1012)

    async def _resume(self, *, immediately: bool) -> None:
        """The connection dropped before the transcript was done: reconnect with ?job_id= and replay."""
        deadline = time.monotonic() + self._resume_window_s
        delay = 0.0 if immediately else 0.5
        while not self._closed.is_set():
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._open(resume=True)
                return
            except StreamError as exc:
                if exc.code in FATAL_ERRORS:
                    self._fatal = exc
                    break
            except (AuthenticationError, RateLimitError) as exc:
                self._fatal = StreamError(str(exc))
                break
            except Exception:
                pass
            if time.monotonic() > deadline:
                self._fatal = StreamError("connection lost and could not be resumed within 60 s")
                break
            delay = min(max(delay * 2, 0.5), 5.0)
        self._closed.set()
        await self._events.put(None)

    # ------------------------------------------------------------ events

    async def _handle(self, ev: StreamEvent) -> None:
        t = ev.type
        if t == "session.created":
            self.job_id = ev.job_id
        elif t in ("session.resumed", "resend"):
            from_ms = ev.raw.get("resume_from_ms", ev.raw.get("from_ms", self._committed_ms))
            await self._rewind(int(from_ms or 0))
        elif t == "audio.committed":
            self._committed_ms = int(ev.raw.get("through_ms") or 0)
            self._buffer = [(s, b) for s, b in self._buffer if s + len(b) // BYTES_PER_MS > self._committed_ms]
        elif t == "final" and ev.index is not None:
            self._finals[ev.index] = Final(ev.index, ev.text or "", ev.start, ev.end)
        elif t == "final.revised" and ev.index is not None:
            f = self._finals.get(ev.index)
            self._finals[ev.index] = Final(ev.index, ev.text or "", f.start if f else None, f.end if f else None)
        elif t == "transcription.done":
            self._done = ev
        if t == "resend":
            return  # internal: the replay is the client's answer, not an event for the caller
        if self._queue_events:
            await self._events.put(ev)
        if self._on_event is not None:
            r = self._on_event(ev)
            if asyncio.iscoroutine(r):
                await r

    async def _rewind(self, from_ms: int) -> None:
        """Re-send everything from `from_ms`: rewind, then the buffer (paced), then carry on live."""
        async with self._send_lock:
            await self._raw_send({"type": "input_audio_buffer.rewind", "from_ms": from_ms})
            self._replay_until = max(self._next_ms, self._replay_until)
            self._next_ms = from_ms
            if self._final_sent:
                self._final_sent = False  # the commit is re-sent once the replay is through
        self._wake.set()

    # ------------------------------------------------------------ sending

    async def _send_json(self, m: dict) -> None:
        async with self._send_lock:
            await self._raw_send(m)

    async def _raw_send(self, m: dict) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(m))
        except Exception:
            pass  # the reader notices the drop and resumes; the buffer still holds the audio

    async def _send_loop(self) -> None:
        try:
            while not self._closed.is_set():
                sent = await self._send_some()
                if not sent:
                    if self._finishing and not self._final_sent and self._next_ms >= self._audio_ms:
                        await self._send_json({"type": "input_audio_buffer.commit", "final": True})
                        self._final_sent = True
                    self._wake.clear()
                    await self._wake.wait()
        except asyncio.CancelledError:
            return

    async def _send_some(self) -> bool:
        """Send the next chunk of buffered audio, if any. Replays are paced; live audio goes at once."""
        chunk = self._chunk_at(self._next_ms)
        if chunk is None:
            return False
        start, data = chunk
        replay = start < self._replay_until
        async with self._send_lock:
            await self._raw_send({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode()})
            self._next_ms = start + len(data) // BYTES_PER_MS
        if replay:
            await asyncio.sleep(len(data) / BYTES_PER_MS / 1000 / REPLAY_SPEED)
        return True

    def _chunk_at(self, ms: int) -> Optional[tuple[int, bytes]]:
        for s, b in self._buffer:
            e = s + len(b) // BYTES_PER_MS
            if e <= ms:
                continue
            off = max(0, ms - s) * BYTES_PER_MS
            piece = b[off: off + CHUNK_MS * BYTES_PER_MS]
            return (s + off // BYTES_PER_MS, piece) if piece else None
        return None

    def _result(self, *, complete: bool) -> StreamResult:
        finals = [self._finals[k] for k in sorted(self._finals)]
        if self._done is not None and self._done.text is not None:
            text = self._done.text
        else:
            text = " ".join(f.text for f in finals if f.text)
        return StreamResult(job_id=self.job_id, text=text, finals=finals, complete=complete)


class LiveSession:
    """Synchronous wrapper around :class:`AsyncLiveSession`, run on a background event loop.

    Iterate it for events (blocking), or pass ``on_event`` (called from the background thread).
    """

    def __init__(self, inner_factory: Callable[[], AsyncLiveSession]) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="sr-live-session", daemon=True)
        self._thread.start()
        self._events: "queue.Queue[Optional[StreamEvent]]" = queue.Queue()
        try:
            self._inner = self._call(self._make(inner_factory))
        except BaseException:
            self._shutdown()
            raise

    async def _make(self, factory: Callable[[], AsyncLiveSession]) -> AsyncLiveSession:
        inner = factory()
        user_cb = inner._on_event

        def relay(ev: StreamEvent) -> Any:
            self._events.put(ev)
            return user_cb(ev) if user_cb is not None else None

        inner._on_event = relay
        inner._queue_events = False
        await inner.connect()
        return inner

    def _call(self, coro: Any, timeout: Optional[float] = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    @property
    def job_id(self) -> Optional[str]:
        return self._inner.job_id

    def send_audio(self, pcm: bytes) -> None:
        self._call(self._inner.send_audio(pcm))

    def end_turn(self) -> None:
        self._call(self._inner.end_turn())

    def finish(self, timeout: float = 120.0) -> StreamResult:
        try:
            return self._call(self._inner.finish(timeout))
        finally:
            self._events.put(None)
            self._shutdown()

    def close(self) -> None:
        try:
            self._call(self._inner.close(), timeout=10)
        finally:
            self._events.put(None)
            self._shutdown()

    def _shutdown(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def __iter__(self) -> Iterator[StreamEvent]:
        while True:
            ev = self._events.get()
            if ev is None:
                return
            yield ev

    def __enter__(self) -> "LiveSession":
        return self

    def __exit__(self, *exc: object) -> None:
        if self._loop.is_running():
            self.close()


#: The server accepts audio at most this far ahead of real time (decoded audio <= wall time since the
#: session started + 10 s); anything faster is refused with ``use_batch``. Kept a little under.
MAX_LEAD_S = 8.0


async def stream_pcm(session: AsyncLiveSession, pcm: bytes, *, speed: float = 1.0) -> StreamResult:
    """Send recorded `pcm` through `session` as live audio, then finish.

    Audio goes out at `speed` x real time but never more than ``MAX_LEAD_S`` ahead of the clock, which is
    the server's limit: the first seconds go out at once, then real time. ``speed`` below 1 slows it down.
    """
    step = CHUNK_MS * BYTES_PER_MS
    t0 = time.monotonic()
    for n, i in enumerate(range(0, len(pcm), step)):
        await session.send_audio(pcm[i: i + step])
        sent_s = (n + 1) * CHUNK_MS / 1000
        due = max(t0 + sent_s / speed, t0 + sent_s - MAX_LEAD_S) if speed > 1 else t0 + sent_s / speed
        wait = due - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
    return await session.finish()
