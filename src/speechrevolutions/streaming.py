"""Realtime streaming transcription (WebSocket /api/v1/stream).

    import asyncio
    from speechrevolutions import AsyncSpeechRevolutions

    async def main():
        client = AsyncSpeechRevolutions()
        async with client.stream(language=None) as session:      # None = detect the language
            asyncio.create_task(session.send_file("call.wav"))   # or: await session.send_audio(pcm16_bytes)
            async for event in session:
                if event.type == "partial":
                    print("\\r" + event.text, end="")
                elif event.type == "final":
                    print("\\n", event.start, event.end, event.text)

Audio is PCM16 mono 16 kHz (`send_file` converts any file with ffmpeg). Events: `partial` (replaces the previous
partial of the same `index`; `stable` = how many leading characters are settled), `final`, `revision` (a final's
corrected text, by `index`), `speech_started`, `turn_ended`, `language`, `done` (the whole text), `error`.
Requires the `streaming` extra: `pip install speechrevolutions[streaming]`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

SAMPLE_RATE = 16000
_CHUNK_S = 0.08


@dataclass
class StreamEvent:
    """One server event, normalised. `raw` holds the original message."""

    type: str
    text: str = ""
    index: int | None = None
    start: float | None = None
    end: float | None = None
    stable: int | None = None
    language: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


_TYPES = {"transcription.partial": "partial", "final": "final", "final.revised": "revision",
          "speech.started": "speech_started", "turn.ended": "turn_ended", "language.detected": "language",
          "transcription.done": "done", "error": "error", "session.created": "session"}


def _event(m: dict[str, Any]) -> StreamEvent:
    t = _TYPES.get(m.get("type", ""), m.get("type", ""))
    return StreamEvent(type=t, text=m.get("text") or (m.get("error") if t == "error" else "") or "",
                       index=m.get("index"), start=m.get("start", m.get("audio_s")), end=m.get("end"),
                       stable=m.get("stable"), language=m.get("language"), raw=m)


class StreamSession:
    """A live transcription session. Use as `async with client.stream(...) as session:`."""

    def __init__(self, url: str, api_key: str, *, language: str | None, vocabulary: list[str] | None,
                 turn_threshold: float | None, max_silence_ms: int | None):
        self._url, self._key = url, api_key
        self._update: dict[str, Any] = {"type": "session.update"}
        if language and language != "auto":
            self._update["language"] = language
        if vocabulary:
            self._update["vocabulary"] = list(vocabulary)
        turn = {k: v for k, v in (("threshold", turn_threshold), ("max_silence_ms", max_silence_ms)) if v is not None}
        if turn:
            self._update["turn_detection"] = turn
        self._ws = None
        self.job_id: str | None = None

    async def __aenter__(self) -> StreamSession:
        try:
            import websockets
        except ImportError as e:  # pragma: no cover
            raise ImportError("streaming needs the `streaming` extra: pip install speechrevolutions[streaming]") from e
        self._ws = await websockets.connect(self._url, additional_headers={"X-API-Key": self._key}, max_size=None)
        await self._ws.send(json.dumps(self._update))
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def send_audio(self, pcm16: bytes) -> None:
        """Send PCM16 mono 16 kHz audio (any chunk size)."""
        if pcm16:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode()}))

    async def end_turn(self) -> None:
        """Finalise the current utterance now (push-to-talk release)."""
        await self._ws.send(json.dumps({"type": "turn.end"}))

    async def finish(self) -> None:
        """No more audio: the server sends the remaining finals, then `done`."""
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

    async def send_file(self, path: str, *, realtime: bool = True) -> None:
        """Convert a file to PCM16 16 kHz mono (ffmpeg) and send it, paced like a microphone; then finish()."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("send_file needs ffmpeg on PATH")
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
                             capture_output=True, check=True).stdout
        step = int(SAMPLE_RATE * _CHUNK_S) * 2
        t0 = time.monotonic()
        for i in range(0, len(pcm), step):
            await self.send_audio(pcm[i:i + step])
            if realtime:
                await asyncio.sleep(max(0.0, t0 + (i + step) / (2 * SAMPLE_RATE) - time.monotonic()))
        await self.finish()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[StreamEvent]:
        import websockets

        try:
            async for raw in self._ws:
                ev = _event(json.loads(raw))
                if ev.type == "session":
                    self.job_id = ev.raw.get("job_id")
                    continue
                yield ev
                if ev.type in ("done", "error"):
                    return
        except websockets.ConnectionClosed:
            return
