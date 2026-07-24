"""Live progress for a web app.

The scenario: you're building a transcription web app and want to show each user
a live progress bar while their file is transcribed (exactly what a dashboard
needs). The SDK surfaces progress through two callbacks — this example shows how
to turn those into a single 0–100 number you can store per job and serve to your
frontend.

Key idea
--------
`transcribe()` accepts:
  * ``on_upload_progress`` — fires while the file uploads   (event.step == "upload")
  * ``on_progress``        — fires while the server transcribes
Each call gets a ``ProgressEvent`` with ``event.percent`` (0–100, or None before
totals are known). You decide what to do with it: write it to your DB, push it
over a WebSocket, or (here) store it in a dict your HTTP endpoint reads.

Run this file directly to see it print; the FastAPI block at the bottom shows how
you'd wire it into a real backend.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

from speechrevolutions import AsyncSpeechRevolutions, ProgressEvent

# Weight the two phases into one bar. Upload is usually quick; give it the first
# slice and let transcription fill the rest. Tune to taste.
_UPLOAD_WEIGHT = 0.15  # upload spans 0–15% of the overall bar
_TRANSCRIBE_WEIGHT = 0.85  # transcription spans 15–100%


@dataclass
class JobProgress:
    """The latest progress for one job — the shape you'd serve to your frontend."""

    phase: str = "starting"  # "upload" | "transcribe" | "done"
    percent: float = 0.0  # overall 0–100 across both phases
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _set(self, phase: str, overall: float) -> None:
        with self.lock:
            self.phase = phase
            # never let the bar go backwards (events can arrive slightly out of order)
            self.percent = max(self.percent, round(overall, 1))

    def on_upload(self, event: ProgressEvent) -> None:
        pct = event.percent or 0.0
        self._set("upload", pct * _UPLOAD_WEIGHT)

    def on_transcribe(self, event: ProgressEvent) -> None:
        pct = event.percent or 0.0
        self._set("transcribe", _UPLOAD_WEIGHT * 100 + pct * _TRANSCRIBE_WEIGHT)

    def snapshot(self) -> dict:
        with self.lock:
            return {"phase": self.phase, "percent": self.percent}


async def transcribe_with_progress(audio: str, store: JobProgress):
    async with AsyncSpeechRevolutions() as client:  # reads SPEECHREVOLUTIONS_API_KEY
        result = await client.transcribe(
            audio,
            on_upload_progress=store.on_upload,   # <- your handler; do anything with event.percent
            on_progress=store.on_transcribe,
        )
    store._set("done", 100.0)
    return result


async def main() -> None:
    store = JobProgress()

    # In a real app you'd read store.snapshot() from an HTTP endpoint on another
    # task/thread. Here a printer task polls it so you can watch the number move.
    async def printer():
        last = None
        while store.phase != "done":
            snap = store.snapshot()
            if snap != last:
                print(f"  [{snap['phase']:>10}] {snap['percent']:5.1f}%")
                last = snap
            await asyncio.sleep(0.2)

    printer_task = asyncio.create_task(printer())
    result = await transcribe_with_progress("audio.mp3", store)
    await printer_task
    print(f"\nDone — {len(result.text)} chars, {len(result.utterances)} utterances")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Wiring it into a real backend (FastAPI sketch)
# ---------------------------------------------------------------------------
# A dict keyed by job_id holds each job's live progress; the transcription runs
# in a background task, and your frontend polls GET /progress/{job_id} (or you
# push over a WebSocket). Percent comes straight from the SDK callbacks.
#
#     from fastapi import FastAPI, BackgroundTasks
#     app = FastAPI()
#     JOBS: dict[str, JobProgress] = {}
#
#     @app.post("/transcribe")
#     async def start(url: str, background: BackgroundTasks):
#         job_id = url  # or your own id
#         JOBS[job_id] = JobProgress()
#         background.add_task(transcribe_with_progress, url, JOBS[job_id])
#         return {"job_id": job_id}
#
#     @app.get("/progress/{job_id}")
#     def progress(job_id: str):
#         return JOBS[job_id].snapshot()   # {"phase": "transcribe", "percent": 63.5}
