"""Tests that talk to the REAL Speech Revolutions API.

Everything else in this repo runs against `mock_api.py`, which is a MODEL of the
contract written by reading the server. A model can be wrong in the same way the
client is wrong, and then the suite is green while production is broken. These
tests exist to close that gap — and they immediately found three places where
the mock had invented a shape the API has never returned (no `status` in a job
summary, a timestamp rather than a job id as the list cursor, and a missing
`llm_download_url`).

They are OPT-IN, because they create real jobs on a real account and cost real
money (a few seconds of audio each, so fractions of a cent):

    SR_LIVE=1 SPEECHREVOLUTIONS_API_KEY=stt_... pytest tests/live -v

The webhook test additionally needs a PUBLICLY ROUTABLE callback URL. The
pipeline refuses anything else — `webhook_service._host_is_public()` resolves
the host and requires every address to be globally routable outside `local`
env — so localhost and 127.0.0.1 are rejected before a request is made. Point
SR_LIVE_WEBHOOK_BASE at a tunnel (cloudflared, ngrok) and the receiver is
started for you:

    SR_LIVE=1 SR_LIVE_WEBHOOK_BASE=https://xyz.trycloudflare.com \
        SR_LIVE_WEBHOOK_PORT=8799 pytest tests/live -v -k webhook
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from speechrevolutions import SpeechRevolutions

pytestmark = pytest.mark.skipif(
    os.environ.get("SR_LIVE") != "1",
    reason="live API tests are opt-in: set SR_LIVE=1 (creates real, billable jobs)",
)

REPO = Path(__file__).resolve().parents[3]
# Shortest first: these create real, billed jobs, so a 1MB clip beats a 68-minute
# one for everything except a deliberate long-file test. Override with
# SR_LIVE_AUDIO=/path/to/clip.mp3
AUDIO_CANDIDATES = [
    REPO / "qa-audio" / "ru_uk_segment.mp3",
    REPO / "qa-audio" / "crawl11.mp3",
    REPO / "stt_unified_repo" / "clusters" / "autoscale_cluster" / "src"
    / "autoscale_cluster" / "assets" / "canary_audio.mp3",
]


@pytest.fixture(scope="module")
def audio() -> Path:
    override = os.environ.get("SR_LIVE_AUDIO")
    if override:
        return Path(override)
    for p in AUDIO_CANDIDATES:
        if p.exists():
            return p
    pytest.skip(f"no test audio found; looked in {[str(p) for p in AUDIO_CANDIDATES]}")


@pytest.fixture(scope="module")
def client():
    if not os.environ.get("SPEECHREVOLUTIONS_API_KEY"):
        pytest.skip("SPEECHREVOLUTIONS_API_KEY is not set")
    with SpeechRevolutions(timeout=900) as c:
        yield c


# ---------------------------------------------------------------------------
# Read-only: no jobs created, no cost
# ---------------------------------------------------------------------------

def test_list_jobs_returns_the_documented_shape(client):
    page = client.list_jobs(limit=3)
    assert set(page) >= {"jobs", "next_before"}
    for job in page["jobs"]:
        # The live JobSummary is exactly {job_id, created_at}. It does NOT
        # carry a status — read one with get_job_status if you need that.
        assert set(job) == {"job_id", "created_at"}, f"unexpected summary shape: {job}"
        uuid.UUID(job["job_id"])
        assert job["created_at"].startswith("20")


def test_list_jobs_cursor_is_a_created_at_timestamp(client):
    page = client.list_jobs(limit=2)
    if not page.get("next_before"):
        pytest.skip("account has fewer than 3 jobs; nothing to paginate")
    # The cursor is the last row's created_at, NOT a job id.
    assert page["next_before"] == page["jobs"][-1]["created_at"]


def test_list_jobs_paginates_without_duplicates(client):
    seen: list[str] = []
    cursor = None
    for _ in range(5):
        page = client.list_jobs(limit=2, before=cursor)
        seen += [j["job_id"] for j in page["jobs"]]
        cursor = page.get("next_before")
        if not cursor:
            break
    if len(seen) < 3:
        pytest.skip("account has too few jobs to exercise pagination")
    assert len(seen) == len(set(seen)), "pagination returned a duplicate"


def test_get_job_status_shape(client):
    page = client.list_jobs(limit=1)
    if not page["jobs"]:
        pytest.skip("account has no jobs")
    status = client.get_job_status(page["jobs"][0]["job_id"])
    assert status.status in {"processing", "completed", "failed"}
    if status.is_completed:
        assert status.download_url


def test_check_failed_against_a_real_job(client):
    page = client.list_jobs(limit=2)
    if not page["jobs"]:
        pytest.skip("account has no jobs")
    ids = [j["job_id"] for j in page["jobs"]]
    flags = client.check_failed(ids)
    assert len(flags) == len(ids)
    assert all(isinstance(f, bool) for f in flags)


def test_a_bad_key_is_rejected():
    from speechrevolutions.exceptions import AuthenticationError

    with SpeechRevolutions(api_key="stt_definitely_not_a_real_key") as c:
        with pytest.raises(AuthenticationError):
            c.list_jobs(limit=1)


# ---------------------------------------------------------------------------
# Real transcription
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_transcribe_a_real_file(client, audio):
    events: list = []
    result = client.transcribe(str(audio), speaker_labels=True,
                              on_progress=events.append)
    assert result.text.strip(), "the API returned an empty transcript"
    assert result.words, "no word timings came back"
    # Progress events are emitted per PIPELINE STEP, and the chunk length is
    # 600s — so a file under ~10 minutes is a single chunk and can finish with
    # ZERO progress events. Verified live: a 12.8s job emitted only `completed`,
    # while a 68-minute job emitted 9 (preprocess, chunk:0..6, aggregation).
    # Requiring events for any file would be wrong; see the multi-chunk test.
    assert all(e.total and e.completed <= e.total for e in events)
    print(f"\n  transcript: {result.text[:120]}")
    print(f"  words: {len(result.words)}  utterances: {len(result.utterances)}")
    print(f"  progress events: {len(events)}")


@pytest.mark.slow
def test_submit_then_poll_to_completion(client, audio):
    """The submit + poll path, which is what batch callers actually use."""
    job_id = client.submit(str(audio))
    uuid.UUID(job_id)

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = client.get_job_status(job_id)
        if status.is_completed:
            break
        assert not status.is_failed, f"job failed: {status.failed_stage} {status.reason}"
        time.sleep(5)
    else:
        pytest.fail("job did not complete within 10 minutes")

    result = client.get_transcript(job_id)
    assert result.text.strip()
    assert result.job_id == job_id


@pytest.mark.slow
def test_live_progress_fires_for_a_multi_chunk_file(client):
    """Live progress is a headline feature; prove it actually arrives.

    Needs a file long enough to span more than one 600s chunk. Skipped unless
    SR_LIVE_LONG_AUDIO points at one, because it is a genuinely billable job.
    """
    long_audio = os.environ.get("SR_LIVE_LONG_AUDIO")
    if not long_audio or not Path(long_audio).exists():
        pytest.skip("set SR_LIVE_LONG_AUDIO to a file longer than 10 minutes")

    events: list = []
    result = client.transcribe(long_audio, on_progress=events.append)
    assert result.text.strip()
    assert len(events) >= 3, f"expected several progress events, got {len(events)}"
    steps = [e.step for e in events]
    assert "preprocess" in steps, f"no preprocess step in {steps}"
    assert any(s and s.startswith("chunk:") for s in steps), f"no chunk steps in {steps}"
    assert "aggregation" in steps, f"no aggregation step in {steps}"
    assert events[-1].completed == events[-1].total
    print(f"\n  progress steps: {steps}")


@pytest.mark.slow
def test_srt_output_from_the_real_api(client, audio):
    result = client.transcribe(str(audio), output_type="srt")
    assert "-->" in result.text, f"not SRT: {result.text[:200]}"


# ---------------------------------------------------------------------------
# Webhooks — the part that has never been exercised at all
# ---------------------------------------------------------------------------

class LiveReceiver:
    """Records deliveries verbatim. Fronted by a public tunnel."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.received: list[dict] = []
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> LiveReceiver:
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: object) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                outer.received.append({
                    "raw": raw,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                })
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self._server.daemon_threads = True
        threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def wait(self, timeout: float = 600.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.received:
                return self.received[0]
            time.sleep(0.5)
        raise AssertionError(f"no webhook arrived within {timeout}s")


@pytest.mark.slow
def test_webhook_is_delivered_by_the_real_pipeline(client, audio):
    """Submit a real job with a callback_url and prove the platform calls back.

    This needs a public tunnel: the pipeline resolves the callback host and
    refuses anything that is not globally routable.
    """
    base = os.environ.get("SR_LIVE_WEBHOOK_BASE")
    if not base:
        pytest.skip(
            "set SR_LIVE_WEBHOOK_BASE to a public tunnel URL "
            "(the pipeline refuses non-public callback hosts)"
        )
    port = int(os.environ.get("SR_LIVE_WEBHOOK_PORT", "8799"))
    callback = f"{base.rstrip('/')}/webhooks/speechrevolutions"

    with LiveReceiver(port) as rcv:
        job_id = client.submit(str(audio), callback_url=callback)
        print(f"\n  submitted {job_id}, awaiting callback at {callback}")
        got = rcv.wait(timeout=900)

    event = json.loads(got["raw"])
    print(f"  delivered: {event}")
    print(f"  headers: { {k: v for k, v in got['headers'].items() if k.startswith('x-sr')} }")

    assert event["job_id"] == job_id
    assert event["status"] in {"completed", "failed"}
    assert got["path"].endswith("/webhooks/speechrevolutions")

    # Headers the documented receivers rely on.
    assert got["headers"].get("x-sr-event") == event["status"]
    assert got["headers"].get("x-sr-delivery"), "no delivery id"
    assert got["headers"].get("user-agent", "").startswith("SpeechRevolutions-Webhook/")

    # If the platform is configured with a signing secret, it must be over the
    # raw bytes. Absent means the secret is unset server-side, which is worth
    # reporting rather than silently passing.
    sig = got["headers"].get("x-sr-signature")
    if sig:
        assert sig.startswith("sha256="), f"unexpected signature format: {sig}"
        print("  signature present and well-formed")
    else:
        print("  NOTE: no X-SR-Signature — webhook_signing_secret is unset server-side")

    if event["status"] == "completed":
        result = client.get_transcript(job_id)
        assert result.text.strip()
