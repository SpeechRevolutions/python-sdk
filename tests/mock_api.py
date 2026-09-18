"""A working stand-in for the Speech Revolutions API, for tests.

Every endpoint the SDKs and the cookbook actually call, implemented against the
real contract in `user_cluster/server/routes/` and
`aggregation_cluster/services/webhook_service.py`:

    POST /api/v1/upload                      create a job (+ presigned PUT)
    POST /api/v1/upload/progress             upload heartbeat
    POST /api/v1/upload/complete             enqueue after the client's PUT
    POST /api/v1/upload/multipart/create     multipart create (+ part URLs)
    POST /api/v1/upload/multipart/complete   multipart complete
    POST /api/v1/upload/multipart/abort      multipart abort
    GET  /api/v1/jobs                        cursor-paginated list
    GET  /api/v1/jobs/{id}                   status (+ fresh download_url)
    GET  /api/v1/jobs/{id}/stream            SSE progress
    POST /api/v1/jobs/cancel                 cancel
    POST /api/v1/jobs/check-failed           batch failure check
    PUT  /_upload/{token}                    the presigned target
    GET  /_result/{job_id}                   the result object

Webhooks are delivered the way the platform delivers them: a compact JSON body
(`separators=(",", ":")`), and the `X-SR-Event`, `X-SR-Delivery` and
`X-SR-Signature: sha256=<hmac>` headers. That matters — a receiver that
re-serializes the parsed body before checking the HMAC will not match, which is
exactly the bug the cookbook recipe's comment warns about.

Behaviour is steered per-instance rather than by monkeypatching the SDK, so a
test exercises the same code path a real caller would:

    api = MockAPI(progress_steps=3)
    api.fail_job_at = "gpu_timestamps"     # make the next job fail
    api.multipart_enabled = False          # force the single-shot fallback
    api.stream_status = 503                # force the SSE->polling fallback
    api.drop_stream_after = 1              # cut the stream to test reconnect

Run it standalone to point a cookbook recipe or another SDK at it:

    python mock_api.py --port 8888
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# A small but complete transcript: two speakers, word timings, a language span.
SAMPLE_TRANSCRIPT = {
    "words": [
        {"word": "Good", "start": 0.00, "end": 0.32, "speaker": "A", "confidence": 0.99,
         "language": "en"},
        {"word": "morning", "start": 0.32, "end": 0.81, "speaker": "A", "confidence": 0.98,
         "language": "en"},
        {"word": "everyone.", "start": 0.81, "end": 1.44, "speaker": "A", "confidence": 0.97,
         "language": "en"},
        {"word": "Thanks", "start": 2.10, "end": 2.48, "speaker": "B", "confidence": 0.96,
         "language": "en"},
        {"word": "for", "start": 2.48, "end": 2.63, "speaker": "B", "confidence": 0.99,
         "language": "en"},
        {"word": "joining.", "start": 2.63, "end": 3.20, "speaker": "B", "confidence": 0.98,
         "language": "en"},
    ],
    "diarization": [
        {"speaker": "A", "start": 0.00, "end": 1.44},
        {"speaker": "B", "start": 2.10, "end": 3.20},
    ],
    "languages": [{"start": 0.0, "end": 3.2, "language": "en"}],
}

SAMPLE_SRT = (
    "1\n00:00:00,000 --> 00:00:01,440\nGood morning everyone.\n\n"
    "2\n00:00:02,100 --> 00:00:03,200\nThanks for joining.\n"
)
SAMPLE_VTT = (
    "WEBVTT\n\n00:00:00.000 --> 00:00:01.440\nGood morning everyone.\n\n"
    "00:00:02.100 --> 00:00:03.200\nThanks for joining.\n"
)
SAMPLE_TXT = "Good morning everyone. Thanks for joining.\n"


@dataclass
class Job:
    job_id: str
    output_type: str = "json"
    callback_url: str | None = None
    status: str = "processing"
    uploaded: bytes = b""
    parts: dict[int, str] = field(default_factory=dict)
    completed_at: float = 0.0
    failed_stage: str | None = None
    reason: str | None = None
    options: dict = field(default_factory=dict)


class MockAPI:
    """Runs the mock on a background thread. Use as a context manager."""

    def __init__(
        self,
        *,
        api_key: str = "test-key",
        progress_steps: int = 2,
        webhook_secret: str | None = None,
        multipart_enabled: bool = True,
        part_size: int = 5 * 1024 * 1024,
    ) -> None:
        self.api_key = api_key
        self.progress_steps = progress_steps
        self.webhook_secret = webhook_secret
        self.multipart_enabled = multipart_enabled
        self.part_size = part_size

        # Steerable failure modes.
        self.fail_job_at: str | None = None
        self.stream_status: int = 200
        self.drop_stream_after: int | None = None
        self.result_available_after: float = 0.0  # seconds; delays GET /_result
        self.require_auth: bool = True

        self.jobs: dict[str, Job] = {}
        self.requests: list[tuple[str, str]] = []   # (method, path)
        self.webhooks_sent: list[dict] = []
        self.upload_heartbeats: list[str] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> MockAPI:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        # poll_interval drives how long shutdown() blocks. The default is 0.5s,
        # which at ~180 tests is a minute and a half of pure teardown.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None, "use MockAPI as a context manager"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # -- helpers for assertions -------------------------------------------

    def paths(self, method: str | None = None) -> list[str]:
        with self._lock:
            return [p for m, p in self.requests if method is None or m == method]

    def count(self, method: str, path: str) -> int:
        return sum(1 for p in self.paths(method) if p.split("?")[0] == path)

    def wait_webhooks(self, n: int = 1, timeout: float = 5.0) -> list[dict]:
        """Block until n deliveries have been ATTEMPTED by the sender thread.

        Distinct from waiting on the receiver: the receiver records on arrival,
        the sender records once the POST returns or raises. Asserting on
        `webhooks_sent` right after the receiver has seen one is a race.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.webhooks_sent) >= n:
                    return list(self.webhooks_sent)
            time.sleep(0.02)
        raise AssertionError(
            f"expected {n} webhook delivery attempt(s), got {len(self.webhooks_sent)}"
        )

    def only_job(self) -> Job:
        with self._lock:
            assert len(self.jobs) == 1, f"expected exactly 1 job, got {len(self.jobs)}"
            return next(iter(self.jobs.values()))

    # -- webhook delivery, matching the platform ---------------------------

    def deliver_webhook(self, job: Job) -> None:
        """POST the completion notification exactly as aggregation_cluster does."""
        if not job.callback_url:
            return
        payload: dict = {"status": job.status}
        if job.status == "failed":
            payload["step"] = job.failed_stage
            payload["reason"] = job.reason
        body = json.dumps({"job_id": job.job_id, **payload}, separators=(",", ":")).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SpeechRevolutions-Webhook/1",
            "X-SR-Event": job.status,
            "X-SR-Delivery": str(uuid.uuid4()),
        }
        if self.webhook_secret:
            headers["X-SR-Signature"] = "sha256=" + hmac.new(
                self.webhook_secret.encode(), body, hashlib.sha256
            ).hexdigest()
        record = {"url": job.callback_url, "body": body, "headers": headers,
                  "job_id": job.job_id, "status": job.status}
        try:
            req = urllib.request.Request(
                job.callback_url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                record["response_status"] = resp.status
        except Exception as exc:  # a receiver that 4xx's still counts as delivered
            record["error"] = str(exc)
        with self._lock:
            self.webhooks_sent.append(record)


def _make_handler(api: MockAPI):  # noqa: C901 - one dispatch table, read top to bottom
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # keep test output clean
            pass

        # -- plumbing --------------------------------------------------

        def _record(self) -> None:
            with api._lock:
                api.requests.append((self.command, self.path))

        def _drain(self) -> bytes:
            """Read the whole request body. MUST happen on every route.

            With HTTP/1.1 keep-alive, a body left unread stays in the socket
            buffer; the next request on that connection then starts parsing
            mid-body and BaseHTTPRequestHandler answers a spurious 400. That
            is a mock bug that looks exactly like an SDK bug, so drain first
            and branch afterwards.
            """
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _body(self) -> dict:
            try:
                return json.loads(self._raw or b"{}")
            except json.JSONDecodeError:
                return {}

        def _send(self, code: int, payload: object = None, *, raw: bytes | None = None,
                  ctype: str = "application/json") -> None:
            data = raw if raw is not None else json.dumps(payload or {}).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Request-Id", str(uuid.uuid4()))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            if not api.require_auth:
                return True
            return self.headers.get("X-API-Key") == api.api_key

        def _unauthorized(self) -> None:
            self._send(401, {"detail": "invalid api key"})

        # -- POST ------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            self._record()
            self._raw = self._drain()
            path = urlparse(self.path).path

            if path.startswith("/api/v1/") and not self._authed():
                return self._unauthorized()

            if path == "/api/v1/upload":
                return self._create_job()
            if path == "/api/v1/upload/multipart/create":
                if not api.multipart_enabled:
                    # The real server 404s this route when multipart is disabled,
                    # which is the signal the SDK falls back to single-shot on.
                    return self._send(404, {"detail": "multipart disabled"})
                return self._create_multipart()
            if path == "/api/v1/upload/progress":
                job_id = self._body().get("job_id", "")
                with api._lock:
                    api.upload_heartbeats.append(job_id)
                return self._send(200, {"ok": True})
            if path == "/api/v1/upload/complete":
                return self._complete_upload()
            if path == "/api/v1/upload/multipart/complete":
                return self._complete_upload()
            if path == "/api/v1/upload/multipart/abort":
                return self._send(200, {"ok": True})
            if path == "/api/v1/jobs/cancel":
                job = api.jobs.get(self._body().get("job_id", ""))
                if job is None:
                    return self._send(404, {"detail": "job not found"})
                job.status = "cancelled"
                return self._send(200, {"cancelled": True})
            if path == "/api/v1/jobs/check-failed":
                ids = self._body().get("job_ids", [])
                return self._send(200, {"failed_jobs": [
                    api.jobs.get(i) is not None and api.jobs[i].status == "failed" for i in ids
                ]})
            return self._send(404, {"detail": "no such route"})

        def _job_from_body(self, body: dict) -> Job:
            job = Job(
                job_id=f"job_{uuid.uuid4().hex[:12]}",
                output_type=body.get("output_type", "json"),
                callback_url=body.get("callback_url"),
                options=body,
            )
            with api._lock:
                api.jobs[job.job_id] = job
            return job

        def _create_job(self) -> None:
            body = self._body()
            job = self._job_from_body(body)
            # A URL input is fetched server-side, so it is immediately enqueued.
            if body.get("audio_url"):
                self._finish(job)
            self._send(200, {
                "job_id": job.job_id,
                "upload_url": f"{api.base_url}/_upload/{job.job_id}",
                "download_url": f"{api.base_url}/_result/{job.job_id}",
                "content_type": "application/octet-stream",
                "expires_in": 900,
            })

        def _create_multipart(self) -> None:
            body = self._body()
            job = self._job_from_body(body)
            size = int(body.get("file_size") or 0)
            n_parts = max(1, -(-size // api.part_size))
            self._send(200, {
                "job_id": job.job_id,
                "part_size": api.part_size,
                "parts": [
                    {"part_number": i, "url": f"{api.base_url}/_upload/{job.job_id}?part={i}"}
                    for i in range(1, n_parts + 1)
                ],
                "download_url": f"{api.base_url}/_result/{job.job_id}",
            })

        def _complete_upload(self) -> None:
            job = api.jobs.get(self._body().get("job_id", ""))
            if job is None:
                return self._send(404, {"detail": "job not found"})
            self._finish(job)
            self._send(200, {"ok": True})

        def _finish(self, job: Job) -> None:
            """Mark the job terminal and fire its webhook, as the platform does."""
            if api.fail_job_at:
                job.status = "failed"
                job.failed_stage = api.fail_job_at
                job.reason = f"simulated failure at {api.fail_job_at}"
            else:
                job.status = "completed"
            job.completed_at = time.time()
            threading.Thread(target=api.deliver_webhook, args=(job,), daemon=True).start()

        # -- PUT (the presigned upload target) --------------------------

        def do_PUT(self) -> None:  # noqa: N802
            self._record()
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/_upload/"):
                self._drain()
                return self._send(404, {"detail": "no such route"})
            job_id = parsed.path.split("/_upload/", 1)[1]
            data = self._drain()
            job = api.jobs.get(job_id)
            if job is None:
                return self._send(404, {"detail": "job not found"})
            part = parse_qs(parsed.query).get("part", [None])[0]
            if part is not None:
                job.parts[int(part)] = hashlib.md5(data).hexdigest()
                self.send_response(200)
                self.send_header("ETag", f'"{job.parts[int(part)]}"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            job.uploaded += data
            self._send(200, {"ok": True})

        # -- GET -------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            self._record()
            parsed = urlparse(self.path)
            path = parsed.path

            if path.startswith("/_result/"):
                return self._serve_result(path.split("/_result/", 1)[1])

            if path.startswith("/api/v1/") and not self._authed():
                return self._unauthorized()

            if path == "/api/v1/jobs":
                return self._list_jobs(parse_qs(parsed.query))
            if path.endswith("/stream") and path.startswith("/api/v1/jobs/"):
                return self._stream(path[len("/api/v1/jobs/"):-len("/stream")])
            if path.startswith("/api/v1/jobs/"):
                return self._job_status(path[len("/api/v1/jobs/"):])
            return self._send(404, {"detail": "no such route"})

        def _serve_result(self, job_id: str) -> None:
            job = api.jobs.get(job_id)
            if job is None or job.status != "completed":
                return self._send(404, {"detail": "not ready"})
            if api.result_available_after and (
                time.time() - job.completed_at < api.result_available_after
            ):
                return self._send(404, {"detail": "not ready"})
            body, ctype = _render(job.output_type)
            self._send(200, raw=body, ctype=ctype)

        def _job_status(self, job_id: str) -> None:
            job = api.jobs.get(job_id)
            if job is None:
                return self._send(404, {"detail": "job not found"})
            out: dict = {"job_id": job.job_id, "status": job.status}
            if job.status == "completed":
                out["download_url"] = f"{api.base_url}/_result/{job.job_id}"
            if job.status == "failed":
                out["failed_stage"] = job.failed_stage
                out["reason"] = job.reason
            self._send(200, out)

        def _list_jobs(self, query: dict) -> None:
            limit = int(query.get("limit", ["50"])[0])
            before = query.get("before", [None])[0]
            with api._lock:
                ordered = sorted(api.jobs.values(), key=lambda j: j.job_id, reverse=True)
            if before:
                ordered = [j for j in ordered if j.job_id < before]
            page, rest = ordered[:limit], ordered[limit:]
            self._send(200, {
                "jobs": [{"job_id": j.job_id, "status": j.status} for j in page],
                "next_before": page[-1].job_id if page and rest else None,
            })

        def _stream(self, job_id: str) -> None:
            job = api.jobs.get(job_id)
            if job is None:
                return self._send(404, {"detail": "job not found"})
            if api.stream_status != 200:
                return self._send(api.stream_status, {"detail": "stream unavailable"})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            resume = self.headers.get("Last-Event-ID")
            start = int(resume) + 1 if resume and resume.isdigit() else 0

            def emit(idx: int, event: str, data: dict) -> None:
                chunk = f"id: {idx}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()

            sent = 0
            for i in range(start, api.progress_steps):
                emit(i, "progress", {"completed": i + 1, "total": api.progress_steps,
                                     "step": "transcribing"})
                sent += 1
                if api.drop_stream_after is not None and sent >= api.drop_stream_after:
                    # Cut mid-stream, the way a dropped connection looks.
                    self.wfile.flush()
                    self.close_connection = True
                    return
                time.sleep(0.01)

            if job.status == "processing":
                self._finish(job)
            if job.status == "failed":
                emit(api.progress_steps, "failed",
                     {"step": job.failed_stage, "reason": job.reason})
            else:
                emit(api.progress_steps, "completed",
                     {"download_url": f"{api.base_url}/_result/{job.job_id}"})
            self.close_connection = True

    return Handler


def _render(output_type: str) -> tuple[bytes, str]:
    if output_type == "srt":
        return SAMPLE_SRT.encode(), "text/plain"
    if output_type == "vtt":
        return SAMPLE_VTT.encode(), "text/vtt"
    if output_type == "txt":
        return SAMPLE_TXT.encode(), "text/plain"
    if output_type in ("docx", "pdf"):
        return b"\x50\x4b\x03\x04binary-placeholder", "application/octet-stream"
    return json.dumps(SAMPLE_TRANSCRIPT).encode(), "application/json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--api-key", default="test-key")
    ap.add_argument("--webhook-secret", default=None)
    ap.add_argument("--progress-steps", type=int, default=3)
    ap.add_argument("--fail-at", default=None,
                    help="fail every job at this stage, e.g. gpu_timestamps")
    ap.add_argument("--stream-status", type=int, default=200,
                    help="status for GET /jobs/{id}/stream; use 503 to force polling")
    args = ap.parse_args()

    api = MockAPI(api_key=args.api_key, webhook_secret=args.webhook_secret,
                  progress_steps=args.progress_steps)
    api.fail_job_at = args.fail_at
    api.stream_status = args.stream_status
    handler = _make_handler(api)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    api._server = server
    print(f"mock Speech Revolutions API on http://127.0.0.1:{args.port}")
    print(f"  SPEECHREVOLUTIONS_API_KEY={args.api_key}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
