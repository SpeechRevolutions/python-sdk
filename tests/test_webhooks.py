"""Webhook delivery, end to end, against the real contract.

Nothing here had ever been tested — not in the SDK, not in the pipeline, not in
the cookbook. These drive a real receiver over a real socket and assert the
things a customer's integration actually depends on:

  * the SDK plumbs `callback_url` through to job creation
  * the payload is `{"job_id", "status"}`, plus `step`/`reason` on failure
  * the headers are `X-SR-Event`, `X-SR-Delivery`, `X-SR-Signature`
  * the signature is `sha256=` + HMAC over the RAW BYTES, compact-separated

That last point is the subtle one. The sender serialises with
`separators=(",", ":")`; a receiver that parses the body and re-serialises it
before computing the HMAC gets different bytes and rejects every delivery.
`test_reserializing_the_body_breaks_the_signature` pins that, so the cookbook's
"compare against the raw bytes received" comment is enforced rather than hoped
for.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mock_api import MockAPI
from speechrevolutions import SpeechRevolutions

SECRET = "whsec_test_secret"


class _CI(dict):
    """Case-insensitive header view, the way an HTTP framework exposes one."""

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self._lower = {k.lower(): v for k, v in data.items()}

    def __getitem__(self, key: str) -> str:
        try:
            return self._lower[key.lower()]
        except KeyError:
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._lower

    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        return self._lower.get(key.lower(), default)


class Receiver:
    """A minimal webhook endpoint that records exactly what arrived."""

    def __init__(self, *, status: int = 200) -> None:
        self.received: list[dict] = []
        self.status = status
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> Receiver:
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: object) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                # Header names are case-INSENSITIVE on the wire and clients do not
                # preserve the casing you wrote: urllib sends "X-Sr-Signature" for
                # "X-SR-Signature". Real frameworks (Starlette, Express) look up
                # case-insensitively, so mirror that rather than pinning a casing
                # no spec guarantees.
                outer.received.append({
                    "raw": raw,
                    "path": self.path,
                    "headers": _CI({k: v for k, v in self.headers.items()}),
                })
                body = b'{"ok":true}'
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @property
    def url(self) -> str:
        assert self._server
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/webhooks/speechrevolutions"

    def wait(self, n: int = 1, timeout: float = 5.0) -> list[dict]:
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.received) >= n:
                return self.received
            time.sleep(0.02)
        raise AssertionError(f"expected {n} webhook(s), got {len(self.received)}")


@pytest.fixture
def receiver():
    with Receiver() as r:
        yield r


# ---------------------------------------------------------------------------
# The SDK plumbs callback_url through
# ---------------------------------------------------------------------------

def test_submit_sends_callback_url_to_the_server(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        assert api.only_job().callback_url == receiver.url


def test_transcribe_also_accepts_callback_url(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.transcribe(b"x" * 64, callback_url=receiver.url)
        assert api.only_job().callback_url == receiver.url


def test_no_callback_url_means_no_delivery():
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3")
        assert api.webhooks_sent == []


# ---------------------------------------------------------------------------
# Delivery and payload shape
# ---------------------------------------------------------------------------

def test_completion_webhook_is_delivered(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            job_id = c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        got = receiver.wait(1)[0]
        event = json.loads(got["raw"])
        # Verified live: a completion carries the presigned download_url, the
        # billed duration and the achieved RTF alongside the status. Assert the
        # contract, not exact equality — extra fields must not break a receiver.
        assert event["job_id"] == job_id
        assert event["status"] == "completed"
        assert event["download_url"].startswith("http")
        assert event["duration_seconds"] > 0
        assert event["rtf"] > 0


def test_a_completion_webhook_carries_everything_needed_to_fetch_the_result(receiver):
    """The payload is self-sufficient: no second API call is required."""
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
            event = json.loads(receiver.wait(1)[0]["raw"])
            # Download straight from the URL in the webhook.
            content = c.download_result(event["download_url"])
    assert b"Good" in content


def test_failure_webhook_carries_step_and_reason(receiver):
    with MockAPI(progress_steps=1) as api:
        api.fail_job_at = "gpu_timestamps"
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            job_id = c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        event = json.loads(receiver.wait(1)[0]["raw"])
        assert event["job_id"] == job_id
        assert event["status"] == "failed"
        assert event["step"] == "gpu_timestamps"
        assert "simulated failure" in event["reason"]


def test_webhook_posts_to_the_exact_path_given(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        assert receiver.wait(1)[0]["path"] == "/webhooks/speechrevolutions"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def test_delivery_headers_are_present(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        h = receiver.wait(1)[0]["headers"]
        assert h["X-SR-Event"] == "completed"
        assert h["X-SR-Delivery"]
        assert h["Content-Type"] == "application/json"
        assert h["User-Agent"].startswith("SpeechRevolutions-Webhook/")


def test_x_sr_event_reflects_failure(receiver):
    with MockAPI(progress_steps=1) as api:
        api.fail_job_at = "preprocess"
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        assert receiver.wait(1)[0]["headers"]["X-SR-Event"] == "failed"


def test_delivery_id_is_unique_per_delivery(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
            c.submit("https://example.com/b.mp3", callback_url=receiver.url)
        got = receiver.wait(2)
        ids = {g["headers"]["X-SR-Delivery"] for g in got}
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# Signature — the part integrations get wrong
# ---------------------------------------------------------------------------

def _expected(raw: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()


def test_signature_header_verifies_against_the_raw_body(receiver):
    with MockAPI(progress_steps=1, webhook_secret=SECRET) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        got = receiver.wait(1)[0]
        assert hmac.compare_digest(got["headers"]["X-SR-Signature"], _expected(got["raw"]))


def test_no_secret_configured_means_no_signature_header(receiver):
    with MockAPI(progress_steps=1, webhook_secret=None) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        assert "X-SR-Signature" not in receiver.wait(1)[0]["headers"]


def test_reserializing_the_body_breaks_the_signature(receiver):
    """The failure mode every webhook integration hits at least once.

    The sender uses compact separators. `json.dumps(json.loads(raw))` produces
    the same OBJECT with different BYTES, so the HMAC no longer matches. This is
    why the cookbook recipe verifies against the raw bytes received.
    """
    with MockAPI(progress_steps=1, webhook_secret=SECRET) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        got = receiver.wait(1)[0]
        raw = got["raw"]
        reserialized = json.dumps(json.loads(raw)).encode()   # default separators

        assert reserialized != raw, "expected compact separators on the wire"
        assert hmac.compare_digest(got["headers"]["X-SR-Signature"], _expected(raw))
        assert not hmac.compare_digest(
            got["headers"]["X-SR-Signature"], _expected(reserialized)
        )


def test_a_tampered_body_fails_verification(receiver):
    with MockAPI(progress_steps=1, webhook_secret=SECRET) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=receiver.url)
        got = receiver.wait(1)[0]
        tampered = got["raw"].replace(b"completed", b"failed___")
        assert not hmac.compare_digest(got["headers"]["X-SR-Signature"], _expected(tampered))


# ---------------------------------------------------------------------------
# Receiver behaviour
# ---------------------------------------------------------------------------

def test_a_receiver_returning_4xx_still_counts_as_delivered():
    # The platform retries 5xx, not 4xx — a 401 from a bad-signature check must
    # not wedge the sender into a retry loop.
    with Receiver(status=401) as rcv, MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            c.submit("https://example.com/a.mp3", callback_url=rcv.url)
        rcv.wait(1)
        assert len(api.wait_webhooks(1)) == 1


def test_webhook_fires_for_the_upload_path_too(receiver):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            job_id = c.submit(b"x" * 128, callback_url=receiver.url)
        event = json.loads(receiver.wait(1)[0]["raw"])
        assert event["job_id"] == job_id
