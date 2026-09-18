"""Synchronous Speech Revolutions STT client."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import requests

from speechrevolutions._audio import is_url, read_audio
from speechrevolutions._config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    POLL_INTERVAL,
    RETRY_BACKOFF_MAX,
    RETRY_STATUS_CODES,
    SSE_MAX_RECONNECTS,
    SSE_RECONNECT_DELAY,
    UPLOAD_BASE_DELAY,
    UPLOAD_MAX_ATTEMPTS,
    UPLOAD_PROGRESS_INTERVAL,
    creates_job as _creates_job,
    extract_request_id,
    parse_retry_after,
    resolve_api_key,
)
from speechrevolutions._progress import resolve_progress as _resolve_progress
from speechrevolutions._upload import ProgressReader
from speechrevolutions._upload import byte_progress_adapter as _byte_progress_adapter
from speechrevolutions._sse import parse_sse_stream
from speechrevolutions.exceptions import (
    APIError,
    AuthenticationError,
    JobFailedError,
    JobNotFoundError,
    RateLimitError,
    STTError,
    TimeoutError,
    UploadError,
)
from speechrevolutions.models import (
    JobStatus,
    OutputType,
    ProcessingTier,
    ProgressCallback,
    ProgressEvent,
    TranscribeOptions,
    UploadJob,
    resolve_options,
)
from speechrevolutions.transcript import Transcript, parse_transcript


logger = logging.getLogger("speechrevolutions")


class _MultipartUnavailable(Exception):
    """Internal signal that a multipart upload should fall back to single-shot."""


class STTClient:
    """
    Synchronous client for the Speech Revolutions speech-to-text API.

    Typical usage (mirrors AssemblyAI / ElevenLabs simplicity)::

        from speechrevolutions import SpeechRevolutions

        client = SpeechRevolutions()  # reads SPEECHREVOLUTIONS_API_KEY or STT_API_KEY
        result = client.transcribe("meeting.mp3", speaker_labels=True)
        print(result.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
        session: requests.Session | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        proxies: dict[str, str] | None = None,
        multipart: bool = True,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        if proxies:
            self._session.proxies.update(proxies)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Prefer multipart uploads (falls back to a single presigned PUT if the
        # server has multipart disabled or a multipart upload fails mid-flight).
        self._multipart = multipart

    # ============================================================
    # HIGH-LEVEL
    # ============================================================

    def transcribe(
        self,
        audio: str | Path | bytes | BinaryIO,
        options: TranscribeOptions | None = None,
        *,
        output_type: OutputType | str | None = None,
        word_timestamps: bool | None = None,
        speaker_labels: bool | None = None,
        diarize: bool | None = None,
        nltk: bool | None = None,
        tier: ProcessingTier | str | None = None,
        custom_vocabulary: list[str] | None = None,
        callback_url: str | None = None,
        on_progress: ProgressCallback | None = None,
        on_upload_progress: ProgressCallback | None = None,
        progress: bool = False,
    ) -> Transcript:
        """
        Transcribe a local file path, remote URL, bytes, or file object.

        Options may be passed via ``options=TranscribeOptions(...)`` and/or
        keyword arguments (kwargs win). ``diarize`` is a Deepgram-compatible
        alias for ``speaker_labels``. ``tier`` is ``standard`` (default) or
        ``economy``.

        Live progress is a Speech Revolutions extra — neither AssemblyAI nor
        Deepgram exposes it for pre-recorded audio. ``on_progress`` fires with
        a :class:`ProgressEvent` for each transcription update (read
        ``event.percent``); ``on_upload_progress`` is the same for byte-level
        upload progress (``event.step == "upload"``). Pass ``progress=True``
        to also render ``tqdm`` console bars (plain-text if tqdm isn't
        installed).

        Returns a :class:`Transcript`; call ``result.save(path)`` to write it
        to disk (this client never writes files on its own).
        """
        opts = resolve_options(
            options,
            output_type=output_type,
            word_timestamps=word_timestamps,
            speaker_labels=speaker_labels,
            diarize=diarize,
            nltk=nltk,
            tier=tier,
            custom_vocabulary=custom_vocabulary,
            callback_url=callback_url,
        )
        job_id, download_url = self._ingest_audio(
            audio, opts, on_upload_progress=on_upload_progress, progress=progress
        )

        progress_cb, printer = _resolve_progress(on_progress, progress)
        try:
            content, result_url = self.wait_for_result(
                job_id,
                download_url,
                on_progress=progress_cb,
            )
        finally:
            if printer is not None:
                printer.close()
        return parse_transcript(
            job_id=job_id,
            content=content,
            output_type=opts.normalized_output_type(),
            download_url=result_url,
        )

    def transcribe_url(self, url: str, *args: Any, **kwargs: Any) -> Transcript:
        """Alias for ``transcribe`` with a remote audio URL (Deepgram-style)."""
        return self.transcribe(url, *args, **kwargs)

    def transcribe_file(self, path: str | Path, *args: Any, **kwargs: Any) -> Transcript:
        """Alias for ``transcribe`` with a local file path (Deepgram-style)."""
        return self.transcribe(path, *args, **kwargs)

    def submit(
        self,
        audio: str | Path | bytes | BinaryIO,
        options: TranscribeOptions | None = None,
        *,
        output_type: OutputType | str | None = None,
        word_timestamps: bool | None = None,
        speaker_labels: bool | None = None,
        diarize: bool | None = None,
        nltk: bool | None = None,
        tier: ProcessingTier | str | None = None,
        custom_vocabulary: list[str] | None = None,
        callback_url: str | None = None,
        on_upload_progress: ProgressCallback | None = None,
        progress: bool = False,
    ) -> str:
        """Upload and enqueue a job, returning its ``job_id`` WITHOUT waiting.

        Unlike :meth:`transcribe` (which blocks until the transcript is ready),
        this returns as soon as the audio is enqueued. Collect the result later
        via a webhook (``callback_url``) or by polling :meth:`get_job_status` /
        :meth:`get_transcript`. Ideal for batch workloads — submit many files,
        then gather — since it holds no long-lived connection per job.
        """
        opts = resolve_options(
            options,
            output_type=output_type,
            word_timestamps=word_timestamps,
            speaker_labels=speaker_labels,
            diarize=diarize,
            nltk=nltk,
            tier=tier,
            custom_vocabulary=custom_vocabulary,
            callback_url=callback_url,
        )
        job_id, _ = self._ingest_audio(
            audio, opts, on_upload_progress=on_upload_progress, progress=progress
        )
        return job_id

    # ============================================================
    # INGESTION (URL fetch / multipart / single-shot)
    # ============================================================

    def _ingest_audio(
        self,
        audio: str | Path | bytes | BinaryIO,
        opts: TranscribeOptions,
        *,
        on_upload_progress: ProgressCallback | None = None,
        progress: bool = False,
    ) -> tuple[str, str]:
        """Get the audio into the platform and return ``(job_id, download_url)``.

        A URL is handed to the server to fetch (no client upload). Local/bytes
        audio is uploaded — multipart when available, otherwise a single PUT.
        """
        # URL inputs are fetched server-side; nothing is uploaded from here.
        if isinstance(audio, (str, Path)) and is_url(str(audio)):
            data = self._api_request(
                "POST", "/api/v1/upload", json=opts.to_payload(audio_url=str(audio))
            )
            return str(data["job_id"]), data["download_url"]

        content, file_size = read_audio(audio, session=self._session)

        upload_cb, upload_printer = _resolve_progress(
            on_upload_progress, progress, label="Uploading", bytes_mode=True
        )
        try:
            if self._multipart:
                try:
                    return self._upload_multipart(content, file_size, opts, on_progress=upload_cb)
                except _MultipartUnavailable:
                    logger.debug("multipart unavailable; falling back to single-shot upload")
            return self._upload_single_shot(content, file_size, opts, on_progress=upload_cb)
        finally:
            if upload_printer is not None:
                upload_printer.close()

    def _upload_single_shot(
        self,
        data: bytes,
        file_size: int,
        opts: TranscribeOptions,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, str]:
        """Classic flow: create job -> PUT the whole file -> complete."""
        job = self.create_upload_job(file_size, opts)
        self.upload_audio(job.upload_url, data, job_id=job.job_id, on_progress=on_progress)
        self.complete_upload(job.job_id)
        return job.job_id, job.download_url

    def _upload_multipart(
        self,
        data: bytes,
        file_size: int,
        opts: TranscribeOptions,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, str]:
        """S3 multipart flow: create -> PUT each part -> complete. Raises
        :class:`_MultipartUnavailable` if the server has multipart disabled (404)
        or a mid-flight failure means we should retry via the single-shot path."""
        try:
            created = self._api_request(
                "POST", "/api/v1/upload/multipart/create", json=opts.to_payload(file_size)
            )
        except JobNotFoundError as e:  # route returns 404 when multipart is disabled
            raise _MultipartUnavailable() from e

        job_id = str(created["job_id"])
        part_size = int(created["part_size"])
        byte_cb = _byte_progress_adapter(on_progress)

        completed_parts: list[dict[str, Any]] = []
        uploaded = 0
        try:
            for part in created["parts"]:
                number = int(part["part_number"])
                start = (number - 1) * part_size
                chunk = data[start : start + part_size]
                etag = self._put_part(part["url"], chunk)
                completed_parts.append({"part_number": number, "etag": etag})
                uploaded += len(chunk)
                if byte_cb is not None:
                    byte_cb(uploaded, file_size)
            self._api_request(
                "POST",
                "/api/v1/upload/multipart/complete",
                json={"job_id": job_id, "parts": completed_parts},
            )
        except Exception as e:
            # Roll back the partial upload, then fall back to a single-shot PUT.
            try:
                self._api_request(
                    "POST", "/api/v1/upload/multipart/abort", json={"job_id": job_id}
                )
            except Exception:
                pass
            raise _MultipartUnavailable() from e

        return job_id, created["download_url"]

    def _put_part(self, url: str, chunk: bytes) -> str:
        """PUT one part to its presigned URL and return the S3 ETag."""
        resp = self._session.put(url, data=chunk, timeout=300)
        if resp.status_code not in (200, 204):
            raise UploadError(f"part upload failed (HTTP {resp.status_code})")
        etag = resp.headers.get("ETag") or resp.headers.get("etag")
        if not etag:
            raise UploadError("part upload response missing ETag header")
        return etag

    # ============================================================
    # UPLOAD FLOW
    # ============================================================

    def create_upload_job(
        self,
        file_size: int,
        options: TranscribeOptions | None = None,
    ) -> UploadJob:
        options = options or TranscribeOptions()
        data = self._api_request("POST", "/api/v1/upload", json=options.to_payload(file_size))
        return UploadJob(
            job_id=str(data["job_id"]),
            upload_url=data["upload_url"],
            download_url=data["download_url"],
            content_type=data.get("content_type", "application/octet-stream"),
            expires_in=int(data.get("expires_in", 0)),
        )

    def touch_upload_progress(self, job_id: str) -> None:
        self._api_request("POST", "/api/v1/upload/progress", json={"job_id": job_id})

    def upload_audio(
        self,
        upload_url: str,
        data: bytes,
        *,
        job_id: str | None = None,
        content_type: str = "application/octet-stream",
        on_progress: ProgressCallback | None = None,
    ) -> None:
        stop_event = threading.Event()
        heartbeat: threading.Thread | None = None
        if job_id:
            heartbeat = threading.Thread(
                target=self._upload_progress_heartbeat,
                args=(job_id, stop_event),
                daemon=True,
            )
            heartbeat.start()

        last_exc: Exception | None = None
        try:
            for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
                try:
                    self._put_upload(upload_url, data, content_type, on_progress)
                    return
                except Exception as exc:
                    last_exc = exc
                    if attempt < UPLOAD_MAX_ATTEMPTS:
                        time.sleep(UPLOAD_BASE_DELAY * (2 ** (attempt - 1)))
        finally:
            stop_event.set()
            if heartbeat is not None:
                heartbeat.join(timeout=5)

        raise UploadError(f"Upload failed after {UPLOAD_MAX_ATTEMPTS} attempts: {last_exc}")

    def complete_upload(self, job_id: str) -> None:
        self._api_request("POST", "/api/v1/upload/complete", json={"job_id": job_id})

    # ============================================================
    # PROGRESS / RESULT
    # ============================================================

    def wait_for_result(
        self,
        job_id: str,
        download_url: str,
        *,
        on_progress: ProgressCallback | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, str]:
        timeout = self.timeout if timeout is None else timeout
        result_url = self._wait_sse(job_id, download_url, on_progress=on_progress, timeout=timeout)
        if result_url is None:
            return self._wait_poll(job_id, download_url, timeout=timeout), download_url
        return self.download_result(result_url), result_url

    def download_result(self, download_url: str) -> bytes:
        try:
            resp = self._session.get(download_url, timeout=60)
        except requests.exceptions.RequestException as exc:
            raise APIError(f"Download failed: {exc}") from exc
        if resp.status_code != 200:
            raise APIError(
                f"Download failed (HTTP {resp.status_code})",
                status_code=resp.status_code,
                body=resp.text[:300],
            )
        return resp.content

    def cancel_job(self, job_id: str) -> None:
        self._api_request("POST", "/api/v1/jobs/cancel", json={"job_id": job_id})

    def check_failed(self, job_ids: list[str]) -> list[bool]:
        data = self._api_request(
            "POST",
            "/api/v1/jobs/check-failed",
            json={"job_ids": [str(j) for j in job_ids]},
        )
        return list(data.get("failed_jobs", []))

    # ============================================================
    # RETRIEVAL (get by id / list)
    # ============================================================

    def get_job_status(self, job_id: str) -> JobStatus:
        """Fetch a job's current status (and a fresh download URL once complete)."""
        data = self._api_request("GET", f"/api/v1/jobs/{job_id}")
        return JobStatus(
            job_id=str(data.get("job_id", job_id)),
            status=str(data.get("status", "")),
            download_url=data.get("download_url"),
            failed_stage=data.get("failed_stage"),
            reason=data.get("reason"),
        )

    def get_transcript(self, job_id: str, *, output_type: str = "json") -> Transcript:
        """Fetch and parse a completed job's transcript by id.

        Raises :class:`JobFailedError` if the job failed, or :class:`STTError`
        if it is still processing (poll ``get_job_status`` for that case).
        """
        status = self.get_job_status(job_id)
        if status.is_failed:
            raise JobFailedError(
                f"Job {job_id} failed", step=status.failed_stage, reason=status.reason
            )
        if not status.is_completed or not status.download_url:
            raise STTError(f"Job {job_id} is not complete (status={status.status})")
        content = self.download_result(status.download_url)
        return parse_transcript(
            job_id=job_id,
            content=content,
            output_type=output_type,
            download_url=status.download_url,
        )

    def list_jobs(self, *, limit: int = 50, before: str | None = None) -> dict[str, Any]:
        """List the caller's most-recent jobs (newest first), cursor-paginated.

        Pass the returned ``next_before`` as ``before`` to fetch the next page.
        """
        path = f"/api/v1/jobs?limit={int(limit)}"
        if before:
            path += f"&before={quote(str(before))}"
        return self._api_request("GET", path)

    # ============================================================
    # INTERNALS
    # ============================================================

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        return headers

    def _api_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        url = f"{self.base_url}{path}"
        # Job-creating calls retry only when the request provably never landed;
        # anything else would risk a duplicate job and a duplicate charge.
        creating = _creates_job(path)
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._session.request(
                    method, url, headers=self._headers(), json=json, timeout=timeout
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                # A connect timeout means no connection was ever established, so
                # the request cannot have been processed — safe to retry even for
                # a create. A read timeout or a mid-flight reset is ambiguous.
                safe = not creating or isinstance(exc, requests.exceptions.ConnectTimeout)
                if safe and attempt <= self.max_retries:
                    time.sleep(self._retry_delay(attempt, None))
                    continue
                if isinstance(exc, requests.exceptions.Timeout):
                    raise TimeoutError(f"Request timed out: {method} {path}") from exc
                raise APIError(f"Cannot connect to {self.base_url}") from exc

            # Retry throttling / transient server errors, honoring Retry-After.
            # For a create, only 429 is safe: the server refused it outright, so
            # no job exists. A 5xx may well have created one before failing.
            if resp.status_code in RETRY_STATUS_CODES and attempt <= self.max_retries:
                if not creating or resp.status_code == 429:
                    retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                    time.sleep(self._retry_delay(attempt, retry_after))
                    continue

            self._raise_for_status(resp)
            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                return {"raw": resp.text}

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, RETRY_BACKOFF_MAX)
        return min(self.retry_backoff * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code in (200, 204):
            return
        request_id = extract_request_id(resp.headers)
        if resp.status_code == 401:
            raise AuthenticationError(
                "Unauthorized — check your API key", status_code=401, request_id=request_id
            )
        if resp.status_code == 404:
            raise JobNotFoundError(
                "Job not found or upload session expired",
                status_code=404,
                request_id=request_id,
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded — try again shortly",
                status_code=429,
                request_id=request_id,
                retry_after=parse_retry_after(resp.headers.get("Retry-After")),
            )
        raise APIError(
            f"Unexpected response (HTTP {resp.status_code})",
            status_code=resp.status_code,
            request_id=request_id,
            body=resp.text[:300],
        )

    def _upload_progress_heartbeat(self, job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(UPLOAD_PROGRESS_INTERVAL):
            try:
                self.touch_upload_progress(job_id)
            except Exception:
                pass

    def _put_upload(
        self,
        upload_url: str,
        data: bytes,
        content_type: str,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        byte_cb = _byte_progress_adapter(on_progress)
        # A fresh reader per call so retries restart progress from 0.
        body: Any = ProgressReader(data, byte_cb) if byte_cb else data
        resp = self._session.put(
            upload_url,
            data=body,
            headers={"Content-Type": content_type},
            timeout=300,
        )
        if resp.status_code not in (200, 204):
            raise UploadError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    def _wait_sse(
        self,
        job_id: str,
        fallback_download_url: str,
        *,
        on_progress: ProgressCallback | None,
        timeout: float,
    ) -> str | None:
        start = time.monotonic()
        last_event_id: str | None = None
        reconnects = 0

        while True:
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"Timed out after {timeout}s waiting for job {job_id}")
            if reconnects > SSE_MAX_RECONNECTS:
                return None
            if reconnects > 0:
                time.sleep(SSE_RECONNECT_DELAY)

            outcome, download_url, last_event_id = self._sse_attempt(
                job_id,
                start,
                timeout,
                last_event_id,
                on_progress=on_progress,
                fallback_download_url=fallback_download_url,
            )

            if outcome == "done":
                return download_url or fallback_download_url
            if outcome == "failed":
                raise JobFailedError("Job failed")
            if outcome == "timeout":
                raise TimeoutError(f"Timed out after {timeout}s waiting for job {job_id}")
            reconnects += 1

    def _sse_attempt(
        self,
        job_id: str,
        start: float,
        timeout: float,
        last_event_id: str | None,
        *,
        on_progress: ProgressCallback | None = None,
        fallback_download_url: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            return "timeout", None, last_event_id

        headers = self._headers(accept="text/event-stream")
        headers.pop("Content-Type", None)
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id

        stream_url = f"{self.base_url}/api/v1/jobs/{job_id}/stream"

        try:
            resp = self._session.get(
                stream_url,
                headers=headers,
                stream=True,
                timeout=(10, max(1.0, timeout - elapsed)),
            )
        except requests.exceptions.Timeout:
            return "timeout", None, last_event_id
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
            return "reconnect", None, last_event_id

        if resp.status_code == 401:
            raise AuthenticationError("Unauthorized — check your API key")
        if resp.status_code == 429:
            raise RateLimitError("Rate limit exceeded on SSE endpoint")
        if resp.status_code != 200:
            return "reconnect", None, last_event_id

        try:
            for sse in parse_sse_stream(resp.iter_lines(decode_unicode=True)):
                if "id" in sse:
                    last_event_id = sse["id"]
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return "timeout", None, last_event_id

                event_type = sse.get("event", "message")
                raw_data = sse.get("data", "{}")
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = {"raw": raw_data}

                if event_type == "progress":
                    # Fire the callback inline so progress is live, not
                    # replayed in a burst after the stream closes.
                    if on_progress:
                        on_progress(
                            ProgressEvent(
                                completed=_as_int(data.get("completed")),
                                total=_as_int(data.get("total")),
                                step=data.get("step"),
                                elapsed_seconds=elapsed,
                                raw=data,
                            )
                        )
                elif event_type == "completed":
                    dl = data.get("download_url") or fallback_download_url
                    return "done", dl, last_event_id
                elif event_type == "failed":
                    step = data.get("step", "unknown")
                    reason = data.get("reason", "unknown")
                    raise JobFailedError(
                        f"Job failed at step={step}: {reason}",
                        step=step,
                        reason=reason,
                    )
            return "reconnect", None, last_event_id
        except JobFailedError:
            raise
        except requests.exceptions.Timeout:
            return "timeout", None, last_event_id
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
            return "reconnect", None, last_event_id
        finally:
            resp.close()

    def _wait_poll(self, job_id: str, download_url: str, *, timeout: float) -> bytes:
        start = time.monotonic()
        max_attempts = max(1, int(timeout / POLL_INTERVAL))
        for attempt in range(1, max_attempts + 1):
            if time.monotonic() - start >= timeout:
                break
            try:
                failed = self.check_failed([job_id])
                if failed and failed[0]:
                    raise JobFailedError(f"Job {job_id} has failed")
            except (AuthenticationError, JobFailedError):
                raise
            except STTError:
                pass

            try:
                resp = self._session.get(download_url, timeout=15)
                if resp.status_code == 200:
                    return resp.content
            except requests.exceptions.RequestException:
                pass

            if attempt < max_attempts:
                time.sleep(POLL_INTERVAL)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> STTClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# Friendly aliases (match DeepgramClient / ElevenLabs naming)
SpeechRevolutions = STTClient
SpeechRevolutionsClient = STTClient


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
