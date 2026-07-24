"""Async Speech Revolutions STT client (httpx)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import httpx

from speechrevolutions._audio import is_url
from speechrevolutions._config import (
    DEFAULT_BASE_URL,
    POLL_INTERVAL,
    SSE_MAX_RECONNECTS,
    SSE_RECONNECT_DELAY,
    UPLOAD_BASE_DELAY,
    UPLOAD_MAX_ATTEMPTS,
    UPLOAD_PROGRESS_INTERVAL,
    resolve_api_key,
)
from speechrevolutions._progress import resolve_progress as _resolve_progress
from speechrevolutions._upload import ProgressReader, aiter_with_progress
from speechrevolutions._upload import byte_progress_adapter as _byte_progress_adapter
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


class AsyncSTTClient:
    """
    Async client for the Speech Revolutions speech-to-text API.

    Typical usage::

        from speechrevolutions import AsyncSpeechRevolutions

        async with AsyncSpeechRevolutions() as client:
            result = await client.transcribe("meeting.mp3", speaker_labels=True)
            print(result.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0))

    # ============================================================
    # HIGH-LEVEL
    # ============================================================

    async def transcribe(
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

        ``diarize`` is a Deepgram-compatible alias for ``speaker_labels``.
        Pass ``on_progress`` / ``on_upload_progress`` to receive
        :class:`ProgressEvent`s (read ``event.percent``) for the transcription
        and upload phases, and/or ``progress=True`` to render ``tqdm`` bars to
        the console. Returns a :class:`Transcript`; call ``result.save()`` to
        write it to disk.
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
        data, file_size = await self._read_audio(audio)
        job = await self.create_upload_job(file_size, opts)

        upload_cb, upload_printer = _resolve_progress(
            on_upload_progress, progress, label="Uploading", bytes_mode=True
        )
        try:
            await self.upload_audio(
                job.upload_url, data, job_id=job.job_id, on_progress=upload_cb
            )
        finally:
            if upload_printer is not None:
                upload_printer.close()
        await self.complete_upload(job.job_id)

        progress_cb, printer = _resolve_progress(on_progress, progress)
        try:
            content, download_url = await self.wait_for_result(
                job.job_id,
                job.download_url,
                on_progress=progress_cb,
            )
        finally:
            if printer is not None:
                printer.close()
        return parse_transcript(
            job_id=job.job_id,
            content=content,
            output_type=opts.normalized_output_type(),
            download_url=download_url,
        )

    async def transcribe_url(self, url: str, *args: Any, **kwargs: Any) -> Transcript:
        return await self.transcribe(url, *args, **kwargs)

    async def transcribe_file(self, path: str | Path, *args: Any, **kwargs: Any) -> Transcript:
        return await self.transcribe(path, *args, **kwargs)

    async def submit(
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

        Collect the result later via a webhook (``callback_url``) or by polling
        :meth:`get_job_status` / :meth:`get_transcript`. Ideal for batch
        workloads — submit many files concurrently, then gather.
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
        data, file_size = await self._read_audio(audio)
        job = await self.create_upload_job(file_size, opts)
        upload_cb, upload_printer = _resolve_progress(
            on_upload_progress, progress, label="Uploading", bytes_mode=True
        )
        try:
            await self.upload_audio(
                job.upload_url, data, job_id=job.job_id, on_progress=upload_cb
            )
        finally:
            if upload_printer is not None:
                upload_printer.close()
        await self.complete_upload(job.job_id)
        return job.job_id

    # ============================================================
    # UPLOAD FLOW
    # ============================================================

    async def create_upload_job(
        self,
        file_size: int,
        options: TranscribeOptions | None = None,
    ) -> UploadJob:
        options = options or TranscribeOptions()
        data = await self._api_request("POST", "/api/v1/upload", json=options.to_payload(file_size))
        return UploadJob(
            job_id=str(data["job_id"]),
            upload_url=data["upload_url"],
            download_url=data["download_url"],
            content_type=data.get("content_type", "application/octet-stream"),
            expires_in=int(data.get("expires_in", 0)),
        )

    async def touch_upload_progress(self, job_id: str) -> None:
        await self._api_request("POST", "/api/v1/upload/progress", json={"job_id": job_id})

    async def upload_audio(
        self,
        upload_url: str | dict[str, Any],
        data: bytes,
        *,
        job_id: str | None = None,
        filename: str | None = None,
        content_type: str = "application/octet-stream",
        on_progress: ProgressCallback | None = None,
    ) -> None:
        stop = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        if job_id:
            heartbeat_task = asyncio.create_task(self._upload_heartbeat(job_id, stop))

        last_exc: Exception | None = None
        try:
            for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
                try:
                    await self._put_or_post_upload(
                        upload_url, data, filename, content_type, on_progress
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    if attempt < UPLOAD_MAX_ATTEMPTS:
                        await asyncio.sleep(UPLOAD_BASE_DELAY * (2 ** (attempt - 1)))
        finally:
            stop.set()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

        raise UploadError(f"Upload failed after {UPLOAD_MAX_ATTEMPTS} attempts: {last_exc}")

    async def complete_upload(self, job_id: str) -> None:
        await self._api_request("POST", "/api/v1/upload/complete", json={"job_id": job_id})

    async def wait_for_result(
        self,
        job_id: str,
        download_url: str,
        *,
        on_progress: ProgressCallback | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, str]:
        timeout = self.timeout if timeout is None else timeout
        result_url = await self._wait_sse(
            job_id, download_url, on_progress=on_progress, timeout=timeout
        )
        if result_url is None:
            content = await self._wait_poll(job_id, download_url, timeout=timeout)
            return content, download_url
        return await self.download_result(result_url), result_url

    async def download_result(self, download_url: str) -> bytes:
        try:
            # follow_redirects: presigned S3 URLs 307-redirect (requests follows
            # by default; httpx does not). Without this the download raises on 307.
            resp = await self._client.get(download_url, timeout=60.0, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise APIError(f"Download failed: {exc}") from exc
        if resp.status_code != 200:
            raise APIError(
                f"Download failed (HTTP {resp.status_code})",
                status_code=resp.status_code,
                body=resp.text[:300],
            )
        return resp.content

    async def cancel_job(self, job_id: str) -> None:
        await self._api_request("POST", "/api/v1/jobs/cancel", json={"job_id": job_id})

    async def check_failed(self, job_ids: list[str]) -> list[bool]:
        data = await self._api_request(
            "POST",
            "/api/v1/jobs/check-failed",
            json={"job_ids": [str(j) for j in job_ids]},
        )
        return list(data.get("failed_jobs", []))

    # ============================================================
    # RETRIEVAL (get by id / list)
    # ============================================================

    async def get_job_status(self, job_id: str) -> JobStatus:
        """Fetch a job's current status (and a fresh download URL once complete)."""
        data = await self._api_request("GET", f"/api/v1/jobs/{job_id}")
        return JobStatus(
            job_id=str(data.get("job_id", job_id)),
            status=str(data.get("status", "")),
            download_url=data.get("download_url"),
            failed_stage=data.get("failed_stage"),
            reason=data.get("reason"),
        )

    async def get_transcript(self, job_id: str, *, output_type: str = "json") -> Transcript:
        """Fetch and parse a completed job's transcript by id.

        Raises :class:`JobFailedError` if the job failed, or :class:`STTError`
        if it is still processing (poll ``get_job_status`` for that case).
        """
        status = await self.get_job_status(job_id)
        if status.is_failed:
            raise JobFailedError(
                f"Job {job_id} failed", step=status.failed_stage, reason=status.reason
            )
        if not status.is_completed or not status.download_url:
            raise STTError(f"Job {job_id} is not complete (status={status.status})")
        content = await self.download_result(status.download_url)
        return parse_transcript(
            job_id=job_id,
            content=content,
            output_type=output_type,
            download_url=status.download_url,
        )

    async def list_jobs(self, *, limit: int = 50, before: str | None = None) -> dict[str, Any]:
        """List the caller's most-recent jobs (newest first), cursor-paginated."""
        path = f"/api/v1/jobs?limit={int(limit)}"
        if before:
            path += f"&before={quote(str(before))}"
        return await self._api_request("GET", path)

    # ============================================================
    # INTERNALS
    # ============================================================

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        return headers

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = await self._client.request(
                method, url, headers=self._headers(), json=json, timeout=timeout
            )
        except httpx.ConnectError as exc:
            raise APIError(f"Cannot connect to {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Request timed out: {method} {path}") from exc

        self._raise_for_status(resp)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code in (200, 204):
            return
        if resp.status_code == 401:
            raise AuthenticationError("Unauthorized — check your API key")
        if resp.status_code == 404:
            raise JobNotFoundError("Job not found or upload session expired")
        if resp.status_code == 429:
            raise RateLimitError("Rate limit exceeded — try again shortly")
        raise APIError(
            f"Unexpected response (HTTP {resp.status_code})",
            status_code=resp.status_code,
            body=resp.text[:300],
        )

    async def _upload_heartbeat(self, job_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=UPLOAD_PROGRESS_INTERVAL)
                break
            except asyncio.TimeoutError:
                try:
                    await self.touch_upload_progress(job_id)
                except Exception:
                    pass

    async def _put_or_post_upload(
        self,
        upload_url: str | dict[str, Any],
        data: bytes,
        filename: str | None,
        content_type: str,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        byte_cb = _byte_progress_adapter(on_progress)
        if isinstance(upload_url, dict):
            # A fresh reader per call so retries restart progress from 0.
            body: Any = ProgressReader(data, byte_cb) if byte_cb else data
            files = {"file": (filename or "audio", body, content_type)}
            resp = await self._client.post(
                upload_url["url"],
                data=upload_url.get("fields", {}),
                files=files,
                timeout=300.0,
            )
        elif byte_cb:
            # Stream in chunks for progress; explicit Content-Length keeps
            # httpx from switching to Transfer-Encoding: chunked (S3 rejects it).
            resp = await self._client.put(
                upload_url,
                content=aiter_with_progress(data, byte_cb),
                headers={"Content-Type": content_type, "Content-Length": str(len(data))},
                timeout=300.0,
            )
        else:
            resp = await self._client.put(
                upload_url,
                content=data,
                headers={"Content-Type": content_type},
                timeout=300.0,
            )
        if resp.status_code not in (200, 204):
            raise UploadError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    async def _wait_sse(
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
                await asyncio.sleep(SSE_RECONNECT_DELAY)

            outcome, download_url, last_event_id = await self._sse_attempt(
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

    async def _sse_attempt(
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
            async with self._client.stream(
                "GET",
                stream_url,
                headers=headers,
                timeout=httpx.Timeout(10.0, read=max(1.0, timeout - elapsed)),
            ) as resp:
                if resp.status_code == 401:
                    raise AuthenticationError("Unauthorized — check your API key")
                if resp.status_code == 429:
                    raise RateLimitError("Rate limit exceeded on SSE endpoint")
                if resp.status_code != 200:
                    return "reconnect", None, last_event_id

                async for sse in _aiter_sse(resp):
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
        except httpx.TimeoutException:
            return "timeout", None, last_event_id
        except httpx.HTTPError:
            return "reconnect", None, last_event_id

    async def _wait_poll(self, job_id: str, download_url: str, *, timeout: float) -> bytes:
        start = time.monotonic()
        max_attempts = max(1, int(timeout / POLL_INTERVAL))
        for attempt in range(1, max_attempts + 1):
            if time.monotonic() - start >= timeout:
                break
            try:
                failed = await self.check_failed([job_id])
                if failed and failed[0]:
                    raise JobFailedError(f"Job {job_id} has failed")
            except (AuthenticationError, JobFailedError):
                raise
            except STTError:
                pass

            try:
                resp = await self._client.get(download_url, timeout=15.0, follow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
            except httpx.HTTPError:
                pass

            if attempt < max_attempts:
                await asyncio.sleep(POLL_INTERVAL)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    async def _read_audio(self, audio: str | Path | bytes | BinaryIO) -> tuple[bytes, int]:
        if isinstance(audio, (str, Path)):
            value = str(audio)
            if is_url(value):
                try:
                    resp = await self._client.get(value, timeout=120.0, follow_redirects=True)
                except httpx.HTTPError as exc:
                    raise APIError(f"Failed to download audio URL: {exc}") from exc
                if resp.status_code != 200:
                    raise APIError(
                        f"Failed to download audio URL (HTTP {resp.status_code})",
                        status_code=resp.status_code,
                        body=resp.text[:300],
                    )
                data = resp.content
            else:
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                data = await asyncio.to_thread(path.read_bytes)
        elif isinstance(audio, bytes):
            data = audio
        elif hasattr(audio, "read"):
            raw = audio.read()
            if asyncio.iscoroutine(raw):
                raw = await raw
            data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)!r}")

        if not data:
            raise ValueError("Audio is empty")
        return data, len(data)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncSTTClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


AsyncSpeechRevolutions = AsyncSTTClient
AsyncSpeechRevolutionsClient = AsyncSTTClient


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def _aiter_sse(resp: httpx.Response):
    """Yield parsed SSE event dicts from an httpx streaming response."""
    event: dict[str, Any] = {}
    async for raw_line in resp.aiter_lines():
        if raw_line == "":
            if event.get("data") is not None:
                yield event
            event = {}
            continue
        if raw_line.startswith(":"):
            continue
        if ":" in raw_line:
            field, _, value = raw_line.partition(":")
            value = value.lstrip(" ")
        else:
            field, value = raw_line, ""
        if field == "id":
            event["id"] = value
        elif field == "event":
            event["event"] = value
        elif field == "data":
            event["data"] = value
