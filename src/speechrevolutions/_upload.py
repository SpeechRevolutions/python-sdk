"""Upload-body helpers that report byte-level progress.

The HTTP client can only observe upload progress if it reads the body through
something we control. For ``requests`` that's a seekable file-like object
(``ProgressReader``) whose ``__len__``/``tell`` let requests set Content-Length.
For ``httpx`` PUTs that's a byte-chunk generator (``iter_with_progress``) paired
with an explicit Content-Length header, which keeps S3 happy (httpx would
otherwise switch to Transfer-Encoding: chunked, which presigned PUTs reject).
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Iterator

from speechrevolutions.models import ProgressCallback, ProgressEvent

# Callback receives (bytes_sent, total_bytes).
ByteProgressFn = Callable[[int, int], None]

CHUNK_SIZE = 64 * 1024


def byte_progress_adapter(on_progress: ProgressCallback | None) -> ByteProgressFn | None:
    """Adapt a :class:`ProgressEvent` callback to a ``(sent, total)`` byte callback.

    Upload events are reported as ``ProgressEvent(step="upload")`` so they share
    the same shape (and ``.percent``) as transcription progress.
    """
    if on_progress is None:
        return None

    def _cb(sent: int, total: int) -> None:
        on_progress(ProgressEvent(completed=sent, total=total, step="upload"))

    return _cb


class ProgressReader:
    """A seekable file-like view over in-memory bytes that reports read progress.

    Works as the ``data=`` body for a ``requests`` PUT (requests reads it in
    chunks and derives Content-Length from ``__len__`` minus ``tell()``) and as
    a multipart file part for both requests and httpx.
    """

    def __init__(self, data: bytes, callback: ByteProgressFn | None = None) -> None:
        self._data = data
        self._total = len(data)
        self._pos = 0
        self._cb = callback

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        if self._cb and chunk:
            self._cb(self._pos, self._total)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._total + offset
        # A fresh reader is created per upload attempt, so a rewind (retry)
        # simply restarts progress from 0 on the next read.
        return self._pos

    def tell(self) -> int:
        return self._pos


def iter_with_progress(
    data: bytes,
    callback: ByteProgressFn | None = None,
) -> Iterator[bytes]:
    """Yield ``data`` in chunks, reporting progress after each — for httpx PUT."""
    total = len(data)
    sent = 0
    for start in range(0, total, CHUNK_SIZE):
        chunk = data[start : start + CHUNK_SIZE]
        sent += len(chunk)
        yield chunk
        if callback:
            callback(sent, total)


async def aiter_with_progress(
    data: bytes,
    callback: ByteProgressFn | None = None,
) -> "AsyncIterator[bytes]":
    """Async variant of :func:`iter_with_progress` for the httpx async PUT."""
    total = len(data)
    sent = 0
    for start in range(0, total, CHUNK_SIZE):
        chunk = data[start : start + CHUNK_SIZE]
        sent += len(chunk)
        yield chunk
        if callback:
            callback(sent, total)
