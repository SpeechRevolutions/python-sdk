"""Audio input helpers: local path, bytes, file objects, or remote URL."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import requests

from speechrevolutions.exceptions import APIError


def is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def read_audio(
    audio: str | Path | bytes | BinaryIO,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> tuple[bytes, int]:
    """
    Normalize audio input to bytes.

    Accepts a local filesystem path, http(s) URL, raw bytes, or binary file object.
    """
    if isinstance(audio, (str, Path)):
        value = str(audio)
        if is_url(value):
            sess = session or requests.Session()
            try:
                resp = sess.get(value, timeout=timeout)
            except requests.exceptions.RequestException as exc:
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
            data = path.read_bytes()
    elif isinstance(audio, bytes):
        data = audio
    elif hasattr(audio, "read"):
        raw = audio.read()
        data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)!r}")

    if not data:
        raise ValueError("Audio is empty")
    return data, len(data)
