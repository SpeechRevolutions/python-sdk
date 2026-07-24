"""Shared SDK constants and API-key resolution."""

from __future__ import annotations

import os

from speechrevolutions.exceptions import AuthenticationError

DEFAULT_BASE_URL = "https://api.speechrevolutions.com"
ENV_API_KEY_NAMES = ("SPEECHREVOLUTIONS_API_KEY", "STT_API_KEY")

UPLOAD_PROGRESS_INTERVAL = 10
UPLOAD_MAX_ATTEMPTS = 4
UPLOAD_BASE_DELAY = 1.0

SSE_MAX_RECONNECTS = 10
SSE_RECONNECT_DELAY = 3.0

POLL_INTERVAL = 5.0

# Transient-failure retry policy for JSON API requests (not uploads/SSE, which
# have their own retry loops). Overridable per-client via STTClient(...).
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 0.5  # seconds; exponential (0.5, 1.0, 2.0, …), capped
RETRY_BACKOFF_MAX = 30.0
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Response headers checked (case-insensitively) for a correlation id.
REQUEST_ID_HEADERS = ("x-request-id", "x-amzn-requestid", "cf-ray")


def extract_request_id(headers: object) -> str | None:
    """Return the first present request-id header value, or None."""
    get = getattr(headers, "get", None)
    if get is None:
        return None
    for name in REQUEST_ID_HEADERS:
        value = get(name)
        if value:
            return str(value)
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds form) into seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None  # HTTP-date form is not honored; caller falls back to backoff


def resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for name in ENV_API_KEY_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    raise AuthenticationError(
        "api_key is required (pass api_key=... or set "
        "SPEECHREVOLUTIONS_API_KEY / STT_API_KEY)"
    )
