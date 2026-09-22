"""SDK exception hierarchy.

Every error carries the HTTP ``status_code`` and the server ``request_id`` (from
the response headers, when present) so failures can be correlated with server
logs. ``RateLimitError`` also exposes ``retry_after`` seconds.
"""

from __future__ import annotations


class SpeechRevolutionsError(Exception):
    """Base exception for all Speech Revolutions SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        body: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.body = body

    def __str__(self) -> str:
        suffix = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{self.message}{suffix}"


class AuthenticationError(SpeechRevolutionsError):
    """Raised when the API key is missing or rejected (HTTP 401)."""


class RateLimitError(SpeechRevolutionsError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: object):
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.retry_after = retry_after


class JobNotFoundError(SpeechRevolutionsError):
    """Raised when a job ID is unknown or its upload session expired (HTTP 404)."""


class JobFailedError(SpeechRevolutionsError):
    """Raised when the transcription pipeline reports a failure."""

    def __init__(
        self,
        message: str,
        *,
        step: str | None = None,
        reason: str | None = None,
        **kwargs: object,
    ):
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.step = step
        self.reason = reason


class UploadError(SpeechRevolutionsError):
    """Raised when the audio upload to object storage fails."""


class TimeoutError(SpeechRevolutionsError):  # noqa: A001 — mirrors stdlib name intentionally
    """Raised when waiting for a job exceeds the configured timeout."""


class APIError(SpeechRevolutionsError):
    """Raised for unexpected non-success HTTP responses from the API."""
