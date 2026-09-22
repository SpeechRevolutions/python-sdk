"""Speech Revolutions STT Python SDK."""

from speechrevolutions.client import SpeechRevolutions, SpeechRevolutionsClient
from speechrevolutions.exceptions import (
    APIError,
    AuthenticationError,
    JobFailedError,
    JobNotFoundError,
    RateLimitError,
    SpeechRevolutionsError,
    TimeoutError,
    UploadError,
)
from speechrevolutions.models import (
    JobStatus,
    OutputType,
    ProcessingTier,
    ProgressEvent,
    TranscribeOptions,
    UploadJob,
)
from speechrevolutions.transcript import LanguageSegment, Transcript, Utterance, Word

__all__ = [
    # Clients
    "SpeechRevolutions",
    "SpeechRevolutionsClient",
    "AsyncSpeechRevolutions",
    "AsyncSpeechRevolutionsClient",
    # Models
    "OutputType",
    "ProcessingTier",
    "TranscribeOptions",
    "ProgressEvent",
    "UploadJob",
    "JobStatus",
    "Transcript",
    "Word",
    "Utterance",
    "LanguageSegment",
    # Errors
    "SpeechRevolutionsError",
    "AuthenticationError",
    "RateLimitError",
    "JobNotFoundError",
    "JobFailedError",
    "UploadError",
    "TimeoutError",
    "APIError",
]

# Read from the installed distribution rather than restated here. 0.2.1 shipped
# with pyproject bumped and this constant left behind, so the package reported a
# version it was not — the kind of drift that only shows up in a bug report.
try:  # pragma: no cover - trivial, and the fallback is only hit from a source tree
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    __version__ = _dist_version("speechrevolutions")
except PackageNotFoundError:  # running from a checkout with nothing installed
    __version__ = "0.0.0.dev0"


def __getattr__(name: str):
    """Lazy-load async client so sync users don't need httpx until they import it."""
    if name in {"AsyncSpeechRevolutions", "AsyncSpeechRevolutionsClient"}:
        from speechrevolutions.async_client import (
            AsyncSpeechRevolutions,
            AsyncSpeechRevolutionsClient,
        )

        mapping = {
            "AsyncSpeechRevolutions": AsyncSpeechRevolutions,
            "AsyncSpeechRevolutionsClient": AsyncSpeechRevolutionsClient,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
