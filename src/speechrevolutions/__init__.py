"""Speech Revolutions STT Python SDK."""

from speechrevolutions.client import SpeechRevolutions, SpeechRevolutionsClient, STTClient
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
    ProgressEvent,
    TranscribeOptions,
    UploadJob,
)
from speechrevolutions.transcript import LanguageSegment, Transcript, Utterance, Word

__all__ = [
    # Clients
    "STTClient",
    "SpeechRevolutions",
    "SpeechRevolutionsClient",
    "AsyncSTTClient",
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
    "STTError",
    "AuthenticationError",
    "RateLimitError",
    "JobNotFoundError",
    "JobFailedError",
    "UploadError",
    "TimeoutError",
    "APIError",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    """Lazy-load async client so sync users don't need httpx until they import it."""
    if name in {"AsyncSTTClient", "AsyncSpeechRevolutions", "AsyncSpeechRevolutionsClient"}:
        from speechrevolutions.async_client import (
            AsyncSpeechRevolutions,
            AsyncSpeechRevolutionsClient,
            AsyncSTTClient,
        )

        mapping = {
            "AsyncSTTClient": AsyncSTTClient,
            "AsyncSpeechRevolutions": AsyncSpeechRevolutions,
            "AsyncSpeechRevolutionsClient": AsyncSpeechRevolutionsClient,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
