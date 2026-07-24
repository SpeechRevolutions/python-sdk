"""Request options and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from speechrevolutions.transcript import LanguageSegment, Transcript, Utterance, Word

__all__ = [
    "OutputType",
    "TranscribeOptions",
    "UploadJob",
    "JobStatus",
    "ProgressEvent",
    "ProgressCallback",
    "Transcript",
    "Word",
    "Utterance",
    "LanguageSegment",
]


class OutputType(str, Enum):
    txt = "txt"
    json = "json"
    srt = "srt"
    vtt = "vtt"
    docx = "docx"
    pdf = "pdf"


class ProcessingTier(str, Enum):
    """Processing / pricing tier."""

    standard = "standard"
    economy = "economy"


@dataclass
class TranscribeOptions:
    """Options controlling how audio is transcribed."""

    output_type: OutputType | str = OutputType.json
    word_timestamps: bool = True
    speaker_labels: bool = True
    nltk: bool = True
    tier: ProcessingTier | str = ProcessingTier.standard
    custom_vocabulary: list[str] | None = None
    callback_url: str | None = None

    def normalized_output_type(self) -> str:
        if isinstance(self.output_type, OutputType):
            return self.output_type.value
        return str(self.output_type)

    def normalized_tier(self) -> str:
        if isinstance(self.tier, ProcessingTier):
            return self.tier.value
        return str(self.tier)

    def to_payload(
        self, file_size: int | None = None, *, audio_url: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output_type": self.normalized_output_type(),
            "word_timestamps": self.word_timestamps,
            "speaker_labels": self.speaker_labels,
            "nltk": self.nltk,
            "tier": self.normalized_tier(),
        }
        if file_size is not None:
            payload["file_size"] = file_size
        if audio_url is not None:
            payload["audio_url"] = audio_url
        if self.custom_vocabulary:
            payload["custom_vocabulary"] = self.custom_vocabulary
        if self.callback_url:
            payload["callback_url"] = self.callback_url
        return payload


def resolve_options(
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
) -> TranscribeOptions:
    """Merge an options object with kwargs (kwargs win). ``diarize`` aliases ``speaker_labels``."""
    base = options or TranscribeOptions()
    speakers = base.speaker_labels
    if speaker_labels is not None:
        speakers = speaker_labels
    if diarize is not None:
        speakers = diarize

    return TranscribeOptions(
        output_type=output_type if output_type is not None else base.output_type,
        word_timestamps=word_timestamps if word_timestamps is not None else base.word_timestamps,
        speaker_labels=speakers,
        nltk=nltk if nltk is not None else base.nltk,
        tier=tier if tier is not None else base.tier,
        custom_vocabulary=(
            custom_vocabulary if custom_vocabulary is not None else base.custom_vocabulary
        ),
        callback_url=callback_url if callback_url is not None else base.callback_url,
    )


@dataclass
class JobStatus:
    """Result of GET /api/v1/jobs/{id} — a job's current state."""

    job_id: str
    status: str  # "processing" | "completed" | "failed"
    download_url: str | None = None
    failed_stage: str | None = None
    reason: str | None = None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"


@dataclass
class UploadJob:
    """Result of POST /api/v1/upload."""

    job_id: str
    upload_url: str | dict[str, Any]
    download_url: str
    content_type: str = "application/octet-stream"
    expires_in: int = 0


@dataclass
class ProgressEvent:
    """A single progress update from the SSE stream (or polling fallback)."""

    completed: int | None = None
    total: int | None = None
    step: str | None = None
    elapsed_seconds: float | None = None
    raw: dict[str, Any] | None = None

    @property
    def percent(self) -> float | None:
        """Completion as a 0–100 float, or ``None`` if it can't be computed yet."""
        if self.completed is None or not self.total:
            return None
        return max(0.0, min(100.0, self.completed / self.total * 100.0))


ProgressCallback = Callable[[ProgressEvent], None]
