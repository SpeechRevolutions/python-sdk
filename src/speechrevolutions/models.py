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
    # ISO 639-1 code (e.g. "en", "ru") to transcribe in, skipping language
    # detection. None (or "auto") = detect automatically. Pinning the wrong
    # language makes the model translate into it.
    language: str | None = None
    callback_url: str | None = None
    #: The caller's own id for the person the audio belongs to; accounts with automatic vocabulary get that
    #: person's learned terms added to custom_vocabulary.
    end_user_id: str | None = None
    #: Replace PII in the transcript (timings kept): email_address, phone_number, credit_card_number,
    #: us_social_security_number, ip_address (default: all). redact_pii_sub: "entity_name" or "hash".
    redact_pii: bool = False
    redact_pii_policies: list[str] | None = None
    redact_pii_sub: str | None = None
    #: json output gains `entities` [{entity_type, text, start, end}].
    entity_detection: bool = False
    #: Mask profanity ("f***").
    filter_profanity: bool = False
    #: json output gains `paragraphs` [{start, end, speaker, text, sentences}].
    paragraphs: bool = False
    #: json output gains `speaker_names` {"SPEAKER_1": "Alice"} when the conversation establishes who is who.
    speaker_names: bool = False
    #: Keep fillers and false starts (verbatim) instead of the default style.
    verbatim: bool | None = None

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
        if self.language is not None:
            payload["language"] = self.language
        if self.callback_url:
            payload["callback_url"] = self.callback_url
        if self.end_user_id:
            payload["end_user_id"] = self.end_user_id
        for flag in ("redact_pii", "entity_detection", "filter_profanity", "paragraphs", "speaker_names"):
            if getattr(self, flag):
                payload[flag] = True
        if self.redact_pii and self.redact_pii_policies:
            payload["redact_pii_policies"] = list(self.redact_pii_policies)
        if self.redact_pii and self.redact_pii_sub:
            payload["redact_pii_sub"] = self.redact_pii_sub
        if self.verbatim is not None:
            payload["verbatim"] = self.verbatim
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
    language: str | None = None,
    callback_url: str | None = None,
    end_user_id: Any = None,
    redact_pii: Any = None,
    redact_pii_policies: Any = None,
    redact_pii_sub: Any = None,
    entity_detection: Any = None,
    filter_profanity: Any = None,
    paragraphs: Any = None,
    speaker_names: Any = None,
    verbatim: Any = None,
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
        language=language if language is not None else base.language,
        callback_url=callback_url if callback_url is not None else base.callback_url,
        end_user_id=end_user_id if end_user_id is not None else base.end_user_id,
        redact_pii=redact_pii if redact_pii is not None else base.redact_pii,
        redact_pii_policies=redact_pii_policies if redact_pii_policies is not None else base.redact_pii_policies,
        redact_pii_sub=redact_pii_sub if redact_pii_sub is not None else base.redact_pii_sub,
        entity_detection=entity_detection if entity_detection is not None else base.entity_detection,
        filter_profanity=filter_profanity if filter_profanity is not None else base.filter_profanity,
        paragraphs=paragraphs if paragraphs is not None else base.paragraphs,
        speaker_names=speaker_names if speaker_names is not None else base.speaker_names,
        verbatim=verbatim if verbatim is not None else base.verbatim,
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
    upload_url: str
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
