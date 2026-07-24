"""Transcript models and format adapters (AssemblyAI / Deepgram-style)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """A single transcribed word with optional timing, speaker, and language."""

    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    language: str | None = None

    @property
    def text(self) -> str:
        """AssemblyAI-compatible alias for ``word``."""
        return self.word

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"word": self.word, "text": self.word}
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.language is not None:
            d["language"] = self.language
        return d


@dataclass
class LanguageSegment:
    """A contiguous time range spoken in a single detected language."""

    start: float
    end: float
    language: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "language": self.language}


@dataclass
class Utterance:
    """A contiguous speaker turn (AssemblyAI-style)."""

    text: str
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None

    @property
    def transcript(self) -> str:
        """Deepgram-compatible alias for ``text``."""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "transcript": self.text,
            "words": [w.to_dict() for w in self.words],
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d


@dataclass
class Transcript:
    """
    High-level transcription result.

    Designed for a familiar DX across providers::

        print(result.text)
        for u in result.utterances:
            print(u.speaker, u.text)

    For non-JSON outputs (srt/vtt/docx/pdf), ``text`` is the decoded file
    contents when UTF-8-decodable, otherwise empty — use ``content`` / ``save``.
    """

    job_id: str
    output_type: str
    content: bytes
    download_url: str = ""
    words: list[Word] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    languages: list[LanguageSegment] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    _text: str = field(default="", repr=False)

    @property
    def text(self) -> str:
        """Full transcript text (AssemblyAI / ElevenLabs-style)."""
        if self._text:
            return self._text
        if self.utterances:
            return " ".join(u.text for u in self.utterances if u.text).strip()
        if self.words:
            return _join_words(self.words)
        return ""

    @property
    def transcript(self) -> str:
        """Deepgram-compatible alias for ``text``."""
        return self.text

    def save(self, path: str) -> str:
        """Write raw content to disk. Appends output_type if path has no extension."""
        name = path.rsplit("/", 1)[-1]
        out = path if "." in name else f"{path}.{self.output_type}"
        with open(out, "wb") as f:
            f.write(self.content)
        return out

    def to_dict(self) -> dict[str, Any]:
        """AssemblyAI-inspired normalized dict."""
        d: dict[str, Any] = {
            "id": self.job_id,
            "status": "completed",
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "utterances": [u.to_dict() for u in self.utterances],
            "output_type": self.output_type,
        }
        if self.languages:
            d["languages"] = [seg.to_dict() for seg in self.languages]
        return d

    def to_deepgram(self) -> dict[str, Any]:
        """
        Rough Deepgram pre-recorded response shape for easier migrations.

        Access path mirrors Deepgram::

            result.to_deepgram()["results"]["channels"][0]["alternatives"][0]["transcript"]
        """
        dg_words = []
        for w in self.words:
            item: dict[str, Any] = {
                "word": w.word.strip(".,!?;:").lower() if w.word else w.word,
                "punctuated_word": w.word,
            }
            if w.start is not None:
                item["start"] = w.start
            if w.end is not None:
                item["end"] = w.end
            if w.confidence is not None:
                item["confidence"] = w.confidence
            if w.speaker is not None:
                item["speaker"] = _speaker_index(w.speaker)
            dg_words.append(item)

        alternative: dict[str, Any] = {
            "transcript": self.text,
            "confidence": 1.0,
            "words": dg_words,
        }

        utterances = []
        for u in self.utterances:
            utt: dict[str, Any] = {
                "transcript": u.text,
                "channel": 0,
                "words": [
                    {
                        "word": w.word,
                        "punctuated_word": w.word,
                        **({"start": w.start} if w.start is not None else {}),
                        **({"end": w.end} if w.end is not None else {}),
                        **({"speaker": _speaker_index(w.speaker)} if w.speaker else {}),
                    }
                    for w in u.words
                ],
            }
            if u.start is not None:
                utt["start"] = u.start
            if u.end is not None:
                utt["end"] = u.end
            if u.speaker is not None:
                utt["speaker"] = _speaker_index(u.speaker)
            utterances.append(utt)

        return {
            "metadata": {
                "request_id": self.job_id,
                "channels": 1,
            },
            "results": {
                "channels": [{"alternatives": [alternative]}],
                "utterances": utterances,
            },
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def parse_transcript(
    *,
    job_id: str,
    content: bytes,
    output_type: str,
    download_url: str = "",
) -> Transcript:
    """Build a Transcript from downloaded result bytes."""
    if output_type != "json":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        return Transcript(
            job_id=job_id,
            output_type=output_type,
            content=content,
            download_url=download_url,
            _text=text,
        )

    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return Transcript(
            job_id=job_id,
            output_type=output_type,
            content=content,
            download_url=download_url,
        )

    if not isinstance(raw, dict):
        return Transcript(
            job_id=job_id,
            output_type=output_type,
            content=content,
            download_url=download_url,
            raw={"value": raw},
            _text=str(raw),
        )

    words = [_parse_word(w) for w in raw.get("words", []) if isinstance(w, dict)]
    diarization = raw.get("diarization") or []
    utterances = (
        _utterances_from_diarization(words, diarization)
        if diarization
        else _utterances_from_words(words)
    )
    languages = [_parse_language_segment(s) for s in raw.get("languages", []) if isinstance(s, dict)]
    text = _join_words(words)

    return Transcript(
        job_id=job_id,
        output_type=output_type,
        content=content,
        download_url=download_url,
        words=words,
        utterances=utterances,
        languages=languages,
        raw=raw,
        _text=text,
    )


def _parse_word(data: dict[str, Any]) -> Word:
    word = data.get("word") or data.get("text") or ""
    return Word(
        word=str(word),
        start=_as_float(data.get("start")),
        end=_as_float(data.get("end")),
        speaker=str(data["speaker"]) if data.get("speaker") is not None else None,
        confidence=_as_float(data.get("confidence")),
        language=str(data["language"]) if data.get("language") is not None else None,
    )


def _parse_language_segment(data: dict[str, Any]) -> LanguageSegment:
    return LanguageSegment(
        start=_as_float(data.get("start")) or 0.0,
        end=_as_float(data.get("end")) or 0.0,
        language=str(data.get("language", "")),
    )


def _utterances_from_words(words: list[Word]) -> list[Utterance]:
    if not words:
        return []

    # If no speaker labels, one utterance for the whole transcript.
    if all(w.speaker is None for w in words):
        return [
            Utterance(
                text=_join_words(words),
                start=words[0].start,
                end=words[-1].end,
                words=list(words),
            )
        ]

    utterances: list[Utterance] = []
    current: list[Word] = [words[0]]
    for w in words[1:]:
        if w.speaker == current[0].speaker:
            current.append(w)
        else:
            utterances.append(_utterance_from_group(current))
            current = [w]
    utterances.append(_utterance_from_group(current))
    return utterances


def _utterances_from_diarization(
    words: list[Word],
    diarization: list[dict[str, Any]],
) -> list[Utterance]:
    """Prefer server-provided diarization segments when present."""
    utterances: list[Utterance] = []
    for seg in diarization:
        if not isinstance(seg, dict):
            continue
        start = _as_float(seg.get("start"))
        end = _as_float(seg.get("end"))
        speaker = str(seg["speaker"]) if seg.get("speaker") is not None else None
        seg_words = [
            w
            for w in words
            if w.start is not None
            and w.end is not None
            and start is not None
            and end is not None
            and w.start >= start - 1e-3
            and w.end <= end + 1e-3
        ]
        if not seg_words:
            # Fallback: words whose midpoint falls in the segment
            seg_words = [
                w
                for w in words
                if w.start is not None
                and w.end is not None
                and start is not None
                and end is not None
                and start <= (w.start + w.end) / 2 <= end
            ]
        text = _join_words(seg_words) if seg_words else ""
        utterances.append(
            Utterance(
                text=text,
                speaker=speaker,
                start=start,
                end=end,
                words=seg_words,
            )
        )
    return utterances or _utterances_from_words(words)


def _utterance_from_group(group: list[Word]) -> Utterance:
    return Utterance(
        text=_join_words(group),
        speaker=group[0].speaker,
        start=group[0].start,
        end=group[-1].end,
        words=list(group),
    )


def _join_words(words: list[Word]) -> str:
    parts: list[str] = []
    for w in words:
        token = w.word
        if not token:
            continue
        if parts and token[0] in ".,!?;:%)]}'\"":
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return " ".join(parts)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _speaker_index(speaker: str | None) -> int | str | None:
    """Map SPEAKER_0 / A → integer when possible (Deepgram-style)."""
    if speaker is None:
        return None
    s = str(speaker)
    if s.upper().startswith("SPEAKER_"):
        try:
            return int(s.split("_", 1)[1])
        except ValueError:
            return s
    if len(s) == 1 and s.isalpha():
        return ord(s.upper()) - ord("A")
    try:
        return int(s)
    except ValueError:
        return s
