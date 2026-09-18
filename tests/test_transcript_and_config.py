"""Transcript parsing and the small config helpers.

These are the paths every caller hits on every successful job, and none of them
had a test before.
"""

from __future__ import annotations

import json

import pytest

from speechrevolutions._config import (
    extract_request_id,
    parse_retry_after,
    resolve_api_key,
)
from speechrevolutions.exceptions import AuthenticationError
from speechrevolutions.transcript import parse_transcript


# --------------------------------------------------------------------------
# API key resolution
# --------------------------------------------------------------------------

def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("SPEECHREVOLUTIONS_API_KEY", "from-env")
    assert resolve_api_key("explicit") == "explicit"


def test_primary_env_var(monkeypatch):
    monkeypatch.delenv("STT_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHREVOLUTIONS_API_KEY", "abc")
    assert resolve_api_key(None) == "abc"


def test_legacy_env_var_still_works(monkeypatch):
    monkeypatch.delenv("SPEECHREVOLUTIONS_API_KEY", raising=False)
    monkeypatch.setenv("STT_API_KEY", "legacy")
    assert resolve_api_key(None) == "legacy"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SPEECHREVOLUTIONS_API_KEY", raising=False)
    monkeypatch.delenv("STT_API_KEY", raising=False)
    with pytest.raises(AuthenticationError):
        resolve_api_key(None)


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [("5", 5.0), ("0", 0.0), ("2.5", 2.5), ("-3", 0.0), (None, None), ("", None)],
)
def test_parse_retry_after_delta_seconds(value, expected):
    assert parse_retry_after(value) == expected


def test_parse_retry_after_http_date_is_not_honoured():
    """Documented limitation: the HTTP-date form falls through to backoff."""
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


def test_extract_request_id_prefers_first_present():
    assert extract_request_id({"x-request-id": "a", "cf-ray": "b"}) == "a"
    assert extract_request_id({"cf-ray": "b"}) == "b"
    assert extract_request_id({}) is None
    assert extract_request_id(None) is None


# --------------------------------------------------------------------------
# Transcript parsing
# --------------------------------------------------------------------------

WORDS_JSON = {
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "A", "confidence": 0.99},
        {"word": "there", "start": 0.4, "end": 0.8, "speaker": "A", "confidence": 0.97},
        {"word": "hi", "start": 1.0, "end": 1.2, "speaker": "B", "confidence": 0.95},
    ],
    "languages": [{"start": 0.0, "end": 1.2, "language": "en"}],
}


def _parse(body, output_type="json"):
    content = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    return parse_transcript(job_id="j1", content=content, output_type=output_type)


def test_parses_words_and_text():
    t = _parse(WORDS_JSON)
    assert len(t.words) == 3
    assert t.text == "Hello there hi"
    assert t.words[0].speaker == "A"
    assert t.words[0].confidence == pytest.approx(0.99)


def test_groups_utterances_by_speaker():
    t = _parse(WORDS_JSON)
    assert [u.speaker for u in t.utterances] == ["A", "B"]
    assert t.utterances[0].text == "Hello there"
    assert t.utterances[1].text == "hi"


def test_parses_language_segments():
    t = _parse(WORDS_JSON)
    assert len(t.languages) == 1
    assert t.languages[0].language == "en"


def test_word_alias_text_key_is_accepted():
    t = _parse({"words": [{"text": "aliased", "start": 0, "end": 1}]})
    assert t.words[0].word == "aliased"


def test_empty_words_yields_empty_transcript():
    t = _parse({"words": []})
    assert t.words == []
    assert t.text == ""


def test_malformed_json_does_not_raise():
    t = _parse(b"{not json at all", output_type="json")
    assert t.words == []
    assert t.content == b"{not json at all"


def test_non_dict_json_is_tolerated():
    t = _parse([1, 2, 3])
    assert t.raw == {"value": [1, 2, 3]}


def test_non_dict_word_entries_are_skipped():
    t = _parse({"words": [{"word": "ok", "start": 0, "end": 1}, "junk", None]})
    assert len(t.words) == 1


@pytest.mark.parametrize("fmt", ["txt", "srt", "vtt"])
def test_text_formats_decode_to_text(fmt):
    t = _parse(b"1\n00:00:00,000 --> 00:00:01,000\nHello\n", output_type=fmt)
    assert "Hello" in t.text
    assert t.output_type == fmt


def test_binary_output_does_not_raise_on_decode():
    t = _parse(b"\x89PNG\r\n\x1a\n\xff\xfe", output_type="pdf")
    assert t.text == ""
    assert t.content.startswith(b"\x89PNG")


def test_save_writes_bytes(tmp_path):
    t = _parse(WORDS_JSON)
    out = tmp_path / "result.json"
    t.save(str(out))
    assert out.exists()
    assert json.loads(out.read_text())["words"][0]["word"] == "Hello"
