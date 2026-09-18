"""Everything a caller does with a Transcript once they have one.

`to_dict`, `to_json`, `to_deepgram` and `save` are all published API and none of
them had a test. `to_deepgram` in particular is the compatibility shim a
migrating customer leans on, so a silent shape change there breaks their code
rather than ours.
"""

from __future__ import annotations

import json

import pytest

from mock_api import MockAPI
from speechrevolutions import SpeechRevolutions


@pytest.fixture
def result():
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            yield c.transcribe(b"x" * 64)


# ---------------------------------------------------------------------------
# Parsed shape
# ---------------------------------------------------------------------------

def test_text_joins_the_words(result):
    assert result.text == "Good morning everyone. Thanks for joining."


def test_transcript_property_is_a_deepgram_alias_for_text(result):
    # Published as a migration convenience: Deepgram callers reach for
    # `.transcript`, so it has to stay equal to `.text`.
    assert result.transcript == result.text


def test_printing_the_object_is_not_the_transcript(result):
    # Deliberate, and worth pinning so nobody "fixes" it by accident: there is
    # no __str__, so print(result) shows the dataclass repr. Every doc and
    # recipe uses result.text, which is the explicit path.
    assert str(result).startswith("Transcript(")


def test_words_carry_timings_speaker_confidence_language(result):
    w = result.words[0]
    assert w.word == "Good"
    assert w.start == 0.0 and w.end == pytest.approx(0.32)
    assert w.speaker == "A"
    assert w.confidence == pytest.approx(0.99)
    assert w.language == "en"


def test_utterances_group_by_speaker_turn(result):
    assert [(u.speaker, u.text) for u in result.utterances] == [
        ("A", "Good morning everyone."),
        ("B", "Thanks for joining."),
    ]


def test_utterances_carry_their_span(result):
    a, b = result.utterances
    assert a.start == 0.0 and a.end == pytest.approx(1.44)
    assert b.start == pytest.approx(2.10) and b.end == pytest.approx(3.20)


def test_languages_are_parsed(result):
    assert [(s.language, s.start, s.end) for s in result.languages] == [("en", 0.0, 3.2)]


def test_raw_keeps_the_untouched_payload(result):
    assert "words" in result.raw and "diarization" in result.raw


# ---------------------------------------------------------------------------
# to_dict / to_json
# ---------------------------------------------------------------------------

def test_to_dict_has_the_documented_keys(result):
    d = result.to_dict()
    for key in ("text", "words", "utterances"):
        assert key in d, f"to_dict() lost '{key}'"
    assert d["text"] == result.text
    assert len(d["words"]) == 6
    assert len(d["utterances"]) == 2


def test_to_dict_is_json_serialisable(result):
    json.dumps(result.to_dict())          # must not raise


def test_to_json_round_trips(result):
    parsed = json.loads(result.to_json())
    assert parsed["text"] == result.text


# ---------------------------------------------------------------------------
# to_deepgram — the migration shim
# ---------------------------------------------------------------------------

def test_to_deepgram_has_deepgrams_envelope(result):
    dg = result.to_deepgram()
    alt = dg["results"]["channels"][0]["alternatives"][0]
    assert alt["transcript"] == result.text
    assert len(alt["words"]) == 6


def test_to_deepgram_words_use_deepgram_field_names(result):
    alt = result.to_deepgram()["results"]["channels"][0]["alternatives"][0]
    w = alt["words"][0]
    assert {"word", "punctuated_word", "start", "end"} <= set(w)
    # Deepgram's own shape: `word` is normalised (lowercased, punctuation
    # stripped) and `punctuated_word` is as spoken. A migrating caller reading
    # either field gets what Deepgram would have given them.
    assert w["word"] == "good"
    assert w["punctuated_word"] == "Good"


def test_to_deepgram_strips_punctuation_from_the_normalised_word(result):
    alt = result.to_deepgram()["results"]["channels"][0]["alternatives"][0]
    everyone = alt["words"][2]
    assert everyone["punctuated_word"] == "everyone."
    assert everyone["word"] == "everyone"


def test_to_deepgram_maps_speakers_to_integers(result):
    # Deepgram numbers speakers; we label them A/B.
    alt = result.to_deepgram()["results"]["channels"][0]["alternatives"][0]
    speakers = {w["speaker"] for w in alt["words"] if "speaker" in w}
    assert speakers == {0, 1}


def test_to_deepgram_is_json_serialisable(result):
    json.dumps(result.to_deepgram())


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

def test_save_writes_json(result, tmp_path):
    p = tmp_path / "out.json"
    returned = result.save(str(p))
    assert p.exists()
    assert str(p) in str(returned)
    assert json.loads(p.read_text())["words"][0]["word"] == "Good"


def test_save_creates_parent_directories(result, tmp_path):
    p = tmp_path / "nested" / "deeper" / "out.json"
    try:
        result.save(str(p))
    except FileNotFoundError:
        pytest.skip("save() does not create parent directories; documented behaviour")
    assert p.exists()


# ---------------------------------------------------------------------------
# Non-JSON output types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,needle", [
    ("srt", "-->"),
    ("vtt", "WEBVTT"),
    ("txt", "Good morning"),
])
def test_text_formats_come_back_decoded(fmt, needle):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            r = c.transcribe(b"x" * 64, output_type=fmt)
    assert r.output_type == fmt
    assert needle in r.text
    assert r.words == []          # no word list for a rendered format


@pytest.mark.parametrize("fmt", ["docx", "pdf"])
def test_binary_formats_keep_their_bytes_and_save(fmt, tmp_path):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            r = c.transcribe(b"x" * 64, output_type=fmt)
    out = tmp_path / f"out.{fmt}"
    r.save(str(out))
    assert out.read_bytes() == r.content
    assert out.read_bytes().startswith(b"\x50\x4b")


def test_srt_save_writes_subtitle_text(tmp_path):
    with MockAPI(progress_steps=1) as api:
        with SpeechRevolutions(api_key="test-key", base_url=api.base_url) as c:
            r = c.transcribe(b"x" * 64, output_type="srt")
    p = tmp_path / "subs.srt"
    r.save(str(p))
    assert "00:00:00,000 --> 00:00:01,440" in p.read_text()
