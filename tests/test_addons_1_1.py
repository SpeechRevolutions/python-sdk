"""1.1.0 add-on options: what is sent, and what comes back."""
from speechrevolutions import PII_POLICIES, TranscribeOptions
from speechrevolutions.models import resolve_options
from speechrevolutions.streaming import StreamEvent, _session_update
from speechrevolutions.transcript import parse_transcript


def test_add_on_options_are_sent_only_when_set():
    base = TranscribeOptions().to_payload()
    for k in ("auto_chapters", "known_speakers", "redact_pii", "redact_pii_policies", "redact_pii_sub", "entity_detection"):
        assert k not in base
    assert base["speaker_labels"] is False                 # explicit, off by default
    p = resolve_options(None, auto_chapters=True, known_speakers=["Agent", "Customer"], redact_pii=True,
                        redact_pii_policies=["person_name", "phone_number"], redact_pii_sub="hash",
                        entity_detection=True, speaker_labels=True).to_payload()
    assert p["auto_chapters"] is True and p["known_speakers"] == ["Agent", "Customer"]
    assert p["redact_pii"] is True and p["redact_pii_policies"] == ["person_name", "phone_number"]
    assert p["redact_pii_sub"] == "hash" and p["entity_detection"] is True and p["speaker_labels"] is True
    assert "redact_pii_policies" not in resolve_options(None, redact_pii_policies=["date"]).to_payload()  # needs redact_pii
    assert "person_name" in PII_POLICIES and len(PII_POLICIES) == 12


def test_transcript_parses_chapters_entities_names_and_pii_status():
    raw = b'''{"words": [{"word": "Hi.", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_1"}],
      "chapters": [{"start": 0.0, "end": 0.4, "headline": "Greeting", "gist": "Hello."}],
      "entities": [{"entity_type": "person_name", "text": "[PERSON_NAME]", "start": 0.0, "end": 0.4}],
      "speaker_names": {"SPEAKER_1": "Agent"}, "pii_status": "patterns_only"}'''
    t = parse_transcript(job_id="j", content=raw, output_type="json")
    assert t.chapters[0].headline == "Greeting" and t.entities[0].entity_type == "person_name"
    assert t.speaker_names == {"SPEAKER_1": "Agent"} and t.pii_status == "patterns_only"
    d = t.to_dict()
    assert d["chapters"][0]["gist"] == "Hello." and d["speaker_names"]["SPEAKER_1"] == "Agent"


def test_streaming_options_reach_session_update():
    m = _session_update("en", None, None, None, None, redact_pii=True, redact_pii_sub="hash",
                        include_partials=True, voiceprint="vp1.x", speaker_policy="strict")
    assert m["redact_pii"] is True and m["redact_pii_sub"] == "hash" and m["include_partials"] is True
    assert m["voiceprint"] == "vp1.x" and m["speaker_policy"] == "strict" and "capture_voiceprint" not in m
    plain = _session_update(None, None, None, None, None)
    assert not ({"redact_pii", "include_partials", "voiceprint", "capture_voiceprint"} & set(plain))
    ev = StreamEvent.from_message({"type": "voice_focus", "active": True})
    assert ev.active is True
    assert StreamEvent.from_message({"type": "final", "index": 0, "text": "[PERSON_NAME]", "pii_status": "patterns_only"}).pii_status == "patterns_only"
