"""New request options (2026-11-14): end_user_id, PII redaction, entities, profanity, paragraphs, verbatim."""
from speechrevolutions.models import TranscribeOptions, resolve_options


def test_defaults_add_nothing():
    p = TranscribeOptions().to_payload(file_size=1)
    for k in ("end_user_id", "redact_pii", "redact_pii_policies", "entity_detection", "filter_profanity", "paragraphs", "verbatim"):
        assert k not in p


def test_new_options_reach_payload():
    o = resolve_options(None, end_user_id="eu-1", redact_pii=True, redact_pii_policies=["phone_number"],
                        redact_pii_sub="hash", entity_detection=True, filter_profanity=True, paragraphs=True, verbatim=True)
    p = o.to_payload(file_size=10)
    assert p["end_user_id"] == "eu-1" and p["redact_pii"] is True and p["redact_pii_policies"] == ["phone_number"]
    assert p["redact_pii_sub"] == "hash" and p["entity_detection"] and p["filter_profanity"] and p["paragraphs"]
    assert p["verbatim"] is True


def test_policies_only_with_redaction():
    p = resolve_options(None, redact_pii_policies=["phone_number"]).to_payload(file_size=1)
    assert "redact_pii_policies" not in p


def test_style_options_reach_payload_only_when_set():
    assert "keep_repeats" not in TranscribeOptions().to_payload(file_size=1)
    assert "conservative_vote" not in TranscribeOptions().to_payload(file_size=1)
    p = resolve_options(None, verbatim=False, keep_repeats=True, conservative_vote=True).to_payload(file_size=1)
    assert p["verbatim"] is False and p["keep_repeats"] is True and p["conservative_vote"] is True
