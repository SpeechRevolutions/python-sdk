"""Streaming client event mapping (speechrevolutions.streaming). Verified live 2026-11-14 against the dev
endpoint: partial / final / revision / done events from a 32 s file sent with send_file."""
from speechrevolutions.streaming import StreamSession, _event


def test_event_mapping():
    e = _event({"type": "transcription.partial", "index": 2, "text": "hel", "stable": 1, "audio_s": 3.5})
    assert (e.type, e.index, e.text, e.stable, e.start) == ("partial", 2, "hel", 1, 3.5)
    e = _event({"type": "final", "index": 2, "text": "Hello.", "start": 1.0, "end": 3.6, "language": "en"})
    assert (e.type, e.start, e.end, e.language) == ("final", 1.0, 3.6, "en")
    assert _event({"type": "final.revised", "index": 2, "text": "Hello!"}).type == "revision"
    assert _event({"type": "error", "error": "at capacity"}).text == "at capacity"
    assert _event({"type": "turn.ended", "index": 2, "start": 1.0, "end": 3.6}).type == "turn_ended"


def test_session_update_contents():
    s = StreamSession("ws://x", "k", language="auto", vocabulary=["Acme"], turn_threshold=0.6, max_silence_ms=700)
    assert s._update == {"type": "session.update", "vocabulary": ["Acme"],
                         "turn_detection": {"threshold": 0.6, "max_silence_ms": 700}}
    assert StreamSession("ws://x", "k", language="ja", vocabulary=None, turn_threshold=None,
                         max_silence_ms=None)._update == {"type": "session.update", "language": "ja"}
