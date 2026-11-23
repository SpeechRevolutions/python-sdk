"""Stream an audio file as if it were live, printing partials and finals as they arrive.

    pip install "speechrevolutions[streaming]"
    export SPEECHREVOLUTIONS_API_KEY=...
    python examples/stream_file.py call.mp3 --lang en

Anything other than a 16 kHz mono 16-bit WAV is converted with ffmpeg. Leave out --lang to have the
language detected. The file streams in real time (live audio can't arrive faster).
"""

import argparse

from speechrevolutions import SpeechRevolutions, StreamEvent


def on_event(e: StreamEvent) -> None:
    if e.type == "session.created":
        print(f"session {e.job_id}")
    elif e.type == "transcription.partial":
        print("\r\033[K" + (e.text or ""), end="", flush=True)  # redraw the open utterance
    elif e.type == "final":
        print(f"\r\033[K[{e.start:6.2f}-{e.end:6.2f}] {e.text}", flush=True)
    elif e.type == "final.revised":
        print(f"         revised #{e.index}: {e.text}", flush=True)
    elif e.type == "language.detected":
        print(f"         language: {e.language}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lang", default=None, help="spoken language, e.g. en (default: detect)")
    ap.add_argument("--vocab", nargs="*", default=None, help="names and terms to recognise")
    a = ap.parse_args()

    with SpeechRevolutions() as client:
        result = client.stream_file(a.audio, language=a.lang, vocabulary=a.vocab, on_event=on_event)
    print(f"\n{len(result.finals)} finals, job {result.job_id}")
    print(result.text)


if __name__ == "__main__":
    main()
