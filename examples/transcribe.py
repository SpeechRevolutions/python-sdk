"""Minimal example — sync client. Shows progress, saves the result.

Every option is spelled out explicitly (no reliance on defaults) so you can
see every knob transcribe() exposes.
"""

from speechrevolutions import SpeechRevolutions


def main() -> None:
    # Reads the key from SPEECHREVOLUTIONS_API_KEY.
    client = SpeechRevolutions()

    result = client.transcribe(
        "audio.mp3",           # audio: local path, URL, bytes, or file object
        output_type="json",            # output format: txt | json | srt | vtt | docx | pdf
        word_timestamps=True,          # include per-word start/end times
        speaker_labels=True,           # label who spoke each segment (alias: diarize=)
        nltk=True,                     # restore punctuation & capitalization
        tier="standard",               # processing tier: standard | economy
        custom_vocabulary=None,        # list[str] of domain terms to bias toward, or None
        on_progress=None,              # callback(ProgressEvent) for transcription %, or None
        on_upload_progress=None,       # callback(ProgressEvent) for upload %, or None
        progress=True,                 # render live upload + transcription bars in the console
    )

    out = result.save("output")        # write to output.<output_type>; returns the path
    print(f"Saved transcript to {out}")


if __name__ == "__main__":
    main()
