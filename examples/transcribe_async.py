"""Minimal example — async client. Shows progress, saves the result.

Mirrors examples/transcribe.py but on the asyncio client. Every option is
spelled out explicitly (no reliance on defaults).
"""

import asyncio

from speechrevolutions import AsyncSpeechRevolutions


async def main() -> None:
    # Reads the key from SPEECHREVOLUTIONS_API_KEY.
    # `async with` closes the underlying httpx client on exit.
    async with AsyncSpeechRevolutions() as client:
        result = await client.transcribe(
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

    out = result.save("output")            # write to output.<output_type>; returns the path
    print(f"Saved transcript to {out}")


if __name__ == "__main__":
    asyncio.run(main())
