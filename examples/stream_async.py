"""Async streaming: feed audio from your own source and handle events in an `async for` loop.

    pip install "speechrevolutions[streaming]"
    python examples/stream_async.py call.mp3

Here the "source" is a file read in 80 ms chunks at real time; in an app it would be a microphone,
a WebRTC track or a phone call. Audio is PCM16 mono 16 kHz.
"""

import asyncio
import sys

from speechrevolutions import AsyncSpeechRevolutions, load_pcm16

CHUNK = 16000 * 2 * 80 // 1000  # 80 ms of PCM16 mono 16 kHz


async def feed(session, pcm: bytes) -> None:
    for i in range(0, len(pcm), CHUNK):
        await session.send_audio(pcm[i : i + CHUNK])
        await asyncio.sleep(0.08)  # real time, like a microphone
    result = await session.finish()
    print(f"\ndone: {result.text}")


async def main(path: str) -> None:
    async with AsyncSpeechRevolutions() as client:
        session = await client.stream(language="en")
        feeder = asyncio.create_task(feed(session, load_pcm16(path)))
        async for event in session:  # ends after transcription.done
            if event.type == "final":
                print(f"[{event.start:.2f}-{event.end:.2f}] {event.text}")
            elif event.type == "turn.ended":
                print(f"   (turn ended, p={event.probability})")
        await feeder


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
