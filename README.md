# Speech Revolutions — Python SDK

Official Python client for the [Speech Revolutions](https://www.speechrevolutions.com) speech-to-text API.

## Install

```bash
pip install speechrevolutions
```

Optional extras:

```bash
pip install "speechrevolutions[streaming]"   # real-time streaming (websockets)
pip install "speechrevolutions[progress]"    # console progress bars (tqdm)
```

## Quick start

```python
from speechrevolutions import SpeechRevolutions

client = SpeechRevolutions()  # reads SPEECHREVOLUTIONS_API_KEY
result = client.transcribe("meeting.mp3", speaker_labels=True)
print(result.text)

for u in result.utterances:
    print(f"{u.speaker}: {u.text}")  # e.g. "SPEAKER_1: Hello."
```

### Async

```python
import asyncio

from speechrevolutions import AsyncSpeechRevolutions


async def main():
    async with AsyncSpeechRevolutions() as client:
        result = await client.transcribe("meeting.mp3", speaker_labels=True)
        print(result.text)


asyncio.run(main())
```

### From a URL (Deepgram-style)

```python
result = client.transcribe_url("https://docs.speechrevolutions.com/samples/diamond-necklace.mp3")
# or, since transcribe() detects http(s) URLs:
# result = client.transcribe("https://docs.speechrevolutions.com/samples/diamond-necklace.mp3")
```

### Options as kwargs or config object

```python
# kwargs (ElevenLabs / Deepgram style). `diarize` is an alias for `speaker_labels`.
result = client.transcribe("meeting.mp3", diarize=True, output_type="json")

# config object (AssemblyAI style)
from speechrevolutions import TranscribeOptions
result = client.transcribe("meeting.mp3", options=TranscribeOptions(speaker_labels=True))
```

Defaults: `output_type="json"`, `word_timestamps=True`, `nltk=True`, `speaker_labels=False`,
`language=None` (detected), everything else off.

### Options and add-ons

Batch is $0.09 per audio hour. Add-ons are charged per audio hour only when a request turns them on:

| Option | What it does | Price |
|---|---|---|
| `speaker_labels=True` (or `diarize=True`) | Speaker diarization: `word.speaker`, `result.utterances` | +$0.02/hr |
| `custom_vocabulary=["Zephyr", "Acme"]` | Keyterms: given to the model and matched by sound | +$0.05/hr |
| `summarize=True` | `result.summary.text` and `result.summary.bullets` | +$0.02/hr |
| `auto_chapters=True` | `result.chapters`: headline + gist sections covering the whole recording | with `summarize`: +$0.02/hr once |
| `filter_profanity=True` | Strong profanity masked (`s***`) in every format | +$0.01/hr |
| `known_speakers=["Agent", "Customer"]` | Speaker identification: `result.speaker_names` maps each speaker to one of them, or leaves it unnamed (needs `speaker_labels`) | +$0.02/hr |
| `redact_pii=True` | Names, organisations, places, addresses, dates, account / card / phone / SSN / IP numbers replaced by `[PERSON_NAME]`-style labels (or `####` with `redact_pii_sub="hash"`); narrow with `redact_pii_policies` (`speechrevolutions.PII_POLICIES`) | +$0.08/hr |
| `entity_detection=True` | `result.entities`: the same types, typed and timed | +$0.08/hr |
| `paragraphs=True` | `result.paragraphs`: speaker-aware, with sentences and times | free |
| `verbatim=True/False` | Keep or drop fillers and false starts | free |
| `language="en"` | Skip detection and transcribe in that language | free |
| `extra={...}` | Any other API option, sent as given | — |

```python
result = client.transcribe(
    "meeting.mp3",
    speaker_labels=True,
    summarize=True,
    paragraphs=True,
    custom_vocabulary=["Northwind"],
)
print(result.summary.text)
for p in result.paragraphs:
    print(f"[{p.start:.1f}s] {p.speaker}: {p.text}")
```

An add-on that could not be delivered is not charged (a summary that failed; PII redaction that could only use
pattern matching, which `result.pii_status == "patterns_only"` reports).

`language` pins an ISO 639-1 code (e.g. `language="ru"`) and skips language
detection; `None` (or `"auto"`) detects it automatically. Pinning the wrong
language makes the model translate into it (Russian audio pinned to `"en"`
comes back as English), and an unsupported code is rejected with HTTP 422.

### Live progress

Unlike AssemblyAI/Deepgram (which give no percentage for pre-recorded audio),
you get real-time progress — for **both** the file upload and the
transcription — as a console bar, a callback, or both.

```python
# 1. Console bars (uses tqdm if installed: pip install "speechrevolutions[progress]")
#    Shows an "Uploading" byte bar, then a "Transcribing" bar.
result = client.transcribe("meeting.mp3", progress=True)

# 2. Programmatic — read event.percent (0–100) to drive your own UI / API
def on_progress(event):        # transcription
    print(event.percent, event.step)      # e.g. 50.0 "chunk:0"

def on_upload(event):          # upload (event.step == "upload")
    print("upload", event.percent)

result = client.transcribe(
    "meeting.mp3",
    on_progress=on_progress,
    on_upload_progress=on_upload,
)
```

`progress=True` and the callbacks compose — the bars render *and* your callbacks
still fire for every event. When the job completes, `on_progress` always receives a
final event at `100` (`event.step == "completed"`), even for a short file that
finished before reporting any progress.

## Result shape

Default `output_type` is `json`. The SDK parses it into a transcript-first object:

| Field | Like |
|-------|------|
| `result.text` | AssemblyAI / ElevenLabs |
| `result.transcript` | Deepgram alias |
| `result.words` | word + start/end/speaker |
| `result.utterances` | AssemblyAI speaker turns |
| `result.to_deepgram()` | Deepgram-shaped dict |
| `result.to_dict()` | normalized JSON |
| `result.content` / `result.save()` | raw bytes / file |

```python
from speechrevolutions import SpeechRevolutions

client = SpeechRevolutions()
result = client.transcribe("meeting.mp3")

# Deepgram's response shape; speakers are numbered from 0.
dg = result.to_deepgram()
print(dg["results"]["channels"][0]["alternatives"][0]["transcript"])

# Save raw content to disk. Appends the output type if the path has no
# extension, e.g. "output" -> "output.json". Returns the written path.
path = result.save("output")
print(f"saved {path}")
```

`to_deepgram()` numbers speakers from 0, as Deepgram does: `SPEAKER_1` becomes speaker `0`,
`SPEAKER_2` speaker `1`. (Before 1.0.0 it kept our 1-based numbers.)

Delete a job's transcripts before the automatic purge with `client.delete_job(job_id)` (a
running job is cancelled first).

## Real-time streaming

Real-time transcription over WebSocket, $0.15 per session hour (early access). Install the
`streaming` extra. Audio is PCM16 mono 16 kHz; `load_pcm16(path)` converts any file with ffmpeg.

```python
from speechrevolutions import SpeechRevolutions

client = SpeechRevolutions()
session = client.stream(language="en")      # or languages=["en", "ja"], or nothing to detect it

for chunk in microphone_chunks():           # your audio source: PCM16 mono 16 kHz bytes
    session.send_audio(chunk)
result = session.finish()                   # waits for transcription.done
print(result.text)
```

Read events as they arrive by passing `on_event`, or by iterating the session from another thread:

```python
def on_event(e):
    if e.type == "transcription.partial":   # the open utterance so far; replaces the last partial
        print("\r" + e.text, end="")
    elif e.type == "final":                  # an utterance, about a second after it ends
        print(f"\n[{e.start:.2f}-{e.end:.2f}] {e.text}")

session = client.stream(language="en", vocabulary=["Northwind"], on_event=on_event)
```

| Event | Fields |
|---|---|
| `session.created` | `job_id` (the billing id, and the handle for resuming) |
| `transcription.partial` | `index`, `text`, `stable` (`text[:stable]` is settled), `audio_s` |
| `final` | `index`, `text`, `start`, `end` |
| `final.revised` | `index`, `text`: a better version of an earlier final |
| `speech.started` / `turn.ended` | `audio_s` / `index`, `start`, `end`, `probability`, `forced` |
| `language.detected` | `language`, `index` (when no language was given) |
| `transcription.done` | `text`: all finals joined |
| `error` | `code`, `message` |

- **Turns:** `turn_detection={"threshold": 0.5, "max_silence_ms": 800}` tunes when a turn ends, and
  `session.end_turn()` ends one now (push-to-talk).
- **PII redaction** (+$0.12 per session hour): `redact_pii=True` redacts finals before they reach you
  (`redact_pii_policies` / `redact_pii_sub` as in batch). Partials are not sent while it is on, unless
  `include_partials=True`, in which case they arrive unredacted.
- **Voice focus** (+$0.10 per session hour): lock the stream to one enrolled voice, so other people and
  background voices are left out. Enrol once with `capture_voiceprint=True`: the result carries
  `result.voiceprint` and `result.voiceprint_quality` (capture again until `ok`). Then pass
  `voiceprint=...` (and optionally `speaker_policy="strict"`). A `voice_focus` event says whether the lock is
  active, and only an active lock is billed. You keep the voiceprint; we keep none after the session.
- **Dropped connections are handled for you:** the session keeps audio the server hasn't acknowledged,
  reconnects with its job id for up to 60 s, and replays the rest. The transcript comes back complete
  under the same `job_id`.
- **Async:** `session = await client.stream(...)` on `AsyncSpeechRevolutions`, then
  `await session.send_audio(...)`, `async for event in session`, `await session.finish()`.
- **Testing with a recording:** `client.stream_file("call.mp3", language="en")` streams a file in
  real time and returns the result. For recordings in production, `transcribe` is faster and cheaper.

Audio must arrive in real time: the server accepts at most about 10 seconds ahead of the clock, and
refuses a faster sender with the error `use_batch`.

## Webhooks & retrieving results later

`submit()` uploads and enqueues a job and returns its id **without waiting** —
ideal for batch/background work. Collect the result later via a webhook
(`callback_url`, a signed POST — verify `X-SR-Signature: sha256=…` against the
raw bytes) or by polling:

```python
job_id = client.submit("meeting.mp3")       # returns immediately, no waiting
# ...or have a webhook notified instead of polling:
client.submit("meeting.mp3", callback_url="https://you.example.com/hook")

status = client.get_job_status(job_id)      # .status: processing|completed|failed
if status.is_completed:
    result = client.get_transcript(job_id)  # downloads + parses
page = client.list_jobs(limit=50)           # {"jobs": [...], "next_before": ...}
```

The signature is HMAC-SHA256 over the raw request body, keyed with **your account's own
webhook signing secret** — find it under API Keys in the
[console](https://console.speechrevolutions.com). Compare it with a constant-time function,
and verify against the bytes you received rather than a re-serialised copy. Worked receivers
for Python, Node, Go and C# are in the
[webhooks guide](https://docs.speechrevolutions.com/guides/webhooks).

See `examples/` for a full submit/poll/webhook walkthrough.

## Robustness

`SpeechRevolutions(max_retries=3, retry_backoff=0.5, proxies={"https": "..."})`.
Transient 429/5xx/network errors are retried (honoring `Retry-After`). Errors are
typed (`RateLimitError`, `AuthenticationError`, …) and carry `.status_code` and
`.request_id` for correlating with support.

## Auth

```bash
export SPEECHREVOLUTIONS_API_KEY=stt_...
```

Or `SpeechRevolutions(api_key="stt_...")`.

## Other languages

Speech Revolutions also publishes SDKs for
[JavaScript/TypeScript](https://github.com/SpeechRevolutions/node-sdk),
[Go](https://github.com/SpeechRevolutions/speechrevolutions-go), and
[C#/.NET](https://github.com/SpeechRevolutions/csharp-sdk) — see
[docs.speechrevolutions.com](https://docs.speechrevolutions.com) for a
cross-language feature comparison.

## License

MIT
