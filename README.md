# Speech Revolutions — Python SDK

Official Python client for the [Speech Revolutions](https://speechrevolutions.com) speech-to-text API.

## Install

```bash
pip install speechrevolutions
```

For console progress bars, install the optional `tqdm` extra:

```bash
pip install "speechrevolutions[progress]"
```

## Quick start

```python
from speechrevolutions import SpeechRevolutions

client = SpeechRevolutions()  # reads SPEECHREVOLUTIONS_API_KEY
result = client.transcribe("meeting.mp3", speaker_labels=True)
print(result.text)

for u in result.utterances:
    print(f"Speaker {u.speaker}: {u.text}")
```

### Async

```python
from speechrevolutions import AsyncSpeechRevolutions

async with AsyncSpeechRevolutions() as client:
    result = await client.transcribe("meeting.mp3", speaker_labels=True)
    print(result.text)
```

### From a URL (Deepgram-style)

```python
result = client.transcribe_url("https://example.com/audio.mp3")
# or
result = client.transcribe("https://example.com/audio.mp3")
```

### Options as kwargs or config object

```python
# kwargs (ElevenLabs / Deepgram style). `diarize` is an alias for `speaker_labels`.
result = client.transcribe("a.mp3", diarize=True, output_type="json")

# config object (AssemblyAI style)
from speechrevolutions import TranscribeOptions
result = client.transcribe("a.mp3", options=TranscribeOptions(speaker_labels=True))
```

Defaults: `output_type="json"`, `word_timestamps`, `speaker_labels`, `nltk` all
`True`, `tier="standard"`, `custom_vocabulary=None`.

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
    print(event.percent, event.step)      # e.g. 42.0 "transcribe"

def on_upload(event):          # upload (event.step == "upload")
    print("upload", event.percent)

result = client.transcribe(
    "meeting.mp3",
    on_progress=on_progress,
    on_upload_progress=on_upload,
)
```

`progress=True` and the callbacks compose — the bars render *and* your callbacks
still fire for every event.

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
dg = result.to_deepgram()
print(dg["results"]["channels"][0]["alternatives"][0]["transcript"])
```

## Webhooks & retrieving results later

`submit()` uploads and enqueues a job and returns its id **without waiting** —
ideal for batch/background work. Collect the result later via a webhook
(`callback_url`, a signed POST — verify `X-SR-Signature: sha256=…` against the
raw bytes) or by polling:

```python
job_id = client.submit("meeting.mp3")       # returns immediately, no waiting
# ...or notify a webhook instead of polling:
client.transcribe("meeting.mp3", callback_url="https://you.example.com/hook")

status = client.get_job_status(job_id)      # .status: processing|completed|failed
if status.is_completed:
    result = client.get_transcript(job_id)  # downloads + parses
page = client.list_jobs(limit=50)           # {"jobs": [...], "next_before": ...}
```

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
