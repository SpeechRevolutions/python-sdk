# Live API tests

Everything else in this repo runs against `tests/mock_api.py`, which is a
**model** of the contract written by reading the server. A model can be wrong in
the same way the client is wrong, and then the suite is green while production
is broken. These tests close that gap.

```sh
SR_LIVE=1 SPEECHREVOLUTIONS_API_KEY=stt_... pytest tests/live -v -m "not slow"   # read-only
SR_LIVE=1 SPEECHREVOLUTIONS_API_KEY=stt_... pytest tests/live -v                 # + real jobs
```

Opt-in, because the non-read-only tests create **real, billable jobs**.

## Webhooks need a public tunnel

`webhook_service._host_is_public()` resolves the callback host and requires every
address to be globally routable outside `local` env — localhost and 127.0.0.1 are
refused before a request is made. So:

```sh
cloudflared tunnel --url http://localhost:8799        # prints https://xxx.trycloudflare.com
SR_LIVE=1 SPEECHREVOLUTIONS_API_KEY=stt_... \
  SR_LIVE_WEBHOOK_BASE=https://xxx.trycloudflare.com SR_LIVE_WEBHOOK_PORT=8799 \
  pytest tests/live -v -k webhook
```

`SR_LIVE_LONG_AUDIO` should point at a file over ten minutes for the progress
test; `SR_LIVE_AUDIO` overrides the short clip used elsewhere.

## What running these against production actually found

The mock was wrong in five places, and every one of them made a test pass
against a shape the API has never returned:

| Mock said | Production says |
|---|---|
| job summary `{job_id, status}` | `{job_id, created_at}` — **no status** |
| `next_before` is a job id | `next_before` is the last row's **created_at timestamp** |
| `GET /jobs/{id}` has no `llm_download_url` | it does, null when absent |
| job ids look like `job_<hex>` | UUIDs |
| webhook payload `{job_id, status}` | also `download_url` (presigned), `duration_seconds`, `rtf` |
| SSE step is `"transcribing"` | `preprocess`, `chunk:0`…`chunk:N`, `aggregation` |
| SSE event id is an integer | a Redis stream id, `"<millis>-<seq>"` |

Two of those changed real conclusions. The webhook payload already carries a
presigned `download_url`, so a receiver does **not** need a second
`get_transcript` call. And progress events are emitted per pipeline step, so a
file under one 600s chunk can legitimately complete with **zero** progress
events — verified live: a 12.8s job emitted only `completed`, a 68-minute job
emitted nine.
