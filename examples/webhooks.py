"""Webhooks: get notified when a job finishes, instead of waiting.

Pass ``callback_url`` and the platform POSTs a signed JSON notification to it on
completion or permanent failure — no need to hold a connection open. This is the
right pattern for server/background workloads.

The POST body is:
    {"job_id": "...", "status": "completed"|"failed",
     "download_url": "...",              # on completed
     "step": "...", "reason": "..."}     # on failed
and is signed with HMAC-SHA256 over the raw body in the header
``X-SR-Signature: sha256=<hex>`` (plus a unique ``X-SR-Delivery`` id).

This file has two parts: submitting a job with a callback, and a receiver that
verifies the signature. Verifying the signature is important — it proves the
request really came from us.
"""

import hashlib
import hmac

from speechrevolutions import SpeechRevolutions


# --- 1. Submit a job with a webhook -----------------------------------------
def submit(audio: str, callback_url: str) -> None:
    client = SpeechRevolutions()  # reads SPEECHREVOLUTIONS_API_KEY
    # transcribe() still waits for the result here; if you only want the webhook,
    # use the lower-level create/upload/complete calls and return immediately.
    client.transcribe(audio, callback_url=callback_url)


# --- 2. Verify an incoming webhook (framework-agnostic) ---------------------
def verify_signature(raw_body: bytes, signature_header: str, signing_secret: str) -> bool:
    """Return True if X-SR-Signature matches the raw request body.

    ALWAYS compare against the raw bytes you received — not a re-serialized dict —
    and use a constant-time comparison.
    """
    expected = "sha256=" + hmac.new(signing_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


# --- FastAPI receiver sketch ------------------------------------------------
# import os, json
# from fastapi import FastAPI, Request, HTTPException
# app = FastAPI()
# SECRET = os.environ["SR_WEBHOOK_SECRET"]  # the same secret configured server-side
#
# @app.post("/webhooks/speechrevolutions")
# async def receive(request: Request):
#     raw = await request.body()
#     if not verify_signature(raw, request.headers.get("X-SR-Signature", ""), SECRET):
#         raise HTTPException(status_code=401, detail="bad signature")
#     event = json.loads(raw)
#     if event["status"] == "completed":
#         ...  # mark the job done in your DB; fetch event["download_url"]
#     else:
#         ...  # event["status"] == "failed": event["step"], event["reason"]
#     return {"ok": True}  # 2xx tells us delivery succeeded (we retry on 5xx)


if __name__ == "__main__":
    # Point this at a URL you control (e.g. an ngrok tunnel to the receiver above).
    submit("audio.mp3", callback_url="https://your-app.example.com/webhooks/speechrevolutions")
    print("Submitted. Your callback_url will be POSTed when the job finishes.")
