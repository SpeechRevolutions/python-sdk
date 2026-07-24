"""Submit without waiting, then retrieve later: submit() + get-by-id + list.

``submit()`` uploads and enqueues a job and returns its id immediately (no
waiting) — ideal for batch/background work. You collect the result later by
polling ``get_job_status`` / ``get_transcript`` (or via a webhook). Results are
fetchable by id long after upload (the download URL is regenerated on demand).
"""

import time

from speechrevolutions import SpeechRevolutions
from speechrevolutions.exceptions import JobFailedError


def poll_until_done(client: SpeechRevolutions, job_id: str, *, interval: float = 3.0):
    """Poll a job's status until it completes or fails."""
    while True:
        status = client.get_job_status(job_id)  # -> JobStatus(status=..., download_url=...)
        print(f"  status: {status.status}")
        if status.is_completed:
            return client.get_transcript(job_id)  # downloads + parses into a Transcript
        if status.is_failed:
            raise JobFailedError(
                f"job {job_id} failed", step=status.failed_stage, reason=status.reason
            )
        time.sleep(interval)


def main() -> None:
    client = SpeechRevolutions()  # reads SPEECHREVOLUTIONS_API_KEY

    # 1. Fire-and-forget: submit() returns a job_id immediately, without waiting.
    job_id = client.submit("audio.mp3")
    print(f"submitted job {job_id}")

    # 2. Collect later: poll status, then fetch the transcript by id.
    result = poll_until_done(client, job_id)
    print(result.text[:500])

    # (Bonus) list your most-recent jobs (newest first), cursor-paginated.
    page = client.list_jobs(limit=10)
    print(f"\n{len(page['jobs'])} recent job(s); next_before={page['next_before']}")


if __name__ == "__main__":
    main()
