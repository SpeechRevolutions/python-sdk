"""Optional console progress rendering for transcription jobs.

Neither AssemblyAI nor Deepgram surfaces live percentage progress for
pre-recorded transcription — our pipeline is chunked and emits SSE progress
events, so this is a Speech Revolutions extra. The same renderer also drives
the byte-level upload bar.

``ProgressPrinter`` renders a ``tqdm`` bar when tqdm is installed, and falls
back to a plain textual bar otherwise, while always forwarding events to a
user-supplied callback.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from speechrevolutions.models import ProgressCallback, ProgressEvent

_DEFAULT_LABEL = "Transcribing"
# Minimum seconds between redraws (a byte upload fires ~thousands of events).
_MIN_REDRAW_INTERVAL = 0.08


class ProgressPrinter:
    """A progress callback that renders to the console and forwards events.

    Wrap an optional user callback::

        printer = ProgressPrinter(forward=on_progress, label="Uploading")
        client.wait_for_result(..., on_progress=printer)
        printer.close()

    ``bytes_mode`` renders sizes (e.g. ``2.5MB/6.0MB``) instead of a bare bar.
    """

    def __init__(
        self,
        *,
        forward: ProgressCallback | None = None,
        label: str = _DEFAULT_LABEL,
        bytes_mode: bool = False,
    ) -> None:
        self._forward = forward
        self._label = label
        self._bytes_mode = bytes_mode
        self._bar: Any = None
        self._last_pct: float | None = None
        self._last_draw: float | None = None
        self._closed = False
        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception:
            tqdm = None
        self._tqdm = tqdm

    def __call__(self, event: ProgressEvent) -> None:
        try:
            self._render(event)
        finally:
            if self._forward is not None:
                self._forward(event)

    def _render(self, event: ProgressEvent) -> None:
        if self._tqdm is not None:
            self._render_tqdm(event)
        else:
            self._render_plain(event)

    def _render_tqdm(self, event: ProgressEvent) -> None:
        # A single, stable bar. For transcription the step name (preprocess /
        # chunk:N / aggregation) is deliberately kept off the bar — chunks
        # finish out of order and make the label jump around; callers who want
        # it read event.step in their callback.
        total = event.total
        if self._bar is None:
            if self._bytes_mode:
                bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
            else:
                bar_format = "{desc}: {percentage:3.0f}%|{bar}| [{elapsed}]"
            self._bar = self._tqdm(
                total=total or None,
                desc=self._label,
                leave=True,
                unit="B" if self._bytes_mode else "it",
                unit_scale=self._bytes_mode,
                unit_divisor=1024,  # 1024-based sizes (MiB) in bytes mode
                bar_format=bar_format,
            )
        # The server may revise the total as chunks are discovered.
        if total and self._bar.total != total:
            self._bar.total = total
        if event.completed is not None:
            # tqdm normally counts increments; set the absolute position.
            self._bar.n = min(event.completed, total) if total else event.completed
        # Throttle redraws (uploads emit thousands of events); always draw the
        # final 100% frame.
        now = time.monotonic()
        complete = bool(total) and event.completed is not None and event.completed >= total
        if complete or self._last_draw is None or now - self._last_draw >= _MIN_REDRAW_INTERVAL:
            self._bar.refresh()
            self._last_draw = now

    def _render_plain(self, event: ProgressEvent) -> None:
        pct = event.percent
        if pct is None:
            return
        # Avoid spamming near-identical lines when events arrive rapidly.
        if self._last_pct is not None and pct < 100 and abs(pct - self._last_pct) < 1.0:
            return
        self._last_pct = pct
        width = 30
        filled = int(width * pct / 100)
        bar = "#" * filled + "-" * (width - filled)
        # Carriage return keeps it on one line, like the tqdm bar.
        end = "\n" if pct >= 100 else ""
        print(f"\r{self._label}: {pct:3.0f}% [{bar}]", end=end, file=sys.stderr, flush=True)

    def close(self) -> None:
        """Finish the bar. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            if self._bar.total:
                self._bar.n = self._bar.total
                self._bar.refresh()
            self._bar.close()


def resolve_progress(
    on_progress: ProgressCallback | None,
    show: bool,
    *,
    label: str = _DEFAULT_LABEL,
    bytes_mode: bool = False,
) -> tuple[ProgressCallback | None, ProgressPrinter | None]:
    """Build the effective progress callback for an upload/transcription phase.

    When ``show`` is true, wrap ``on_progress`` in a :class:`ProgressPrinter`
    that renders to the console and still forwards to the user callback. The
    returned printer (or ``None``) must have ``.close()`` called when done.
    """
    if not show:
        return on_progress, None
    printer = ProgressPrinter(forward=on_progress, label=label, bytes_mode=bytes_mode)
    return printer, printer
