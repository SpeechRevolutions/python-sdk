"""SSE stream parsing helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def parse_sse_stream(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """
    Yield parsed SSE events from an iterable of decoded lines.

    Each yielded dict may contain keys 'id', 'event', and 'data'.
    Comment/heartbeat lines (starting with ':') are skipped.
    """
    event: dict[str, Any] = {}
    for raw_line in lines:
        if raw_line == "":
            if event.get("data") is not None:
                yield event
            event = {}
            continue

        if raw_line.startswith(":"):
            continue

        if ":" in raw_line:
            field, _, value = raw_line.partition(":")
            value = value.lstrip(" ")
        else:
            field, value = raw_line, ""

        if field == "id":
            event["id"] = value
        elif field == "event":
            event["event"] = value
        elif field == "data":
            event["data"] = value
