"""Machine-readable event stream written alongside the human log.

The daily ``.log`` file is written for a person reading over someone's
shoulder, and it drops the values needed to explain a decision afterwards:
detections appear as counts without coordinates, a planner that rejects every
candidate prints one line regardless of which of its eight conditions did the
rejecting, and layout detectors keep their deciding ratios to themselves.
Reconstructing a single stuck session from that meant reverse-engineering
confidence values by hand.

This module records the same run as one JSON object per event so the reasoning
can be queried instead of inferred. It is deliberately append-only, bounded,
and free of anything a person would mind sharing - coordinates, ratios and
type names only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

from .models import ActionCommand, Detection, Frame, Target, VerificationResult

SCHEMA_VERSION = 1


class EventLog(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...

    def close(self) -> None: ...


class NullEventLog:
    """Drop every event; used when the feature is switched off."""

    def emit(self, event: str, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlEventLog:
    """Append one JSON object per line to a daily ``events-YYYYMMDD.jsonl``.

    A bot left running for days must not fill a disk, so the file rolls over
    once at the size cap: the current file becomes ``.1`` and a fresh one
    starts. That keeps between one and two caps' worth of history, where
    truncating in place would leave nothing right after the roll - which is
    exactly when someone reaches for it.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int = 16 * 1024 * 1024,
        encoding: str = "utf-8",
    ) -> None:
        self.directory = directory
        self.max_bytes = max(0, max_bytes)
        self.encoding = encoding
        self._date = ""
        self._stream: TextIO | None = None
        self._written = 0
        self._sequence = 0

    @property
    def cycle(self) -> int:
        return self._sequence

    def start_cycle(self) -> int:
        self._sequence += 1
        return self._sequence

    def _ensure_stream(self) -> TextIO:
        today = datetime.now().strftime("%Y%m%d")
        if self._stream is None or self._date != today:
            self._close_stream()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"events-{today}.jsonl"
            self._written = path.stat().st_size if path.exists() else 0
            self._stream = path.open("a", encoding=self.encoding, buffering=1)
            self._date = today
        elif self.max_bytes and self._written >= self.max_bytes:
            self._close_stream()
            path = self.directory / f"events-{self._date}.jsonl"
            # Sorts before the live file, so readers taking the newest lines
            # walk the current file first and fall back to this one.
            previous = self.directory / f"events-{self._date}.1.jsonl"
            try:
                previous.unlink(missing_ok=True)
                path.replace(previous)
            except OSError:
                pass
            self._stream = path.open("w", encoding=self.encoding, buffering=1)
            self._written = 0
        assert self._stream is not None
        return self._stream

    def emit(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "t": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "c": self._sequence,
            "e": event,
        }
        record.update({key: value for key, value in fields.items() if value is not None})
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            stream = self._ensure_stream()
            stream.write(line + "\n")
            self._written += len(line) + 1
        except (OSError, TypeError, ValueError):
            # Diagnostics must never take the bot down with them.
            return None

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def close(self) -> None:
        self._close_stream()


# A single frame has been seen carrying 111 route markers. Recording every one
# of them would make the stream mostly noise, so the list is capped and the
# true count reported alongside it.
MAX_DETECTIONS_PER_EVENT = 40


def detection_payload(
    detections: Sequence[Detection],
    *,
    limit: int = MAX_DETECTIONS_PER_EVENT,
) -> list[dict[str, Any]]:
    """Render detections with the coordinates the text log leaves out."""

    payload: list[dict[str, Any]] = []
    for item in detections[:limit]:
        entry: dict[str, Any] = {
            "type": item.type,
            "x": item.x,
            "y": item.y,
            "conf": round(float(item.confidence), 4),
        }
        if item.metadata:
            # The deciding ratios of the layout detectors live here and are
            # the only record of why one of them fired.
            entry["meta"] = {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in item.metadata.items()
            }
        payload.append(entry)
    return payload


def target_payload(target: Target | None) -> dict[str, Any] | None:
    if target is None:
        return None
    return {
        "type": target.type,
        "x": target.x,
        "y": target.y,
        "conf": round(float(target.confidence), 4),
    }


def action_payload(action: ActionCommand | None) -> dict[str, Any] | None:
    if action is None:
        return None
    payload: dict[str, Any] = {"kind": action.kind.value}
    if action.x is not None:
        payload["x"] = action.x
    if action.y is not None:
        payload["y"] = action.y
    return payload


def verification_payload(result: VerificationResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": result.success, "reason": result.reason}
    if result.pixel_change is not None:
        payload["pixel_change"] = round(float(result.pixel_change), 4)
    return payload


def frame_payload(frame: Frame) -> dict[str, Any]:
    return {"w": frame.width, "h": frame.height, "seq": frame.sequence}
