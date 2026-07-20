"""Shared immutable data models used across the bot pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True, slots=True)
class Frame:
    image: Image
    captured_at: datetime = field(default_factory=utc_now)
    source: str = "unknown"
    sequence: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class Detection:
    type: str
    x: int
    y: int
    confidence: float
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bbox(
        cls,
        type: str,
        bbox: BoundingBox,
        confidence: float,
        **metadata: Any,
    ) -> Detection:
        x, y = bbox.center
        return cls(type=type, x=x, y=y, confidence=confidence, bbox=bbox, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "confidence": round(float(self.confidence), 6),
        }
        if self.bbox:
            result["bbox"] = {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True, slots=True)
class Target:
    type: str
    x: int
    y: int
    confidence: float
    detection: Detection


class ActionKind(StrEnum):
    TAP = "tap"
    SWIPE = "swipe"
    LONG_PRESS = "long_press"
    SLEEP = "sleep"


@dataclass(frozen=True, slots=True)
class ActionCommand:
    kind: ActionKind
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    duration_ms: int = 0

    @classmethod
    def tap(cls, x: int, y: int) -> ActionCommand:
        return cls(kind=ActionKind.TAP, x=x, y=y)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    success: bool
    reason: str
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ActionRecord:
    timestamp: datetime
    action: ActionCommand
    target: Target | None
    result: VerificationResult
    attempt: int
