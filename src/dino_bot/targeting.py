"""Small, shared helpers for turning detections into planner targets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Detection, Target


def best_detection(items: Iterable[Detection] | None) -> Detection | None:
    """Return the highest-confidence detection, including for an empty input."""

    if items is None:
        return None
    return max(items, key=lambda item: item.confidence, default=None)


def detection_target(detection: Detection, *, target_type: str | None = None) -> Target:
    """Preserve detection evidence while optionally naming a workflow action."""

    return Target(
        type=detection.type if target_type is None else target_type,
        x=detection.x,
        y=detection.y,
        confidence=detection.confidence,
        detection=detection,
    )


def synthetic_target(
    target_type: str,
    x: int,
    y: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> Target:
    """Create a fixed-coordinate action with explicit synthetic provenance."""

    detection = Detection(
        type=target_type,
        x=x,
        y=y,
        confidence=1.0,
        metadata={"synthetic": True, **(metadata or {})},
    )
    return detection_target(detection)


def swipe_target(
    target_type: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    duration_ms: int = 400,
) -> Target:
    """Create a synthetic swipe action using the target metadata contract."""

    return synthetic_target(
        target_type,
        x1,
        y1,
        metadata={"swipe": {"x2": x2, "y2": y2, "duration_ms": duration_ms}},
    )
