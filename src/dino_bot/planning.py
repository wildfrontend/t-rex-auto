"""Reusable target-selection strategies."""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot

from .models import Detection, Frame, Target


class TargetPlanner:
    def __init__(
        self,
        target_types: Sequence[str] = ("resource",),
        strategy: str = "nearest_center",
    ) -> None:
        self.target_types = frozenset(target_types)
        self.strategy = strategy

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        candidates = [item for item in detections if item.type in self.target_types]
        if not candidates:
            return None
        if self.strategy == "highest_confidence":
            selected = max(candidates, key=lambda item: item.confidence)
        else:
            center_x, center_y = frame.width / 2, frame.height / 2
            selected = min(
                candidates,
                key=lambda item: (
                    hypot(item.x - center_x, item.y - center_y),
                    -item.confidence,
                ),
            )
        return Target(
            type=selected.type,
            x=selected.x,
            y=selected.y,
            confidence=selected.confidence,
            detection=selected,
        )
