"""Post-action verification rules."""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot

import cv2
import numpy as np

from .models import Detection, Frame, Target, VerificationResult


class TargetChangedVerifier:
    def __init__(
        self,
        max_distance: float = 35.0,
        pixel_change_threshold: float = 0.08,
        failure_types: Sequence[str] = (),
    ):
        self.max_distance = max_distance
        self.pixel_change_threshold = pixel_change_threshold
        self.failure_types = frozenset(failure_types)

    def verify(
        self,
        before: Frame,
        after: Frame,
        target: Target,
        before_detections: Sequence[Detection],
        after_detections: Sequence[Detection],
    ) -> VerificationResult:
        failures = sorted(
            {item.type for item in after_detections if item.type in self.failure_types}
        )
        if failures:
            return VerificationResult(
                success=False,
                reason=f"failure indicator detected: {', '.join(failures)}",
                confidence=1.0,
            )
        nearby = [
            item
            for item in after_detections
            if item.type == target.type
            and hypot(item.x - target.x, item.y - target.y) <= self.max_distance
        ]
        change = self._target_region_change(before, after, target)
        if not nearby:
            return VerificationResult(
                success=True,
                reason=f"target disappeared; pixel_change={change:.3f}",
                confidence=max(0.75, min(0.99, 0.75 + change)),
            )
        if change >= self.pixel_change_threshold:
            return VerificationResult(
                success=True,
                reason=(
                    f"target interaction changed UI; pixel_change={change:.3f} "
                    f">= {self.pixel_change_threshold:.3f}"
                ),
                confidence=min(0.95, 0.65 + change),
            )
        best = max(nearby, key=lambda item: item.confidence)
        return VerificationResult(
            success=False,
            reason=(
                f"target still detected at ({best.x},{best.y}); "
                f"confidence={best.confidence:.3f}; pixel_change={change:.3f}"
            ),
            confidence=best.confidence,
        )

    def _target_region_change(self, before: Frame, after: Frame, target: Target) -> float:
        if before.image.shape != after.image.shape:
            return 1.0
        bbox = target.detection.bbox
        if bbox is None:
            half = 24
            x1, y1 = max(0, target.x - half), max(0, target.y - half)
            x2, y2 = min(before.width, target.x + half), min(before.height, target.y + half)
        else:
            margin = 4
            x1, y1 = max(0, bbox.x - margin), max(0, bbox.y - margin)
            x2 = min(before.width, bbox.x + bbox.width + margin)
            y2 = min(before.height, bbox.y + bbox.height + margin)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        left = cv2.cvtColor(before.image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(after.image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(np.mean(cv2.absdiff(left, right)) / 255.0)
