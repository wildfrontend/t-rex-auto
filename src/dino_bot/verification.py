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
        success_transitions: dict[str, Sequence[str]] | None = None,
        black_mean_threshold: float = 2.0,
    ):
        self.max_distance = max_distance
        self.pixel_change_threshold = pixel_change_threshold
        self.failure_types = frozenset(failure_types)
        self.success_transitions = {
            target_type: frozenset(successors)
            for target_type, successors in (success_transitions or {}).items()
        }
        self.black_mean_threshold = black_mean_threshold

    def relevant_detection_types(self, target_type: str) -> frozenset[str]:
        expected = self.success_transitions.get(target_type)
        return frozenset(
            {
                *(expected or (target_type,)),
                *self.failure_types,
            }
        )

    def verify(
        self,
        before: Frame,
        after: Frame,
        target: Target,
        before_detections: Sequence[Detection],
        after_detections: Sequence[Detection],
    ) -> VerificationResult:
        sample = after.image[::8, ::8]
        if float(np.mean(sample)) <= self.black_mean_threshold:
            return VerificationResult(
                success=False,
                reason="verification frame is black",
                confidence=1.0,
            )
        failures = sorted(
            {item.type for item in after_detections if item.type in self.failure_types}
        )
        if failures:
            return VerificationResult(
                success=False,
                reason=f"failure indicator detected: {', '.join(failures)}",
                confidence=1.0,
            )
        expected_successors = self.success_transitions.get(target.type, frozenset())
        visible_successors = sorted(
            {item.type for item in after_detections if item.type in expected_successors}
        )
        if visible_successors:
            return VerificationResult(
                success=True,
                reason=f"next UI detected: {', '.join(visible_successors)}",
                confidence=1.0,
            )
        if expected_successors:
            return VerificationResult(
                success=False,
                reason=(
                    "expected next UI not detected: "
                    f"{', '.join(sorted(expected_successors))}"
                ),
                confidence=0.9,
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
