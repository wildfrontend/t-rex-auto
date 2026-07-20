"""Simple detector facade and detector implementation exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dino_bot.config import load_config
from dino_bot.detection import (
    CompositeDetector,
    DetectorAssetError,
    HuntCapacityDetector,
    HuntTeamAvailabilityDetector,
    OpenCvDetector,
    TargetTooStrongDetector,
)
from dino_bot.interfaces import Detector
from dino_bot.models import Detection, Frame

_detector: Detector | None = None


def configure(detector: Detector) -> None:
    global _detector
    _detector = detector


def _default_detector() -> Detector:
    config = load_config(Path(__file__).with_name("config.json"))
    return CompositeDetector(
        OpenCvDetector(
            config.detector.manifest,
            config.detector.default_threshold,
            config.detector.nms_iou,
        ),
        HuntTeamAvailabilityDetector(),
        HuntCapacityDetector(),
        TargetTooStrongDetector(),
    )


def detect(frame: Frame | np.ndarray) -> list[Detection]:
    global _detector
    if _detector is None:
        _detector = _default_detector()
    normalized = frame if isinstance(frame, Frame) else Frame(frame)
    return _detector.detect(normalized)

__all__ = [
    "CompositeDetector",
    "DetectorAssetError",
    "OpenCvDetector",
    "configure",
    "detect",
]
