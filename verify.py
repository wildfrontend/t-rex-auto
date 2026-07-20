"""Simple verification facade."""

from __future__ import annotations

from collections.abc import Sequence

from dino_bot.models import Detection, Frame, Target, VerificationResult
from dino_bot.verification import TargetChangedVerifier

_verifier = TargetChangedVerifier()


def configure(verifier: TargetChangedVerifier) -> None:
    global _verifier
    _verifier = verifier


def verify(
    before: Frame,
    after: Frame,
    target: Target,
    before_detections: Sequence[Detection],
    after_detections: Sequence[Detection],
) -> VerificationResult:
    return _verifier.verify(
        before,
        after,
        target,
        before_detections,
        after_detections,
    )

__all__ = ["TargetChangedVerifier", "configure", "verify"]
