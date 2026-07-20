"""Replaceable boundaries between core logic and platform/feature implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import (
    ActionCommand,
    ActionRecord,
    Detection,
    Frame,
    Target,
    VerificationResult,
)


class CaptureProvider(Protocol):
    def capture(self) -> Frame: ...

    def close(self) -> None: ...


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


class Planner(Protocol):
    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None: ...


class ActionDriver(Protocol):
    def execute(self, action: ActionCommand, frame: Frame) -> None: ...


class Verifier(Protocol):
    def verify(
        self,
        before: Frame,
        after: Frame,
        target: Target,
        before_detections: Sequence[Detection],
        after_detections: Sequence[Detection],
    ) -> VerificationResult: ...


class ModeObserver(Protocol):
    def on_frame(self, frame: Frame) -> None: ...

    def on_action_complete(self, record: ActionRecord, before: Frame, after: Frame) -> None: ...

    def close(self) -> None: ...
