"""State-machine implementation of the Sense -> Think -> Act loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .interfaces import ActionDriver, CaptureProvider, Detector, ModeObserver, Planner, Verifier
from .models import (
    ActionCommand,
    ActionRecord,
    Detection,
    Frame,
    Target,
    VerificationResult,
    utc_now,
)


class BotState(StrEnum):
    IDLE = "idle"
    CAPTURE = "capture"
    DETECT = "detect"
    PLANNING = "planning"
    ACTION = "action"
    VERIFY = "verify"
    RECOVER = "recover"
    STOPPED = "stopped"


@dataclass(slots=True)
class BotContext:
    capture_provider: CaptureProvider
    detector: Detector
    planner: Planner
    action_driver: ActionDriver
    verifier: Verifier
    observer: ModeObserver
    logger: logging.Logger
    click_delay_ms: int = 200
    idle_delay_ms: int = 500
    verify_retries: int = 3
    max_actions: int = 0
    state: BotState = BotState.IDLE
    stop_requested: bool = False
    frame: Frame | None = None
    detections: list[Detection] = field(default_factory=list)
    target: Target | None = None
    action: ActionCommand | None = None
    before_frame: Frame | None = None
    before_detections: list[Detection] = field(default_factory=list)
    after_frame: Frame | None = None
    after_detections: list[Detection] = field(default_factory=list)
    last_result: VerificationResult | None = None
    attempt: int = 0
    action_count: int = 0


class StateHandler(Protocol):
    def execute(self, context: BotContext) -> BotState: ...


class IdleState:
    def execute(self, context: BotContext) -> BotState:
        if context.stop_requested:
            return BotState.STOPPED
        if context.max_actions and context.action_count >= context.max_actions:
            context.logger.info("Stop | max_actions=%d reached", context.max_actions)
            return BotState.STOPPED
        return BotState.CAPTURE


class CaptureState:
    def execute(self, context: BotContext) -> BotState:
        frame = context.capture_provider.capture()
        context.observer.on_frame(frame)
        context.frame = frame
        context.logger.debug("Capture | %dx%d | #%d", frame.width, frame.height, frame.sequence)
        return BotState.DETECT


class DetectState:
    def execute(self, context: BotContext) -> BotState:
        if context.frame is None:
            raise RuntimeError("Detect state entered without a frame")
        context.detections = context.detector.detect(context.frame)
        counts: dict[str, int] = {}
        for item in context.detections:
            counts[item.type] = counts.get(item.type, 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
        context.logger.info("Detect | %s", summary or "no targets")
        return BotState.PLANNING


class PlanningState:
    def execute(self, context: BotContext) -> BotState:
        if context.frame is None:
            raise RuntimeError("Planning state entered without a frame")
        context.target = context.planner.choose(context.frame, context.detections)
        if context.target is None:
            context.logger.debug("Planning | no actionable target")
            if context.idle_delay_ms:
                time.sleep(context.idle_delay_ms / 1000)
            return BotState.IDLE
        context.action = ActionCommand.tap(context.target.x, context.target.y)
        context.logger.info(
            "Planning | %s at (%d,%d) confidence=%.3f",
            context.target.type,
            context.target.x,
            context.target.y,
            context.target.confidence,
        )
        return BotState.ACTION


class ActionState:
    def execute(self, context: BotContext) -> BotState:
        if context.frame is None or context.target is None or context.action is None:
            raise RuntimeError("Action state entered without frame, target, or command")
        context.before_frame = context.frame
        context.before_detections = list(context.detections)
        context.attempt += 1
        context.logger.info(
            "Action | %s (%d,%d) | attempt=%d",
            context.action.kind.value,
            context.action.x,
            context.action.y,
            context.attempt,
        )
        context.action_driver.execute(context.action, context.frame)
        context.action_count += 1
        if context.click_delay_ms:
            time.sleep(context.click_delay_ms / 1000)
        return BotState.VERIFY


class VerifyState:
    def execute(self, context: BotContext) -> BotState:
        if (
            context.before_frame is None
            or context.target is None
            or context.action is None
        ):
            raise RuntimeError("Verify state entered without a pending action")
        after = context.capture_provider.capture()
        context.observer.on_frame(after)
        after_detections = context.detector.detect(after)
        context.after_frame = after
        context.after_detections = after_detections
        result = context.verifier.verify(
            context.before_frame,
            after,
            context.target,
            context.before_detections,
            after_detections,
        )
        context.last_result = result
        record = ActionRecord(
            timestamp=utc_now(),
            action=context.action,
            target=context.target,
            result=result,
            attempt=context.attempt,
        )
        context.observer.on_action_complete(record, context.before_frame, after)
        if result.success:
            context.logger.info("Verify | Success | %s", result.reason)
            context.attempt = 0
            context.frame = after
            context.detections = after_detections
            return BotState.IDLE
        context.logger.warning("Verify | Failed | %s", result.reason)
        if context.max_actions and context.action_count >= context.max_actions:
            context.logger.info("Verify | retry skipped because max_actions was reached")
            context.attempt = 0
            return BotState.IDLE
        if context.attempt <= context.verify_retries:
            return BotState.RECOVER
        context.logger.error("Verify | retry limit exhausted after %d attempts", context.attempt)
        context.attempt = 0
        return BotState.IDLE


class RecoverState:
    def execute(self, context: BotContext) -> BotState:
        context.logger.info("Recover | refresh and re-plan")
        context.frame = context.after_frame
        context.detections = list(context.after_detections)
        context.target = None
        context.action = None
        return BotState.CAPTURE


class StoppedState:
    def execute(self, context: BotContext) -> BotState:
        return BotState.STOPPED


DEFAULT_STATES: dict[BotState, StateHandler] = {
    BotState.IDLE: IdleState(),
    BotState.CAPTURE: CaptureState(),
    BotState.DETECT: DetectState(),
    BotState.PLANNING: PlanningState(),
    BotState.ACTION: ActionState(),
    BotState.VERIFY: VerifyState(),
    BotState.RECOVER: RecoverState(),
    BotState.STOPPED: StoppedState(),
}


class BotEngine:
    def __init__(
        self,
        context: BotContext,
        states: dict[BotState, StateHandler] | None = None,
    ) -> None:
        self.context = context
        self.states = states or DEFAULT_STATES

    def step(self) -> BotState:
        handler = self.states[self.context.state]
        self.context.state = handler.execute(self.context)
        return self.context.state

    def run(self) -> None:
        self.context.logger.info("Bot started | Sense -> Think -> Act")
        try:
            while self.context.state != BotState.STOPPED:
                self.step()
        except KeyboardInterrupt:
            self.context.logger.info("Stop requested by user")
            self.context.state = BotState.STOPPED
        finally:
            self.close()

    def stop(self) -> None:
        self.context.stop_requested = True

    def close(self) -> None:
        self.context.capture_provider.close()
        self.context.observer.close()
        self.context.logger.info("Bot stopped | actions=%d", self.context.action_count)
