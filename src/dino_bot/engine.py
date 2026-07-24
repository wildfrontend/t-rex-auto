"""State-machine implementation of the Sense -> Think -> Act loop."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .interfaces import (
    ActionDriver,
    CaptureProvider,
    Detector,
    ModeObserver,
    Planner,
    RuntimeRecovery,
    Verifier,
)
from .models import (
    ActionCommand,
    ActionKind,
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
    post_action_delays_ms: dict[str, int] = field(default_factory=dict)
    target_action_kinds: dict[str, ActionKind] = field(default_factory=dict)
    idle_delay_ms: int = 500
    transition_poll_interval_ms: int = 250
    verify_retries: int = 3
    max_actions: int = 0
    max_cycles: int = 0
    cycle_complete_targets: tuple[str, ...] = ()
    runtime_recovery: RuntimeRecovery | None = None
    state: BotState = BotState.IDLE
    stop_requested: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
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
    cycle_count: int = 0
    verification_deadline: float | None = None


class StateHandler(Protocol):
    def execute(self, context: BotContext) -> BotState: ...


def _wait_for_delay(context: BotContext, delay_ms: int) -> bool:
    """Wait for a delay and return True when a stop request interrupts it."""

    if context.stop_requested or context.stop_event.is_set():
        return True
    return context.stop_event.wait(max(0, delay_ms) / 1000)


class IdleState:
    def execute(self, context: BotContext) -> BotState:
        if context.stop_requested or context.stop_event.is_set():
            return BotState.STOPPED
        if context.max_actions and context.action_count >= context.max_actions:
            context.logger.info("Stop | max_actions=%d reached", context.max_actions)
            return BotState.STOPPED
        if context.max_cycles and context.cycle_count >= context.max_cycles:
            context.logger.info("Stop | max_cycles=%d reached", context.max_cycles)
            return BotState.STOPPED
        return BotState.CAPTURE


class CaptureState:
    def execute(self, context: BotContext) -> BotState:
        frame = context.capture_provider.capture()
        context.observer.on_frame(frame)
        context.frame = frame
        context.logger.debug("Capture | %dx%d | #%d", frame.width, frame.height, frame.sequence)
        if context.runtime_recovery is not None:
            if context.runtime_recovery.observe(frame):
                _reset_after_runtime_recovery(context)
                return BotState.IDLE
            if _runtime_recovery_is_blocking(context.runtime_recovery):
                if context.idle_delay_ms and _wait_for_delay(
                    context, context.idle_delay_ms
                ):
                    return BotState.STOPPED
                return BotState.IDLE
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
            delay_ms = context.idle_delay_ms
            next_ready_delay = getattr(context.planner, "next_ready_delay_ms", None)
            if callable(next_ready_delay):
                cooldown_ms = int(next_ready_delay())
                if cooldown_ms > delay_ms:
                    context.logger.info(
                        "Planning | cooldown | remaining=%dms",
                        cooldown_ms,
                    )
                delay_ms = max(delay_ms, cooldown_ms)
            if delay_ms and _wait_for_delay(context, delay_ms):
                return BotState.STOPPED
            return BotState.IDLE
        action_kind = context.target_action_kinds.get(
            context.target.type,
            ActionKind.TAP,
        )
        context.action = (
            ActionCommand.back()
            if action_kind == ActionKind.BACK
            else ActionCommand.tap(context.target.x, context.target.y)
        )
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
        if context.action.x is None or context.action.y is None:
            context.logger.info(
                "Action | %s | attempt=%d",
                context.action.kind.value,
                context.attempt,
            )
        else:
            context.logger.info(
                "Action | %s (%d,%d) | attempt=%d",
                context.action.kind.value,
                context.action.x,
                context.action.y,
                context.attempt,
            )
        context.action_driver.execute(context.action, context.frame)
        context.action_count += 1
        delay_ms = context.post_action_delays_ms.get(
            context.target.type,
            context.click_delay_ms,
        )
        context.verification_deadline = context.clock() + max(0, delay_ms) / 1000
        initial_poll_ms = min(
            max(0, delay_ms),
            max(1, context.transition_poll_interval_ms),
        )
        if initial_poll_ms:
            context.logger.debug(
                "Action | poll in %dms; transition timeout=%dms | target=%s",
                initial_poll_ms,
                max(0, delay_ms),
                context.target.type,
            )
            if _wait_for_delay(context, initial_poll_ms):
                context.logger.info("Stop | interrupted post-action wait")
                return BotState.STOPPED
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
        if context.runtime_recovery is not None:
            if context.runtime_recovery.observe(after):
                context.after_frame = after
                context.after_detections = []
                _reset_after_runtime_recovery(context)
                return BotState.IDLE
            if _runtime_recovery_is_blocking(context.runtime_recovery):
                context.after_frame = after
                context.after_detections = []
                if context.idle_delay_ms and _wait_for_delay(
                    context, context.idle_delay_ms
                ):
                    return BotState.STOPPED
                return BotState.VERIFY
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
        explicit_failure = result.reason.startswith("failure indicator detected:")
        deadline = context.verification_deadline
        if (
            not result.success
            and not explicit_failure
            and deadline is not None
            and context.clock() < deadline
        ):
            now = context.clock()
            remaining_ms = max(1, round((deadline - now) * 1000))
            poll_ms = min(max(1, context.transition_poll_interval_ms), remaining_ms)
            context.logger.debug(
                "Verify | Pending | %s | poll=%dms | remaining=%dms",
                result.reason,
                poll_ms,
                remaining_ms,
            )
            if _wait_for_delay(context, poll_ms):
                return BotState.STOPPED
            return BotState.VERIFY
        context.verification_deadline = None
        record = ActionRecord(
            timestamp=utc_now(),
            action=context.action,
            target=context.target,
            result=result,
            attempt=context.attempt,
        )
        context.observer.on_action_complete(record, context.before_frame, after)
        if result.success:
            on_action_success = getattr(context.planner, "on_action_success", None)
            if callable(on_action_success):
                on_action_success(context.target.type)
            context.logger.info("Verify | Success | %s", result.reason)
            if context.target.type in context.cycle_complete_targets:
                context.cycle_count += 1
                context.logger.info(
                    "Workflow | completed cycle %d/%s",
                    context.cycle_count,
                    context.max_cycles or "unlimited",
                )
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


def _reset_after_runtime_recovery(context: BotContext) -> None:
    context.logger.info("Recovery | clearing transient workflow state")
    reset_workflow = getattr(context.planner, "reset_workflow", None)
    if callable(reset_workflow):
        reset_workflow()
    context.frame = None
    context.detections = []
    context.target = None
    context.action = None
    context.before_frame = None
    context.before_detections = []
    context.after_frame = None
    context.after_detections = []
    context.last_result = None
    context.attempt = 0
    context.verification_deadline = None


def _runtime_recovery_is_blocking(runtime_recovery: RuntimeRecovery) -> bool:
    """Return True while recovery is intentionally holding a black frame."""

    return bool(getattr(runtime_recovery, "is_black", False))


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
        self.context.stop_event.set()

    def close(self) -> None:
        self.context.capture_provider.close()
        self.context.observer.close()
        self.context.logger.info(
            "Bot stopped | actions=%d | cycles=%d",
            self.context.action_count,
            self.context.cycle_count,
        )
