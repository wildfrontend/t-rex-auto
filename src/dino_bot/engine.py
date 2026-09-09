"""State-machine implementation of the Sense -> Think -> Act loop."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .actions import AdbError
from .capture import AdbCaptureError
from .events import (
    EventLog,
    NullEventLog,
    action_payload,
    detection_payload,
    frame_payload,
    target_payload,
    verification_payload,
)
from .interfaces import (
    ActionDriver,
    CaptureProvider,
    Detector,
    DinosaurFailureRecorder,
    HuntProgressRecovery,
    ModeObserver,
    Planner,
    RuntimeRecovery,
    StallRecorder,
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


_GAME_RESTART_AFTER_SUCCESS: dict[str, tuple[str, str]] = {
    "duplicate_login_close_button": (
        "duplicate login dialog closed",
        "duplicate_login",
    ),
}


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
    verification_minimum_checks: int = 2
    verify_retries: int = 3
    max_actions: int = 0
    max_cycles: int = 0
    cycle_complete_targets: tuple[str, ...] = ()
    runtime_recovery: RuntimeRecovery | None = None
    hunt_progress_recovery: HuntProgressRecovery | None = None
    stall_snapshots: StallRecorder | None = None
    dinosaur_failure_snapshots: DinosaurFailureRecorder | None = None
    event_log: EventLog = field(default_factory=NullEventLog, repr=False)
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
    attempt_target_type: str | None = None
    action_count: int = 0
    cycle_count: int = 0
    verification_timeout_ms: int = 0
    verification_deadline: float | None = None
    verification_started_at: float | None = None
    verification_checks: int = 0
    reuse_verified_detections: bool = False
    inert_taps: int = 0
    inert_target: tuple[str, int, int] | None = None
    escalate_to_back: bool = False
    inert_tap_threshold: int = 2
    inert_pixel_change: float = 0.001
    action_failure_stage_threshold: int = 2
    action_failure_restart_threshold: int = 3
    max_restarts_without_progress: int = 3
    action_stage: str = ""
    action_failure_episodes: dict[tuple[str, str], int] = field(default_factory=dict)
    recovery_restarts_without_progress: int = 0
    action_transport_failures: int = 0
    action_transport_failure_limit: int = 2
    capture_transport_failures: int = 0
    capture_transport_failure_limit: int = 6
    capture_transport_retry_delay_ms: int = 5_000


class StateHandler(Protocol):
    def execute(self, context: BotContext) -> BotState: ...


def _planner_stage(planner: Planner) -> str:
    method = getattr(planner, "last_stage", None)
    return str(method()) if callable(method) else ""


def _failure_stage_key(stage: str) -> str:
    """Drop volatile child detail while preserving the owning workflow stage."""

    parts = stage.split(":")
    if stage.startswith("hatch_hunt_") and len(parts) > 1:
        return ":".join(parts[:2])
    return parts[0]


def _is_recovery_progress(context: BotContext, target_type: str) -> bool:
    method = getattr(context.planner, "is_recovery_progress", None)
    if callable(method):
        return bool(method(target_type))
    return bool(
        target_type in context.cycle_complete_targets
        or target_type == getattr(context.planner, "completion_type", None)
    )


def _record_action_success(context: BotContext, target_type: str) -> None:
    """Clear only recovered behaviour failures; milestones clear the fuse."""

    for key in tuple(context.action_failure_episodes):
        if key[1] == target_type:
            context.action_failure_episodes.pop(key, None)
    if not _is_recovery_progress(context, target_type):
        return
    if context.action_failure_episodes or context.recovery_restarts_without_progress:
        context.logger.info(
            "Recovery | progress milestone restored | target=%s | restart fuse reset",
            target_type,
        )
        context.event_log.emit(
            "failure_escalation",
            level=0,
            action="progress_restored",
            target=target_type,
            restarts=context.recovery_restarts_without_progress,
        )
    context.action_failure_episodes.clear()
    context.recovery_restarts_without_progress = 0


def _request_stage_recovery(
    context: BotContext,
    target: Target,
    *,
    stage: str,
    episodes: int,
) -> bool:
    method = getattr(context.planner, "recover_from_action_failures", None)
    if not callable(method):
        return False
    return bool(
        method(
            target,
            stage,
            episodes,
            context.after_frame,
            context.after_detections,
        )
    )


def _escalate_retry_exhaustion(context: BotContext, target: Target) -> BotState | None:
    """Escalate repeated exhausted retry budgets across every action type."""

    stage = context.action_stage or _planner_stage(context.planner) or "unknown"
    key = (_failure_stage_key(stage), target.type)
    episodes = context.action_failure_episodes.get(key, 0) + 1
    context.action_failure_episodes[key] = episodes
    stage_threshold = max(1, context.action_failure_stage_threshold)
    restart_threshold = max(stage_threshold + 1, context.action_failure_restart_threshold)

    if episodes < stage_threshold:
        level, action = 1, "local_exhausted"
    elif episodes < restart_threshold:
        level = 2
        action = (
            "stage_recovery"
            if _request_stage_recovery(
                context,
                target,
                stage=stage,
                episodes=episodes,
            )
            else "stage_recovery_unavailable"
        )
    else:
        if (
            context.recovery_restarts_without_progress
            >= max(1, context.max_restarts_without_progress)
        ):
            context.logger.critical(
                "Recovery | action failure fuse opened | stage=%s target=%s"
                " episodes=%d restarts=%d | stopping bot",
                stage,
                target.type,
                episodes,
                context.recovery_restarts_without_progress,
            )
            context.event_log.emit(
                "failure_escalation",
                level=5,
                action="stop_bot",
                stage=stage,
                target=target.type,
                episodes=episodes,
                restarts=context.recovery_restarts_without_progress,
            )
            context.stop_requested = True
            context.stop_event.set()
            return BotState.STOPPED

        recovery = context.runtime_recovery
        restarted = bool(
            recovery is not None
            and recovery.request_restart(
                f"repeated action failure at {stage}: {target.type}"
                f" ({episodes} exhausted retry budgets)",
                reason_key="action_failure",
            )
        )
        if restarted:
            context.recovery_restarts_without_progress += 1
            context.logger.error(
                "Recovery | action failure escalation | level=4 stage=%s"
                " target=%s episodes=%d restart=%d/%d",
                stage,
                target.type,
                episodes,
                context.recovery_restarts_without_progress,
                context.max_restarts_without_progress,
            )
            context.event_log.emit(
                "failure_escalation",
                level=4,
                action="restart_game",
                stage=stage,
                target=target.type,
                episodes=episodes,
                restarts=context.recovery_restarts_without_progress,
            )
            _reset_after_runtime_recovery(context)
            return BotState.IDLE

        level = 3
        recovered = _request_stage_recovery(
            context,
            target,
            stage=stage,
            episodes=episodes,
        )
        action = "restart_deferred" if recovery is not None else "restart_unavailable"
        if recovered:
            action += "_stage_recovery"

    context.logger.warning(
        "Recovery | action failure escalation | level=%d action=%s"
        " stage=%s target=%s episodes=%d restarts=%d",
        level,
        action,
        stage,
        target.type,
        episodes,
        context.recovery_restarts_without_progress,
    )
    context.event_log.emit(
        "failure_escalation",
        level=level,
        action=action,
        stage=stage,
        target=target.type,
        episodes=episodes,
        restarts=context.recovery_restarts_without_progress,
    )
    return None


def _build_action(action_kind: ActionKind, target: Target) -> ActionCommand:
    if action_kind == ActionKind.BACK:
        return ActionCommand.back()
    if action_kind == ActionKind.SWIPE:
        swipe = target.detection.metadata.get("swipe") or {}
        x2 = swipe.get("x2")
        y2 = swipe.get("y2")
        if x2 is None or y2 is None:
            # A swipe target without an endpoint is a planner bug; a tap at the
            # start point is the safest thing that still makes progress.
            return ActionCommand.tap(target.x, target.y)
        return ActionCommand(
            kind=ActionKind.SWIPE,
            x=target.x,
            y=target.y,
            x2=int(x2),
            y2=int(y2),
            duration_ms=int(swipe.get("duration_ms", 300)),
        )
    return ActionCommand.tap(target.x, target.y)


def _wait_for_delay(context: BotContext, delay_ms: int) -> bool:
    """Wait for a delay and return True when a stop request interrupts it."""

    if context.stop_requested or context.stop_event.is_set():
        return True
    return context.stop_event.wait(max(0, delay_ms) / 1000)


class IdleState:
    def execute(self, context: BotContext) -> BotState:
        if context.stop_requested or context.stop_event.is_set():
            return BotState.STOPPED
        is_complete = getattr(context.planner, "is_complete", None)
        if callable(is_complete) and is_complete():
            context.logger.info("Stop | planner completed its bounded workflow")
            return BotState.STOPPED
        if context.max_actions and context.action_count >= context.max_actions:
            context.logger.info("Stop | max_actions=%d reached", context.max_actions)
            return BotState.STOPPED
        if context.max_cycles and context.cycle_count >= context.max_cycles:
            context.logger.info("Stop | max_cycles=%d reached", context.max_cycles)
            return BotState.STOPPED
        if context.reuse_verified_detections:
            context.reuse_verified_detections = False
            context.logger.debug("Idle | reuse verified detections")
            return BotState.PLANNING
        return BotState.CAPTURE


class CaptureState:
    def execute(self, context: BotContext) -> BotState:
        started = time.perf_counter()
        frame = context.capture_provider.capture()
        if context.capture_transport_failures:
            context.logger.info("Capture | ADB screenshot recovered; resuming fresh planning")
            context.event_log.emit("capture_transport_recovered")
            context.capture_transport_failures = 0
        capture_ms = round((time.perf_counter() - started) * 1000)
        context.observer.on_frame(frame)
        context.frame = frame
        start_cycle = getattr(context.event_log, "start_cycle", None)
        if callable(start_cycle):
            start_cycle()
        context.event_log.emit("capture", ms=capture_ms, frame=frame_payload(frame))
        context.logger.debug(
            "Capture | %dx%d | #%d | %dms",
            frame.width,
            frame.height,
            frame.sequence,
            capture_ms,
        )
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
        # Most of a planning scan is spent on screens the current stage cannot
        # reach - the launch dialogs alone are a quarter of the bill. When the
        # planner can name what it is able to act on, scan only that; it widens
        # the request itself whenever the narrow view stops paying off.
        scoped_types: frozenset[str] | None = None
        detect_types = getattr(context.detector, "detect_types", None)
        planning_types = getattr(context.planner, "planning_detection_types", None)
        if callable(detect_types) and callable(planning_types):
            scoped_types = planning_types()
        started = time.perf_counter()
        if scoped_types is None:
            context.detections = context.detector.detect(context.frame)
        else:
            context.detections = detect_types(context.frame, scoped_types)
        detect_ms = round((time.perf_counter() - started) * 1000)
        counts: dict[str, int] = {}
        for item in context.detections:
            counts[item.type] = counts.get(item.type, 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
        context.event_log.emit(
            "detect",
            ms=detect_ms,
            n=len(context.detections),
            scoped=len(scoped_types) if scoped_types is not None else None,
            det=detection_payload(context.detections),
        )
        context.logger.info(
            "Detect | %s | %dms | %s",
            summary or "no targets",
            detect_ms,
            f"scoped={len(scoped_types)}" if scoped_types is not None else "full scan",
        )
        # Layout detectors decide from a handful of ratios that never reach the
        # log, so a misfire can only be diagnosed by reverse-engineering the
        # confidence value afterwards. Emit the ratios they already record.
        for item in context.detections:
            if not item.metadata:
                continue
            values = ", ".join(
                f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in sorted(item.metadata.items())
                if key != "detector"
            )
            if values:
                context.logger.debug(
                    "Detect | %s | %s | confidence=%.3f | %s",
                    item.type,
                    item.metadata.get("detector", "unknown"),
                    item.confidence,
                    values,
                )
        return BotState.PLANNING


class PlanningState:
    def execute(self, context: BotContext) -> BotState:
        if context.frame is None:
            raise RuntimeError("Planning state entered without a frame")
        context.target = context.planner.choose(context.frame, context.detections)
        cooldown_ms = 0
        if context.target is None:
            next_ready_delay = getattr(context.planner, "next_ready_delay_ms", None)
            if callable(next_ready_delay):
                cooldown_ms = int(next_ready_delay())
        rejections: dict[str, int] = {}
        last_rejections = getattr(context.planner, "last_rejections", None)
        if callable(last_rejections):
            rejections = last_rejections()
        # Which branch of the planner ran is what separates a cycle spent
        # waiting out a capacity cooldown from one spent settling the map. Both
        # plan nothing, and only one of them is worth tuning.
        stage = ""
        last_stage = getattr(context.planner, "last_stage", None)
        if callable(last_stage):
            stage = str(last_stage())
        idle_ms = 0
        last_idle_seconds = getattr(context.planner, "last_idle_seconds", None)
        if callable(last_idle_seconds):
            idle_ms = round(float(last_idle_seconds()) * 1000)
        recenter_reason = None
        last_recenter_reason = getattr(context.planner, "last_recenter_reason", None)
        if callable(last_recenter_reason):
            recenter_reason = last_recenter_reason()
        blind_ms = 0
        last_blind_seconds = getattr(context.planner, "last_blind_seconds", None)
        if callable(last_blind_seconds):
            blind_ms = round(float(last_blind_seconds()) * 1000)
        # Supply is what recentering exists to restore, and whether the anchor
        # was measured decides which rejection rules were even allowed to run.
        # Neither is recoverable from the rejection counts afterwards.
        supply = None
        last_supply = getattr(context.planner, "last_supply", None)
        if callable(last_supply):
            supply = int(last_supply())
        anchor = None
        anchor_measured = getattr(context.planner, "anchor_measured", None)
        if callable(anchor_measured):
            anchor = "measured" if anchor_measured() else "predicted"
        context.event_log.emit(
            "plan",
            stage=stage or None,
            target=target_payload(context.target),
            reject=rejections or None,
            cooldown_ms=cooldown_ms or None,
            idle_ms=idle_ms or None,
            blind_ms=blind_ms or None,
            supply=supply,
            anchor=anchor,
            recenter_reason=recenter_reason,
        )
        _report_blind_stall(context, stage)
        if (
            context.hunt_progress_recovery is not None
            and context.hunt_progress_recovery.observe(
                context.detections,
                context.target,
                cooldown_ms=cooldown_ms,
            )
        ):
            _reset_after_runtime_recovery(context)
            return BotState.IDLE
        if context.target is None:
            if rejections:
                context.logger.debug(
                    "Planning | no actionable target | %s",
                    " ".join(
                        f"{name}={count}"
                        for name, count in sorted(rejections.items())
                    ),
                )
            else:
                context.logger.debug("Planning | no actionable target")
            delay_ms = context.idle_delay_ms
            if cooldown_ms > delay_ms:
                context.logger.info(
                    "Planning | cooldown | remaining=%dms",
                    cooldown_ms,
                )
            delay_ms = max(delay_ms, cooldown_ms)
            if delay_ms and _wait_for_delay(context, delay_ms):
                return BotState.STOPPED
            return BotState.IDLE
        context.action_stage = stage
        action_kind = context.target_action_kinds.get(
            context.target.type,
            ActionKind.TAP,
        )
        target_key = (context.target.type, context.target.x, context.target.y)
        if context.inert_target != target_key:
            # The inert run belongs to one coordinate; carrying it across would
            # send the back key on the first attempt at an unrelated target.
            context.inert_taps = 0
            context.inert_target = None
            context.escalate_to_back = False
        if context.escalate_to_back and action_kind != ActionKind.BACK:
            # Repeated taps moved nothing at all, so the coordinate is inert -
            # either a phantom detection or an overlay that ignores taps. Try
            # the hardware back key once before giving up on this target.
            context.escalate_to_back = False
            context.logger.warning(
                "Planning | %s ignored %d taps; escalating to back",
                context.target.type,
                context.inert_taps,
            )
            action_kind = ActionKind.BACK
        context.action = _build_action(action_kind, context.target)
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
        if context.attempt_target_type != context.target.type:
            context.attempt = 0
            context.attempt_target_type = context.target.type
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
        context.event_log.emit(
            "action",
            act=action_payload(context.action),
            target=target_payload(context.target),
            attempt=context.attempt,
        )
        try:
            context.action_driver.execute(context.action, context.frame)
        except AdbError as exc:
            # A shell/input failure says nothing about whether the UI changed.
            # Do not let it take down an unattended run, and do not claim a
            # successful gesture.  The planner can return to a known-safe
            # screen after one failure; two consecutive transport failures
            # stop cleanly instead of endlessly issuing blind commands.
            context.action_transport_failures += 1
            failures = context.action_transport_failures
            context.logger.warning(
                "Action | ADB failed | target=%s | consecutive=%d/%d | error=%s",
                context.target.type,
                failures,
                context.action_transport_failure_limit,
                exc,
            )
            context.event_log.emit(
                "action_transport_failure",
                target=target_payload(context.target),
                consecutive=failures,
                limit=context.action_transport_failure_limit,
                error=str(exc),
            )
            on_action_failure = getattr(context.planner, "on_action_failure", None)
            if callable(on_action_failure):
                on_action_failure(context.target.type)
            context.target = None
            context.action = None
            context.attempt = 0
            context.attempt_target_type = None
            if failures >= max(1, context.action_transport_failure_limit):
                context.logger.error(
                    "Action | stopping after consecutive ADB failures | count=%d",
                    failures,
                )
                return BotState.STOPPED
            return BotState.CAPTURE
        context.action_transport_failures = 0
        context.action_count += 1
        delay_ms = context.post_action_delays_ms.get(
            context.target.type,
            context.click_delay_ms,
        )
        context.verification_timeout_ms = max(0, delay_ms)
        context.verification_started_at = context.clock()
        context.verification_deadline = None
        context.verification_checks = 0
        initial_poll_ms = min(
            max(0, delay_ms),
            max(1, context.transition_poll_interval_ms),
        )
        if initial_poll_ms:
            context.logger.debug(
                "Action | adaptive verify in %dms; max timeout=%dms | target=%s",
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
        capture_started = time.perf_counter()
        after = context.capture_provider.capture()
        capture_ms = round((time.perf_counter() - capture_started) * 1000)
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
        relevant_types: set[str] = set()
        verifier_types = getattr(context.verifier, "relevant_detection_types", None)
        if callable(verifier_types):
            relevant_types.update(verifier_types(context.target.type))
        planner_types = getattr(context.planner, "verification_detection_types", None)
        if callable(planner_types):
            relevant_types.update(planner_types(context.target.type))
        detect_started = time.perf_counter()
        detect_types = getattr(context.detector, "detect_types", None)
        if relevant_types and callable(detect_types):
            after_detections = detect_types(after, relevant_types)
        else:
            after_detections = context.detector.detect(after)
        detect_ms = round((time.perf_counter() - detect_started) * 1000)
        context.logger.debug(
            "Verify | performance | capture=%dms | detect=%dms | types=%d",
            capture_ms,
            detect_ms,
            len(relevant_types),
        )
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
        context.verification_checks += 1
        explicit_failure = result.reason.startswith("failure indicator detected:")
        now = context.clock()
        deadline = context.verification_deadline
        if (
            not result.success
            and not explicit_failure
            and deadline is None
            and context.verification_timeout_ms
        ):
            deadline = now + context.verification_timeout_ms / 1000
            context.verification_deadline = deadline
        within_deadline = deadline is not None and now < deadline
        minimum_checks_pending = (
            context.verification_timeout_ms > 0
            and context.verification_checks
            < max(1, context.verification_minimum_checks)
        )
        early_finalize_method = getattr(
            context.planner,
            "should_finalize_verification_early",
            None,
        )
        early_finalize = bool(
            callable(early_finalize_method)
            and early_finalize_method(
                context.target,
                result,
                context.verification_checks,
            )
        )
        pending = (
            not result.success
            and not explicit_failure
            and not early_finalize
            and (within_deadline or minimum_checks_pending)
        )
        remaining_ms = (
            max(0, round((deadline - now) * 1000))
            if deadline is not None
            else 0
        )
        elapsed_ms = (
            max(0, round((now - context.verification_started_at) * 1000))
            if context.verification_started_at is not None
            else None
        )
        context.event_log.emit(
            "verify",
            phase="pending" if pending else "final",
            check=context.verification_checks,
            elapsed_ms=elapsed_ms,
            remaining_ms=remaining_ms if pending else None,
            target=target_payload(context.target),
            attempt=context.attempt,
            result=verification_payload(result),
            n=len(after_detections),
            det=detection_payload(after_detections),
            capture_ms=capture_ms,
            detect_ms=detect_ms,
        )
        if pending:
            poll_interval_ms = max(1, context.transition_poll_interval_ms)
            poll_ms = (
                min(poll_interval_ms, max(1, remaining_ms))
                if within_deadline
                else poll_interval_ms
            )
            context.logger.debug(
                "Verify | Pending | %s | check=%d | poll=%dms | remaining=%dms",
                result.reason,
                context.verification_checks,
                poll_ms,
                remaining_ms,
            )
            if _wait_for_delay(context, poll_ms):
                return BotState.STOPPED
            return BotState.VERIFY
        context.verification_deadline = None
        context.verification_timeout_ms = 0
        context.verification_started_at = None
        context.verification_checks = 0
        record = ActionRecord(
            timestamp=utc_now(),
            action=context.action,
            target=context.target,
            result=result,
            attempt=context.attempt,
        )
        context.observer.on_action_complete(record, context.before_frame, after)
        target_key = (context.target.type, context.target.x, context.target.y)
        if (
            not result.success
            and result.pixel_change is not None
            and result.pixel_change < context.inert_pixel_change
        ):
            if context.inert_target != target_key:
                context.inert_taps = 0
                context.inert_target = target_key
            context.inert_taps += 1
            context.escalate_to_back = (
                context.inert_taps >= context.inert_tap_threshold
            )
        else:
            context.inert_taps = 0
            context.inert_target = None
            context.escalate_to_back = False
        if result.success:
            target_type = context.target.type
            on_action_success_context = getattr(
                context.planner,
                "on_action_success_context",
                None,
            )
            if callable(on_action_success_context):
                on_action_success_context(
                    context.target,
                    after,
                    after_detections,
                    result,
                )
            else:
                on_action_success = getattr(context.planner, "on_action_success", None)
                if callable(on_action_success):
                    on_action_success(target_type)
            _record_action_success(context, target_type)
            # A confirmed hunt is the only thing the stall watchdog accepts as
            # progress; everything else on screen can stay unchanged for a
            # quarter of an hour while the bot produces nothing. It also ends
            # the startup phase for detectors that only apply during launch.
            if target_type == getattr(
                context.planner, "completion_type", None
            ):
                for component in (
                    context.hunt_progress_recovery,
                    context.detector,
                ):
                    on_hunt_completed = getattr(component, "on_hunt_completed", None)
                    if callable(on_hunt_completed):
                        on_hunt_completed()
            context.logger.info("Verify | Success | %s", result.reason)
            if target_type in context.cycle_complete_targets:
                context.cycle_count += 1
                context.logger.info(
                    "Workflow | completed cycle %d/%s",
                    context.cycle_count,
                    context.max_cycles or "unlimited",
                )
            restart_reason = _GAME_RESTART_AFTER_SUCCESS.get(target_type)
            if restart_reason is not None:
                recovery = context.runtime_recovery
                if recovery is None:
                    context.logger.error(
                        "Recovery | cannot restart after %s; recovery is disabled",
                        target_type,
                    )
                elif recovery.request_restart(
                    restart_reason[0],
                    reason_key=restart_reason[1],
                    bypass_cooldown=True,
                ):
                    _reset_after_runtime_recovery(context)
                    return BotState.IDLE
                else:
                    context.logger.error(
                        "Recovery | restart failed after verified %s",
                        target_type,
                    )
            context.attempt = 0
            context.attempt_target_type = None
            context.frame = after
            context.detections = after_detections
            can_reuse = getattr(
                context.planner,
                "can_reuse_verification_result",
                None,
            )
            context.reuse_verified_detections = bool(
                callable(can_reuse)
                and can_reuse(context.target.type, after_detections)
            )
            return BotState.IDLE
        context.logger.warning("Verify | Failed | %s", result.reason)
        if context.attempt <= context.verify_retries:
            recovery_types = getattr(
                context.planner,
                "failure_recovery_detection_types",
                None,
            )
            detect_types = getattr(context.detector, "detect_types", None)
            if callable(recovery_types) and callable(detect_types):
                requested = recovery_types(context.target.type)
                if requested:
                    started = time.perf_counter()
                    context.after_detections = detect_types(after, requested)
                    detect_ms = round((time.perf_counter() - started) * 1000)
                    context.logger.info(
                        "Recover | rescan failed frame | %d targets | %dms | scoped=%d",
                        len(context.after_detections),
                        detect_ms,
                        len(requested),
                    )
                    context.event_log.emit(
                        "recover_detect",
                        ms=detect_ms,
                        scoped=len(requested),
                        n=len(context.after_detections),
                        det=detection_payload(context.after_detections),
                    )
        if (
            context.dinosaur_failure_snapshots is not None
            and context.target.type == "dinosaur"
        ):
            context.dinosaur_failure_snapshots.capture(
                context.before_frame,
                after,
                context.target,
                context.after_detections,
                result,
                attempt=context.attempt,
            )
        on_action_failure_context = getattr(
            context.planner,
            "on_action_failure_context",
            None,
        )
        if callable(on_action_failure_context):
            on_action_failure_context(
                context.target,
                context.after_frame,
                context.after_detections,
                context.attempt,
            )
        on_action_failure = getattr(context.planner, "on_action_failure", None)
        if callable(on_action_failure):
            on_action_failure(context.target.type)
        if context.max_actions and context.action_count >= context.max_actions:
            context.logger.info("Verify | retry skipped because max_actions was reached")
            context.attempt = 0
            context.attempt_target_type = None
            return BotState.IDLE
        on_blocked_action_context = getattr(
            context.planner,
            "on_blocked_action_context",
            None,
        )
        if callable(on_blocked_action_context) and on_blocked_action_context(
            context.target,
            context.after_detections,
            context.attempt,
        ):
            context.logger.warning(
                "Recovery | hunt confirmation remained blocked after %d attempts; "
                "closing the hunt dialog and collecting mailbox rewards",
                context.attempt,
            )
            context.event_log.emit(
                "mailbox_full_recovery",
                target=target_payload(context.target),
                attempts=context.attempt,
            )
            context.attempt = 0
            context.attempt_target_type = None
            return BotState.IDLE
        if context.attempt <= context.verify_retries:
            can_reuse_failed = getattr(
                context.planner,
                "can_reuse_failed_verification_result",
                None,
            )
            if callable(can_reuse_failed) and can_reuse_failed(
                context.target.type,
                context.after_detections,
            ):
                context.logger.info(
                    "Recover | reuse failed verification frame and re-plan"
                )
                context.frame = after
                context.detections = list(context.after_detections)
                context.target = None
                context.action = None
                return BotState.PLANNING
            return BotState.RECOVER
        context.logger.error("Verify | retry limit exhausted after %d attempts", context.attempt)
        context.event_log.emit(
            "retry_exhausted",
            target=target_payload(context.target),
            attempts=context.attempt,
        )
        on_retry_exhausted = getattr(context.planner, "on_retry_exhausted", None)
        if callable(on_retry_exhausted):
            on_retry_exhausted(context.target)
            context.logger.warning(
                "Planning | retry exhausted hook handled %s at (%d,%d)",
                context.target.type,
                context.target.x,
                context.target.y,
            )
        on_retry_exhausted_context = getattr(
            context.planner,
            "on_retry_exhausted_context",
            None,
        )
        if callable(on_retry_exhausted_context) and on_retry_exhausted_context(
            context.target,
            context.after_detections,
        ):
            context.logger.warning(
                "Recovery | hunt confirmation was blocked; "
                "closing the hunt dialog and collecting mailbox rewards"
            )
            context.event_log.emit(
                "mailbox_full_recovery",
                target=target_payload(context.target),
            )
        escalation_state = _escalate_retry_exhaustion(context, context.target)
        context.attempt = 0
        context.attempt_target_type = None
        return escalation_state or BotState.IDLE


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


def _report_blind_stall(context: BotContext, stage: str) -> None:
    """Record a stall the planner could neither act on nor name a wait for.

    The planner has already released its stage machines by the time this runs;
    what is left to do is leave evidence. The snapshot is the point: the cause
    of these episodes is a screen the detector has no name for, so the event
    stream - which can only report what matched - cannot describe it.
    """

    take_blind_escape = getattr(context.planner, "take_blind_escape", None)
    if not callable(take_blind_escape):
        return
    escape = take_blind_escape()
    if not escape:
        return
    context.event_log.emit("blind_stall", **escape)
    context.logger.warning(
        "Planning | no actionable target for %.0fs | stage=%s | releasing stages",
        float(escape.get("seconds", 0.0)),
        escape.get("stage") or stage or "unknown",
    )
    if context.stall_snapshots is None or context.frame is None:
        return
    context.stall_snapshots.capture(
        context.frame,
        context.detections,
        seconds=float(escape.get("seconds", 0.0)),
        stage=str(escape.get("stage") or stage or ""),
        escapes=int(escape.get("escapes", 0)),
    )


def _reset_after_runtime_recovery(context: BotContext) -> None:
    context.logger.info("Recovery | clearing transient workflow state")
    context.event_log.emit("recovery", action="reset_workflow")
    if context.hunt_progress_recovery is not None:
        context.hunt_progress_recovery.reset()
    # The app is launching again, so startup-only modals are back in scope.
    on_app_restart = getattr(context.detector, "on_app_restart", None)
    if callable(on_app_restart):
        on_app_restart()
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
    context.attempt_target_type = None
    context.action_stage = ""
    context.verification_timeout_ms = 0
    context.verification_deadline = None
    context.verification_started_at = None
    context.verification_checks = 0
    context.reuse_verified_detections = False
    context.inert_taps = 0
    context.inert_target = None
    context.escalate_to_back = False


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
        self._game_restart_requested = threading.Event()
        self._control_lock = threading.Lock()

    def step(self) -> BotState:
        if self._game_restart_requested.is_set():
            return self._handle_game_restart()
        handler = self.states[self.context.state]
        try:
            self.context.state = handler.execute(self.context)
        except AdbCaptureError as exc:
            self.context.state = self._handle_capture_transport_failure(exc)
        return self.context.state

    def _handle_capture_transport_failure(self, error: AdbCaptureError) -> BotState:
        """Retry screenshots, never a click whose result is now unknown.

        AdbClient reconnects only the configured serial. Clearing transient
        state also covers a disconnect during action verification: the next
        successful screenshot must be planned afresh, not replay the old tap.
        """
        context = self.context
        context.capture_transport_failures += 1
        failures = context.capture_transport_failures
        limit = max(1, context.capture_transport_failure_limit)
        context.logger.warning(
            "Capture | ADB failed | state=%s | consecutive=%d/%d | error=%s",
            context.state, failures, limit, error,
        )
        context.event_log.emit(
            "capture_transport_failure", state=str(context.state),
            consecutive=failures, limit=limit, error=str(error),
        )
        _reset_after_runtime_recovery(context)
        if failures >= limit:
            context.logger.error(
                "Capture | stopping after consecutive ADB failures | count=%d", failures,
            )
            return BotState.STOPPED
        if _wait_for_delay(context, context.capture_transport_retry_delay_ms):
            return BotState.STOPPED
        return BotState.CAPTURE

    def _handle_game_restart(self) -> BotState:
        try:
            recovery = self.context.runtime_recovery
            if recovery is None:
                self.context.logger.error(
                    "Control | game restart unavailable; recovery is disabled"
                )
                return self.context.state
            recovery.request_restart(
                "manual control request",
                reason_key="manual_control",
                bypass_cooldown=True,
            )
            _reset_after_runtime_recovery(self.context)
            self.context.state = BotState.IDLE
            return self.context.state
        finally:
            self._game_restart_requested.clear()

    def run(self) -> None:
        self.context.logger.info("Bot started | Sense -> Think -> Act")
        self.context.event_log.emit("session", action="start")
        try:
            while self.context.state != BotState.STOPPED:
                self.step()
        except KeyboardInterrupt:
            self.context.logger.info("Stop requested by user")
            self.context.state = BotState.STOPPED
        except Exception:
            self.context.logger.exception(
                "Bot crashed | state=%s | actions=%d | cycles=%d",
                self.context.state,
                self.context.action_count,
                self.context.cycle_count,
            )
            raise
        finally:
            self.close()

    def stop(self) -> None:
        self.context.stop_requested = True
        self.context.stop_event.set()

    def request_game_restart(self) -> bool:
        """Queue one fixed-package game restart for the Bot thread."""

        with self._control_lock:
            if (
                self.context.runtime_recovery is None
                or self.context.stop_requested
                or self.context.state == BotState.STOPPED
                or self._game_restart_requested.is_set()
            ):
                return False
            self._game_restart_requested.set()
        self.context.logger.warning("Control | game restart requested")
        return True

    def close(self) -> None:
        self.context.capture_provider.close()
        self.context.observer.close()
        self.context.event_log.emit(
            "session",
            action="stop",
            actions=self.context.action_count,
            cycles=self.context.cycle_count,
        )
        self.context.event_log.close()
        self.context.logger.info(
            "Bot stopped | actions=%d | cycles=%d",
            self.context.action_count,
            self.context.cycle_count,
        )
