"""Hatch management with hunting during egg cooldowns."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from math import hypot
from typing import Any

from .full_hatch import (
    HOME_PILE_TOLERANCE,
    STARTUP_DETECTION_TYPES,
    _is_bright_outdoor_map,
    home_pile_offset,
    is_centered_home_screen,
    nest_mask_close_target,
)
from . import hatch as hatch_feature
from .models import Detection, Frame, Target, VerificationResult
from .parent_open import NEST_TITLE
from .planning import HuntPlanner

# How many consecutive handoff frames may show only the hunt map's centre egg
# before the handoff stops waiting and leaves the map by its own controls.
# Detection runs at roughly one frame every 2-6s here, so a few frames absorb a
# genuinely transient miss while capping the stall at seconds, not the full
# 90s deadline.
MAX_ANCHOR_ONLY_HANDOFF_FRAMES = 3

# How much of the deadline a verified step towards home buys back.  An S9 trace
# tapped the exit at 10:28:55, verified it at 10:28:57 and was killed at
# 10:29:04: a working exit sequence guillotined 2s in.  One step is worth a
# fresh window, but no more than the deadline itself.
HANDOFF_PROGRESS_EXTENSION_SECONDS = 30.0

# Total extension a single handoff may earn.  Progress that never arrives at a
# centred home is still a stall; without this cap a map that alternates exit
# and recentre taps forever would keep renewing its own deadline, which is the
# exact silent burn the deadline exists to stop.
MAX_HANDOFF_PROGRESS_EXTENSIONS = 3
# How many frames the handoff waits for a visible hatch home to stop moving
# before it touches the camera again. The cloud wipe between the cave and the
# home map spans a few frames; anything longer than that is a map that really
# is stuck, and the deadline still covers it.
MAX_HOME_ANCHOR_SETTLE_FRAMES = 6


def _describe_home_proof(frame: Frame, detections: Sequence[Detection]) -> str:
    """Say which gate of the centred-home proof rejected the final frame.

    The proof is all-or-nothing, so a timeout alone cannot tell a map parked
    in the cave apart from one sitting on the home screen a few pixels off.
    Naming the failing gate is the difference between guessing and knowing.
    """

    try:
        bright = _is_bright_outdoor_map(frame)
        if not bright:
            return f"proof=not-outdoor-map | mean_brightness={frame.image.mean():.0f}"
        pile = hatch_feature.has_home_pile_structure(
            frame,
            egg_pile_point=(450.0, 1330.0),
            reference_width=900.0,
        )
        if not pile:
            return "proof=no-home-pile-structure"
        offset = home_pile_offset(frame)
        if offset is None:
            return "proof=pile-offset-unmeasurable"
        scale = frame.width / 900.0
        worst = max(abs(offset[0]), abs(offset[1]))
        return (
            f"proof=pile-off-centre | offset=({offset[0]:.0f},{offset[1]:.0f})"
            f" | worst={worst:.0f} | tolerance={HOME_PILE_TOLERANCE * scale:.0f}"
        )
    except Exception as error:  # diagnostics must never break the handoff
        return f"proof=undiagnosable | {type(error).__name__}: {error}"


class HatchHuntPlanner:
    """Run hunts while a hatch workflow is in its no-ready-egg rescan wait.

    The final handoff window is reserved for finishing the current hunt and
    returning the collection map to its measured centre.  Full Hatch receives
    control only after both the hatch home anchor and hunt centre egg are
    visible for two consecutive frames.
    """

    def __init__(
        self,
        hatch: Any,
        hunt: HuntPlanner,
        *,
        handoff_seconds: float = 30.0,
        handoff_timeout_seconds: float = 90.0,
        nest_close_attempt_limit: int = 6,
        clock: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.clock = clock or time.monotonic
        self.hatch = hatch
        self.hunt = hunt
        self.handoff_ms = max(0, round(handoff_seconds * 1000))
        # Handoff ends on a measurement, not on a countdown, so a measurement
        # that can never succeed has no exit at all: one observed run spent 14
        # minutes and 227 actions toggling between the map and home before a
        # human stopped it. A deadline converts that silent burn into a logged
        # decision.  A healthy handoff finishes in well under 30s.
        self.handoff_timeout_seconds = max(0.0, handoff_timeout_seconds)
        self.nest_close_attempt_limit = max(1, nest_close_attempt_limit)
        self.logger = logger or logging.getLogger("dino_bot")
        self._mode = "hatch"
        self._action_owner: Any = None
        self._centered_frames = 0
        # Consecutive handoff frames whose only home-ish proof was the hunt
        # map's centre egg.  Bounded so a centred map cannot stall the handoff
        # for its whole deadline.
        self._anchor_only_frames = 0
        self._nest_close_attempts = 0
        self._handoff_deadline: float | None = None
        self._handoff_reason = ""
        self._handoff_extensions = 0
        self._last_handoff_progress: str | None = None
        self._home_anchor_waits = 0
        # 狩獵閒置差事:狩獵側全目標冷卻時,把空窗拿去收巢蛋。
        self.errand_min_idle_ms = 15_000
        self.errand_margin_ms = 90_000
        self.errand_interval_seconds = 180.0
        self._next_errand_at = 0.0

    @property
    def completion_type(self) -> str:
        """Expose verified hunts to detector and event hooks."""

        return self.hunt.completion_type

    def is_complete(self) -> bool:
        # 內層孵蛋流程宣告結束(含恢復重試耗盡)時,讓引擎乾淨停止,
        # 而不是留下一個只掃描不動作的殭屍程序。
        # Egg-pile calibration is a recoverable hatch failure in combined
        # mode: keep the hunt side alive while the hatch side is fused off.
        if self._hatch_is_blocked():
            return not self._continue_hunting_when_blocked()
        is_complete = getattr(self.hatch, "is_complete", None)
        return bool(is_complete()) if callable(is_complete) else False

    def last_stage(self) -> str:
        child = self.hatch if self._mode == "hatch" else self.hunt
        stage = child.last_stage()
        return f"hatch_hunt_{self._mode}:{stage}"

    def workflow_status(self) -> dict[str, Any]:
        """Live state for Dashboard; unlike log inference this cannot age out."""
        blocked = self._hatch_is_blocked()
        state: dict[str, Any] = {"hatch_blocked": blocked, "planner_stage": self.last_stage()}
        if getattr(self.hatch, "population_limit_reached", False):
            state.update(stage="population_limit_hunt", label="已達安全人口，持續狩獵")
        elif blocked:
            state.update(stage="hatch_blocked_hunt", label="孵蛋已鎖住，僅繼續狩獵")
        elif self._mode == "hunt" and getattr(self.hatch, "capacity_retry_pending", False):
            state.update(stage="capacity_retry_hunt", label="人口讀取失敗，狩獵後重試")
        elif self._mode == "hunt":
            state.update(stage="cooldown_hunt", label="冷卻期間狩獵")
        elif self._mode == "handoff":
            state.update(stage="handoff", label="返回孵蛋首頁")
        elif self._mode == "capacity_camera_refresh_depart":
            state.update(stage="capacity_camera_refresh", label="刷新人口地圖位置")
        else:
            return state
        state["cooldown_remaining_seconds"] = (
            max(0, self._hatch_cooldown_delay_ms() // 1000) if not blocked else None
        )
        return state

    def choose(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        # A login/restart screen invalidates the old cooldown/map context.
        # Re-enter hatch-first mode so the incubator is always checked before
        # hunting resumes. FullHatchPlanner owns the startup shortcut choice.
        if self._mode != "hatch" and any(
            item.type in STARTUP_DETECTION_TYPES for item in detections
        ):
            self.logger.info(
                "Hatch+Hunt | startup interruption detected | restarting hatch-first flow"
            )
            self.hatch.reset_workflow()
            self.hunt.reset_workflow()
            self._mode = "hatch"
            self._centered_frames = 0
            self._nest_close_attempts = 0

        # 狩獵側的字彙裡沒有「我的巢」面板。面板一開著,planner 照地圖劇本
        # 找恐龍、點離開巢穴鈕,兩者都被面板蓋住,於是整輪空轉到重試耗盡。
        # 面板是孵蛋側的畫面:先關掉它,再把控制權交還孵蛋側重新判斷冷卻,
        # 而不是在原地接著狩獵。
        if self._mode != "hatch":
            if any(item.type == NEST_TITLE for item in detections):
                return self._close_stray_nest_panel(frame, detections)
            if self._nest_close_attempts:
                self._nest_close_attempts = 0
                self.logger.info(
                    "Hatch+Hunt | stray nest panel closed | returning to hatch"
                )
                self.hunt.reset_workflow()
                self._mode = "hatch"
                self._centered_frames = 0
                return self._choose_owned(self.hatch, frame, detections)

        if self._mode == "hatch" and self._begin_capacity_camera_refresh():
            self._mode = "capacity_camera_refresh_depart"
            self._centered_frames = 0
            self._anchor_only_frames = 0
            self._handoff_extensions = 0
            self._last_handoff_progress = None
            self._home_anchor_waits = 0
            self._handoff_reason = "capacity camera refresh"
            self._handoff_deadline = (
                self.clock() + self.handoff_timeout_seconds
                if self.handoff_timeout_seconds
                else None
            )
            self.logger.warning(
                "Hatch+Hunt | capacity unreadable; entering hunt map for one camera refresh"
            )
            return self._choose_capacity_camera_refresh_depart(frame, detections)

        if self._mode == "capacity_camera_refresh_depart":
            if self._handoff_expired():
                return self._fail_capacity_camera_refresh(frame, detections, "hunt map not reached")
            return self._choose_capacity_camera_refresh_depart(frame, detections)

        blocked = self._hatch_is_blocked()
        if blocked:
            if not self._continue_hunting_when_blocked():
                return None
            if self._mode != "hunt":
                self._mode = "hunt"
                self._centered_frames = 0
                if getattr(self.hatch, "population_limit_reached", False):
                    self.logger.info("Hatch+Hunt | safe population reached; switching to hunt")
                else:
                    self.logger.error(
                        "Hatch+Hunt | hatch calibration blocked; switching to hunt"
                    )
            return self._choose_owned(self.hunt, frame, detections)

        if self._mode == "hatch":
            target = self._choose_owned(self.hatch, frame, detections)
            if target is not None:
                return target
            if not self.hatch.is_hunt_cooldown_active():
                return None
            remaining = self._hatch_cooldown_delay_ms()
            if remaining <= self.handoff_ms:
                return None
            self._mode = "hunt"
            self._centered_frames = 0
            if getattr(self.hatch, "capacity_retry_pending", False):
                self.logger.info(
                    "Hatch+Hunt | capacity recheck in %.0fs | switching to hunt",
                    remaining / 1000,
                )
            else:
                self.logger.info(
                    "Hatch+Hunt | egg cooldown %.0fs | switching to hunt",
                    remaining / 1000,
                )
            return self._choose_owned(self.hunt, frame, detections)

        if self._mode == "hunt":
            remaining = self._hatch_cooldown_delay_ms()
            if (
                not self.hatch.is_hunt_cooldown_active()
                or remaining <= self.handoff_ms
            ):
                self._enter_handoff("cooldown")
                self.logger.info(
                    "Hatch+Hunt | cooldown handoff window | remaining=%.0fs",
                    remaining / 1000,
                )
                return self._choose_handoff(frame, detections)
            hunt_idle = self.hunt.next_ready_delay_ms()
            if (
                hunt_idle >= self.errand_min_idle_ms
                and remaining > self.handoff_ms + self.errand_margin_ms
                and self.clock() >= self._next_errand_at
                and self.hatch.begin_interim_collection()
            ):
                # 狩獵側全目標都在冷卻;把這段空窗換成一趟回家收蛋,
                # 收完由既有的冷卻切換邏輯自動回到狩獵。
                self._next_errand_at = self.clock() + self.errand_interval_seconds
                self._enter_handoff("errand")
                self.logger.info(
                    "Hatch+Hunt | hunt idle %.0fs | interim collection errand",
                    hunt_idle / 1000,
                )
                return self._choose_handoff(frame, detections)
            return self._choose_owned(self.hunt, frame, detections)

        if self._handoff_expired():
            return self._abandon_handoff(frame, detections)
        return self._choose_handoff(frame, detections)

    def next_ready_delay_ms(self) -> int:
        if self._mode == "hatch":
            return self.hatch.next_ready_delay_ms()
        if self._mode in {"handoff", "capacity_camera_refresh_depart"}:
            return 0
        until_handoff = max(
            0,
            self._hatch_cooldown_delay_ms() - self.handoff_ms,
        )
        hunt_delay = self.hunt.next_ready_delay_ms()
        if not hunt_delay:
            return 0
        delay = hunt_delay if self._hatch_is_blocked() else min(hunt_delay, until_handoff)
        return delay

    def _hatch_cooldown_delay_ms(self) -> int:
        method = getattr(self.hatch, "hunt_cooldown_delay_ms", None)
        return method() if callable(method) else self.hatch.next_ready_delay_ms()

    def planning_detection_types(self) -> frozenset[str] | None:
        if self._mode == "hunt":
            hunt_types = self.hunt.planning_detection_types()
            if hunt_types is not None:
                # NEST_TITLE is not part of any hunting stage, but without it
                # in the scan a stray My Nest panel is invisible and the hunt
                # burns its retries on controls the panel covers.
                return frozenset({*hunt_types, NEST_TITLE})
            # ``None`` means a full scan to a standalone planner. The combined
            # detector also owns all hatch templates, so translate that request
            # into the complete hunting vocabulary when the planner exposes it.
            full_method = getattr(self.hunt, "full_detection_types", None)
            return full_method() if callable(full_method) else None
        hatch_method = getattr(self.hatch, "planning_detection_types", None)
        hatch_types = hatch_method() if callable(hatch_method) else None
        if self._mode not in {"handoff", "capacity_camera_refresh_depart"}:
            return hatch_types

        # Handoff still needs the hunt map evidence to decide whether the
        # viewport must be recentered before returning to hatch mode. Keep the
        # hatch safety controls, but add the hunt planner's scoped set instead
        # of falling back to an expensive combined full scan.
        hunt_method = getattr(self.hunt, "planning_detection_types", None)
        hunt_types = hunt_method() if callable(hunt_method) else None
        if hatch_types is None or hunt_types is None:
            return None
        return frozenset({*hatch_types, *hunt_types, NEST_TITLE})

    def verification_detection_types(self, target_type: str) -> frozenset[str]:
        method = getattr(self._action_owner, "verification_detection_types", None)
        return method(target_type) if callable(method) else frozenset()

    def on_action_success(self, target_type: str) -> None:
        self._extend_handoff_for_progress(target_type)
        method = getattr(self._action_owner, "on_action_success", None)
        if callable(method):
            method(target_type)

    def on_action_success_context(
        self,
        target: Target,
        frame: Frame,
        detections: Sequence[Detection],
        result: VerificationResult,
    ) -> None:
        method = getattr(self._action_owner, "on_action_success_context", None)
        if callable(method):
            # ``on_action_success`` is skipped on this path, so the handoff
            # needs its own credit for the step.
            self._extend_handoff_for_progress(target.type)
            method(target, frame, detections, result)
            return
        self.on_action_success(target.type)

    def on_action_failure(self, target_type: str) -> None:
        method = getattr(self._action_owner, "on_action_failure", None)
        if callable(method):
            method(target_type)

    def on_action_failure_context(
        self,
        target: Target,
        frame: Frame | None,
        detections: Sequence[Detection],
        attempts: int,
    ) -> None:
        method = getattr(self._action_owner, "on_action_failure_context", None)
        if callable(method):
            method(target, frame, detections, attempts)

    def should_finalize_verification_early(
        self,
        target: Target,
        result: VerificationResult,
        checks: int,
    ) -> bool:
        method = getattr(
            self._action_owner,
            "should_finalize_verification_early",
            None,
        )
        return bool(method(target, result, checks)) if callable(method) else False

    def on_retry_exhausted(self, target: Target) -> None:
        method = getattr(self._action_owner, "on_retry_exhausted", None)
        if callable(method):
            method(target)

    def recover_from_action_failures(
        self,
        target: Target,
        stage: str,
        episodes: int,
        frame: Frame | None,
        detections: Sequence[Detection],
    ) -> bool:
        method = getattr(self._action_owner, "recover_from_action_failures", None)
        return bool(
            method(target, stage, episodes, frame, detections)
        ) if callable(method) else False

    def is_recovery_progress(self, target_type: str) -> bool:
        for owner in (self._action_owner, self.hatch, self.hunt):
            method = getattr(owner, "is_recovery_progress", None)
            if callable(method) and method(target_type):
                return True
        return False

    def on_blocked_action_context(
        self,
        target: Target,
        detections: Sequence[Detection],
        attempt: int,
    ) -> bool:
        method = getattr(self._action_owner, "on_blocked_action_context", None)
        return bool(method(target, detections, attempt)) if callable(method) else False

    def on_retry_exhausted_context(
        self,
        target: Target,
        detections: Sequence[Detection],
    ) -> bool:
        method = getattr(self._action_owner, "on_retry_exhausted_context", None)
        return bool(method(target, detections)) if callable(method) else False

    def can_reuse_verification_result(
        self,
        target_type: str,
        detections: Sequence[Detection],
    ) -> bool:
        method = getattr(self._action_owner, "can_reuse_verification_result", None)
        return bool(method(target_type, detections)) if callable(method) else False

    def failure_recovery_detection_types(
        self,
        target_type: str,
    ) -> frozenset[str]:
        method = getattr(self._action_owner, "failure_recovery_detection_types", None)
        return method(target_type) if callable(method) else frozenset()

    def can_reuse_failed_verification_result(
        self,
        target_type: str,
        detections: Sequence[Detection],
    ) -> bool:
        method = getattr(
            self._action_owner,
            "can_reuse_failed_verification_result",
            None,
        )
        return bool(method(target_type, detections)) if callable(method) else False

    def reset_workflow(self) -> None:
        self.hatch.reset_workflow()
        self.hunt.reset_workflow()
        self._mode = "hatch"
        self._action_owner = None
        self._centered_frames = 0
        self._nest_close_attempts = 0
        self._handoff_deadline = None
        self._handoff_reason = ""

    # Hunt diagnostics remain available to the shared engine while combined.
    def take_blind_escape(self) -> dict[str, Any] | None:
        return self.hunt.take_blind_escape() if self._mode in {"hunt", "handoff"} else None

    def last_rejections(self) -> dict[str, int]:
        return self.hunt.last_rejections() if self._mode in {"hunt", "handoff"} else {}

    def last_idle_seconds(self) -> float:
        return self.hunt.last_idle_seconds() if self._mode in {"hunt", "handoff"} else 0.0

    def last_recenter_reason(self) -> str | None:
        return self.hunt.last_recenter_reason() if self._mode in {"hunt", "handoff"} else None

    def last_blind_seconds(self) -> float:
        return self.hunt.last_blind_seconds() if self._mode in {"hunt", "handoff"} else 0.0

    def last_supply(self) -> int:
        return self.hunt.last_supply() if self._mode in {"hunt", "handoff"} else 0

    def anchor_measured(self) -> bool:
        return self.hunt.anchor_measured() if self._mode in {"hunt", "handoff"} else False

    def _hatch_is_blocked(self) -> bool:
        method = getattr(self.hatch, "is_hatch_blocked", None)
        return bool(method()) if callable(method) else False

    def _continue_hunting_when_blocked(self) -> bool:
        method = getattr(self.hatch, "continue_hunting_when_blocked", None)
        return bool(method()) if callable(method) else True

    def _close_stray_nest_panel(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        """Dismiss a My Nest panel found open while the hunt side has control.

        Bounded: a panel that refuses to close is a screen this planner cannot
        read, so hand the whole workflow back to the hatch side's recovery
        rather than tapping the mask forever.
        """

        self._nest_close_attempts += 1
        if self._nest_close_attempts > self.nest_close_attempt_limit:
            self.logger.error(
                "Hatch+Hunt | nest panel still open after %d mask taps"
                " | handing back to hatch recovery",
                self.nest_close_attempt_limit,
            )
            self.reset_workflow()
            return self._choose_owned(self.hatch, frame, detections)
        self.logger.info(
            "Hatch+Hunt | nest panel open during %s | closing (%d/%d)",
            self._mode,
            self._nest_close_attempts,
            self.nest_close_attempt_limit,
        )
        # The tap belongs to the hatch vocabulary, so its success, failure and
        # verification callbacks have to reach the hatch planner.
        self._action_owner = self.hatch
        return nest_mask_close_target(frame, self.hatch.reference_width)

    def _enter_handoff(self, reason: str) -> None:
        self._mode = "handoff"
        self._centered_frames = 0
        self._anchor_only_frames = 0
        self._handoff_extensions = 0
        self._last_handoff_progress = None
        self._home_anchor_waits = 0
        self._handoff_reason = reason
        self._handoff_deadline = (
            self.clock() + self.handoff_timeout_seconds
            if self.handoff_timeout_seconds
            else None
        )

    def _handoff_progress_types(self) -> frozenset[str]:
        """Taps that prove the map is walking back towards the hatch home.

        Read off the hunt planner rather than hardcoded, so a renamed asset
        cannot silently turn the extension off.
        """

        return frozenset(
            {self.hunt.map_exit_type, self.hunt.forest_recenter_type},
        )

    def _extend_handoff_for_progress(self, target_type: str) -> None:
        """Give a verified step towards home more time to finish.

        The deadline measures wall clock, not progress, so it cannot tell a
        map toggling in place apart from one that is two taps from the hatch
        home.  A verified exit or recentre is proof of the latter, so it buys
        a fresh window -- capped, because progress that never reaches a centred
        home is still a stall.
        """

        if self._mode != "handoff" or self._handoff_deadline is None:
            return
        if target_type not in self._handoff_progress_types():
            return
        # Exit and recentre sit on the same map corner and each brings the
        # other back: tapping exit reveals recentre, tapping recentre reveals
        # exit. Verified alternation therefore looks exactly like progress and
        # used to buy a fresh window every time, spending all three extensions
        # on a map that never moved. Only a step that differs from the last one
        # counts; repeating the same target still does, since that is a retry
        # of one real step rather than a loop between two.
        if (
            self._last_handoff_progress is not None
            and target_type != self._last_handoff_progress
        ):
            self.logger.info(
                "Hatch+Hunt | handoff alternating %s <-> %s | not progress",
                self._last_handoff_progress,
                target_type,
            )
            self._last_handoff_progress = target_type
            return
        self._last_handoff_progress = target_type
        if self._handoff_extensions >= MAX_HANDOFF_PROGRESS_EXTENSIONS:
            return
        extended = max(
            self._handoff_deadline,
            self.clock() + HANDOFF_PROGRESS_EXTENSION_SECONDS,
        )
        if extended <= self._handoff_deadline:
            return
        self._handoff_extensions += 1
        self._handoff_deadline = extended
        self.logger.info(
            "Hatch+Hunt | handoff progress via %s | deadline extended %.0fs"
            " (%d/%d)",
            target_type,
            HANDOFF_PROGRESS_EXTENSION_SECONDS,
            self._handoff_extensions,
            MAX_HANDOFF_PROGRESS_EXTENSIONS,
        )

    def _handoff_expired(self) -> bool:
        return (
            self._handoff_deadline is not None
            and self.clock() >= self._handoff_deadline
        )

    def _abandon_handoff(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        """Leave a handoff that the centred-home test will never end.

        An errand is optional, so drop it and go back to hunting; the cooldown
        handoff is not, so give the hatch side its own recovery instead. Either
        way the deadline is cleared here, because both destinations re-arm one
        of their own when they need it.
        """

        reason = self._handoff_reason
        self._handoff_deadline = None
        self.logger.error(
            "Hatch+Hunt | handoff to centered home timed out after %.0fs | source=%s"
            " | %s",
            self.handoff_timeout_seconds,
            reason or "unknown",
            _describe_home_proof(frame, detections),
        )
        if reason == "errand":
            abort = getattr(self.hatch, "abort_interim_collection", None)
            if callable(abort) and abort("centered home never confirmed"):
                # 差事沒跑成,別讓下一輪空窗立刻再試一次同樣的路。
                self._next_errand_at = (
                    self.clock() + self.errand_interval_seconds
                )
                self._mode = "hunt"
                self._centered_frames = 0
                return self._choose_owned(self.hunt, frame, detections)

        if reason == "capacity camera refresh":
            return self._fail_capacity_camera_refresh(
                frame,
                detections,
                "centered home handoff timed out",
            )

        self.hunt.reset_workflow()
        self._mode = "hatch"
        self._centered_frames = 0
        recover = getattr(self.hatch, "begin_home_recovery", None)
        if callable(recover):
            recover("hatch+hunt handoff could not confirm centered home")
        return self._choose_owned(self.hatch, frame, detections)

    def _choose_handoff(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        center_anchor = next(
            (
                item
                for item in detections
                if item.type == self.hunt.center_anchor_type
                and hypot(item.x - frame.width / 2, item.y - frame.height / 2)
                <= 100
            ),
            None,
        )
        # ``is_centered_home_screen`` already proves the unobscured hatch HUD
        # and measures the cyan base of the central incubator. Requiring the
        # hunting map's small egg anchor here deadlocks a successful handoff:
        # that landmark is legitimately off-screen once the hatch home is
        # centered.
        if is_centered_home_screen(frame, detections):
            self._centered_frames += 1
            if self._centered_frames < 2:
                return None
            self.logger.info(
                "Hatch+Hunt | centered home confirmed | resuming hatch",
            )
            self.hunt.reset_workflow()
            self._mode = "hatch"
            self._centered_frames = 0
            self._handoff_deadline = None
            complete_refresh = getattr(
                self.hatch,
                "complete_hunt_map_capacity_refresh",
                None,
            )
            if callable(complete_refresh) and complete_refresh():
                self.logger.info(
                    "Hatch+Hunt | hunt-map camera refresh complete | retrying capacity"
                )
                return self._choose_owned(self.hatch, frame, detections)
            begin_home_collection = getattr(
                self.hatch,
                "begin_home_collection",
                None,
            )
            if callable(begin_home_collection):
                begin_home_collection()
            return self._choose_owned(self.hatch, frame, detections)

        self._centered_frames = 0
        # A centred hunt egg with a temporarily missed hatch HUD anchor is
        # already safe; wait for the second proof instead of leaving the map.
        #
        # Only briefly, though.  The hunt map's centre egg proves the map is
        # in a normal state, never that the hatch home is reachable, so a map
        # parked with that egg centred satisfies this test on every frame: an
        # S9 trace sat 13px from centre and burned the whole 90s deadline in a
        # 152s loop (91s stalled, 61s hunting) while the nest button it needed
        # was visible at 0.957 the entire time.  Wait out a transient miss,
        # then fall through to the map-exit path below.
        # A frame that simply failed to match the animated egg is not evidence
        # the map moved, so it must not restart the wait. Only an egg seen
        # somewhere else proves the map is being worked.
        anchor_elsewhere = center_anchor is None and any(
            item.type == self.hunt.center_anchor_type for item in detections
        )
        if anchor_elsewhere:
            self._anchor_only_frames = 0
        elif center_anchor is not None or self._anchor_only_frames:
            # Count a centred egg, and carry the streak across a frame that
            # merely missed it -- but never start one from a map that has not
            # shown the egg centred at all.
            self._anchor_only_frames += 1
            if self._anchor_only_frames <= MAX_ANCHOR_ONLY_HANDOFF_FRAMES:
                return None

        hunt_controls = any(
            item.type in self.hunt.hunt_button_types for item in detections
        )
        map_evidence = any(
            item.type
            in {
                self.hunt.map_exit_type,
                self.hunt.forest_recenter_type,
                self.hunt.center_anchor_type,
                self.hunt.mailbox_type,
                self.hunt.dinosaur_type,
                *self.hunt.own_path_types,
            }
            for item in detections
        )
        # The hatch home anchor is already on screen: the map arrived, and the
        # only thing left is for the transition to settle so the geometric
        # proof can run. Asking for another recenter here taps the Forest
        # control, which starts a fresh camera move whose cloud wipe covers the
        # frame - the proof then fails, the anchor is seen again, and the tap
        # repeats. On s9 that loop lost every one of seven handoffs, and when
        # the cooldown handoff hit it the hatch side fused off entirely. Let
        # the settled frame come to us instead.
        home_anchor_visible = any(
            item.type == hatch_feature.HOME_ANCHOR for item in detections
        )
        if home_anchor_visible:
            self._home_anchor_waits += 1
            if self._home_anchor_waits <= MAX_HOME_ANCHOR_SETTLE_FRAMES:
                return None
            # The map settled and the home is here, just not centred well
            # enough. Recentring cannot close that gap - it resets the camera,
            # and s9 measured the identical (-4,146) residue afterwards on two
            # separate handoffs. The hatch side owns the measured drag that
            # does close it, and its docstring names this caller.
            if self._offer_home_to_recovery(frame, detections):
                return self.choose(frame, detections)
        else:
            self._home_anchor_waits = 0
        if map_evidence and not hunt_controls:
            recenter_reason = (
                "capacity camera refresh"
                if self._handoff_reason == "capacity camera refresh"
                else "hatch cooldown handoff"
            )
            self.hunt.request_external_recenter(recenter_reason)
        return self._choose_owned(self.hunt, frame, detections)

    def _offer_home_to_recovery(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> bool:
        """Hand a measurably off-centre home to the hatch recovery planner.

        Only when the pile is visible and merely off-centre. A frame with no
        pile at all is a different fault - the map never left the cave - and
        recovery's measured drag would have nothing to aim at, so that case
        keeps the existing recenter path.
        """

        if self._handoff_reason == "capacity camera refresh":
            return False
        try:
            if not _is_bright_outdoor_map(frame):
                return False
            if not hatch_feature.has_home_pile_structure(
                frame,
                egg_pile_point=(450.0, 1330.0),
                reference_width=900.0,
            ):
                return False
            offset = home_pile_offset(frame)
        except Exception:  # this path must never break the handoff
            return False
        if offset is None:
            return False
        recover = getattr(self.hatch, "begin_home_recovery", None)
        if not callable(recover):
            return False
        self.logger.info(
            "Hatch+Hunt | home visible but off-centre by (%.0f,%.0f)px"
            " | handing to hatch recovery",
            offset[0],
            offset[1],
        )
        self.hunt.reset_workflow()
        self._mode = "hatch"
        self._centered_frames = 0
        self._home_anchor_waits = 0
        self._handoff_deadline = None
        recover("hatch+hunt handoff found an off-centre home")
        return True

    def _begin_capacity_camera_refresh(self) -> bool:
        method = getattr(self.hatch, "begin_hunt_map_capacity_refresh", None)
        return bool(method()) if callable(method) else False

    def _fail_capacity_camera_refresh(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        reason: str,
    ) -> Target | None:
        fail = getattr(self.hatch, "fail_hunt_map_capacity_refresh", None)
        if callable(fail):
            fail(reason)
        self.hunt.reset_workflow()
        self._mode = "hatch"
        self._centered_frames = 0
        self._handoff_deadline = None
        return self.choose(frame, detections)

    def _choose_capacity_camera_refresh_depart(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        """Enter the hunt map, then hand straight back without selecting prey."""

        hunt_map_visible = any(
            item.type
            in {
                self.hunt.map_exit_type,
                self.hunt.center_anchor_type,
                self.hunt.mailbox_type,
                self.hunt.dinosaur_type,
                *self.hunt.own_path_types,
            }
            for item in detections
        )
        if hunt_map_visible:
            self._enter_handoff("capacity camera refresh")
            self.logger.info(
                "Hatch+Hunt | hunt map reached for capacity camera refresh; returning home"
            )
            return self._choose_handoff(frame, detections)
        return self._choose_owned(self.hunt, frame, detections)

    def _choose_owned(
        self,
        owner: Any,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        target = owner.choose(frame, detections)
        if target is not None:
            self._action_owner = owner
        return target
