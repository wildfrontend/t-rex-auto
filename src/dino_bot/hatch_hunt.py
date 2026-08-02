"""Alternate full hatch management with hunting during egg cooldowns."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import hypot
from typing import Any

from .full_hatch import FullHatchPlanner, is_centered_home_screen
from .models import Detection, Frame, Target
from .planning import HuntPlanner


class HatchHuntPlanner:
    """Run hunts while Full Hatch is in its no-ready-egg rescan wait.

    The final handoff window is reserved for finishing the current hunt and
    returning the collection map to its measured centre.  Full Hatch receives
    control only after both the hatch home anchor and hunt centre egg are
    visible for two consecutive frames.
    """

    def __init__(
        self,
        hatch: FullHatchPlanner,
        hunt: HuntPlanner,
        *,
        handoff_seconds: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.hatch = hatch
        self.hunt = hunt
        self.handoff_ms = max(0, round(handoff_seconds * 1000))
        self.logger = logger or logging.getLogger("dino_bot")
        self._mode = "hatch"
        self._action_owner: Any = None
        self._centered_frames = 0

    @property
    def completion_type(self) -> str:
        """Expose verified hunts to detector and event hooks."""

        return self.hunt.completion_type

    def is_complete(self) -> bool:
        return False

    def last_stage(self) -> str:
        child = self.hatch if self._mode == "hatch" else self.hunt
        stage = child.last_stage()
        return f"hatch_hunt_{self._mode}:{stage}"

    def choose(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        if self._mode == "hatch":
            target = self._choose_owned(self.hatch, frame, detections)
            if target is not None:
                return target
            if not self.hatch.is_hunt_cooldown_active():
                return None
            remaining = self.hatch.next_ready_delay_ms()
            if remaining <= self.handoff_ms:
                return None
            self._mode = "hunt"
            self._centered_frames = 0
            self.logger.info(
                "Hatch+Hunt | egg cooldown %.0fs | switching to hunt",
                remaining / 1000,
            )
            return self._choose_owned(self.hunt, frame, detections)

        if self._mode == "hunt":
            remaining = self.hatch.next_ready_delay_ms()
            if (
                not self.hatch.is_hunt_cooldown_active()
                or remaining <= self.handoff_ms
            ):
                self._mode = "handoff"
                self._centered_frames = 0
                self.logger.info(
                    "Hatch+Hunt | cooldown handoff window | remaining=%.0fs",
                    remaining / 1000,
                )
                return self._choose_handoff(frame, detections)
            return self._choose_owned(self.hunt, frame, detections)

        return self._choose_handoff(frame, detections)

    def next_ready_delay_ms(self) -> int:
        if self._mode == "hatch":
            return self.hatch.next_ready_delay_ms()
        if self._mode == "handoff":
            return 0
        until_handoff = max(
            0,
            self.hatch.next_ready_delay_ms() - self.handoff_ms,
        )
        hunt_delay = self.hunt.next_ready_delay_ms()
        if not hunt_delay:
            return 0
        return min(hunt_delay, until_handoff)

    def planning_detection_types(self) -> frozenset[str] | None:
        if self._mode != "hunt":
            return None
        return self.hunt.planning_detection_types()

    def verification_detection_types(self, target_type: str) -> frozenset[str]:
        method = getattr(self._action_owner, "verification_detection_types", None)
        return method(target_type) if callable(method) else frozenset()

    def on_action_success(self, target_type: str) -> None:
        method = getattr(self._action_owner, "on_action_success", None)
        if callable(method):
            method(target_type)

    def on_action_failure(self, target_type: str) -> None:
        method = getattr(self._action_owner, "on_action_failure", None)
        if callable(method):
            method(target_type)

    def on_retry_exhausted(self, target: Target) -> None:
        method = getattr(self._action_owner, "on_retry_exhausted", None)
        if callable(method):
            method(target)

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

    def reset_workflow(self) -> None:
        self.hatch.reset_workflow()
        self.hunt.reset_workflow()
        self._mode = "hatch"
        self._action_owner = None
        self._centered_frames = 0

    # Hunt diagnostics remain available to the shared engine while combined.
    def take_blind_escape(self) -> dict[str, Any] | None:
        return self.hunt.take_blind_escape() if self._mode != "hatch" else None

    def last_rejections(self) -> dict[str, int]:
        return self.hunt.last_rejections() if self._mode != "hatch" else {}

    def last_idle_seconds(self) -> float:
        return self.hunt.last_idle_seconds() if self._mode != "hatch" else 0.0

    def last_recenter_reason(self) -> str | None:
        return self.hunt.last_recenter_reason() if self._mode != "hatch" else None

    def last_blind_seconds(self) -> float:
        return self.hunt.last_blind_seconds() if self._mode != "hatch" else 0.0

    def last_supply(self) -> int:
        return self.hunt.last_supply() if self._mode != "hatch" else 0

    def anchor_measured(self) -> bool:
        return self.hunt.anchor_measured() if self._mode != "hatch" else False

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
            self.logger.info("Hatch+Hunt | centered home confirmed | resuming hatch")
            self.hunt.reset_workflow()
            self._mode = "hatch"
            self._centered_frames = 0
            return self._choose_owned(self.hatch, frame, detections)

        self._centered_frames = 0
        # A centred hunt egg with a temporarily missed hatch HUD anchor is
        # already safe; wait for the second proof instead of leaving the map.
        if center_anchor is not None:
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
        if map_evidence and not hunt_controls:
            self.hunt.request_external_recenter("hatch cooldown handoff")
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
