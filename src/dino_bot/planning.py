"""Reusable target-selection strategies."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any

from .models import Detection, ExclusionZone, Frame, Target


class TargetPlanner:
    def __init__(
        self,
        target_types: Sequence[str] = ("resource",),
        strategy: str = "nearest_center",
        *,
        blocking_types: Sequence[str] = (),
        deduplicate_types: Sequence[str] = (),
        dedup_radius: float = 60.0,
        history_file: Path | None = None,
        history_limit: int = 500,
        retry_exhausted_cooldown_ms: int = 60_000,
        suppression_radius: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target_types = tuple(target_types)
        self.strategy = strategy
        self.blocking_types = frozenset(blocking_types)
        self.deduplicate_types = frozenset(deduplicate_types)
        self.dedup_radius = dedup_radius
        self.history_file = history_file
        self.history_limit = history_limit
        self.retry_exhausted_cooldown_ms = max(0, retry_exhausted_cooldown_ms)
        self.suppression_radius = max(0.0, suppression_radius)
        self.clock = clock
        # (type, x, y, radius, expires_at)
        self._suppressed: list[tuple[str, float, float, float, float]] = []
        self._history = self._load_history()
        self._stage = ""

    def last_stage(self) -> str:
        """Return what the planner was doing during the previous ``choose``.

        A cycle that plans nothing looks identical in the log whether the bot
        was waiting out a capacity cooldown, letting the map settle, or working
        through the mail flow. Those cost very different amounts of throughput,
        so the waiting branch names itself and the event stream can attribute
        idle time instead of reporting one undifferentiated "no target".
        """

        return self._stage

    def on_retry_exhausted(self, target: Target) -> None:
        """Stop re-selecting a target whose retry budget ran out.

        Without this the planner sees an unchanged frame on the next tick and
        picks the very same dead target again, so a single misdetection
        deadlocks the whole bot instead of costing one retry cycle.
        """

        self.suppress(target.type, target.x, target.y)

    def suppress(
        self,
        target_type: str,
        x: float,
        y: float,
        *,
        cooldown_ms: int | None = None,
        radius: float | None = None,
    ) -> None:
        cooldown = (
            self.retry_exhausted_cooldown_ms if cooldown_ms is None else cooldown_ms
        )
        if cooldown <= 0:
            return
        self._suppressed.append(
            (
                target_type,
                float(x),
                float(y),
                self.suppression_radius if radius is None else max(0.0, radius),
                self.clock() + cooldown / 1000,
            )
        )

    def is_suppressed(self, target_type: str, x: float, y: float) -> bool:
        now = self.clock()
        self._suppressed = [entry for entry in self._suppressed if entry[4] > now]
        return any(
            entry[0] == target_type
            and hypot(x - entry[1], y - entry[2]) <= entry[3]
            for entry in self._suppressed
        )

    def filter_suppressed(
        self,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        return [
            item
            for item in detections
            if not self.is_suppressed(item.type, item.x, item.y)
        ]

    def clear_suppressed(self) -> None:
        self._suppressed.clear()

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if any(item.type in self.blocking_types for item in detections):
            return None

        detections = self.filter_suppressed(detections)
        candidates: list[Detection] = []
        for target_type in self.target_types:
            candidates = [item for item in detections if item.type == target_type]
            if target_type in self.deduplicate_types:
                candidates = [
                    item
                    for item in candidates
                    if not self._was_selected(frame, item)
                ]
            if candidates:
                break
        if not candidates:
            return None
        if self.strategy == "highest_confidence":
            selected = max(candidates, key=lambda item: item.confidence)
        else:
            center_x, center_y = frame.width / 2, frame.height / 2
            selected = min(
                candidates,
                key=lambda item: (
                    hypot(item.x - center_x, item.y - center_y),
                    -item.confidence,
                ),
            )
        target = Target(
            type=selected.type,
            x=selected.x,
            y=selected.y,
            confidence=selected.confidence,
            detection=selected,
        )
        if selected.type in self.deduplicate_types:
            self._remember(frame, selected)
        return target

    def _was_selected(self, frame: Frame, detection: Detection) -> bool:
        for entry in self._history:
            if entry.get("type") != detection.type:
                continue
            previous_x = float(entry.get("x_ratio", -1)) * frame.width
            previous_y = float(entry.get("y_ratio", -1)) * frame.height
            if hypot(detection.x - previous_x, detection.y - previous_y) <= self.dedup_radius:
                return True
        return False

    def _remember(self, frame: Frame, detection: Detection) -> None:
        self._history.append(
            {
                "type": detection.type,
                "x_ratio": detection.x / frame.width,
                "y_ratio": detection.y / frame.height,
            }
        )
        self._history = self._history[-self.history_limit :]
        self._save_history()

    def _load_history(self) -> list[dict[str, Any]]:
        if self.history_file is None or not self.history_file.exists():
            return []
        try:
            payload = json.loads(self.history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)][-self.history_limit :]

    def _save_history(self) -> None:
        if self.history_file is None:
            return
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.history_file)

    def clear_history(self) -> None:
        self._history.clear()
        self._save_history()


class HuntPlanner(TargetPlanner):
    """Feature planner that never chains dinosaur taps while the map is moving."""

    def __init__(
        self,
        *args: Any,
        dinosaur_type: str = "dinosaur",
        hunt_button_types: Sequence[str] = (
            "hunt_button",
            "hunt_max_group_button",
            "hunt_confirm_button",
        ),
        completion_type: str = "hunt_confirm_button",
        map_exit_type: str = "map_exit_nest_button",
        forest_recenter_type: str = "forest_recenter_button",
        center_anchor_type: str = "map_center_egg",
        recovery_button_types: Sequence[str] = ("hunt_team_return_button",),
        interrupt_button_types: Sequence[str] = (
            "duplicate_login_close_button",
            "device_history_confirm_button",
            "startup_offer_dismiss",
            "startup_growth_result_back",
            "startup_auto_battle_close",
        ),
        own_path_types: Sequence[str] = ("own_hunt_path",),
        own_path_radius: float = 90.0,
        anchor_exclusion_radius: float = 50.0,
        dinosaur_failure_cooldown_ms: int = 5_000,
        dinosaur_failure_radius: float = 80.0,
        recenter_every: int = 10,
        mail_after_hunts: int = 30,
        mail_failure_limit: int = 3,
        mailbox_type: str = "mailbox_button",
        mail_collect_all_type: str = "mail_collect_all_button",
        mail_reward_collect_type: str = "mail_reward_collect_button",
        mail_close_type: str = "mail_close_button",
        hunt_dialog_close_type: str = "hunt_dialog_close_button",
        no_available_type: str = "no_available_dinosaurs",
        target_too_strong_type: str = "target_too_strong",
        capacity_full_type: str = "hunt_capacity_full",
        capacity_wait_seconds: float = 300.0,
        ring_width: float = 150.0,
        own_path_angle_degrees: float = 7.0,
        stalled_recenter_frames: int = 8,
        map_settle_frames: int = 2,
        map_settle_tolerance_px: float = 20.0,
        map_settle_max_frames: int = 12,
        safe_margin: int = 80,
        bottom_exclusion_px: int = 180,
        exclusion_zones: Sequence[ExclusionZone] = (),
        action_cooldowns_ms: dict[str, int] | None = None,
        await_hunt_frames: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dinosaur_type = dinosaur_type
        self.hunt_button_types = frozenset(hunt_button_types)
        self.completion_type = completion_type
        self.map_exit_type = map_exit_type
        self.forest_recenter_type = forest_recenter_type
        self.center_anchor_type = center_anchor_type
        self.recovery_button_types = frozenset(recovery_button_types)
        self.interrupt_button_types = frozenset(interrupt_button_types)
        self.own_path_types = frozenset(own_path_types)
        self.own_path_radius = max(0.0, own_path_radius)
        self.anchor_exclusion_radius = max(0.0, anchor_exclusion_radius)
        self.dinosaur_failure_cooldown_ms = max(
            0,
            dinosaur_failure_cooldown_ms,
        )
        self.dinosaur_failure_radius = max(0.0, dinosaur_failure_radius)
        self.recenter_every = max(1, recenter_every)
        self.mail_after_hunts = max(1, mail_after_hunts)
        self.mail_failure_limit = max(1, mail_failure_limit)
        self.mailbox_type = mailbox_type
        self.mail_collect_all_type = mail_collect_all_type
        self.mail_reward_collect_type = mail_reward_collect_type
        self.mail_close_type = mail_close_type
        self.hunt_dialog_close_type = hunt_dialog_close_type
        self.no_available_type = no_available_type
        self.target_too_strong_type = target_too_strong_type
        self.capacity_full_type = capacity_full_type
        self.capacity_wait_seconds = max(0.0, capacity_wait_seconds)
        self.ring_width = max(1.0, ring_width)
        self.own_path_angle_degrees = max(0.0, own_path_angle_degrees)
        self.stalled_recenter_frames = max(1, stalled_recenter_frames)
        self.map_settle_frames = max(1, map_settle_frames)
        self.map_settle_tolerance_px = max(0.0, map_settle_tolerance_px)
        self.map_settle_max_frames = max(
            self.map_settle_frames,
            map_settle_max_frames,
        )
        self.safe_margin = max(0, safe_margin)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.exclusion_zones = tuple(exclusion_zones)
        self.action_cooldowns_ms = dict(action_cooldowns_ms or {})
        self.await_hunt_frames = max(1, await_hunt_frames)
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._recenter_stage = 0
        self._recenter_dinosaur_frames = 0
        self._pending_hunt_return = False
        self._hunt_count = 0
        self._total_hunt_count = 0
        self._last_anchor: tuple[float, float] | None = None
        self._mail_stage = 0
        self._mail_failures = 0
        self._mailbox_full_recovery = False
        self._capacity_cooldown_until = 0.0
        self._action_cooldown_until = 0.0
        self._map_idle_frames = 0
        self._map_settle_active = False
        self._map_settle_observed_frames = 0
        self._map_settle_stable_frames = 0
        self._map_settle_anchor: tuple[float, float] | None = None
        self._map_settle_dinosaur: tuple[float, float] | None = None
        self._failed_dinosaur_positions: list[tuple[float, float, float]] = []
        self._last_selected_dinosaur: tuple[float, float] | None = None
        self._anchor_before_dinosaur: tuple[float, float] | None = None
        self._rejections: dict[str, int] = {}

    def on_action_success(self, target_type: str) -> None:
        """Commit hunt counters only after the confirmation tap is verified."""

        if target_type == self.hunt_dialog_close_type and self._mailbox_full_recovery:
            self._mailbox_full_recovery = False
            self._mail_stage = 1
            self._mail_failures = 0
            self._pending_hunt_return = False
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            self.clear_suppressed()
            return
        if target_type == self.dinosaur_type:
            self._last_selected_dinosaur = None
            self._anchor_before_dinosaur = None
        if self._mail_stage and target_type in self._mail_stage_by_type:
            # Progress clears the budget: a cycle that is advancing has not
            # earned the give-up that a repeatedly failing one has.
            self._mail_failures = 0
        cooldown_ms = self.action_cooldowns_ms.get(target_type, 0)
        if cooldown_ms:
            self._action_cooldown_until = time.monotonic() + cooldown_ms / 1000
        if target_type != self.completion_type:
            return
        self.clear_history()
        if not self._pending_hunt_return:
            self._hunt_count += 1
            self._total_hunt_count += 1
        self._pending_hunt_return = True
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._map_settle_active = True
        self._map_settle_observed_frames = 0
        self._map_settle_stable_frames = 0
        self._map_settle_anchor = None
        self._map_settle_dinosaur = None

    def on_action_failure(self, target_type: str) -> None:
        """Roll back state that ``choose`` committed before verification ran.

        Several stages advance the moment an action is *planned*, so a failed
        action leaves the planner believing it reached a screen that never
        appeared. Every such stage needs an entry here, not just the dinosaur.
        """

        if target_type == self.hunt_dialog_close_type and self._mailbox_full_recovery:
            # Stay in recovery stage so the dialog close can be retried.
            return

        if target_type == self.dinosaur_type:
            if (
                self._last_selected_dinosaur is not None
                and self.dinosaur_failure_cooldown_ms
            ):
                self._failed_dinosaur_positions.append(
                    (
                        *self._last_selected_dinosaur,
                        time.monotonic()
                        + self.dinosaur_failure_cooldown_ms / 1000,
                    )
                )
            if self._anchor_before_dinosaur is not None:
                self._last_anchor = self._anchor_before_dinosaur
            self._last_selected_dinosaur = None
            self._anchor_before_dinosaur = None
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            return

        if target_type in {self.forest_recenter_type, self.map_exit_type}:
            # Both taps set `_recenter_stage` while planning. Leaving it set
            # after a failure parks the planner in the anchor stage, where it
            # ignores every hunt control still on screen.
            self._recenter_stage = 0
            self._recenter_dinosaur_frames = 0
            return

        if self._mail_stage and target_type in self._mail_stage_by_type:
            # A mail cycle that cannot finish must not consume the run. Once
            # the rewards are claimed the mailbox stops offering the buttons
            # this flow waits for, so every retry fails the same way while
            # hunting - the actual job - is stopped. Spend a bounded number of
            # attempts and go back to the map.
            self._mail_failures += 1
            if self._mail_failures >= self.mail_failure_limit:
                self._abandon_mail()
                return
            # Step back to the stage that owns this button so the mail flow
            # retries it instead of waiting for a screen it never reached.
            self._mail_stage = self._mail_stage_by_type[target_type]

    def on_retry_exhausted_context(
        self,
        target: Target,
        detections: Sequence[Detection],
    ) -> bool:
        """Turn the observed full-mailbox hunt failure into a recovery flow.

        The game does not expose a stable machine-readable error flag. The
        reliable evidence from the real failure screen is the combination of
        an exhausted hunt-confirm action and the hunt dialog's red close
        button still being present. Normal confirmation dialogs are unaffected
        because recovery is armed only after the retry budget is exhausted.
        """

        if target.type != self.completion_type or not any(
            item.type == self.hunt_dialog_close_type for item in detections
        ):
            return False
        self._mailbox_full_recovery = True
        self._mail_stage = 0
        self._mail_failures = 0
        self._pending_hunt_return = False
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        return True

    def _abandon_mail(self) -> None:
        """End this mail cycle without completing it.

        The hunt counter is cleared along with the stage: leaving it armed
        re-enters the same failing flow at the next recenter, which is how a
        single missed screen turns into a bot that never hunts again.
        """

        self._mail_stage = 0
        self._mail_failures = 0
        self._mailbox_full_recovery = False
        self._total_hunt_count = 0

    @property
    def _mail_stage_by_type(self) -> dict[str, int]:
        return {
            self.mailbox_type: 1,
            self.mail_collect_all_type: 2,
            self.mail_reward_collect_type: 3,
            self.mail_close_type: 4,
        }

    def reset_workflow(self) -> None:
        """Discard screen-dependent state after the Android app is restarted."""
        self.clear_history()
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._recenter_stage = 0
        self._recenter_dinosaur_frames = 0
        self._pending_hunt_return = False
        self._last_anchor = None
        self._mail_stage = 0
        self._mail_failures = 0
        # A restart invalidates where the workflow stood, and the hunt counter
        # is part of that: leaving it armed sends the bot straight back into
        # the mail flow that caused the restart, restart after restart.
        self._total_hunt_count = 0
        self._capacity_cooldown_until = 0.0
        self._action_cooldown_until = 0.0
        self._map_idle_frames = 0
        self._map_settle_active = False
        self._map_settle_observed_frames = 0
        self._map_settle_stable_frames = 0
        self._map_settle_anchor = None
        self._map_settle_dinosaur = None
        self._failed_dinosaur_positions.clear()
        self._last_selected_dinosaur = None
        self._anchor_before_dinosaur = None
        self.clear_suppressed()

    def next_ready_delay_ms(self) -> int:
        """Return the remaining non-UI cooldown without blocking the engine.

        Suppression deadlines are deliberately excluded: the engine feeds this
        value to the stall watchdog as "expected wait", so counting them would
        let a suppressed phantom mute the very watchdog that has to rescue it.
        """

        now = time.monotonic()
        deadline = max(
            self._capacity_cooldown_until,
            self._action_cooldown_until,
        )
        return max(0, round((deadline - now) * 1000))

    def _observe_map_settle(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> bool:
        """Hold new dinosaur taps until the post-hunt map motion settles."""

        if not self._map_settle_active:
            return False

        self._map_settle_observed_frames += 1
        anchors = [
            item for item in detections if item.type == self.center_anchor_type
        ]
        if anchors:
            anchor = min(
                anchors,
                key=lambda item: hypot(
                    item.x - frame.width / 2,
                    item.y - frame.height / 2,
                ),
            )
            current = (float(anchor.x), float(anchor.y))
            previous = self._map_settle_anchor
            if (
                previous is not None
                and hypot(current[0] - previous[0], current[1] - previous[1])
                <= self.map_settle_tolerance_px
            ):
                self._map_settle_stable_frames += 1
            else:
                self._map_settle_stable_frames = 1
            self._map_settle_anchor = current
            self._map_settle_dinosaur = None
        else:
            dinosaurs = [
                item for item in detections if item.type == self.dinosaur_type
            ]
            if dinosaurs:
                previous = self._map_settle_dinosaur
                if previous is None:
                    dinosaur = min(
                        dinosaurs,
                        key=lambda item: hypot(
                            item.x - frame.width / 2,
                            item.y - frame.height / 2,
                        ),
                    )
                else:
                    dinosaur = min(
                        dinosaurs,
                        key=lambda item: hypot(
                            item.x - previous[0],
                            item.y - previous[1],
                        ),
                    )
                current = (float(dinosaur.x), float(dinosaur.y))
                if (
                    previous is not None
                    and hypot(current[0] - previous[0], current[1] - previous[1])
                    <= self.map_settle_tolerance_px
                ):
                    self._map_settle_stable_frames += 1
                else:
                    self._map_settle_stable_frames = 1
                self._map_settle_dinosaur = current
            else:
                self._map_settle_stable_frames = 0
                self._map_settle_anchor = None
                self._map_settle_dinosaur = None

        if (
            self._map_settle_stable_frames >= self.map_settle_frames
            or self._map_settle_observed_frames >= self.map_settle_max_frames
        ):
            self._map_settle_active = False
            self._map_settle_observed_frames = 0
            self._map_settle_stable_frames = 0
            self._map_settle_anchor = None
            self._map_settle_dinosaur = None
            return False
        return True

    def verification_detection_types(self, target_type: str) -> frozenset[str]:
        """Return active hunt exceptions that verification must not filter out."""

        target_types = {
            *self.recovery_button_types,
            self.no_available_type,
            self.target_too_strong_type,
        }
        if target_type == self.completion_type:
            target_types.update(
                {
                    self.dinosaur_type,
                    *self.own_path_types,
                    self.map_exit_type,
                    self.center_anchor_type,
                    self.mailbox_type,
                    self.capacity_full_type,
                    self.hunt_dialog_close_type,
                }
            )
        return frozenset(target_types)

    def can_reuse_verification_result(
        self,
        target_type: str,
        detections: Sequence[Detection],
    ) -> bool:
        """Return True when filtered verification already found the next action."""

        visible_types = {item.type for item in detections}
        if target_type == self.dinosaur_type:
            return bool(visible_types & self.hunt_button_types)
        if (
            target_type in self.hunt_button_types
            and target_type != self.completion_type
        ):
            return self.completion_type in visible_types
        if target_type == self.completion_type:
            return bool(
                visible_types
                & {
                    self.dinosaur_type,
                    self.map_exit_type,
                    self.center_anchor_type,
                    self.mailbox_type,
                }
            )
        return False

    @staticmethod
    def _angle_distance(left: float, right: float) -> float:
        difference = abs(left - right) % 360.0
        return min(difference, 360.0 - difference)

    def _in_exclusion_zone(self, frame: Frame, item: Detection) -> bool:
        return any(
            zone.contains(item.x, item.y, frame.width)
            for zone in self.exclusion_zones
        )

    def _reject_dinosaur(
        self,
        frame: Frame,
        item: Detection,
        anchor_x: float,
        anchor_y: float,
        detections: Sequence[Detection],
        established_angles: Sequence[float],
        team_status_buttons: Sequence[Detection],
    ) -> str | None:
        """Name the first rule that disqualifies a dinosaur, or None.

        Eight independent rules can empty the candidate list, and the text log
        reported all eight as a single "no actionable target". Naming them is
        what makes a stall explainable without re-deriving the anchor maths by
        hand afterwards.
        """

        # Screen-space guard: the bottom navigation area can resemble a
        # dinosaur and must never receive a hunting tap.
        if not (
            self.safe_margin <= item.x <= frame.width - self.safe_margin
            and self.safe_margin
            <= item.y
            <= frame.height - self.bottom_exclusion_px
        ):
            return "screen_margin"
        # The center egg is fixed near the viewport center, but its template
        # can disappear for an animated frame while the predicted map anchor
        # remains offset. Keep a screen-space guard as a second line of
        # defense against that false target.
        if (
            hypot(item.x - frame.width / 2, item.y - frame.height / 2)
            <= self.anchor_exclusion_radius
        ):
            return "screen_center"
        if not (
            self.safe_margin
            <= anchor_x + frame.width / 2 - item.x
            <= frame.width - self.safe_margin
            and self.safe_margin
            <= anchor_y + frame.height / 2 - item.y
            <= frame.height - self.safe_margin
        ):
            return "anchor_window"
        if hypot(item.x - anchor_x, item.y - anchor_y) <= self.anchor_exclusion_radius:
            return "anchor_radius"
        if any(
            hypot(item.x - failed_x, item.y - failed_y)
            <= self.dinosaur_failure_radius
            for failed_x, failed_y, _ in self._failed_dinosaur_positions
        ):
            return "failure_cooldown"
        if any(
            hypot(item.x - marker.x, item.y - marker.y) <= self.own_path_radius
            for marker in detections
            if marker.type in self.own_path_types
        ):
            return "own_path_marker"
        if any(
            self._angle_distance(
                degrees(atan2(item.y - anchor_y, item.x - anchor_x)) % 360.0,
                path_angle,
            )
            <= self.own_path_angle_degrees
            for path_angle in established_angles
        ):
            return "own_path_angle"
        if any(
            status.x - 190 <= item.x <= status.x + 190
            and status.y - 330 <= item.y <= status.y + 100
            for status in team_status_buttons
        ):
            return "team_status_panel"
        return None

    def last_rejections(self) -> dict[str, int]:
        """Return why the previous ``choose`` discarded each dinosaur."""

        return dict(self._rejections)

    def _established_path_angles(
        self,
        anchor_x: float,
        anchor_y: float,
        detections: Sequence[Detection],
    ) -> list[float]:
        marker_angles = [
            degrees(atan2(item.y - anchor_y, item.x - anchor_x)) % 360.0
            for item in detections
            if item.type in self.own_path_types
            and 50 <= hypot(item.x - anchor_x, item.y - anchor_y) <= 750
        ]
        cluster_tolerance = 4.0
        return [
            angle
            for angle in marker_angles
            if sum(
                self._angle_distance(angle, other) <= cluster_tolerance
                for other in marker_angles
            )
            >= 3
        ]

    def _choose_mail_target(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        by_type = {
            target_type: [item for item in detections if item.type == target_type]
            for target_type in (
                self.mailbox_type,
                self.mail_collect_all_type,
                self.mail_reward_collect_type,
                self.mail_close_type,
            )
        }
        if self._mail_stage <= 1:
            candidates = by_type[self.mailbox_type]
            target = super().choose(frame, candidates)
            if target is not None:
                self._mail_stage = 2
                return target
            self._resume_hunting_after_mail(frame, detections, by_type)
            return None
        if self._mail_stage == 2:
            candidates = by_type[self.mail_collect_all_type]
            if not candidates:
                candidates = by_type[self.mailbox_type]
            target = super().choose(frame, candidates)
            if target is not None:
                if target.type == self.mail_collect_all_type:
                    self._mail_stage = 3
                return target
            self._resume_hunting_after_mail(frame, detections, by_type)
            return None
        if self._mail_stage == 3:
            candidates = by_type[self.mail_reward_collect_type]
            if not candidates:
                candidates = by_type[self.mail_collect_all_type]
            target = super().choose(frame, candidates)
            if target is not None:
                if target.type == self.mail_reward_collect_type:
                    self._mail_stage = 4
                return target
            self._resume_hunting_after_mail(frame, detections, by_type)
            return None
        if self._mail_stage == 4:
            candidates = by_type[self.mail_close_type]
            if not candidates:
                candidates = by_type[self.mail_reward_collect_type]
            target = super().choose(frame, candidates)
            if target is not None:
                if target.type == self.mail_close_type:
                    self._mail_stage = 5
                return target
            # Stage 4 is where a missed close verification lands: the tap
            # worked, the overlay is gone, and neither button this stage waits
            # for will ever appear again. Without this the planner idles here
            # until the stall watchdog restarts the game.
            self._resume_hunting_after_mail(frame, detections, by_type)
            return None
        on_centered_map = any(
            item.type == self.center_anchor_type
            and hypot(item.x - frame.width / 2, item.y - frame.height / 2) <= 100
            for item in detections
        )
        if on_centered_map:
            self._mail_stage = 0
            self._mail_failures = 0
            self._total_hunt_count = 0
            return None
        # Planning advances to stage 5 before verification. BlueStacks can
        # occasionally ignore a tap, so keep choosing the close button while
        # the mail overlay is still visible. Only a centered map confirms that
        # the mail workflow has actually completed.
        close_target = super().choose(frame, by_type[self.mail_close_type])
        if close_target is not None:
            return close_target
        self._resume_hunting_after_mail(frame, detections, by_type)
        return None

    def _resume_hunting_after_mail(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        by_type: Mapping[str, Sequence[Detection]],
    ) -> bool:
        """Leave the mail flow when the map is back but the egg was missed.

        Every mail stage waits for a specific button, and none of them will
        appear once the overlay is closed. The map landmarks plus dinosaurs are
        proof enough that the flow is over: the animated centre egg misses
        template matching often enough that waiting for it is how the bot stops
        hunting entirely.
        """

        mail_overlay_visible = any(
            by_type[mail_type]
            for mail_type in (
                self.mail_collect_all_type,
                self.mail_reward_collect_type,
                self.mail_close_type,
            )
        )
        has_hunt_control = any(
            item.type in self.hunt_button_types for item in detections
        )
        has_map_landmark = any(
            item.type in {self.map_exit_type, self.mailbox_type}
            for item in detections
        )
        has_dinosaur = any(
            item.type == self.dinosaur_type for item in detections
        )
        if (
            mail_overlay_visible
            or has_hunt_control
            or not (has_map_landmark and has_dinosaur)
        ):
            return False
        self.clear_history()
        self._mail_stage = 0
        self._mail_failures = 0
        self._mailbox_full_recovery = False
        self._total_hunt_count = 0
        self._recenter_stage = 0
        self._last_anchor = (frame.width / 2, frame.height / 2)
        self._map_idle_frames = 0
        return True

    def _choose_map_exit(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        exit_buttons = [
            item for item in detections if item.type == self.map_exit_type
        ]
        target = super().choose(frame, exit_buttons)
        if target is not None:
            return target

        # The nest is animated and can briefly miss exact template matching.
        # A visible mailbox is a stable map-only landmark, so it safely
        # authorizes the fixed bottom-right nest coordinate as a fallback.
        own_path_map_evidence = (
            self._last_anchor is not None
            and frame.height > frame.width
            and any(item.type in self.own_path_types for item in detections)
        )
        if any(item.type == self.mailbox_type for item in detections) or own_path_map_evidence:
            fallback = Detection(
                type=self.map_exit_type,
                x=round(frame.width * 841 / 900),
                y=round(frame.height * 1295 / 1600),
                confidence=0.7,
                metadata={"detector": "map_landmark_fallback"},
            )
            # This synthetic target bypasses the base planner, so the
            # suppression list has to be consulted explicitly.
            if self.is_suppressed(fallback.type, fallback.x, fallback.y):
                return None
            return Target(
                type=fallback.type,
                x=fallback.x,
                y=fallback.y,
                confidence=fallback.confidence,
                detection=fallback,
            )
        return None

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        self._rejections = {}
        self._stage = "hunting"
        # Login/device-switch prompts and startup offers can interrupt any
        # workflow stage. Always clear them before resuming mail or hunting.
        #
        # Suppressed entries are dropped here rather than inside the base
        # planner: this branch returns unconditionally, so a misdetected
        # interrupt that survives to the `if` would keep the bot idling on a
        # phantom instead of falling through to the hunting flow below.
        interruptions = self.filter_suppressed(
            [item for item in detections if item.type in self.interrupt_button_types]
        )
        if interruptions:
            self._stage = "interrupt"
            return super().choose(frame, interruptions)

        if any(item.type in self.blocking_types for item in detections):
            self._stage = "blocked"
            return None

        if (
            self._action_cooldown_until
            and time.monotonic() < self._action_cooldown_until
        ):
            self._stage = "action_cooldown"
            return None

        visible_anchors = [
            item for item in detections if item.type == self.center_anchor_type
        ]
        if visible_anchors:
            anchor = min(
                visible_anchors,
                key=lambda item: hypot(
                    item.x - frame.width / 2,
                    item.y - frame.height / 2,
                ),
            )
            self._last_anchor = (float(anchor.x), float(anchor.y))

        if self._observe_map_settle(frame, detections):
            self._stage = "map_settle"
            return None

        if self._mailbox_full_recovery:
            self._stage = "mailbox_full_recovery"
            close_buttons = [
                item
                for item in detections
                if item.type == self.hunt_dialog_close_type
            ]
            close_target = super().choose(frame, close_buttons)
            if close_target is not None:
                return close_target
            # If the close tap took effect but verification missed the animated
            # transition, a visible mailbox proves we are back on the map.
            if any(item.type == self.mailbox_type for item in detections):
                self._mailbox_full_recovery = False
                self._mail_stage = 1
                self._mail_failures = 0
                self.clear_suppressed()
            else:
                return None

        if self._mail_stage:
            self._stage = "mail"
            return self._choose_mail_target(frame, detections)

        actionable_hunt_controls = self.filter_suppressed(
            [item for item in detections if item.type in self.hunt_button_types]
        )
        # "Target too strong" is the game's verdict on the hunt already on
        # screen, not a statement about an idle map, so it outranks the confirm
        # button underneath it. Deferring to that button cost 26 seconds of taps
        # that could never work, and burned its retry budget: the confirm button
        # sits at a fixed coordinate, so the suppression that finally broke the
        # loop also disabled the next hunt's confirmation for a minute.
        too_strong = self.filter_suppressed(
            [item for item in detections if item.type == self.target_too_strong_type]
        )
        if too_strong:
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            self._stage = "hunt_unavailable"
            return super().choose(frame, too_strong)

        # "No available dinosaurs" must still never beat a hunt control that is
        # already on screen. The team sheet opens showing an empty roster, so
        # reading it before the max-group tap turns every hunt into a cancel and
        # the confirm button is never reached at all.
        #
        # A suppressed hunt control does not count: once it has burned its
        # retries there is nothing left to defer to, and the sheet still has to
        # be closed to get back to the map.
        unavailable = self.filter_suppressed(
            [item for item in detections if item.type == self.no_available_type]
        )
        if unavailable and not actionable_hunt_controls:
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            self._stage = "hunt_unavailable"
            return super().choose(frame, unavailable)

        team_status_buttons = [
            item for item in detections if item.type in self.recovery_button_types
        ]
        if team_status_buttons:
            self._awaiting_hunt_button = False
            self._waited_frames = 0

        if self._recenter_stage == 0:
            forest = [
                item for item in detections if item.type == self.forest_recenter_type
            ]
            if forest:
                self._stage = "recenter"
                target = super().choose(frame, forest)
                if target is not None:
                    self._recenter_stage = 2
                    self._recenter_dinosaur_frames = 0
                return target

        if self._recenter_stage == 1:
            self._stage = "recenter"
            forest = [
                item for item in detections if item.type == self.forest_recenter_type
            ]
            if forest:
                target = super().choose(frame, forest)
                if target is not None:
                    self._recenter_stage = 2
                    self._recenter_dinosaur_frames = 0
                return target
            return self._choose_map_exit(frame, detections)

        # A visible hunt control proves the recenter already left the map view,
        # so waiting for the egg anchor would ignore an action that is ready
        # right now. Release the stage instead of stalling on a missed anchor.
        if self._recenter_stage == 2 and actionable_hunt_controls:
            self._recenter_stage = 0
            self._recenter_dinosaur_frames = 0

        if self._recenter_stage == 2:
            self._stage = "recenter"
            anchors = [
                item for item in detections if item.type == self.center_anchor_type
            ]
            centered = any(
                hypot(item.x - frame.width / 2, item.y - frame.height / 2) <= 100
                for item in anchors
            )
            if centered:
                self.clear_history()
                self._recenter_stage = 0
                self._recenter_dinosaur_frames = 0
                centered_anchor = min(
                    anchors,
                    key=lambda item: hypot(
                        item.x - frame.width / 2,
                        item.y - frame.height / 2,
                    ),
                )
                self._last_anchor = (
                    float(centered_anchor.x),
                    float(centered_anchor.y),
                )
                if self._total_hunt_count >= self.mail_after_hunts:
                    self._mail_stage = 1
                    self._mail_failures = 0
                return None
            # Planning the forest tap advances the state before verification.
            # If BlueStacks ignores that tap, the forest button remains visible
            # and no egg anchor appears. Retry the same safe transition instead
            # of waiting forever in the anchor stage.
            if not anchors:
                forest = [
                    item
                    for item in detections
                    if item.type == self.forest_recenter_type
                ]
                if forest:
                    self._recenter_dinosaur_frames = 0
                    return super().choose(frame, forest)
                has_hunt_control = any(
                    item.type in self.hunt_button_types for item in detections
                )
                has_map_landmark = any(
                    item.type in {self.map_exit_type, self.mailbox_type}
                    for item in detections
                )
                has_dinosaur = any(
                    item.type == self.dinosaur_type for item in detections
                )
                if has_dinosaur and not has_hunt_control:
                    self._recenter_dinosaur_frames += 1
                else:
                    self._recenter_dinosaur_frames = 0
                dinosaur_only_confirmed = (
                    self._recenter_dinosaur_frames >= self.map_settle_frames
                )
                if (
                    has_dinosaur
                    and not has_hunt_control
                    and (has_map_landmark or dinosaur_only_confirmed)
                ):
                    # The forest transition succeeded, but the animated egg
                    # anchor and map landmarks can miss template matching. Two
                    # consecutive dinosaur-only frames prove that the forest
                    # button disappeared into the collection map. Retain a
                    # safe synthetic center instead of waiting forever.
                    self.clear_history()
                    self._recenter_stage = 0
                    self._recenter_dinosaur_frames = 0
                    self._last_anchor = (frame.width / 2, frame.height / 2)
                    self._map_idle_frames = 0
                    if self._total_hunt_count >= self.mail_after_hunts:
                        self._mail_stage = 1
                    self._mail_failures = 0
                    return None
            return super().choose(frame, anchors)

        has_hunt_control = any(
            item.type in self.hunt_button_types for item in detections
        )
        on_collect_map = any(
            item.type in {
                self.map_exit_type,
                self.center_anchor_type,
                self.mailbox_type,
            }
            for item in detections
        )
        # A map landmark can miss for one animated frame. Seeing dinosaurs
        # without any hunt control is itself sufficient evidence that the
        # previous confirmed hunt returned to the collection map. This keeps
        # the batch counter exact instead of carrying the pending flag into
        # the next confirmation.
        if not has_hunt_control and any(
            item.type == self.dinosaur_type for item in detections
        ):
            on_collect_map = True
        if (
            self._last_anchor is not None
            and frame.height > frame.width
            and any(item.type in self.own_path_types for item in detections)
        ):
            on_collect_map = True
        if self._pending_hunt_return and on_collect_map and not has_hunt_control:
            self._pending_hunt_return = False
            if self._hunt_count >= self.recenter_every:
                self._hunt_count = 0
                self._recenter_stage = 1
                self._stage = "recenter"
                return self._choose_map_exit(frame, detections)

        now = time.monotonic()
        if now < self._capacity_cooldown_until:
            self._stage = "capacity_wait"
            return None
        if any(item.type == self.capacity_full_type for item in detections):
            self._capacity_cooldown_until = now + self.capacity_wait_seconds
            self._stage = "capacity_wait"
            return None

        if has_hunt_control:
            self._map_idle_frames = 0
            self._stage = "hunt_control"
            hunt_controls = [
                item for item in detections if item.type in self.hunt_button_types
            ]
            target = super().choose(frame, hunt_controls)
            return target

        if self._awaiting_hunt_button:
            self._waited_frames += 1
            if self._waited_frames < self.await_hunt_frames:
                self._stage = "await_hunt"
                return None
            self._awaiting_hunt_button = False
            self._waited_frames = 0

        navigation_types = {
            self.map_exit_type,
            self.forest_recenter_type,
            self.center_anchor_type,
            self.mailbox_type,
            self.mail_collect_all_type,
            self.mail_reward_collect_type,
            self.mail_close_type,
            self.hunt_dialog_close_type,
            *self.recovery_button_types,
        }
        # Fixed overlays such as the left buff stack sit on top of the map and
        # keep scrolling dinosaurs behind them. Drop those candidates before any
        # anchor logic so both the anchored and the fallback path stay covered.
        actionable = []
        for item in detections:
            if item.type in navigation_types:
                continue
            if item.type == self.dinosaur_type and self._in_exclusion_zone(frame, item):
                self._rejections["exclusion_zone"] = (
                    self._rejections.get("exclusion_zone", 0) + 1
                )
                continue
            actionable.append(item)
        anchor_position = self._last_anchor
        self._failed_dinosaur_positions = [
            entry
            for entry in self._failed_dinosaur_positions
            if entry[2] > now
        ]
        if anchor_position is not None:
            anchor_x, anchor_y = anchor_position
            established_angles = self._established_path_angles(
                anchor_x,
                anchor_y,
                detections,
            )
            # Treat every visible blue route marker as a buffered no-click zone.
            # If every dinosaur is inside that corridor, wait for another frame
            # instead of falling back to an unsafe dinosaur.
            safe_dinosaurs = []
            for item in actionable:
                if item.type != self.dinosaur_type:
                    continue
                reason = self._reject_dinosaur(
                    frame,
                    item,
                    anchor_x,
                    anchor_y,
                    detections,
                    established_angles,
                    team_status_buttons,
                )
                if reason is None:
                    safe_dinosaurs.append(item)
                else:
                    self._rejections[reason] = self._rejections.get(reason, 0) + 1
            actionable = [
                item for item in actionable if item.type != self.dinosaur_type
            ]
            if safe_dinosaurs:
                def radial_key(item: Detection) -> tuple[float, float, float, float]:
                    distance = hypot(item.x - anchor_x, item.y - anchor_y)
                    angle = degrees(atan2(item.y - anchor_y, item.x - anchor_x)) % 360.0
                    clearance = min(
                        (
                            self._angle_distance(angle, path_angle)
                            for path_angle in established_angles
                        ),
                        default=180.0,
                    )
                    return (
                        distance // self.ring_width,
                        -clearance,
                        distance,
                        -item.confidence,
                    )

                nearest = min(
                    safe_dinosaurs,
                    key=radial_key,
                )
                actionable.append(nearest)
        elif on_collect_map:
            self._recenter_stage = 1
            return self._choose_map_exit(frame, detections)
        target = super().choose(frame, actionable)
        if target is None and on_collect_map:
            self._map_idle_frames += 1
            if self._map_idle_frames >= self.stalled_recenter_frames:
                self._map_idle_frames = 0
                self._recenter_stage = 1
                self._stage = "recenter"
                return self._choose_map_exit(frame, detections)
        elif target is not None:
            self._map_idle_frames = 0
        if target is not None and target.type == self.dinosaur_type:
            self._last_selected_dinosaur = (float(target.x), float(target.y))
            self._anchor_before_dinosaur = anchor_position
            if anchor_position is not None:
                self._last_anchor = (
                    anchor_position[0] + frame.width / 2 - target.x,
                    anchor_position[1] + frame.height / 2 - target.y,
                )
            self._awaiting_hunt_button = True
            self._waited_frames = 0
        return target
