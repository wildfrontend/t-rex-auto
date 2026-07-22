"""Reusable target-selection strategies."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any

from .models import Detection, Frame, Target


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
    ) -> None:
        self.target_types = tuple(target_types)
        self.strategy = strategy
        self.blocking_types = frozenset(blocking_types)
        self.deduplicate_types = frozenset(deduplicate_types)
        self.dedup_radius = dedup_radius
        self.history_file = history_file
        self.history_limit = history_limit
        self._history = self._load_history()

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if any(item.type in self.blocking_types for item in detections):
            return None

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
        recenter_every: int = 10,
        mail_after_hunts: int = 30,
        mailbox_type: str = "mailbox_button",
        mail_collect_all_type: str = "mail_collect_all_button",
        mail_reward_collect_type: str = "mail_reward_collect_button",
        mail_close_type: str = "mail_close_button",
        no_available_type: str = "no_available_dinosaurs",
        target_too_strong_type: str = "target_too_strong",
        capacity_full_type: str = "hunt_capacity_full",
        capacity_wait_seconds: float = 300.0,
        ring_width: float = 150.0,
        own_path_angle_degrees: float = 7.0,
        stalled_recenter_frames: int = 8,
        safe_margin: int = 80,
        bottom_exclusion_px: int = 180,
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
        self.recenter_every = max(1, recenter_every)
        self.mail_after_hunts = max(1, mail_after_hunts)
        self.mailbox_type = mailbox_type
        self.mail_collect_all_type = mail_collect_all_type
        self.mail_reward_collect_type = mail_reward_collect_type
        self.mail_close_type = mail_close_type
        self.no_available_type = no_available_type
        self.target_too_strong_type = target_too_strong_type
        self.capacity_full_type = capacity_full_type
        self.capacity_wait_seconds = max(0.0, capacity_wait_seconds)
        self.ring_width = max(1.0, ring_width)
        self.own_path_angle_degrees = max(0.0, own_path_angle_degrees)
        self.stalled_recenter_frames = max(1, stalled_recenter_frames)
        self.safe_margin = max(0, safe_margin)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.await_hunt_frames = max(1, await_hunt_frames)
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._recenter_stage = 0
        self._pending_hunt_return = False
        self._hunt_count = 0
        self._total_hunt_count = 0
        self._last_anchor: tuple[float, float] | None = None
        self._mail_stage = 0
        self._capacity_cooldown_until = 0.0
        self._map_idle_frames = 0

    def on_action_success(self, target_type: str) -> None:
        """Commit hunt counters only after the confirmation tap is verified."""

        if target_type != self.completion_type:
            return
        self.clear_history()
        if not self._pending_hunt_return:
            self._hunt_count += 1
            self._total_hunt_count += 1
        self._pending_hunt_return = True
        self._awaiting_hunt_button = False
        self._waited_frames = 0

    def reset_workflow(self) -> None:
        """Discard screen-dependent state after the Android app is restarted."""
        self.clear_history()
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._recenter_stage = 0
        self._pending_hunt_return = False
        self._last_anchor = None
        self._mail_stage = 0
        self._capacity_cooldown_until = 0.0
        self._map_idle_frames = 0

    @staticmethod
    def _angle_distance(left: float, right: float) -> float:
        difference = abs(left - right) % 360.0
        return min(difference, 360.0 - difference)

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
        if self._mail_stage == 2:
            candidates = by_type[self.mail_collect_all_type]
            if not candidates:
                candidates = by_type[self.mailbox_type]
            target = super().choose(frame, candidates)
            if target is not None and target.type == self.mail_collect_all_type:
                self._mail_stage = 3
            return target
        if self._mail_stage == 3:
            candidates = by_type[self.mail_reward_collect_type]
            if not candidates:
                candidates = by_type[self.mail_collect_all_type]
            target = super().choose(frame, candidates)
            if target is not None and target.type == self.mail_reward_collect_type:
                self._mail_stage = 4
            return target
        if self._mail_stage == 4:
            candidates = by_type[self.mail_close_type]
            if not candidates:
                candidates = by_type[self.mail_reward_collect_type]
            target = super().choose(frame, candidates)
            if target is not None and target.type == self.mail_close_type:
                self._mail_stage = 5
            return target
        on_centered_map = any(
            item.type == self.center_anchor_type
            and hypot(item.x - frame.width / 2, item.y - frame.height / 2) <= 100
            for item in detections
        )
        if on_centered_map:
            self._mail_stage = 0
            self._total_hunt_count = 0
            return None
        # Planning advances to stage 5 before verification. BlueStacks can
        # occasionally ignore a tap, so keep choosing the close button while
        # the mail overlay is still visible. Only a centered map confirms that
        # the mail workflow has actually completed.
        close_target = super().choose(frame, by_type[self.mail_close_type])
        if close_target is not None:
            return close_target
        return None

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
            return Target(
                type=fallback.type,
                x=fallback.x,
                y=fallback.y,
                confidence=fallback.confidence,
                detection=fallback,
            )
        return None

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        # Login/device-switch prompts and startup offers can interrupt any
        # workflow stage. Always clear them before resuming mail or hunting.
        interruptions = [
            item for item in detections if item.type in self.interrupt_button_types
        ]
        if interruptions:
            return super().choose(frame, interruptions)

        if any(item.type in self.blocking_types for item in detections):
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

        if self._mail_stage:
            return self._choose_mail_target(frame, detections)

        unavailable = [
            item
            for item in detections
            if item.type in {self.no_available_type, self.target_too_strong_type}
        ]
        if unavailable:
            self._awaiting_hunt_button = False
            self._waited_frames = 0
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
                target = super().choose(frame, forest)
                if target is not None:
                    self._recenter_stage = 2
                return target

        if self._recenter_stage == 1:
            forest = [
                item for item in detections if item.type == self.forest_recenter_type
            ]
            if forest:
                target = super().choose(frame, forest)
                if target is not None:
                    self._recenter_stage = 2
                return target
            return self._choose_map_exit(frame, detections)

        if self._recenter_stage == 2:
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
                    return super().choose(frame, forest)
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
                return self._choose_map_exit(frame, detections)

        now = time.monotonic()
        if now < self._capacity_cooldown_until:
            return None
        if any(item.type == self.capacity_full_type for item in detections):
            self._capacity_cooldown_until = now + self.capacity_wait_seconds
            return None

        if has_hunt_control:
            self._map_idle_frames = 0
            hunt_controls = [
                item for item in detections if item.type in self.hunt_button_types
            ]
            target = super().choose(frame, hunt_controls)
            return target

        if self._awaiting_hunt_button:
            self._waited_frames += 1
            if self._waited_frames < self.await_hunt_frames:
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
            *self.recovery_button_types,
        }
        actionable = [item for item in detections if item.type not in navigation_types]
        anchor_position = self._last_anchor
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
            safe_dinosaurs = [
                item
                for item in actionable
                if item.type == self.dinosaur_type
                # Screen-space guard: the bottom navigation area can resemble
                # a dinosaur and must never receive a hunting tap.
                and self.safe_margin <= item.x <= frame.width - self.safe_margin
                and self.safe_margin
                <= item.y
                <= frame.height - self.bottom_exclusion_px
                and self.safe_margin
                <= anchor_x + frame.width / 2 - item.x
                <= frame.width - self.safe_margin
                and self.safe_margin
                <= anchor_y + frame.height / 2 - item.y
                <= frame.height - self.safe_margin
                and all(
                    hypot(item.x - marker.x, item.y - marker.y)
                    > self.own_path_radius
                    for marker in detections
                    if marker.type in self.own_path_types
                )
                and all(
                    self._angle_distance(
                        degrees(atan2(item.y - anchor_y, item.x - anchor_x))
                        % 360.0,
                        path_angle,
                    )
                    > self.own_path_angle_degrees
                    for path_angle in established_angles
                )
                and all(
                    not (
                        status.x - 190 <= item.x <= status.x + 190
                        and status.y - 330 <= item.y <= status.y + 100
                    )
                    for status in team_status_buttons
                )
            ]
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
                return self._choose_map_exit(frame, detections)
        elif target is not None:
            self._map_idle_frames = 0
        if target is not None and target.type == self.dinosaur_type:
            if anchor_position is not None:
                self._last_anchor = (
                    anchor_position[0] + frame.width / 2 - target.x,
                    anchor_position[1] + frame.height / 2 - target.y,
                )
            self._awaiting_hunt_button = True
            self._waited_frames = 0
        return target
