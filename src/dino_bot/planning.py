"""Reusable target-selection strategies."""

from __future__ import annotations

import json
from collections.abc import Sequence
from math import hypot
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
        own_path_types: Sequence[str] = ("own_hunt_path",),
        own_path_radius: float = 90.0,
        recenter_every: int = 10,
        safe_margin: int = 80,
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
        self.own_path_types = frozenset(own_path_types)
        self.own_path_radius = max(0.0, own_path_radius)
        self.recenter_every = max(1, recenter_every)
        self.safe_margin = max(0, safe_margin)
        self.await_hunt_frames = max(1, await_hunt_frames)
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._recenter_stage = 0
        self._pending_hunt_return = False
        self._hunt_count = 0
        self._last_anchor: tuple[float, float] | None = None

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
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
            exit_buttons = [
                item for item in detections if item.type == self.map_exit_type
            ]
            return super().choose(frame, exit_buttons)

        if self._recenter_stage == 2:
            anchors = [
                item for item in detections if item.type == self.center_anchor_type
            ]
            centered = any(
                hypot(item.x - frame.width / 2, item.y - frame.height / 2) <= 100
                for item in anchors
            )
            on_collect_map = any(
                item.type == self.map_exit_type for item in detections
            )
            if centered and on_collect_map:
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
            else:
                forest = [
                    item
                    for item in detections
                    if item.type == self.forest_recenter_type
                ]
                return super().choose(frame, forest)

        has_hunt_control = any(
            item.type in self.hunt_button_types for item in detections
        )
        if has_hunt_control:
            hunt_controls = [
                item for item in detections if item.type in self.hunt_button_types
            ]
            target = super().choose(frame, hunt_controls)
            if target is not None and target.type == self.completion_type:
                # A successful dinosaur interaction recenters the map. Coordinate
                # exclusions are only valid for the current attempt/viewport.
                self.clear_history()
                if not self._pending_hunt_return:
                    self._hunt_count += 1
                self._pending_hunt_return = True
                self._awaiting_hunt_button = False
                self._waited_frames = 0
            return target

        on_collect_map = any(
            item.type in {self.map_exit_type, self.center_anchor_type}
            for item in detections
        )
        if self._pending_hunt_return and on_collect_map:
            self._pending_hunt_return = False
            if self._hunt_count >= self.recenter_every:
                self._hunt_count = 0
                self._recenter_stage = 1
                exit_buttons = [
                    item for item in detections if item.type == self.map_exit_type
                ]
                return super().choose(frame, exit_buttons)

        if self._awaiting_hunt_button:
            self._waited_frames += 1
            if self._waited_frames < self.await_hunt_frames:
                return None
            self._awaiting_hunt_button = False
            self._waited_frames = 0

        navigation_types = {
            self.map_exit_type,
            self.forest_recenter_type,
            *self.recovery_button_types,
        }
        actionable = [item for item in detections if item.type not in navigation_types]
        anchor_position = self._last_anchor
        if anchor_position is not None:
            anchor_x, anchor_y = anchor_position
            # Treat every visible blue route marker as a buffered no-click zone.
            # If every dinosaur is inside that corridor, wait for another frame
            # instead of falling back to an unsafe dinosaur.
            safe_dinosaurs = [
                item
                for item in actionable
                if item.type == self.dinosaur_type
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
                nearest = min(
                    safe_dinosaurs,
                    key=lambda item: (
                        hypot(item.x - anchor_x, item.y - anchor_y),
                        -item.confidence,
                    ),
                )
                actionable.append(nearest)
        elif on_collect_map:
            self._recenter_stage = 1
            exit_buttons = [
                item for item in detections if item.type == self.map_exit_type
            ]
            return super().choose(frame, exit_buttons)
        target = super().choose(frame, actionable)
        if target is not None and target.type == self.dinosaur_type:
            if anchor_position is not None:
                self._last_anchor = (
                    anchor_position[0] + frame.width / 2 - target.x,
                    anchor_position[1] + frame.height / 2 - target.y,
                )
            self._awaiting_hunt_button = True
            self._waited_frames = 0
        return target
