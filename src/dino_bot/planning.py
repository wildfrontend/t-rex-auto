"""Reusable target-selection strategies."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any

from .models import Detection, ExclusionZone, Frame, Target, VerificationResult


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
        logger: logging.Logger | None = None,
    ) -> None:
        # Every escalation path this planner owns reports what it did before
        # acting on it; without a logger the repeated-failure recovery raised
        # AttributeError and took the whole run down with it.
        self.logger = logger or logging.getLogger("dino_bot")
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

    # Stages that plan nothing because they are deliberately waiting out a
    # deadline they set themselves. Every other empty cycle is the planner
    # failing to find work, which is what the blind-idle timer measures.
    #
    # `mail` is here because it carries its own deadline, not because it never
    # stalls. A stage with a specific timeout has to own its own release: with
    # both timers on the default 20 seconds they fired on the same cycle, so a
    # mail flow that had just handled itself still raised a blind-stall event
    # and a snapshot - and raising `mail_stage_timeout_seconds` above
    # `blind_idle_seconds` did nothing at all, because the generic release got
    # there first and abandoned the flow anyway.
    _BOUNDED_WAIT_STAGES = frozenset(
        {
            "capacity_wait",
            "action_cooldown",
            "map_settle",
            "blocked",
            "hunt_unavailable",
            "interrupt",
            "mail",
        }
    )

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
            "server_error_restart_button",
        ),
        launch_only_types: Sequence[str] = (
            "duplicate_login_close_button",
            "device_history_confirm_button",
            "startup_offer_dismiss",
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
        autoplace_cancel_type: str = "hunt_autoplace_cancel_button",
        autoplace_refused_wait_seconds: float = 180.0,
        capacity_wait_seconds: float = 300.0,
        ring_width: float = 150.0,
        own_path_angle_degrees: float = 7.0,
        stalled_recenter_seconds: float = 10.0,
        recenter_min_candidates: int = 1,
        empty_supply_recenter_frames: int = 2,
        blind_idle_seconds: float = 20.0,
        mail_stage_timeout_seconds: float = 20.0,
        map_settle_frames: int = 2,
        map_settle_tolerance_px: float = 20.0,
        map_settle_max_frames: int = 12,
        safe_margin: int = 80,
        max_center_distance_px: float = 600.0,
        bottom_exclusion_px: int = 180,
        exclusion_zones: Sequence[ExclusionZone] = (),
        action_cooldowns_ms: dict[str, int] | None = None,
        await_hunt_frames: int = 5,
        stage_scoped_scan: bool = True,
        full_scan_interval_seconds: float = 30.0,
        full_scan_after_idle_cycles: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dinosaur_type = dinosaur_type
        ordered_hunt_button_types = tuple(hunt_button_types)
        self.hunt_button_types = frozenset(ordered_hunt_button_types)
        self.hunt_entry_type = (
            ordered_hunt_button_types[0] if ordered_hunt_button_types else "hunt_button"
        )
        self.completion_type = completion_type
        self.map_exit_type = map_exit_type
        self.forest_recenter_type = forest_recenter_type
        self.center_anchor_type = center_anchor_type
        self.recovery_button_types = frozenset(recovery_button_types)
        self.interrupt_button_types = frozenset(interrupt_button_types)
        # The subset of the interrupts only the launch sequence can raise.
        # They are template matches and the priciest part of a scan; the
        # rest are layout checks costing under a millisecond, so those stay
        # visible to every scan.
        self.launch_only_types = frozenset(launch_only_types)
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
        self.autoplace_cancel_type = autoplace_cancel_type
        self.autoplace_refused_wait_seconds = max(
            0.0,
            autoplace_refused_wait_seconds,
        )
        self.capacity_wait_seconds = max(0.0, capacity_wait_seconds)
        self.ring_width = max(1.0, ring_width)
        self.own_path_angle_degrees = max(0.0, own_path_angle_degrees)
        self.stalled_recenter_seconds = max(0.001, stalled_recenter_seconds)
        # Recentering is a resupply operation: it exists so the next few scans
        # have enough dinosaurs to choose from, not to put the egg anywhere in
        # particular. This is the supply floor that triggers it.
        self.recenter_min_candidates = max(1, recenter_min_candidates)
        self.empty_supply_recenter_frames = max(
            1,
            empty_supply_recenter_frames,
        )
        self.blind_idle_seconds = max(0.001, blind_idle_seconds)
        self.mail_stage_timeout_seconds = max(0.001, mail_stage_timeout_seconds)
        self.map_settle_frames = max(1, map_settle_frames)
        self.map_settle_tolerance_px = max(0.0, map_settle_tolerance_px)
        self.map_settle_max_frames = max(
            self.map_settle_frames,
            map_settle_max_frames,
        )
        self.safe_margin = max(0, safe_margin)
        # Tapping a dinosaur recenters the map on it, and the further the tap
        # lands from the viewport center the less often the hunt panel opens
        # at all. A measured 161-minute run: taps within 300 px succeeded 86%
        # of the time, 300-500 px 69%, and beyond 500 px only 21%. Past 600 px
        # the whole band produced 2 hunts out of 31 taps, so the candidates it
        # removes are almost pure waste. 0 disables the limit.
        self.max_center_distance_px = max(0.0, max_center_distance_px)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.exclusion_zones = tuple(exclusion_zones)
        self.action_cooldowns_ms = dict(action_cooldowns_ms or {})
        self.await_hunt_frames = max(1, await_hunt_frames)
        self.stage_scoped_scan = bool(stage_scoped_scan)
        self.full_scan_interval_seconds = max(0.0, full_scan_interval_seconds)
        self.full_scan_after_idle_cycles = max(1, full_scan_after_idle_cycles)
        self._scoped_idle_cycles = 0
        self._last_full_scan: float | None = None
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
        # 拒絕過「巢的自動配置」之後,隊伍面板還開著,必須先收掉才能等待。
        self._autoplace_refused = False
        self._capacity_cooldown_until = 0.0
        self._action_cooldown_until = 0.0
        self._map_idle_since: float | None = None
        self._last_map_idle_seconds = 0.0
        self._recenter_reason: str | None = None
        # Whether `_last_anchor` came from a detected egg or was inferred from
        # the last tap. Only a measured anchor is worth rejecting candidates
        # over; a predicted one accumulates a fresh error every hunt.
        self._anchor_measured = False
        self._last_supply = 0
        self._empty_supply_frames = 0
        self._no_target_since: float | None = None
        self._last_blind_seconds = 0.0
        self._blind_escapes = 0
        self._pending_blind_escape: dict[str, Any] | None = None
        self._mail_progress_since: float | None = None
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

        if target_type == self.autoplace_cancel_type:
            # The dialog is gone; the sheet underneath still is not. Mark the
            # refusal so the close rung fires even though Hunt looks pressable.
            self._autoplace_refused = True
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            return
        if target_type == self.hunt_dialog_close_type and self._autoplace_refused:
            # Back on the map with the sheet shut. The only team the game will
            # field right now is one it refuses to send without reshuffling the
            # nest, so stop asking until the hunting parties return.
            self._autoplace_refused = False
            self._capacity_cooldown_until = (
                time.monotonic() + self.autoplace_refused_wait_seconds
            )
            self._stage = "capacity_wait"
            self.logger.warning(
                "Hunt | only nest parents are selectable | refused nest"
                " auto-arrange and waiting %.0fs",
                self.autoplace_refused_wait_seconds,
            )
            return
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

    def on_action_failure_context(
        self,
        target: Target,
        frame: Frame | None,
        detections: Sequence[Detection],
        attempts: int,
    ) -> None:
        """Abandon an inert hunt entry after two map-confirmed failures.

        A stale hunt-button match can be nearly perfect while sitting on the
        live collection map. Two failed taps plus stable map landmarks prove
        that retrying the same coordinate cannot open the hunt sheet. Suppress
        only that coordinate and let normal map planning choose another target.
        """

        if (
            target.type != self.hunt_entry_type
            or attempts < 2
            or frame is None
            or not any(
                item.type
                in {
                    self.center_anchor_type,
                    self.map_exit_type,
                    self.mailbox_type,
                }
                for item in detections
            )
        ):
            return
        self.suppress(target.type, target.x, target.y)
        self._awaiting_hunt_button = False
        self._waited_frames = 0

    def on_retry_exhausted(self, target: Target) -> None:
        """Suppress a spent target and release hunt-entry waiting state."""

        super().on_retry_exhausted(target)
        if target.type == self.hunt_entry_type:
            self._awaiting_hunt_button = False
            self._waited_frames = 0

    def recover_from_action_failures(
        self,
        target: Target,
        stage: str,
        episodes: int,
        frame: Frame | None,
        detections: Sequence[Detection],
    ) -> bool:
        """Unwind all hunt stages before another failed budget is attempted."""

        del detections
        if frame is None:
            return False
        self.suppress(target.type, target.x, target.y)
        self._release_stage_machines(frame)
        self._begin_recenter("repeated_action_failure")
        self.logger.warning(
            "Hunt recovery | repeated action failure | stage=%s target=%s"
            " episodes=%d | recentering map",
            stage,
            target.type,
            episodes,
        )
        return True

    def is_recovery_progress(self, target_type: str) -> bool:
        """Only a verified completed hunt proves the recovery really worked."""

        return target_type == self.completion_type

    def should_finalize_verification_early(
        self,
        target: Target,
        result: VerificationResult,
        checks: int,
    ) -> bool:
        """Fail a hunt-entry tap once two frames prove it changed nothing."""

        return bool(
            target.type == self.hunt_entry_type
            and checks >= 2
            and result.pixel_change is not None
            and result.pixel_change < 0.001
        )

    def on_blocked_action_context(
        self,
        target: Target,
        detections: Sequence[Detection],
        attempt: int,
    ) -> bool:
        """Recover once hunt confirmation remains blocked for two attempts.

        The game does not expose a stable machine-readable error flag. The
        reliable evidence from the real failure screen is the combination of a
        repeatedly failed hunt-confirm action and the hunt dialog's red close
        button still being present. One retry remains available for an
        occasional missed tap; a second failure switches directly to mailbox
        cleanup instead of spending the full four-attempt budget.
        """

        if attempt < 2 or target.type != self.completion_type or not any(
            item.type == self.hunt_dialog_close_type for item in detections
        ):
            return False
        return self._arm_mailbox_full_recovery()

    def on_retry_exhausted_context(
        self,
        target: Target,
        detections: Sequence[Detection],
    ) -> bool:
        """Fallback for callers that only report context after all retries."""

        if target.type != self.completion_type or not any(
            item.type == self.hunt_dialog_close_type for item in detections
        ):
            return False
        return self._arm_mailbox_full_recovery()

    def _arm_mailbox_full_recovery(self) -> bool:
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
        self._mail_progress_since = None

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
        self._anchor_measured = False
        self._mail_stage = 0
        self._mail_failures = 0
        # A restart invalidates where the workflow stood, and the hunt counter
        # is part of that: leaving it armed sends the bot straight back into
        # the mail flow that caused the restart, restart after restart.
        self._total_hunt_count = 0
        self._autoplace_refused = False
        self._capacity_cooldown_until = 0.0
        self._action_cooldown_until = 0.0
        self._map_idle_since = None
        self._last_map_idle_seconds = 0.0
        self._recenter_reason = None
        self._empty_supply_frames = 0
        self._no_target_since = None
        self._last_blind_seconds = 0.0
        self._mail_progress_since = None
        # A relaunch walks back through the login and startup dialogs, and
        # those are exactly what a stage-scoped scan leaves out.
        self._scoped_idle_cycles = 0
        self._last_full_scan = None
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
        # The outer counterpart to that guard. Distance from the viewport
        # center is how far the map has to travel when the tap lands, and the
        # measured success rate falls off a cliff with it. Rejecting the far
        # band costs almost nothing because those taps rarely open the panel.
        if self.max_center_distance_px and (
            hypot(item.x - frame.width / 2, item.y - frame.height / 2)
            > self.max_center_distance_px
        ):
            return "center_distance"
        # Would tapping this dinosaur push the egg off screen? That only
        # disqualifies it while the egg is the thing being protected, and it
        # is not: recentering exists to restore the supply of reachable
        # dinosaurs for the next few scans, and the egg is merely how the bot
        # recognises that the reset finished. Enforcing it against a *predicted*
        # anchor is worse than useless - the prediction gains a fresh error
        # every hunt, and a measured run threw away 1344 candidates this way,
        # emptying 22% of all planning cycles and then paying for a recenter to
        # refill them.
        if self._anchor_measured and not (
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
        # Same gate, same reason. These angles are measured from the anchor, so
        # a predicted anchor turns the corridor test into noise. The radius rule
        # above needs no origin and covers the same ground: the route markers
        # run 30 to a frame and are present in 95% of scans, already rejecting a
        # quarter of every dinosaur on screen.
        if self._anchor_measured and any(
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

    def anchor_measured(self) -> bool:
        """Whether the last anchor came from a detected egg or was inferred."""

        return self._anchor_measured

    def last_supply(self) -> int:
        """Dinosaurs that survived every rejection rule on the previous cycle.

        This is what recentering is for, so it is what the diagnostic has to
        show: a run whose supply sits at zero is not slow, it is starving.
        """

        return self._last_supply

    def last_idle_seconds(self) -> float:
        """Return how long the map had gone without a target when last planned."""

        return self._last_map_idle_seconds

    def last_recenter_reason(self) -> str | None:
        """Return why the previous planning decision started recentering."""

        return self._recenter_reason

    def _begin_recenter(self, reason: str) -> None:
        """Start a fresh recenter cycle and reset its batch budget."""

        self._hunt_count = 0
        self._map_idle_since = None
        self._empty_supply_frames = 0
        self._recenter_stage = 1
        self._stage = "recenter"
        self._recenter_reason = reason

    def request_external_recenter(self, reason: str) -> None:
        """Request a safe map recenter for a feature handoff.

        External workflows use this only after map evidence is visible.  A
        long per-action cooldown must not delay leaving the hunting map once
        another feature's deadline has arrived.
        """

        self._action_cooldown_until = 0.0
        if self._recenter_stage == 0:
            self._begin_recenter(reason)

    def planning_detection_types(self) -> frozenset[str] | None:
        """Name what the next planning decision can act on, or None for everything.

        A full scan prices every template in the manifest, and the launch-only
        dialogs are a quarter of that bill on screens that cannot appear again
        once a run is under way. Narrowing the scan to the stage the planner is
        actually in is safe exactly while the narrow view keeps producing work,
        so a scan that plans nothing twice running widens the next one back to
        everything. The timer only accelerates that fallback after an empty
        scoped cycle; it must not interrupt a productive map every 30 seconds,
        because a live full scan can cost more than 15 seconds.
        """

        if not self.stage_scoped_scan:
            return None
        now = self.clock()
        timer_due_after_idle = (
            self.full_scan_interval_seconds > 0
            and self._scoped_idle_cycles > 0
            and self._last_full_scan is not None
            and now - self._last_full_scan >= self.full_scan_interval_seconds
        )
        if (
            self._last_full_scan is None
            or self._scoped_idle_cycles >= self.full_scan_after_idle_cycles
            or timer_due_after_idle
        ):
            self._last_full_scan = now
            self._scoped_idle_cycles = 0
            return None
        return self._stage_detection_types()

    def full_detection_types(self) -> frozenset[str]:
        """Return the complete vocabulary owned by the hunting workflow.

        A standalone hunting detector may interpret ``None`` as its complete
        manifest. In hatch-hunt mode, however, the shared detector also owns
        dozens of hatch templates. Naming the hunting vocabulary keeps a
        hunting fallback complete without paying for the inactive workflow.
        """

        return frozenset(
            {
                *self.target_types,
                *self.blocking_types,
                self.dinosaur_type,
                *self.hunt_button_types,
                self.completion_type,
                self.map_exit_type,
                self.forest_recenter_type,
                self.center_anchor_type,
                *self.recovery_button_types,
                *self.interrupt_button_types,
                *self.own_path_types,
                self.mailbox_type,
                self.mail_collect_all_type,
                self.mail_reward_collect_type,
                self.mail_close_type,
                self.hunt_dialog_close_type,
                self.no_available_type,
                self.target_too_strong_type,
                self.capacity_full_type,
            }
        )

    def failure_recovery_detection_types(
        self,
        target_type: str,
    ) -> frozenset[str]:
        """Return a cheap recovery scan for a map action that missed.

        Verification normally looks only for the expected next control. If it
        never appears after a dinosaur or hunt-entry tap, scan that same frame
        for map evidence so recovery can skip an inert coordinate safely.
        """

        if target_type not in {self.dinosaur_type, self.hunt_entry_type}:
            return frozenset()
        return self._stage_detection_types()

    def can_reuse_failed_verification_result(
        self,
        target_type: str,
        detections: Sequence[Detection],
    ) -> bool:
        """Reuse a failed dinosaur frame when it still contains map work."""

        if target_type != self.dinosaur_type:
            return False
        visible = {item.type for item in detections}
        return bool(
            visible
            & {
                self.dinosaur_type,
                *self.hunt_button_types,
                *self.recovery_button_types,
                *self.interrupt_button_types,
            }
        )

    def _stage_detection_types(self) -> frozenset[str]:
        """Return the detections the planner's current stage can act on.

        The set stays deliberately wide. Leaving out the launch dialogs and the
        mail overlay is unambiguous - the game cannot raise a device-history
        prompt mid-run, and the mail buttons only exist inside the mail flow -
        and measured 42% off a full scan. Narrowing further, to the point of
        hiding the hunt sheet while the map is worked, buys another 18% and
        costs a wasted cycle every time a sheet opens on its own.
        """

        # Exceptions the game can raise at any point of a hunt cycle. They are
        # cheap to look for and expensive to miss: an unseen capacity warning
        # spends the next five minutes tapping a hunt the game keeps refusing.
        always = {
            *self.blocking_types,
            *self.recovery_button_types,
            *(self.interrupt_button_types - self.launch_only_types),
            self.no_available_type,
            self.target_too_strong_type,
            self.capacity_full_type,
            # The nest auto-arrange prompt is exactly the case this paragraph
            # warns about: unseen, it cost 14 minutes of refused hunts.
            self.autoplace_cancel_type,
        }
        map_view = {
            self.dinosaur_type,
            *self.own_path_types,
            self.center_anchor_type,
            self.map_exit_type,
            self.forest_recenter_type,
            self.mailbox_type,
        }
        mail_view = {
            self.mailbox_type,
            self.mail_collect_all_type,
            self.mail_reward_collect_type,
            self.mail_close_type,
        }
        if self._mailbox_full_recovery:
            return frozenset(
                always | map_view | mail_view | {self.hunt_dialog_close_type}
            )
        if self._mail_stage:
            return frozenset(
                always | mail_view | {self.map_exit_type, self.center_anchor_type}
            )
        # The map, the recenter round trip and the hunt sheet all read the same
        # screen, and the anchor rides along with them: a scan that drops
        # `map_center_egg` leaves the planner predicting the anchor instead of
        # measuring it, which is what starves `anchor_window` of candidates.
        return frozenset(
            always | map_view | self.hunt_button_types | {self.hunt_dialog_close_type}
        )

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
        """Advance the mail flow, and give up on it if it stops advancing.

        Each stage waits for one specific button, and the release condition -
        ``_resume_hunting_after_mail`` - needs a map landmark to fire. On a
        screen showing neither, the flow waits forever: one run spent 55
        consecutive cycles in stage 1 with no mailbox in sight. The deadline is
        measured from the last stage change rather than from entry, so a slow
        mail flow is never cut short, only a motionless one.

        A returned target is not progress. Stages 2 to 4 deliberately re-offer
        the *previous* stage's button when their own is missing, and stage 5
        re-taps close for as long as the overlay is up, so "planned something"
        describes a retry loop exactly as well as it describes advancing. With
        the timer reset on every target, a mailbox that never opens its
        collect-all button kept stage 2 tapping it for 117 measured seconds
        without the deadline ever being consulted.
        """

        entry_stage = self._mail_stage
        started = self.clock()
        if self._mail_progress_since is None:
            self._mail_progress_since = started
        target = self._advance_mail_stage(frame, detections)
        if self._mail_stage != entry_stage:
            self._mail_progress_since = None if self._mail_stage == 0 else self.clock()
            return target
        if started - self._mail_progress_since >= self.mail_stage_timeout_seconds:
            self._abandon_mail()
            # Dropping the target with the flow: it is a mail button, and the
            # decision just taken was to stop pressing those.
            return None
        return target

    def _advance_mail_stage(
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
        self._anchor_measured = False
        self._map_idle_since = None
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

        # The fixed coordinate below is only safe on a clear map. A hunt sheet
        # leaves the upper third of the map showing - own-path segments and all
        # - so the landmark evidence still passes while the sheet covers the
        # very spot this would tap.
        if any(
            item.type == self.hunt_dialog_close_type for item in detections
        ):
            return None

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
        target = self._choose_target(frame, detections)
        # A scoped scan earns the next one by producing work. Counting the
        # empty ones is what lets `planning_detection_types` widen the view
        # before something it cannot see turns into a stall.
        if target is None and self._stage not in self._BOUNDED_WAIT_STAGES:
            self._scoped_idle_cycles += 1
        else:
            self._scoped_idle_cycles = 0
        self._observe_blind_idle(frame, target)
        return target

    def _observe_blind_idle(self, frame: Frame, target: Target | None) -> None:
        """Time the cycles where the planner can neither act nor name a wait.

        Every stage escape is written as "leave once the expected control is
        visible", so a screen carrying none of them holds all of them at once:
        the recenter timeout at ``_choose_hunt_target`` only runs while a map
        landmark is in frame, and ``_resume_hunting_after_mail`` needs the same
        landmark to release the mail flow. A run measured 699 seconds - 42% of
        its wall clock - spread over six such episodes, each ended only by the
        watchdog restarting the game. This timer is deliberately gated on
        nothing the screen has to supply.
        """

        if target is not None or self._stage in self._BOUNDED_WAIT_STAGES:
            self._no_target_since = None
            self._last_blind_seconds = 0.0
            self._blind_escapes = 0
            return

        now = self.clock()
        if self._no_target_since is None:
            self._no_target_since = now
            self._last_blind_seconds = 0.0
            return

        self._last_blind_seconds = max(0.0, now - self._no_target_since)
        if self._last_blind_seconds < self.blind_idle_seconds:
            return

        self._blind_escapes += 1
        self._pending_blind_escape = {
            "seconds": round(self._last_blind_seconds, 1),
            "stage": self._stage,
            # Consecutive escapes within this episode. A second one means the
            # release did not reach the cause, which is what the engine uses to
            # decide the episode is worth a snapshot.
            "escapes": self._blind_escapes,
        }
        self._release_stage_machines(frame)
        # Re-arm rather than latch: an episode the release does not end has to
        # keep reporting, both to escalate and to record how long it ran.
        self._no_target_since = now

    def _release_stage_machines(self, frame: Frame) -> None:
        """Drop every stage that is parked waiting for a control it cannot see.

        Releasing the stages is not enough on its own, and replaying a measured
        stall is what showed why. With no anchor the hunting branch reads
        dinosaurs-without-hunt-controls as "on the collection map, anchor lost"
        and calls `_begin_recenter("missing_anchor")`; that recenter waits for
        the nest or forest button; and the fallback which would tap the nest
        coordinate anyway is itself gated on having an anchor. Releasing the
        stage just feeds the same loop again - the replay went round it ten
        times in four minutes.

        Adopting the frame centre breaks it, and is the same move the recenter
        path already makes after two dinosaur-only frames. It costs no blind
        tap: the planner still has to find a dinosaur that passes every
        rejection rule before it acts.
        """

        if self._last_anchor is None:
            self._last_anchor = (frame.width / 2, frame.height / 2)
            self._anchor_measured = False
        self._abandon_mail()
        self._mail_progress_since = None
        self._recenter_stage = 0
        self._recenter_dinosaur_frames = 0
        self._awaiting_hunt_button = False
        self._waited_frames = 0
        self._pending_hunt_return = False
        self._map_settle_active = False
        self._map_settle_observed_frames = 0
        self._map_settle_stable_frames = 0
        self._map_settle_anchor = None
        self._map_settle_dinosaur = None
        self._map_idle_since = None
        now = self.clock()
        self._suppressed = [
            entry
            for entry in self._suppressed
            if entry[0] in self.hunt_button_types and entry[4] > now
        ]

    def take_blind_escape(self) -> dict[str, Any] | None:
        """Hand the engine the escape it has not reported yet, once."""

        escape, self._pending_blind_escape = self._pending_blind_escape, None
        return escape

    def last_blind_seconds(self) -> float:
        """Seconds the planner has been unable to act or name a wait."""

        return self._last_blind_seconds

    def _choose_target(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        previous_stage = self._stage
        self._rejections = {}
        # Cleared per cycle: a stage that returns before counting candidates
        # would otherwise report the previous cycle's supply as its own.
        self._last_supply = 0
        self._last_map_idle_seconds = 0.0
        self._recenter_reason = None
        if previous_stage != "hunting":
            self._map_idle_since = None
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
            self._anchor_measured = True

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

        # Hunting a group that includes a nest parent makes the game ask
        # whether to auto-arrange the nest and carry on. Saying yes would
        # reshuffle the parents the hatch side spent its rounds selecting, so
        # this always answers Cancel. Nothing recognised this dialog before:
        # the confirm tap looked like it simply failed, which sent the planner
        # into the "mailbox must be full" recovery - 25 round trips to an empty
        # mailbox, 14 minutes, zero hunts.
        autoplace_cancel = self.filter_suppressed(
            [
                item
                for item in detections
                if item.type == self.autoplace_cancel_type
            ]
        )
        if autoplace_cancel:
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            self._stage = "refuse_nest_autoplace"
            return super().choose(frame, autoplace_cancel)

        # A team sheet that cannot field anyone greys out its Hunt button, so
        # no hunt control matches, yet the sheet still covers the map's exit
        # control in the bottom right. Its own close button is the only thing
        # left on screen that can be pressed. Without this rung the planner
        # reached for the exit underneath the sheet instead: s13 spent 100
        # seconds there, recognising this X on every single frame while
        # tapping a coordinate the sheet was sitting on top of.
        #
        # `_autoplace_refused` forces the same exit after a refusal: the sheet
        # is still usable, so tapping Hunt again would only raise the dialog
        # again. Leave the sheet, then wait for dinosaurs to come home.
        stranded_dialog = self.filter_suppressed(
            [
                item
                for item in detections
                if item.type == self.hunt_dialog_close_type
            ]
        )
        if stranded_dialog and (
            self._autoplace_refused or not actionable_hunt_controls
        ):
            self._awaiting_hunt_button = False
            self._waited_frames = 0
            self._stage = "close_hunt_dialog"
            return super().choose(frame, stranded_dialog)

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
                self._anchor_measured = True
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
                has_hunt_control = bool(actionable_hunt_controls)
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
                    self._anchor_measured = False
                    self._map_idle_since = None
                    if self._total_hunt_count >= self.mail_after_hunts:
                        self._mail_stage = 1
                    self._mail_failures = 0
                    return None
            return super().choose(frame, anchors)

        has_hunt_control = bool(actionable_hunt_controls)
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
                self._begin_recenter("batch")
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
            self._map_idle_since = None
            self._stage = "hunt_control"
            target = super().choose(frame, actionable_hunt_controls)
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
        supply = 0
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
            # Supply has to mean "could be tapped right now", which is two
            # filters further on than "passed the rejection rules". A dinosaur
            # already hunted this map stays on screen and keeps passing every
            # rule, but `choose` drops it as a duplicate - so a map whose last
            # candidates were all spent reported supply and planned nothing,
            # and the resupply trigger never fired. A measured run sat on
            # `supply=2` for twenty seconds that way before the blind-stall
            # timer had to rescue it.
            safe_dinosaurs = [
                item
                for item in self.filter_suppressed(safe_dinosaurs)
                if not (
                    self.dinosaur_type in self.deduplicate_types
                    and self._was_selected(frame, item)
                )
            ]
            supply = len(safe_dinosaurs)
            if safe_dinosaurs:
                def radial_key(item: Detection) -> tuple[float, float, float, float]:
                    # Rank by displacement, which is what a tap actually costs:
                    # the map re-centres on the dinosaur, so how far it sits
                    # from the *screen* centre is exactly how far the view - and
                    # the egg with it - is about to move. Measuring from the
                    # anchor instead was measuring the wrong thing as soon as
                    # the anchor stopped being centred, and needs an anchor at
                    # all. Right after a reset the egg is the screen centre, so
                    # this is the ring-around-the-egg ordering; several hunts
                    # later it is still the cheapest tap available.
                    distance = hypot(
                        item.x - frame.width / 2,
                        item.y - frame.height / 2,
                    )
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
            self._begin_recenter("missing_anchor")
            return self._choose_map_exit(frame, detections)
        target = super().choose(frame, actionable)
        self._last_supply = supply
        if on_collect_map and supply == 0:
            self._empty_supply_frames += 1
        else:
            self._empty_supply_frames = 0
        # Recentering is resupply, so run it off the supply rather than off
        # "did this cycle plan anything". Those differed by a lot: with
        # `anchor_window` rejecting against a predicted anchor, cycles reported
        # nothing to do while the map was still full, and 31 of a run's 32
        # resets were spent refilling a map that had never emptied.
        #
        # The grace period stays. A corridor full of the bot's own routes
        # clears itself as hunts return, and resetting the moment supply dips
        # would trade a few seconds of waiting for a whole map reload.
        if supply < self.recenter_min_candidates and on_collect_map:
            if self._empty_supply_frames >= self.empty_supply_recenter_frames:
                self._begin_recenter("empty_supply")
                return self._choose_map_exit(frame, detections)
            idle_now = self.clock()
            if self._map_idle_since is None:
                self._map_idle_since = idle_now
            self._last_map_idle_seconds = max(0.0, idle_now - self._map_idle_since)
            if self._last_map_idle_seconds >= self.stalled_recenter_seconds:
                self._begin_recenter("low_supply")
                return self._choose_map_exit(frame, detections)
        else:
            self._map_idle_since = None
            self._empty_supply_frames = 0
        if target is not None and target.type == self.dinosaur_type:
            self._last_selected_dinosaur = (float(target.x), float(target.y))
            self._anchor_before_dinosaur = anchor_position
            if anchor_position is not None:
                self._last_anchor = (
                    anchor_position[0] + frame.width / 2 - target.x,
                    anchor_position[1] + frame.height / 2 - target.y,
                )
                # Inferred from where the map was told to go, not seen.
                self._anchor_measured = False
            self._awaiting_hunt_button = True
            self._waited_frames = 0
        return target
