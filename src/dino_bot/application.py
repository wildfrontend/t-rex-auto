"""Composition root for the bot features (hunt and hatch)."""

from __future__ import annotations

import logging

from . import hatch as hatch_feature
from . import nest_filter as nest_filter_feature
from .actions import AdbActionDriver, AdbClient
from .capture import AdbScreencapCapture, MssEmulatorCapture
from .config import AppConfig
from .detection import (
    CompositeDetector,
    HuntCapacityDetector,
    HuntTeamAvailabilityDetector,
    OpenCvDetector,
    StartupAutoBattleDialogDetector,
    StartupGrowthResultDetector,
    StartupLayoutGuard,
    TargetTooStrongDetector,
)
from .engine import BotContext, BotEngine
from .events import EventLog, JsonlEventLog, NullEventLog
from .hatch import HatchPlanner
from .logging import configure_logging
from .models import ActionKind
from .modes import create_mode
from .nest_filter import NestTagFilterTestPlanner
from .planning import HuntPlanner
from .recovery import AdbAppRestarter, BlackScreenRecovery, HuntProgressWatchdog
from .stalls import StallSnapshotWriter
from .verification import TargetChangedVerifier


def create_engine(
    config: AppConfig,
    *,
    verbose: bool = False,
    feature: str = "hunt",
) -> BotEngine:
    if feature == "hatch":
        return _create_hatch_engine(config, verbose=verbose)
    if feature == "hatch-filter-test":
        return _create_hatch_engine(config, verbose=verbose, filter_test=True)
    if feature != "hunt":
        raise ValueError(f"unknown feature: {feature}")
    return _create_hunt_engine(config, verbose=verbose)


def _create_hunt_engine(config: AppConfig, *, verbose: bool = False) -> BotEngine:
    logger = configure_logging(
        config.logs_dir,
        verbose=verbose,
        max_bytes=config.log_max_bytes,
        backup_count=config.log_backup_count,
    )
    logger.info(
        "Timing | poll=%dms | click=%dms | dinosaur=%dms | hunt=%dms"
        " | confirm=%dms | idle=%dms",
        config.transition_poll_interval,
        config.click_delay,
        config.post_action_delays.get("dinosaur", config.click_delay),
        config.post_action_delays.get("hunt_button", config.click_delay),
        config.post_action_delays.get("hunt_confirm_button", config.click_delay),
        config.idle_delay,
    )
    adb = AdbClient(config.adb)
    device = adb.ensure_ready()
    logger.info("ADB | device=%s | %s", device.serial, device.description)

    if config.capture.backend == "adb":
        capture = AdbScreencapCapture(adb)
    else:
        capture = MssEmulatorCapture(
            config.capture.window_titles,
            process_names=config.capture.process_names,
            viewport=config.capture.viewport,
            auto_viewport=config.capture.auto_viewport,
            chrome_insets=config.capture.chrome_insets,
        )

    open_cv_detector = OpenCvDetector(
        config.detector.manifest,
        default_threshold=config.detector.default_threshold,
        nms_iou=config.detector.nms_iou,
    )
    if open_cv_detector.asset_count == 0:
        logger.warning("Detector has no assets; bot will observe but cannot choose targets")
    detector = CompositeDetector(
        open_cv_detector,
        HuntTeamAvailabilityDetector(),
        HuntCapacityDetector(),
        TargetTooStrongDetector(),
        StartupLayoutGuard(StartupGrowthResultDetector(), logger=logger),
        StartupLayoutGuard(StartupAutoBattleDialogDetector(), logger=logger),
        reference_size=open_cv_detector.reference_size,
    )
    planner = HuntPlanner(
        config.planner.target_types,
        config.planner.strategy,
        blocking_types=config.planner.blocking_types,
        deduplicate_types=config.planner.deduplicate_types,
        dedup_radius=config.planner.dedup_radius,
        history_file=config.planner.history_file,
        history_limit=config.planner.history_limit,
        recenter_every=config.planner.recenter_every,
        own_path_radius=config.planner.own_path_radius,
        anchor_exclusion_radius=config.planner.anchor_exclusion_radius,
        dinosaur_failure_cooldown_ms=config.planner.dinosaur_failure_cooldown_ms,
        dinosaur_failure_radius=config.planner.dinosaur_failure_radius,
        mail_after_hunts=config.planner.mail_after_hunts,
        mail_failure_limit=config.planner.mail_failure_limit,
        capacity_wait_seconds=config.planner.capacity_wait_seconds,
        ring_width=config.planner.ring_width,
        own_path_angle_degrees=config.planner.own_path_angle_degrees,
        stalled_recenter_seconds=config.planner.stalled_recenter_seconds,
        recenter_min_candidates=config.planner.recenter_min_candidates,
        blind_idle_seconds=config.planner.blind_idle_seconds,
        mail_stage_timeout_seconds=config.planner.mail_stage_timeout_seconds,
        map_settle_frames=config.planner.map_settle_frames,
        map_settle_tolerance_px=config.planner.map_settle_tolerance_px,
        map_settle_max_frames=config.planner.map_settle_max_frames,
        max_center_distance_px=config.planner.max_center_distance_px,
        bottom_exclusion_px=config.planner.bottom_exclusion_px,
        exclusion_zones=config.planner.exclusion_zones,
        retry_exhausted_cooldown_ms=config.planner.retry_exhausted_cooldown_ms,
        suppression_radius=config.planner.suppression_radius,
        action_cooldowns_ms=config.planner.action_cooldowns_ms,
        stage_scoped_scan=config.planner.stage_scoped_scan,
        full_scan_interval_seconds=config.planner.full_scan_interval_seconds,
        full_scan_after_idle_cycles=config.planner.full_scan_after_idle_cycles,
    )
    action = AdbActionDriver(adb)
    verifier = TargetChangedVerifier(
        max_distance=config.verify.max_distance,
        pixel_change_threshold=config.verify.pixel_change_threshold,
        failure_types=config.verify.failure_types,
        success_transitions=config.verify.success_transitions,
        black_mean_threshold=config.recovery.black_mean_threshold,
        success_requires_target_absence=(
            config.verify.success_requires_target_absence
        ),
    )
    observer = create_mode(
        config.mode,
        debug_dir=config.debug_dir,
        training_dir=config.training_dir,
        save_debug_image=config.save_debug_image or config.mode == "debug",
        training_fps=config.training.fps,
        training_max_images=config.training.max_images,
    )
    runtime_recovery = None
    hunt_progress_recovery = None
    if config.recovery.enabled:
        runtime_recovery = BlackScreenRecovery(
            AdbAppRestarter(
                adb,
                config.recovery.package,
                config.recovery.activity,
            ),
            logger,
            timeout_seconds=config.recovery.black_screen_timeout_seconds,
            mean_threshold=config.recovery.black_mean_threshold,
            cooldown_seconds=config.recovery.restart_cooldown_seconds,
            launch_wait_seconds=config.recovery.launch_wait_seconds,
        )
        hunt_progress_recovery = HuntProgressWatchdog(
            runtime_recovery,
            logger,
            timeout_seconds=config.recovery.no_hunt_progress_timeout_seconds,
            suspend_budget_seconds=(
                config.recovery.hunt_progress_suspend_budget_seconds
            ),
        )
    event_log: EventLog = (
        JsonlEventLog(
            config.logs_dir,
            max_bytes=config.event_log.max_bytes,
            backup_count=config.event_log.backup_count,
        )
        if config.event_log.enabled
        else NullEventLog()
    )
    stall_snapshots = (
        StallSnapshotWriter(
            config.stalls_dir,
            logger,
            limit=config.stalls.snapshot_limit,
            min_interval_seconds=config.stalls.snapshot_min_interval_seconds,
        )
        if config.stalls.snapshots_enabled
        else None
    )
    context = BotContext(
        capture_provider=capture,
        detector=detector,
        planner=planner,
        action_driver=action,
        verifier=verifier,
        observer=observer,
        logger=logger,
        click_delay_ms=config.click_delay,
        post_action_delays_ms=config.post_action_delays,
        target_action_kinds={
            target_type: ActionKind(action)
            for target_type, action in config.target_actions.items()
        },
        idle_delay_ms=config.idle_delay,
        transition_poll_interval_ms=config.transition_poll_interval,
        verification_minimum_checks=config.verify.minimum_checks,
        verify_retries=config.verify_retry,
        max_actions=config.max_actions,
        max_cycles=config.workflow.max_cycles,
        cycle_complete_targets=config.workflow.complete_on,
        runtime_recovery=runtime_recovery,
        hunt_progress_recovery=hunt_progress_recovery,
        stall_snapshots=stall_snapshots,
        event_log=event_log,
    )
    return BotEngine(context)


def _create_hatch_engine(
    config: AppConfig,
    *,
    verbose: bool = False,
    filter_test: bool = False,
) -> BotEngine:
    """Wire the Auto Hatch feature onto the shared capture/act/verify core.

    Everything platform-shaped (ADB, capture, modes, black-screen recovery,
    event log, stall snapshots) is identical to hunt; only the detector
    manifest, the planner, and the verification vocabulary change. The hunt
    progress watchdog stays out: it counts hunts, and a hatch session would
    look permanently stalled to it.
    """

    logger = configure_logging(
        config.logs_dir,
        verbose=verbose,
        max_bytes=config.log_max_bytes,
        backup_count=config.log_backup_count,
    )
    hatch = config.hatch
    if filter_test:
        logger.info("Feature | hatch-filter-test | target=攻擊特化 | safe T7 subset")
    else:
        logger.info(
            "Feature | hatch | rescan=%.0fs | egg_pile=(%.0f,%.0f)@%.0fw | scrolls<=%d",
            hatch.rescan_interval_seconds,
            hatch.egg_pile[0],
            hatch.egg_pile[1],
            hatch.reference_width,
            hatch.max_scrolls,
        )
    adb = AdbClient(config.adb)
    device = adb.ensure_ready()
    logger.info("ADB | device=%s | %s", device.serial, device.description)

    if config.capture.backend == "adb":
        capture = AdbScreencapCapture(adb)
    else:
        capture = MssEmulatorCapture(
            config.capture.window_titles,
            process_names=config.capture.process_names,
            viewport=config.capture.viewport,
            auto_viewport=config.capture.auto_viewport,
            chrome_insets=config.capture.chrome_insets,
        )

    open_cv_detector = OpenCvDetector(
        hatch.manifest,
        default_threshold=config.detector.default_threshold,
        nms_iou=config.detector.nms_iou,
    )
    if open_cv_detector.asset_count == 0:
        logger.warning(
            "Hatch detector has no assets; capture templates into %s first",
            hatch.manifest,
        )
    detector = CompositeDetector(
        open_cv_detector,
        reference_size=open_cv_detector.reference_size,
    )
    planner = (
        NestTagFilterTestPlanner(reference_width=hatch.reference_width)
        if filter_test
        else HatchPlanner(
            egg_pile_point=(hatch.egg_pile[0], hatch.egg_pile[1]),
            reference_width=hatch.reference_width,
            scroll_vector=hatch.scroll_vector,
            scroll_duration_ms=hatch.scroll_duration_ms,
            max_scrolls=hatch.max_scrolls,
            rescan_interval_seconds=hatch.rescan_interval_seconds,
            require_home_anchor=hatch.require_home_anchor,
            home_failure_limit=hatch.home_failure_limit,
            home_backoff_seconds=hatch.home_backoff_seconds,
            logger=logger,
        )
    )
    action = AdbActionDriver(adb)
    defaults = nest_filter_feature if filter_test else hatch_feature
    success_transitions = dict(defaults.DEFAULT_SUCCESS_TRANSITIONS)
    success_transitions.update(config.verify.success_transitions)
    verifier = TargetChangedVerifier(
        max_distance=config.verify.max_distance,
        pixel_change_threshold=config.verify.pixel_change_threshold,
        failure_types=(),
        success_transitions=success_transitions,
        black_mean_threshold=config.recovery.black_mean_threshold,
    )
    observer = create_mode(
        config.mode,
        debug_dir=config.debug_dir,
        training_dir=config.training_dir,
        save_debug_image=config.save_debug_image or config.mode == "debug",
        training_fps=config.training.fps,
        training_max_images=config.training.max_images,
    )
    runtime_recovery = None
    if config.recovery.enabled:
        runtime_recovery = BlackScreenRecovery(
            AdbAppRestarter(
                adb,
                config.recovery.package,
                config.recovery.activity,
            ),
            logger,
            timeout_seconds=config.recovery.black_screen_timeout_seconds,
            mean_threshold=config.recovery.black_mean_threshold,
            cooldown_seconds=config.recovery.restart_cooldown_seconds,
            launch_wait_seconds=config.recovery.launch_wait_seconds,
        )
    event_log: EventLog = (
        JsonlEventLog(
            config.logs_dir,
            max_bytes=config.event_log.max_bytes,
            backup_count=config.event_log.backup_count,
        )
        if config.event_log.enabled
        else NullEventLog()
    )
    stall_snapshots = (
        StallSnapshotWriter(
            config.stalls_dir,
            logger,
            limit=config.stalls.snapshot_limit,
            min_interval_seconds=config.stalls.snapshot_min_interval_seconds,
        )
        if config.stalls.snapshots_enabled
        else None
    )
    post_action_delays = dict(defaults.DEFAULT_POST_ACTION_DELAYS_MS)
    post_action_delays.update(config.post_action_delays)
    target_actions = dict(defaults.DEFAULT_TARGET_ACTIONS)
    target_actions.update(config.target_actions)
    context = BotContext(
        capture_provider=capture,
        detector=detector,
        planner=planner,
        action_driver=action,
        verifier=verifier,
        observer=observer,
        logger=logger,
        click_delay_ms=config.click_delay,
        post_action_delays_ms=post_action_delays,
        target_action_kinds={
            target_type: ActionKind(action_name)
            for target_type, action_name in target_actions.items()
        },
        idle_delay_ms=config.idle_delay,
        transition_poll_interval_ms=config.transition_poll_interval,
        verification_minimum_checks=config.verify.minimum_checks,
        verify_retries=config.verify_retry,
        max_actions=config.max_actions,
        max_cycles=config.workflow.max_cycles,
        # Hatch cycles are completed only by a verified claim. The shared
        # config normally contains hunt's mailbox completion target, which
        # must not leak into this feature or --max-cycles can never stop it.
        cycle_complete_targets=defaults.DEFAULT_CYCLE_COMPLETE_TARGETS,
        runtime_recovery=runtime_recovery,
        hunt_progress_recovery=None,
        stall_snapshots=stall_snapshots,
        event_log=event_log,
    )
    return BotEngine(context)


def close_logging() -> None:
    logging.shutdown()
