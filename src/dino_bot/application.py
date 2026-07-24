"""Composition root for the Auto Collect MVP."""

from __future__ import annotations

import logging

from .actions import AdbActionDriver, AdbClient
from .capture import AdbScreencapCapture, MssBlueStacksCapture
from .config import AppConfig
from .detection import (
    CompositeDetector,
    HuntCapacityDetector,
    HuntTeamAvailabilityDetector,
    OpenCvDetector,
    StartupAutoBattleDialogDetector,
    StartupGrowthResultDetector,
    TargetTooStrongDetector,
)
from .engine import BotContext, BotEngine
from .logging import configure_logging
from .models import ActionKind
from .modes import create_mode
from .planning import HuntPlanner
from .recovery import AdbAppRestarter, BlackScreenRecovery
from .verification import TargetChangedVerifier


def create_engine(config: AppConfig, *, verbose: bool = False) -> BotEngine:
    logger = configure_logging(config.logs_dir, verbose=verbose)
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
        capture = MssBlueStacksCapture(
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
        StartupGrowthResultDetector(),
        StartupAutoBattleDialogDetector(),
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
        mail_after_hunts=config.planner.mail_after_hunts,
        capacity_wait_seconds=config.planner.capacity_wait_seconds,
        ring_width=config.planner.ring_width,
        own_path_angle_degrees=config.planner.own_path_angle_degrees,
        stalled_recenter_frames=config.planner.stalled_recenter_frames,
        bottom_exclusion_px=config.planner.bottom_exclusion_px,
        action_cooldowns_ms=config.planner.action_cooldowns_ms,
    )
    action = AdbActionDriver(adb)
    verifier = TargetChangedVerifier(
        config.verify.max_distance,
        config.verify.pixel_change_threshold,
        config.verify.failure_types,
        config.verify.success_transitions,
        config.recovery.black_mean_threshold,
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
        verify_retries=config.verify_retry,
        max_actions=config.max_actions,
        max_cycles=config.workflow.max_cycles,
        cycle_complete_targets=config.workflow.complete_on,
        runtime_recovery=runtime_recovery,
    )
    return BotEngine(context)


def close_logging() -> None:
    logging.shutdown()
