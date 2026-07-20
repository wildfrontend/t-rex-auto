"""Composition root for the Auto Collect MVP."""

from __future__ import annotations

import logging

from .actions import AdbActionDriver, AdbClient
from .capture import AdbScreencapCapture, MssBlueStacksCapture
from .config import AppConfig
from .detection import OpenCvDetector
from .engine import BotContext, BotEngine
from .logging import configure_logging
from .models import ActionKind
from .modes import create_mode
from .planning import HuntPlanner
from .verification import TargetChangedVerifier


def create_engine(config: AppConfig, *, verbose: bool = False) -> BotEngine:
    logger = configure_logging(config.logs_dir, verbose=verbose)
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

    detector = OpenCvDetector(
        config.detector.manifest,
        default_threshold=config.detector.default_threshold,
        nms_iou=config.detector.nms_iou,
    )
    if detector.asset_count == 0:
        logger.warning("Detector has no assets; bot will observe but cannot choose targets")
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
    )
    action = AdbActionDriver(adb)
    verifier = TargetChangedVerifier(
        config.verify.max_distance,
        config.verify.pixel_change_threshold,
        config.verify.failure_types,
    )
    observer = create_mode(
        config.mode,
        debug_dir=config.debug_dir,
        training_dir=config.training_dir,
        save_debug_image=config.save_debug_image or config.mode == "debug",
        training_fps=config.training.fps,
        training_max_images=config.training.max_images,
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
        verify_retries=config.verify_retry,
        max_actions=config.max_actions,
        max_cycles=config.workflow.max_cycles,
        cycle_complete_targets=config.workflow.complete_on,
    )
    return BotEngine(context)


def close_logging() -> None:
    logging.shutdown()
