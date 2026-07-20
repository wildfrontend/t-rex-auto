"""Composition root for the Auto Collect MVP."""

from __future__ import annotations

import logging

from .actions import AdbActionDriver, AdbClient
from .capture import AdbScreencapCapture, MssBlueStacksCapture
from .config import AppConfig
from .detection import OpenCvDetector
from .engine import BotContext, BotEngine
from .logging import configure_logging
from .modes import create_mode
from .planning import TargetPlanner
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
    planner = TargetPlanner(config.planner.target_types, config.planner.strategy)
    action = AdbActionDriver(adb)
    verifier = TargetChangedVerifier(
        config.verify.max_distance,
        config.verify.pixel_change_threshold,
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
        idle_delay_ms=config.idle_delay,
        verify_retries=config.verify_retry,
        max_actions=config.max_actions,
    )
    return BotEngine(context)


def close_logging() -> None:
    logging.shutdown()
