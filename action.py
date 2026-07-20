"""Simple action facade and ADB action implementation exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dino_bot.actions import AdbActionDriver, AdbClient, AdbError
from dino_bot.config import load_config
from dino_bot.interfaces import ActionDriver
from dino_bot.models import ActionCommand, ActionKind, Frame

_driver: ActionDriver | None = None
_default_frame: Frame | None = None


def configure(driver: ActionDriver, frame: Frame | np.ndarray | None = None) -> None:
    global _driver, _default_frame
    _driver = driver
    _default_frame = _as_frame(frame) if frame is not None else None


def _as_frame(frame: Frame | np.ndarray) -> Frame:
    return frame if isinstance(frame, Frame) else Frame(frame)


def _resources(frame: Frame | np.ndarray | None) -> tuple[ActionDriver, Frame]:
    global _driver, _default_frame
    if _driver is None:
        config = load_config(Path(__file__).with_name("config.json"))
        client = AdbClient(config.adb)
        client.ensure_ready()
        _driver = AdbActionDriver(client)
    if frame is not None:
        resolved_frame = _as_frame(frame)
    elif _default_frame is not None:
        resolved_frame = _default_frame
    else:
        from capture import capture_frame

        resolved_frame = capture_frame()
    return _driver, resolved_frame


def tap(x: int, y: int, frame: Frame | np.ndarray | None = None) -> None:
    driver, resolved_frame = _resources(frame)
    driver.execute(ActionCommand.tap(x, y), resolved_frame)


def swipe(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration_ms: int = 300,
    frame: Frame | np.ndarray | None = None,
) -> None:
    driver, resolved_frame = _resources(frame)
    driver.execute(
        ActionCommand(ActionKind.SWIPE, x1, y1, x2, y2, duration_ms),
        resolved_frame,
    )


def long_press(
    x: int,
    y: int,
    duration_ms: int = 800,
    frame: Frame | np.ndarray | None = None,
) -> None:
    driver, resolved_frame = _resources(frame)
    driver.execute(
        ActionCommand(ActionKind.LONG_PRESS, x, y, duration_ms=duration_ms),
        resolved_frame,
    )


def sleep(duration_ms: int) -> None:
    AdbActionDriver.sleep(duration_ms)

__all__ = [
    "AdbActionDriver",
    "AdbClient",
    "AdbError",
    "configure",
    "long_press",
    "sleep",
    "swipe",
    "tap",
]
