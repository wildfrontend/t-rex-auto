"""Simple ``capture.capture()`` facade plus replaceable capture backends."""

from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray

from dino_bot.actions import AdbClient
from dino_bot.capture import AdbScreencapCapture, CaptureError, MssBlueStacksCapture
from dino_bot.config import load_config
from dino_bot.interfaces import CaptureProvider
from dino_bot.models import Frame

_provider: CaptureProvider | None = None


def configure(provider: CaptureProvider) -> None:
    """Inject a provider, primarily for independent modules and tests."""
    global _provider
    _provider = provider


def _default_provider() -> CaptureProvider:
    config = load_config(Path(__file__).with_name("config.json"))
    if config.capture.backend == "adb":
        client = AdbClient(config.adb)
        client.ensure_ready()
        return AdbScreencapCapture(client)
    return MssBlueStacksCapture(
        config.capture.window_titles,
        config.capture.process_names,
        config.capture.viewport,
        config.capture.auto_viewport,
        config.capture.chrome_insets,
    )


def capture_frame() -> Frame:
    global _provider
    if _provider is None:
        _provider = _default_provider()
    return _provider.capture()


def capture() -> NDArray:
    """Capture BlueStacks and return the BGR numpy image kept in RAM."""
    return capture_frame().image


def close() -> None:
    global _provider
    if _provider is not None:
        _provider.close()
        _provider = None

__all__ = [
    "AdbScreencapCapture",
    "CaptureError",
    "MssBlueStacksCapture",
    "capture",
    "capture_frame",
    "close",
    "configure",
]
