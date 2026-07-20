"""Environment diagnostics and capture throughput checks."""

from __future__ import annotations

import importlib.util
import platform
import sys
import time
from dataclasses import dataclass

from .actions import AdbClient
from .capture import AdbScreencapCapture, BlueStacksWindowFinder, MssBlueStacksCapture
from .config import AppConfig
from .detection import OpenCvDetector


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_checks(config: AppConfig) -> list[Check]:
    checks = [
        Check("Python", sys.version_info >= (3, 12), platform.python_version()),
        Check("Operating system", platform.system() == "Windows", platform.platform()),
    ]
    for module in ("numpy", "cv2", "mss"):
        checks.append(
            Check(f"Dependency {module}", importlib.util.find_spec(module) is not None, "installed")
        )
    if platform.system() == "Windows":
        checks.append(
            Check(
                "Dependency win32gui",
                importlib.util.find_spec("win32gui") is not None,
                "installed",
            )
        )
    try:
        detector = OpenCvDetector(
            config.detector.manifest,
            config.detector.default_threshold,
            config.detector.nms_iou,
        )
        checks.append(
            Check(
                "Detector assets",
                detector.asset_count > 0,
                f"{detector.asset_count} configured",
                required=False,
            )
        )
    except Exception as exc:
        checks.append(Check("Detector assets", False, str(exc)))
    try:
        adb = AdbClient(config.adb)
        checks.append(Check("ADB executable", True, adb.executable))
        devices = adb.devices()
        ready = [item for item in devices if item.state == "device"]
        checks.append(
            Check("ADB device", bool(ready), ", ".join(item.serial for item in ready) or "none")
        )
    except Exception as exc:
        checks.append(Check("ADB", False, str(exc)))
    if platform.system() == "Windows":
        try:
            hwnd = BlueStacksWindowFinder(
                config.capture.window_titles,
                config.capture.process_names,
            ).find()
            checks.append(Check("BlueStacks window", True, f"HWND={hwnd}"))
        except Exception as exc:
            checks.append(Check("BlueStacks window", False, str(exc)))
    return checks


def benchmark_capture(config: AppConfig, frame_count: int = 100) -> tuple[float, tuple[int, int]]:
    if config.capture.backend == "adb":
        adb = AdbClient(config.adb)
        adb.ensure_ready()
        capture = AdbScreencapCapture(adb)
    else:
        capture = MssBlueStacksCapture(
            config.capture.window_titles,
            config.capture.process_names,
            config.capture.viewport,
            config.capture.auto_viewport,
            config.capture.chrome_insets,
        )
    started = time.perf_counter()
    size = (0, 0)
    try:
        for _ in range(frame_count):
            frame = capture.capture()
            size = (frame.width, frame.height)
    finally:
        capture.close()
    elapsed = time.perf_counter() - started
    return frame_count / elapsed, size
