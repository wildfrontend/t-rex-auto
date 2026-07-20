"""ADB transport and coordinate-aware BlueStacks action driver."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import AdbConfig
from .models import ActionCommand, ActionKind, Frame


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    serial: str
    state: str
    description: str = ""


class AdbClient:
    def __init__(self, config: AdbConfig):
        self.config = config
        self.executable = self._resolve_executable(config.executable)

    @staticmethod
    def _resolve_executable(configured: str | None) -> str:
        candidates: list[str] = []
        if configured:
            candidates.append(configured)
        discovered = shutil.which("adb")
        if discovered:
            candidates.append(discovered)
        candidates.extend(
            [
                r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
                r"C:\Program Files\BlueStacks\HD-Adb.exe",
            ]
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return str(Path(candidate))
        raise AdbError(
            "ADB executable not found. Set adb.executable in config.json to HD-Adb.exe or adb.exe."
        )

    def _command(self, args: Sequence[str], use_serial: bool = True) -> list[str]:
        command = [self.executable]
        if use_serial and self.config.serial:
            command.extend(["-s", self.config.serial])
        command.extend(args)
        return command

    def run(
        self,
        args: Sequence[str],
        *,
        use_serial: bool = True,
        binary: bool = False,
        check: bool = True,
    ) -> bytes | str:
        try:
            completed = subprocess.run(
                self._command(args, use_serial),
                capture_output=True,
                timeout=self.config.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdbError(f"ADB command failed to start: {exc}") from exc
        if check and completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise AdbError(f"ADB exited with {completed.returncode}: {message}")
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace").strip()

    def connect(self) -> str:
        if not self.config.serial:
            return "serial not configured"
        return str(self.run(["connect", self.config.serial], use_serial=False))

    def devices(self) -> list[DeviceInfo]:
        output = str(self.run(["devices", "-l"], use_serial=False))
        devices: list[DeviceInfo] = []
        for line in output.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            devices.append(
                DeviceInfo(
                    serial=parts[0],
                    state=parts[1] if len(parts) > 1 else "unknown",
                    description=parts[2] if len(parts) > 2 else "",
                )
            )
        return devices

    def ensure_ready(self) -> DeviceInfo:
        if self.config.connect_on_start and self.config.serial:
            self.connect()
        devices = self.devices()
        ready = [item for item in devices if item.state == "device"]
        if self.config.serial:
            ready = [item for item in ready if item.serial == self.config.serial]
        if not ready:
            wanted = self.config.serial or "any device"
            raise AdbError(
                f"No ready ADB device for {wanted}. "
                "Enable Android Debug Bridge in BlueStacks settings."
            )
        return ready[0]

    def display_size(self) -> tuple[int, int]:
        output = str(self.run(["shell", "wm", "size"]))
        matches = re.findall(r"(?:Override|Physical) size:\s*(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"Cannot parse device size from: {output!r}")
        width, height = matches[-1]
        return int(width), int(height)

    def screencap_png(self) -> bytes:
        return bytes(self.run(["exec-out", "screencap", "-p"], binary=True))


class AdbActionDriver:
    def __init__(self, client: AdbClient, device_size: tuple[int, int] | None = None):
        self.client = client
        self._device_size = device_size

    @property
    def device_size(self) -> tuple[int, int]:
        if self._device_size is None:
            self._device_size = self.client.display_size()
        return self._device_size

    def _map(self, x: int, y: int, frame: Frame) -> tuple[int, int]:
        width, height = self.device_size
        if (frame.width > frame.height) != (width > height):
            width, height = height, width
        mapped_x = round(x * width / frame.width)
        mapped_y = round(y * height / frame.height)
        return max(0, min(width - 1, mapped_x)), max(0, min(height - 1, mapped_y))

    def execute(self, action: ActionCommand, frame: Frame) -> None:
        if action.kind == ActionKind.SLEEP:
            time.sleep(action.duration_ms / 1000)
            return
        if action.x is None or action.y is None:
            raise ValueError(f"{action.kind} requires x and y")
        x1, y1 = self._map(action.x, action.y, frame)
        if action.kind == ActionKind.TAP:
            self.client.run(["shell", "input", "tap", str(x1), str(y1)])
            return
        if action.kind in {ActionKind.SWIPE, ActionKind.LONG_PRESS}:
            if action.kind == ActionKind.LONG_PRESS:
                x2, y2 = x1, y1
            else:
                if action.x2 is None or action.y2 is None:
                    raise ValueError("swipe requires x2 and y2")
                x2, y2 = self._map(action.x2, action.y2, frame)
            duration = max(1, action.duration_ms)
            self.client.run(
                [
                    "shell",
                    "input",
                    "swipe",
                    str(x1),
                    str(y1),
                    str(x2),
                    str(y2),
                    str(duration),
                ]
            )
            return
        raise ValueError(f"Unsupported action: {action.kind}")

    def tap(self, x: int, y: int, frame: Frame) -> None:
        self.execute(ActionCommand.tap(x, y), frame)

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int, frame: Frame
    ) -> None:
        self.execute(
            ActionCommand(ActionKind.SWIPE, x1, y1, x2, y2, duration_ms),
            frame,
        )

    def long_press(self, x: int, y: int, duration_ms: int, frame: Frame) -> None:
        self.execute(
            ActionCommand(ActionKind.LONG_PRESS, x, y, duration_ms=duration_ms),
            frame,
        )

    @staticmethod
    def sleep(duration_ms: int) -> None:
        time.sleep(duration_ms / 1000)


class RecordingActionDriver:
    """Non-mutating driver for tests and dry integration checks."""

    def __init__(self) -> None:
        self.actions: list[ActionCommand] = []

    def execute(self, action: ActionCommand, frame: Frame) -> None:
        self.actions.append(action)
