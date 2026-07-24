"""ADB transport and coordinate-aware BlueStacks action driver."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
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
        self._selected_serial: str | None = None

    @staticmethod
    def _resolve_executable(configured: str | None) -> str:
        candidates: list[str] = []
        if configured:
            candidates.append(configured)
        discovered = shutil.which("adb")
        if discovered:
            candidates.append(discovered)
        for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
            sdk_root = os.environ.get(variable)
            if sdk_root:
                executable_name = "adb.exe" if sys.platform == "win32" else "adb"
                candidates.append(str(Path(sdk_root) / "platform-tools" / executable_name))
        if sys.platform == "darwin":
            candidates.append(
                str(Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb")
            )
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(
                str(Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe")
            )
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
            "ADB executable not found. Install Android Platform Tools or enable BlueStacks ADB."
        )

    def _command(self, args: Sequence[str], use_serial: bool = True) -> list[str]:
        command = [self.executable]
        configured_serial = self._configured_serial()
        serial = self._selected_serial or (
            configured_serial if configured_serial != "auto" else None
        )
        if use_serial and serial:
            command.extend(["-s", serial])
        command.extend(args)
        return command

    def _configured_serial(self) -> str:
        return (self.config.serial or "auto").strip() or "auto"

    def _configured_fallback_serial(self) -> str | None:
        serial = (self.config.fallback_serial or "").strip()
        return serial or None

    @staticmethod
    def _is_network_or_emulator_serial(serial: str) -> bool:
        return (
            serial.startswith("emulator-")
            or serial.startswith("adb-")
            or ":" in serial
        )

    @property
    def selected_serial(self) -> str | None:
        return self._selected_serial

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
        serial = self._configured_serial()
        if serial == "auto":
            serial = self._configured_fallback_serial() or ""
        if not serial:
            return "no ADB network fallback configured"
        if not self._is_network_or_emulator_serial(serial):
            return "USB device does not require adb connect"
        if serial.startswith("emulator-"):
            return "local emulator does not require adb connect"
        return str(self.run(["connect", serial], use_serial=False))

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
        configured_serial = self._configured_serial()
        if (
            self.config.connect_on_start
            and configured_serial != "auto"
            and self._is_network_or_emulator_serial(configured_serial)
        ):
            self.connect()
        devices = self.devices()
        ready = [item for item in devices if item.state == "device"]
        if configured_serial != "auto":
            ready = [item for item in ready if item.serial == configured_serial]
            if not ready:
                raise AdbError(
                    f"No ready ADB device for {configured_serial}. "
                    "Authorize USB debugging or enable emulator ADB."
                )
            selected = ready[0]
            if (
                self.config.require_usb
                and self._is_network_or_emulator_serial(selected.serial)
            ):
                raise AdbError(
                    f"Configured ADB device {selected.serial} is not a USB device."
                )
        else:
            if self.config.require_usb:
                selected = self._select_usb_device(ready)
            elif (
                not ready
                and self.config.connect_on_start
                and self._configured_fallback_serial()
            ):
                with suppress(AdbError):
                    self.connect()
                devices = self.devices()
                ready = [item for item in devices if item.state == "device"]
                selected = self._select_automatic_device(ready)
            else:
                selected = self._select_automatic_device(ready)
        self._selected_serial = selected.serial
        return selected

    @classmethod
    def _select_usb_device(cls, ready: Sequence[DeviceInfo]) -> DeviceInfo:
        usb_devices = [
            item
            for item in ready
            if not cls._is_network_or_emulator_serial(item.serial)
        ]
        if len(usb_devices) == 1:
            return usb_devices[0]
        if len(usb_devices) > 1:
            serials = ", ".join(item.serial for item in usb_devices)
            raise AdbError(
                f"Multiple USB ADB devices are ready: {serials}. "
                "Set adb.serial explicitly to avoid controlling the wrong device."
            )
        raise AdbError(
            "No USB Android device is ready. Connect one device, enable USB debugging, "
            "and authorize this computer."
        )

    @classmethod
    def _select_automatic_device(cls, ready: Sequence[DeviceInfo]) -> DeviceInfo:
        usb_devices = [
            item
            for item in ready
            if not cls._is_network_or_emulator_serial(item.serial)
        ]
        if usb_devices:
            return cls._select_usb_device(ready)
        if len(ready) == 1:
            return ready[0]
        if not ready:
            raise AdbError(
                "No ready ADB device. Authorize USB debugging or enable emulator ADB."
            )
        serials = ", ".join(item.serial for item in ready)
        raise AdbError(
            f"Multiple non-USB ADB devices are ready: {serials}. "
            "Set adb.serial explicitly to avoid controlling the wrong device."
        )

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
        if action.kind == ActionKind.BACK:
            self.client.run(["shell", "input", "keyevent", "4"])
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

    def back(self, frame: Frame) -> None:
        self.execute(ActionCommand.back(), frame)

    @staticmethod
    def sleep(duration_ms: int) -> None:
        time.sleep(duration_ms / 1000)


class RecordingActionDriver:
    """Non-mutating driver for tests and dry integration checks."""

    def __init__(self) -> None:
        self.actions: list[ActionCommand] = []

    def execute(self, action: ActionCommand, frame: Frame) -> None:
        self.actions.append(action)
