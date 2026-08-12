"""ADB transport and coordinate-aware Android emulator action driver."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import adb_discovery
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
        executable_name = "adb.exe" if sys.platform == "win32" else "adb"
        app_root = Path(__file__).resolve().parents[2]
        candidates.append(
            str(app_root / "tools" / "platform-tools" / executable_name)
        )
        discovered = shutil.which("adb")
        if discovered:
            candidates.append(discovered)
        for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
            sdk_root = os.environ.get(variable)
            if sdk_root:
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
                r"C:\Program Files (x86)\Nemu\vmonitor\bin\adb_server.exe",
                r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
                r"C:\Program Files\BlueStacks\HD-Adb.exe",
            ]
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return str(Path(candidate))
        raise AdbError(
            "ADB executable not found. Install Android Platform Tools or configure the "
            "ADB executable supplied by your emulator."
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
        completed = self._invoke(args, use_serial=use_serial)
        reconnect_attempted = False
        if (
            check
            and completed.returncode != 0
            and use_serial
            and self.config.serial
            and args
            and args[0] in {"shell", "exec-out"}
            and self._is_closed_transport(completed)
        ):
            # Some emulator ADB daemons remain listed as `device` while their
            # shell transport has already been closed.  The next tap used to
            # crash the Bot immediately in that state.  ADB's own `reconnect`
            # command closes the selected host-side transport and forces a new
            # connection; keep this recovery strictly bounded to one attempt.
            reconnect_attempted = True
            self._recover_closed_transport()
            completed = self._invoke(args, use_serial=use_serial)

        if check and completed.returncode != 0:
            message = self._error_message(completed)
            if reconnect_attempted:
                message += (
                    " (ADB reconnect was attempted once; restart the emulator"
                    " if its shell remains closed)"
                )
            raise AdbError(f"ADB exited with {completed.returncode}: {message}")
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace").strip()

    def _invoke(
        self,
        args: Sequence[str],
        *,
        use_serial: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                self._command(args, use_serial),
                capture_output=True,
                timeout=self.config.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdbError(f"ADB command failed to start: {exc}") from exc

    @staticmethod
    def _error_message(completed: subprocess.CompletedProcess[bytes]) -> str:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        return stderr or stdout or "unknown ADB error"

    @classmethod
    def _is_closed_transport(
        cls,
        completed: subprocess.CompletedProcess[bytes],
    ) -> bool:
        return cls._error_message(completed).lower() == "error: closed"

    def _recover_closed_transport(self) -> None:
        # Ignore the recovery command's own exit code: the original command is
        # retried exactly once and remains the authoritative result.
        self._invoke(["reconnect"], use_serial=True)
        time.sleep(0.5)
        if self.config.serial:
            self._invoke(
                ["connect", self.config.serial],
                use_serial=False,
            )
            time.sleep(0.5)

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

    def discover(self) -> list[adb_discovery.DiscoveredDevice]:
        """Probe the known emulator ports and report every reachable device."""

        ports = (
            tuple((port, "") for port in self.config.discovery_ports)
            if self.config.discovery_ports
            else None
        )
        return adb_discovery.discover(
            lambda args: str(self.run(args, use_serial=False)),
            ports=ports,
        )

    def ensure_ready(self) -> DeviceInfo:
        if self.config.connect_on_start and self.config.serial:
            self.connect()
        devices = self.devices()
        ready = [item for item in devices if item.state == "device"]
        if self.config.serial:
            ready = [item for item in ready if item.serial == self.config.serial]
        if ready:
            return ready[0]

        # Nothing matched. A configured serial that is simply wrong looks
        # exactly like a closed emulator from here, so scan before failing:
        # either the answer is one probe away, or the error can finally name
        # what *is* connected instead of only what was asked for.
        if not self.config.auto_discover:
            raise AdbError(self._not_ready_message(()))
        found = [device for device in self.discover() if device.ready]
        chosen: adb_discovery.DiscoveredDevice | None = None
        if self.config.serial:
            # The configured port can be right while adb simply had not
            # connected to it yet; the scan does that as a side effect.
            chosen = next(
                (device for device in found if device.serial == self.config.serial),
                None,
            )
        elif len(found) == 1:
            chosen = found[0]
        if chosen is None:
            raise AdbError(self._not_ready_message(found))
        return DeviceInfo(chosen.serial, chosen.state, chosen.description)

    def _not_ready_message(
        self,
        found: Sequence[adb_discovery.DiscoveredDevice],
    ) -> str:
        wanted = self.config.serial or "any device"
        lines = [f"No ready ADB device for {wanted}."]
        if found:
            lines.append(f"Reachable now: {adb_discovery.describe(found)}.")
            if self.config.serial:
                lines.append(
                    "Set adb.serial to one of those, or clear it to use the only"
                    " one found. `dino-bot adb --auto` writes it for you."
                )
            else:
                lines.append(
                    "More than one device answered; set adb.serial to the one you"
                    " want. `dino-bot adb` lists them."
                )
        else:
            lines.append(
                "No emulator answered on the known ADB ports. Start the emulator,"
                " enable ADB in its settings, then run `dino-bot adb`."
            )
        return " ".join(lines)

    def display_size(self) -> tuple[int, int]:
        output = str(self.run(["shell", "wm", "size"]))
        matches = re.findall(r"(?:Override|Physical) size:\s*(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"Cannot parse device size from: {output!r}")
        width, height = matches[-1]
        return int(width), int(height)

    def screencap_png(self) -> bytes:
        return bytes(self.run(["exec-out", "screencap", "-p"], binary=True))

    def screencap_raw(self) -> bytes:
        """Uncompressed framebuffer dump; skips the device-side PNG encoder."""

        return bytes(self.run(["exec-out", "screencap"], binary=True))


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
