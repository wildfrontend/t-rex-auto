"""Find the emulator's ADB endpoint instead of asking the user for a port.

A wrong port produces the same "no ready device" error as ADB being switched
off in the emulator, as the emulator not running, and as a firewall blocking
loopback. The port number is also not something an emulator puts in front of
its user, so the one question the error asks is the one the user cannot
answer. Probing the small set of ports the common emulators use turns it into
something the bot can answer for itself.

Dead ports are rejected with a TCP connect first. ``adb connect`` to a port
nobody is listening on blocks for about a second, so probing a dozen
candidates that way would cost more than the rest of startup put together.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

# Port-to-brand mapping drifts between emulator versions, so the label is a
# hint for the person reading the list and nothing here depends on it being
# right. Being on this list only means "worth a TCP probe".
DEFAULT_DISCOVERY_PORTS: tuple[tuple[int, str], ...] = (
    (5555, "BlueStacks / LDPlayer / 一般 adb over TCP"),
    (5557, "LDPlayer 第二個實例"),
    (5559, "LDPlayer 第三個實例"),
    (5565, "BlueStacks 多開"),
    (5575, "BlueStacks 多開"),
    (7555, "MuMu Player 6"),
    (16384, "BlueStacks 5 較新版 / MuMu Player 12"),
    (16416, "BlueStacks 5 多開"),
    (16448, "BlueStacks 5 多開"),
    (21503, "MEmu"),
    (62001, "Nox"),
    (62025, "Nox 多開"),
)

DEFAULT_PROBE_TIMEOUT = 0.15


@dataclass(frozen=True, slots=True)
class DiscoveredDevice:
    """One ADB endpoint that reported itself ready."""

    serial: str
    state: str
    description: str = ""
    hint: str = ""
    # Devices adb already knew about are listed even when their port is not a
    # candidate: a USB phone or an Android Studio AVD is a legitimate target
    # and would otherwise be invisible to a port-based search.
    preexisting: bool = False

    @property
    def ready(self) -> bool:
        return self.state == "device"

    def as_dict(self) -> dict[str, object]:
        return {
            "serial": self.serial,
            "state": self.state,
            "description": self.description,
            "hint": self.hint,
            "preexisting": self.preexisting,
            "ready": self.ready,
        }


def is_listening(
    host: str,
    port: int,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> bool:
    """Whether something accepts a TCP connection on ``host:port``."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def listening_ports(
    host: str,
    ports: Iterable[tuple[int, str]],
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> list[tuple[int, str]]:
    return [(port, hint) for port, hint in ports if is_listening(host, port, timeout)]


def discover(
    run: Callable[[Sequence[str]], str],
    *,
    host: str = "127.0.0.1",
    ports: Sequence[tuple[int, str]] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> list[DiscoveredDevice]:
    """Return every ADB device reachable now, connecting to candidates first.

    ``run`` takes ADB arguments without ``-s`` and returns stdout; the caller
    owns which adb binary that is.
    """

    candidates = tuple(DEFAULT_DISCOVERY_PORTS if ports is None else ports)
    hints = {f"{host}:{port}": hint for port, hint in candidates}

    before = {device.serial for device in _list_devices(run)}
    for port, _hint in listening_ports(host, candidates, timeout):
        serial = f"{host}:{port}"
        if serial in before:
            continue
        try:
            run(["connect", serial])
        except Exception:  # noqa: BLE001 - one dead candidate must not end the scan
            continue

    found: list[DiscoveredDevice] = []
    for device in _list_devices(run):
        found.append(
            DiscoveredDevice(
                serial=device.serial,
                state=device.state,
                description=device.description,
                hint=hints.get(device.serial, ""),
                preexisting=device.serial in before,
            )
        )
    return found


@dataclass(frozen=True, slots=True)
class _RawDevice:
    serial: str
    state: str
    description: str


def _list_devices(run: Callable[[Sequence[str]], str]) -> list[_RawDevice]:
    try:
        output = str(run(["devices", "-l"]))
    except Exception:  # noqa: BLE001 - report "nothing found", not a crash
        return []
    devices: list[_RawDevice] = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        devices.append(
            _RawDevice(
                serial=parts[0],
                state=parts[1] if len(parts) > 1 else "unknown",
                description=parts[2] if len(parts) > 2 else "",
            )
        )
    return devices


def describe(devices: Sequence[DiscoveredDevice]) -> str:
    """One-line summary of a scan, for logs and error messages."""

    if not devices:
        return "none"
    return ", ".join(
        f"{device.serial}({device.state}{'; ' + device.hint if device.hint else ''})"
        for device in devices
    )
