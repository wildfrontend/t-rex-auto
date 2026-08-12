from __future__ import annotations

import json
from pathlib import Path

import pytest

from dino_bot import adb_discovery
from dino_bot.actions import AdbClient, AdbError
from dino_bot.cli import build_parser, main
from dino_bot.config import AdbConfig, ConfigError, load_config


class FakeAdb:
    """An adb whose reachable endpoints are declared, not discovered."""

    def __init__(self, *, listening: dict[int, str] | None = None) -> None:
        # port -> state reported once connected
        self.listening = dict(listening or {})
        self.connected: dict[str, str] = {}
        self.calls: list[list[str]] = []

    def run(self, args, use_serial: bool = True) -> str:  # noqa: ANN001
        args = list(args)
        self.calls.append(args)
        if args[0] == "connect":
            serial = args[1]
            port = int(serial.rsplit(":", 1)[1])
            if port in self.listening:
                self.connected[serial] = self.listening[port]
                return f"connected to {serial}"
            return f"cannot connect to {serial}"
        if args[0] == "devices":
            lines = ["List of devices attached"]
            for serial, state in sorted(self.connected.items()):
                lines.append(f"{serial}\t{state} product:test")
            return "\n".join(lines)
        raise AssertionError(f"unexpected adb call: {args}")


def _client(config: AdbConfig, fake: FakeAdb) -> AdbClient:
    client = AdbClient.__new__(AdbClient)
    client.config = config
    client.executable = "adb"
    client.run = fake.run  # type: ignore[method-assign]
    return client


@pytest.fixture
def only_listening(monkeypatch: pytest.MonkeyPatch):
    """Make the TCP pre-probe answer from a declared set instead of the host."""

    def install(ports: set[int]) -> None:
        monkeypatch.setattr(
            adb_discovery,
            "is_listening",
            lambda host, port, timeout=0.0: port in ports,
        )

    return install


def test_discovery_only_dials_ports_that_are_actually_listening(only_listening) -> None:
    only_listening({5555})
    fake = FakeAdb(listening={5555: "device"})

    found = adb_discovery.discover(lambda args: fake.run(args))

    connects = [call for call in fake.calls if call[0] == "connect"]
    # `adb connect` to a closed port blocks for about a second, so a scan that
    # dialled every candidate would cost more than the rest of startup.
    assert connects == [["connect", "127.0.0.1:5555"]]
    assert [device.serial for device in found] == ["127.0.0.1:5555"]
    assert found[0].ready
    assert found[0].hint


def test_discovery_reports_devices_adb_already_knew_about(only_listening) -> None:
    only_listening(set())
    fake = FakeAdb()
    # A USB phone or an Android Studio AVD is a legitimate target whose serial
    # is on no candidate port; a port-only search would never see it.
    fake.connected["emulator-5554"] = "device"

    found = adb_discovery.discover(lambda args: fake.run(args))

    assert [device.serial for device in found] == ["emulator-5554"]
    assert found[0].preexisting is True
    assert found[0].hint == ""


def test_empty_serial_adopts_the_only_device_that_answers(only_listening) -> None:
    only_listening({16384})
    fake = FakeAdb(listening={16384: "device"})
    client = _client(AdbConfig(serial=None), fake)

    device = client.ensure_ready()

    assert device.serial == "127.0.0.1:16384"


def test_empty_serial_refuses_to_guess_between_two_devices(only_listening) -> None:
    only_listening({5555, 62001})
    fake = FakeAdb(listening={5555: "device", 62001: "device"})
    client = _client(AdbConfig(serial=None), fake)

    with pytest.raises(AdbError) as excinfo:
        client.ensure_ready()

    # Picking one would silently drive whichever emulator sorted first.
    message = str(excinfo.value)
    assert "127.0.0.1:5555" in message
    assert "127.0.0.1:62001" in message


def test_configured_serial_still_works_when_adb_had_not_connected(only_listening) -> None:
    only_listening({7555})
    fake = FakeAdb(listening={7555: "device"})
    client = _client(
        AdbConfig(serial="127.0.0.1:7555", connect_on_start=False),
        fake,
    )

    assert client.ensure_ready().serial == "127.0.0.1:7555"


def test_wrong_port_error_names_what_is_actually_connected(only_listening) -> None:
    only_listening({5555})
    fake = FakeAdb(listening={5555: "device"})
    client = _client(AdbConfig(serial="127.0.0.1:16384"), fake)

    with pytest.raises(AdbError) as excinfo:
        client.ensure_ready()

    message = str(excinfo.value)
    assert "127.0.0.1:16384" in message
    # The old error only repeated the port that failed, which is the one piece
    # of information the user already had and could not act on.
    assert "127.0.0.1:5555" in message
    assert "dino-bot adb" in message


def test_scan_is_skipped_when_auto_discover_is_off(only_listening) -> None:
    only_listening({5555})
    fake = FakeAdb(listening={5555: "device"})
    client = _client(
        AdbConfig(serial="127.0.0.1:16384", auto_discover=False),
        fake,
    )

    with pytest.raises(AdbError):
        client.ensure_ready()

    # connect_on_start still dials the configured serial once; what must not
    # happen is the candidate sweep finding 5555 behind the user's back.
    connects = [call[1] for call in fake.calls if call[0] == "connect"]
    assert connects == ["127.0.0.1:16384"]


def test_configured_discovery_ports_replace_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[int] = []
    monkeypatch.setattr(
        adb_discovery,
        "is_listening",
        lambda host, port, timeout=0.0: probed.append(port) or False,
    )
    fake = FakeAdb()
    client = _client(AdbConfig(serial=None, discovery_ports=(1234, 5678)), fake)

    client.discover()

    assert probed == [1234, 5678]


def test_config_rejects_a_discovery_port_that_cannot_be_one(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"adb": {"discovery_ports": [5555, 99999]}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="between 1 and 65535"):
        load_config(config_path)


def test_cli_adb_auto_writes_the_found_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"capture": {"backend": "adb"}, "adb": {"serial": "127.0.0.1:1"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        adb_discovery, "is_listening", lambda host, port, timeout=0.0: port == 5555
    )
    fake = FakeAdb(listening={5555: "device"})
    monkeypatch.setattr(AdbClient, "_resolve_executable", staticmethod(lambda _: "adb"))
    monkeypatch.setattr(AdbClient, "run", lambda self, args, use_serial=True: fake.run(args))

    exit_code = main(["--config", str(config_path), "adb", "--auto"])

    assert exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["adb"]["serial"] == "127.0.0.1:5555"
    # Everything else in the file has to survive being rewritten.
    assert saved["capture"] == {"backend": "adb"}
    assert "127.0.0.1:5555" in capsys.readouterr().out


def test_cli_adb_clear_switches_the_config_to_automatic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"adb": {"serial": "127.0.0.1:16384"}}), encoding="utf-8"
    )

    assert main(["--config", str(config_path), "adb", "--clear"]) == 0

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["adb"]["serial"] is None


def test_cli_parses_adb_command_options() -> None:
    args = build_parser().parse_args(["adb", "--set", "127.0.0.1:5555"])

    assert args.command == "adb"
    assert args.set_serial == "127.0.0.1:5555"
