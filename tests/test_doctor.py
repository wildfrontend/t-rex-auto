from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import dino_bot.doctor as doctor
from dino_bot.config import AppConfig


class _Detector:
    asset_count = 1


class _AdbClient:
    executable = "adb.exe"

    def ensure_ready(self) -> SimpleNamespace:
        return SimpleNamespace(serial="127.0.0.1:16384")


@pytest.fixture
def windows_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(doctor.platform, "platform", lambda: "Windows-11")
    monkeypatch.setattr(doctor, "OpenCvDetector", lambda *_args: _Detector())
    monkeypatch.setattr(doctor, "AdbClient", lambda _config: _AdbClient())


def test_adb_backend_does_not_require_visible_emulator_window(
    monkeypatch: pytest.MonkeyPatch,
    windows_checks: None,
) -> None:
    base_config = AppConfig(root=Path("."), emulator="custom")
    config = replace(
        base_config,
        capture=replace(base_config.capture, backend="adb"),
    )

    def unexpected_window_finder(*_args: object) -> object:
        raise AssertionError("ADB diagnostics must not inspect desktop windows")

    monkeypatch.setattr(doctor, "EmulatorWindowFinder", unexpected_window_finder)

    checks = doctor.run_checks(config)

    assert all(check.name != "custom window" for check in checks)
    assert all(check.ok for check in checks)


def test_mss_backend_still_requires_visible_emulator_window(
    monkeypatch: pytest.MonkeyPatch,
    windows_checks: None,
) -> None:
    config = AppConfig(root=Path("."))

    class Finder:
        def __init__(self, *_args: object) -> None:
            pass

        def find(self) -> int:
            return 123

    monkeypatch.setattr(doctor, "EmulatorWindowFinder", Finder)

    checks = doctor.run_checks(config)

    window_check = next(check for check in checks if check.name == "bluestacks window")
    assert window_check.ok
    assert window_check.detail == "HWND=123"
