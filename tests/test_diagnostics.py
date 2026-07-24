from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from dino_bot.config import AppConfig
from dino_bot.diagnostics import create_diagnostic_bundle, redact_text
from dino_bot.doctor import Check


def test_diagnostic_bundle_contains_sanitized_evidence(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    logs_dir.joinpath("20260723.log").write_text(
        "\n".join(
            [
                "08:00:00 | INFO | Bot started | Sense -> Think -> Act",
                "08:00:01 | INFO | token=do-not-share contact=user@example.com",
                r"08:00:02 | ERROR | Failed at C:\Users\Alice\Dino\config.json",
                "08:00:03 | INFO | Planning | hunt_confirm_button at (451,1412) confidence=1.000",
                "08:00:03 | INFO | Action | tap (451,1412) | attempt=1",
                "08:00:04 | INFO | Verify | Success | next UI detected: map_exit_nest_button",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = AppConfig(root=tmp_path)
    output = create_diagnostic_bundle(
        config,
        Path("diagnostics/result.zip"),
        config_path=tmp_path / "config.json",
        checks=[Check("ADB device", False, r"C:\Users\Alice\adb.exe token=hidden")],
        generated_at=datetime(2026, 7, 23, 8, 1, tzinfo=UTC),
    )

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        status = json.loads(archive.read("status.json"))
        summary = json.loads(archive.read("summary.json"))
        settings = json.loads(archive.read("settings.json"))
        combined = "\n".join(
            archive.read(name).decode("utf-8", errors="replace")
            for name in names
            if name.endswith((".json", ".log", ".md", ".txt"))
        )

    assert {
        "README_FOR_CODEX.md",
        "manifest.json",
        "summary.json",
        "status.json",
        "doctor.json",
        "settings.json",
        "logs/recent.log",
    } <= names
    assert manifest["bot_version"] == "0.2.1"
    assert manifest["snapshot_included"] is False
    assert status["successful_hunts"] == 1
    assert settings["root"] == "<app-root>"
    assert summary["needs_attention"] is True
    assert "required_environment_checks_failed" in {
        item["code"] for item in summary["issues"]
    }
    assert "do-not-share" not in combined
    assert "user@example.com" not in combined
    assert r"C:\Users\Alice" not in combined
    assert "<redacted>" in combined
    assert "<email>" in combined
    assert "%USERPROFILE%" in combined


def test_diagnostic_bundle_preserves_config_error_and_snapshot_failure(tmp_path: Path) -> None:
    output = create_diagnostic_bundle(
        None,
        tmp_path / "broken-config",
        config_path=tmp_path / "config.json",
        config_error="Invalid JSON: token=private-value",
        snapshot_requested=True,
        snapshot_error="ADB screenshot failed",
        checks=[],
    )

    with ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        summary = json.loads(archive.read("summary.json"))
        config_error = archive.read("config_error.txt").decode("utf-8")
        snapshot_error = archive.read("snapshot_error.txt").decode("utf-8")

    assert output.suffix == ".zip"
    assert manifest["config_loaded"] is False
    assert manifest["snapshot_requested"] is True
    assert manifest["snapshot_included"] is False
    assert {item["code"] for item in summary["issues"]} >= {
        "configuration_invalid",
        "snapshot_unavailable",
    }
    assert "private-value" not in config_error
    assert "ADB screenshot failed" in snapshot_error


def test_diagnostic_bundle_includes_only_explicit_snapshot(tmp_path: Path) -> None:
    output = create_diagnostic_bundle(
        AppConfig(root=tmp_path),
        tmp_path / "with-snapshot.zip",
        config_path=tmp_path / "config.json",
        checks=[],
        snapshot_requested=True,
        snapshot_png=b"fake-png",
    )

    with ZipFile(output) as archive:
        assert archive.read("snapshot.png") == b"fake-png"
        assert "snapshot_error.txt" not in archive.namelist()


def test_redact_text_removes_bearer_and_home_paths(tmp_path: Path) -> None:
    value = redact_text(
        f"Bearer abc.def path={tmp_path}/logs mail=person@example.org",
        tmp_path,
    )

    assert "abc.def" not in value
    assert str(tmp_path) not in value
    assert "person@example.org" not in value
    assert "Bearer <redacted>" in value
    assert "<app-root>/logs" in value


def test_redact_text_removes_macos_user_home() -> None:
    value = redact_text(
        "ADB=/Users/Alice/Library/Android/sdk/platform-tools/adb "
        "log=/Users/Bob/Desktop/bot.log"
    )

    assert "Alice" not in value
    assert "Bob" not in value
    assert value == "ADB=$HOME/Library/Android/sdk/platform-tools/adb log=$HOME/Desktop/bot.log"
