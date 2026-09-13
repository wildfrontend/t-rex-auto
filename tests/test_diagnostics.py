from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from dino_bot import __version__
from dino_bot.config import AppConfig
from dino_bot.diagnostics import create_diagnostic_bundle, redact_text
from dino_bot.doctor import Check
from dino_bot.optimization import summarize_events


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
    # Assert against the single source of truth, not a literal: pinning the
    # string here is what let the reported version drift from the shipped one.
    assert manifest["bot_version"] == __version__
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


def test_optimization_counts_pending_checks_separately() -> None:
    target = {"type": "dinosaur", "x": 80, "y": 50}
    records = [
        {
            "t": "08:00:00.000",
            "c": 1,
            "e": "verify",
            "target": target,
            "attempt": 1,
            "result": {"ok": False, "reason": "legacy pending"},
        },
        {
            "t": "08:00:00.500",
            "c": 1,
            "e": "verify",
            "target": target,
            "attempt": 1,
            "result": {"ok": True, "reason": "legacy final"},
        },
        {
            "t": "08:00:01.000",
            "c": 2,
            "e": "verify",
            "phase": "pending",
            "target": target,
            "attempt": 1,
            "result": {"ok": False, "reason": "pending"},
        },
        {
            "t": "08:00:02.000",
            "c": 2,
            "e": "verify",
            "phase": "final",
            "target": target,
            "attempt": 1,
            "result": {"ok": False, "reason": "final"},
        },
    ]

    summary = summarize_events(json.dumps(record) for record in records)

    assert summary["verify"] == {
        "checks_total": 4,
        "pending": 2,
        "total": 2,
        "failed": 1,
        "failure_rate": 0.5,
        "retry_exhausted": 0,
    }


def test_optimization_aggregates_blind_stall_episodes() -> None:
    """Each record covers only the interval since the last one in its episode."""

    records = [
        {"t": "08:00:20.000", "c": 8, "e": "blind_stall", "seconds": 20.0,
         "stage": "recenter", "escapes": 1},
        {"t": "08:00:40.000", "c": 15, "e": "blind_stall", "seconds": 20.0,
         "stage": "recenter", "escapes": 2},
        {"t": "08:01:00.000", "c": 22, "e": "blind_stall", "seconds": 20.0,
         "stage": "recenter", "escapes": 3},
        {"t": "08:05:00.000", "c": 90, "e": "blind_stall", "seconds": 25.0,
         "stage": "mail", "escapes": 1},
    ]

    summary = summarize_events(json.dumps(record) for record in records)

    assert summary["blind_stalls"] == {
        "episodes": 2,
        "seconds_total": 85.0,
        "seconds_longest": 60.0,
        "stages": {"recenter": 3, "mail": 1},
    }
    codes = {item["code"] for item in summary["suggestions"]}
    assert "planner_blind_on_unknown_screen" in codes


def test_optimization_reports_how_often_each_control_was_visible() -> None:
    """A template that stops matching reads exactly like a button that is gone."""

    records = [
        {"t": "08:00:00.000", "c": 1, "e": "detect", "ms": 2300, "n": 2, "det": [
            {"type": "mailbox_button", "x": 841, "y": 1210, "conf": 0.99},
            {"type": "dinosaur", "x": 500, "y": 620, "conf": 0.9},
        ]},
        {"t": "08:00:03.000", "c": 2, "e": "detect", "ms": 2300, "n": 2, "det": [
            {"type": "dinosaur", "x": 505, "y": 615, "conf": 0.9},
            {"type": "dinosaur", "x": 300, "y": 700, "conf": 0.9},
        ]},
    ]

    summary = summarize_events(json.dumps(record) for record in records)

    assert summary["detection_visibility"] == {
        "detect_cycles": 2,
        # Cycles the type appeared in, not how many of it were matched: two
        # dinosaurs in one frame is still one cycle of visibility.
        "seen_share": {"dinosaur": 1.0, "mailbox_button": 0.5},
    }


def test_optimization_splits_idle_cycles_out_of_each_stage() -> None:
    target = {"type": "map_exit_nest_button", "x": 841, "y": 1295}
    records = [
        {"t": "08:00:00.000", "c": 1, "e": "plan", "stage": "recenter",
         "target": target},
        {"t": "08:00:03.000", "c": 2, "e": "plan", "stage": "recenter"},
        {"t": "08:00:06.000", "c": 3, "e": "plan", "stage": "recenter"},
    ]

    summary = summarize_events(json.dumps(record) for record in records)

    assert summary["stage_cycles"] == {"recenter": 3}
    assert summary["stage_idle_cycles"] == {"recenter": 2}


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


def test_diagnostic_bundle_includes_recent_failure_evidence(tmp_path: Path) -> None:
    stalls_dir = tmp_path / "logs" / "stalls"
    stalls_dir.mkdir(parents=True)
    for index in range(4):
        stem = f"parent-stats-20260811-0044{index:02d}"
        stalls_dir.joinpath(f"{stem}.png").write_bytes(f"frame-{index}".encode())
        stalls_dir.joinpath(f"{stem}-left-hp.png").write_bytes(
            f"crop-{index}".encode()
        )
        stalls_dir.joinpath(f"{stem}.json").write_text(
            json.dumps({"reason": "parent_stats_unreadable", "token": "hidden"}),
            encoding="utf-8",
        )
        os.utime(stalls_dir / f"{stem}.json", (index + 1, index + 1))

    output = create_diagnostic_bundle(
        AppConfig(root=tmp_path),
        tmp_path / "with-evidence.zip",
        config_path=tmp_path / "config.json",
        checks=[],
    )

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        newest_json = archive.read(
            "logs/stalls/parent-stats-20260811-004403.json"
        ).decode("utf-8")

    assert "logs/stalls/parent-stats-20260811-004403.png" in names
    assert "logs/stalls/parent-stats-20260811-004403-left-hp.png" in names
    assert "logs/stalls/parent-stats-20260811-004400.json" not in names
    assert manifest["failure_evidence"]["by_type"] == {"parent-stats": 3}
    assert "hidden" not in newest_json
    assert "<redacted>" in newest_json


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


def test_event_log_keeps_deep_history_as_gzipped_generations(tmp_path: Path) -> None:
    """A seven-hour run must not lose its first six hours.

    The 16 MB cap fills in about 97 minutes of real running, so one generation
    covered barely three hours. Generations past the first are gzipped, which
    is what makes keeping twenty of them cost 21 MB instead of 320 MB.
    """

    import gzip

    from dino_bot.events import JsonlEventLog

    logs_dir = tmp_path / "logs"
    log = JsonlEventLog(logs_dir, max_bytes=200, backup_count=4)
    for index in range(60):
        log.emit("plan", marker=index)
    log.close()

    date = datetime.now().strftime("%Y%m%d")
    live = logs_dir / f"events-{date}.jsonl"
    first = logs_dir / f"events-{date}.1.jsonl"
    assert live.exists(), "the live file is always plain"
    assert first.exists(), "the newest backup stays plain for the readers"
    assert not first.name.endswith(".gz")

    archived = sorted(logs_dir.glob(f"events-{date}.*.jsonl.gz"))
    assert archived, "older generations are compressed"
    assert len(archived) <= 3, "nothing is kept past backup_count"
    # A higher generation number is older, the way logrotate numbers them and
    # the way the pre-existing `.1` already behaved, so chronological order is
    # the reverse of filename order.
    markers = []
    for path in reversed(archived):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            markers += [json.loads(line)["marker"] for line in handle if line.strip()]
    assert markers == sorted(markers), "history stays in order across generations"
    assert markers, "the archives carry real events, not empty files"

    # The diagnostic bundle globs events-*.jsonl; the archives must not match
    # it, or it would ship gzip bytes as text.
    assert sorted(p.name for p in logs_dir.glob("events-*.jsonl")) == [
        f"events-{date}.1.jsonl",
        f"events-{date}.jsonl",
    ]


def test_text_log_rolls_into_compressed_generations(tmp_path: Path) -> None:
    import gzip
    import logging as std_logging

    from dino_bot.logging import DailyFileHandler

    logs_dir = tmp_path / "logs"
    handler = DailyFileHandler(logs_dir, max_bytes=200, backup_count=3)
    handler.setFormatter(std_logging.Formatter("%(message)s"))
    for index in range(60):
        handler.emit(
            std_logging.LogRecord(
                "dino_bot", std_logging.INFO, __file__, index, f"line {index:04d}", None, None
            )
        )
    handler.close()

    date = datetime.now().strftime("%Y%m%d")
    assert (logs_dir / f"{date}.log").exists()
    assert (logs_dir / f"{date}.1.log").exists()
    archived = sorted(logs_dir.glob(f"{date}.*.log.gz"))
    assert archived and len(archived) <= 2
    with gzip.open(archived[0], "rt", encoding="utf-8") as handle:
        assert handle.read().startswith("line ")
    # _recent_log_text globs 20*.log and must not pick up the archives.
    assert all(not p.name.endswith(".gz") for p in logs_dir.glob("20*.log"))
