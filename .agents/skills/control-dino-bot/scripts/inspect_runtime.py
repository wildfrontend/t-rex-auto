#!/usr/bin/env python3
"""Read-only Dino Bot log fallback for environments without Windows interop."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PORTS = {
    "hunt": 8765,
    "hatch": 8766,
    "hatch-full": 8772,
    "hatch-hunt": 8773,
    "hatch-stage": 8774,
}
FEATURE_RE = re.compile(r"Feature \| (?P<mode>[a-z0-9-]+)")
STATUS_PORT_RE = re.compile(r"Status API \| http://127\.0\.0\.1:(?P<port>\d+)/status")
ADB_SERIAL_RE = re.compile(r"ADB \| device=(?P<serial>[^ |]+)")


def _tail_text(path: Path, limit: int = 2_000_000) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                return stream.read()[-limit:]
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _latest_valid_event(path: Path) -> dict[str, Any] | None:
    for line in reversed(_tail_text(path).splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _latest_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    groups = match.groupdict()
    return groups.get("mode") or groups.get("port") or groups.get("serial")


def _local_config_path(runtime_root: Path, raw_config: str) -> Path:
    normalized = raw_config.replace("\\", "/")
    windows_absolute = re.fullmatch(r"(?P<drive>[A-Za-z]):/(?P<path>.+)", normalized)
    if windows_absolute:
        return (
            Path("/mnt")
            / windows_absolute.group("drive").lower()
            / windows_absolute.group("path")
        )
    config_path = Path(normalized)
    return config_path if config_path.is_absolute() else runtime_root / config_path


def _resolve_instance(
    runtime_root: Path,
    instance_id: str | None,
) -> tuple[Path, dict[str, Any] | None]:
    if not instance_id:
        return runtime_root / "app" / "logs", None
    registry_path = runtime_root / "instances.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read instance registry: {registry_path}") from exc
    instances = registry.get("instances") if isinstance(registry, dict) else None
    if not isinstance(instances, list):
        raise ValueError(f"invalid instance registry: {registry_path}")
    for item in instances:
        if not isinstance(item, dict) or item.get("id") != instance_id:
            continue
        raw_config = item.get("config")
        if not isinstance(raw_config, str):
            break
        config_path = _local_config_path(runtime_root, raw_config)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        adb = config.get("adb") if isinstance(config, dict) else {}
        return config_path.parent / "logs", {
            "id": instance_id,
            "name": item.get("name"),
            "config": str(config_path.resolve()),
            "status_port": item.get("status_port"),
            "serial": adb.get("serial") if isinstance(adb, dict) else None,
        }
    raise ValueError(f"unknown instance: {instance_id}")


def inspect(
    runtime_root: Path,
    requested_mode: str,
    instance_id: str | None = None,
) -> dict[str, Any]:
    logs_dir, instance = _resolve_instance(runtime_root, instance_id)
    event_files = sorted(
        [*logs_dir.glob("events-*.jsonl"), *logs_dir.glob("events-*.jsonl.gz")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    text_files = sorted(
        [*logs_dir.glob("20*.log"), *logs_dir.glob("20*.log.gz")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    latest_event_file = event_files[0] if event_files else None
    latest_event = _latest_valid_event(latest_event_file) if latest_event_file else None
    latest_text_file = text_files[0] if text_files else None
    recent_text = _tail_text(latest_text_file) if latest_text_file else ""

    detected_mode = _latest_match(FEATURE_RE, recent_text)
    detected_port_text = _latest_match(STATUS_PORT_RE, recent_text)
    detected_port = int(detected_port_text) if detected_port_text else None
    detected_serial = _latest_match(ADB_SERIAL_RE, recent_text)
    mode = requested_mode if requested_mode != "auto" else detected_mode
    registry_port = instance.get("status_port") if instance else None
    expected_port = registry_port or (PORTS.get(mode) if mode else detected_port)

    last_session_action = None
    if latest_event_file:
        for line in reversed(_tail_text(latest_event_file).splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("e") == "session" and event.get("action") in {"start", "stop"}:
                last_session_action = event["action"]
                break

    modified_at = (
        datetime.fromtimestamp(latest_event_file.stat().st_mtime, UTC)
        if latest_event_file
        else None
    )
    age_seconds = (
        max(0.0, (datetime.now(UTC) - modified_at).total_seconds())
        if modified_at
        else None
    )
    if last_session_action == "stop":
        state = "stopped_by_log"
    elif last_session_action == "start" and age_seconds is not None and age_seconds <= 30:
        state = "active_recently"
    else:
        state = "unknown"

    return {
        "ok": bool(latest_event_file or latest_text_file),
        "evidence": "shared_log_fallback",
        "state": state,
        "process_identity_verified": False,
        "instance": instance,
        "logged_serial": detected_serial,
        "mode": mode,
        "expected_status_port": expected_port,
        "logged_status_port": detected_port,
        "last_session_action": last_session_action,
        "last_event_age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "last_event": latest_event,
        "event_log": str(latest_event_file) if latest_event_file else None,
        "text_log": str(latest_text_file) if latest_text_file else None,
        "warning": (
            "Log evidence cannot prove Windows process identity or distinguish concurrent "
            "processes. Use the Windows controller for confirmed running/stopped state."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--instance")
    parser.add_argument("--mode", choices=["auto", *PORTS], default="auto")
    args = parser.parse_args()
    try:
        result = inspect(args.runtime_root.resolve(), args.mode, args.instance)
    except ValueError as exc:
        print(
            json.dumps(
                {"ok": False, "error": "instance_resolution_failed", "message": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
