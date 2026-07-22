"""Structured runtime status derived from the Bot's human-readable logs."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

_LOG_LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[^|]+) \| (?P<message>.*)$"
)
_TIMING = re.compile(
    r"Timing \| click=(?P<click>\d+)ms \| dinosaur=(?P<dinosaur>\d+)ms"
    r" \| hunt=(?P<hunt>\d+)ms \| confirm=(?P<confirm>\d+)ms"
    r" \| idle=(?P<idle>\d+)ms"
)
_PLANNING = re.compile(r"Planning \| (?P<target>\S+) at \(")
_ACTION = re.compile(r"Action \| (?P<action>.+?) \| attempt=(?P<attempt>\d+)$")


def _timestamp(log_file: Path, time_text: str) -> str:
    try:
        date_text = datetime.strptime(log_file.stem, "%Y%m%d").date().isoformat()
    except ValueError:
        date_text = log_file.stem
    return f"{date_text}T{time_text}"


def _read_recent_entries(logs_dir: Path) -> list[dict[str, str]]:
    files = sorted(logs_dir.glob("20*.log"))[-2:]
    entries: list[dict[str, str]] = []
    for log_file in files:
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _LOG_LINE.match(line)
            if match is None:
                continue
            entries.append(
                {
                    "timestamp": _timestamp(log_file, match.group("time")),
                    "level": match.group("level").strip(),
                    "message": match.group("message"),
                    "log_file": str(log_file),
                }
            )
    return entries


def _latest_session(entries: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    start_index = -1
    for index, entry in enumerate(entries):
        if entry["message"].startswith("Bot started |"):
            start_index = index

    timing: dict[str, int] = {}
    search_end = start_index + 1 if start_index >= 0 else len(entries)
    for entry in reversed(entries[:search_end]):
        match = _TIMING.search(entry["message"])
        if match is not None:
            timing = {
                "click_delay_ms": int(match.group("click")),
                "dinosaur_delay_ms": int(match.group("dinosaur")),
                "hunt_button_delay_ms": int(match.group("hunt")),
                "hunt_confirm_delay_ms": int(match.group("confirm")),
                "idle_delay_ms": int(match.group("idle")),
            }
            break
    return (entries[start_index:] if start_index >= 0 else entries), timing


def _stage_from_message(message: str, running: bool) -> str:
    if not running:
        return "stopped"
    if "Recovery | black screen persisted" in message or "game restarted" in message:
        return "recovering"
    if "Recovery | black screen detected" in message or "restart deferred" in message:
        return "waiting_for_picture"
    if message.startswith("Verify |"):
        return "verifying"
    if message.startswith("Action |"):
        return "executing"
    if message.startswith("Recover |"):
        return "retrying"
    if message.startswith("Planning |"):
        return "planning"
    if message.startswith("Detect |") or message.startswith("Capture |"):
        return "scanning"
    if message.startswith("Bot started |"):
        return "starting"
    return "active"


def build_runtime_status(logs_dir: Path, recent_action_limit: int = 10) -> dict[str, Any]:
    """Return statistics for the most recent Bot session."""

    entries, timing = _latest_session(_read_recent_entries(logs_dir))
    if not entries:
        return {
            "running": False,
            "current_stage": "not_started",
            "session_started": None,
            "last_log_time": None,
            "last_successful_hunt": None,
            "successful_hunts": 0,
            "mailbox_cycles": 0,
            "total_actions": 0,
            "verification_failures": 0,
            "retry_exhausted": 0,
            "black_screen_detections": 0,
            "black_screen_persisted": 0,
            "game_restarts": 0,
            "game_restart_failures": 0,
            "timing": timing,
            "recent_actions": [],
            "log_file": None,
            "generated_at": datetime.now().astimezone().isoformat(),
        }

    actions: list[dict[str, Any]] = []
    pending_action: dict[str, Any] | None = None
    current_target: str | None = None
    successful_hunts = 0
    mailbox_cycles = 0
    verification_failures = 0
    retry_exhausted = 0
    black_screen_detections = 0
    black_screen_persisted = 0
    game_restarts = 0
    game_restart_failures = 0
    last_successful_hunt: str | None = None

    for entry in entries:
        message = entry["message"]
        planning = _PLANNING.search(message)
        if planning is not None:
            current_target = planning.group("target")

        action_match = _ACTION.search(message)
        if action_match is not None:
            pending_action = {
                "timestamp": entry["timestamp"],
                "target": current_target or "unknown",
                "action": action_match.group("action"),
                "attempt": int(action_match.group("attempt")),
                "result": "pending",
            }
            actions.append(pending_action)
        elif message.startswith("Verify | Success |"):
            if pending_action is not None:
                pending_action["result"] = "success"
                pending_action["result_time"] = entry["timestamp"]
                if pending_action["target"] == "hunt_confirm_button":
                    successful_hunts += 1
                    last_successful_hunt = entry["timestamp"]
                pending_action = None
        elif message.startswith("Verify | Failed |"):
            verification_failures += 1
            if pending_action is not None:
                pending_action["result"] = "failed"
                pending_action["result_time"] = entry["timestamp"]
                pending_action = None

        if message.startswith("Workflow | completed cycle"):
            mailbox_cycles += 1
        if "Verify | retry limit exhausted" in message:
            retry_exhausted += 1
        if "Recovery | black screen detected" in message:
            black_screen_detections += 1
        if "Recovery | black screen persisted" in message:
            black_screen_persisted += 1
        if "Recovery | game restarted;" in message:
            game_restarts += 1
        if "Recovery | game restart failed:" in message:
            game_restart_failures += 1

    stopped = any(entry["message"].startswith("Bot stopped |") for entry in entries)
    running = entries[0]["message"].startswith("Bot started |") and not stopped
    last_entry = entries[-1]
    limit = max(0, recent_action_limit)
    return {
        "running": running,
        "current_stage": _stage_from_message(last_entry["message"], running),
        "session_started": entries[0]["timestamp"] if running or stopped else None,
        "last_log_time": last_entry["timestamp"],
        "last_successful_hunt": last_successful_hunt,
        "successful_hunts": successful_hunts,
        "mailbox_cycles": mailbox_cycles,
        "total_actions": len(actions),
        "verification_failures": verification_failures,
        "retry_exhausted": retry_exhausted,
        "black_screen_detections": black_screen_detections,
        "black_screen_persisted": black_screen_persisted,
        "game_restarts": game_restarts,
        "game_restart_failures": game_restart_failures,
        "timing": timing,
        "recent_actions": actions[-limit:] if limit else [],
        "log_file": last_entry["log_file"],
        "generated_at": datetime.now().astimezone().isoformat(),
    }
