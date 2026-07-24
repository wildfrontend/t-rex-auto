#!/usr/bin/env python3
"""Allowlisted local controller for the macOS Dino Mutant Bot."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CONTROLLER_DIR = Path(__file__).resolve().parent
SOURCE_APP_ROOT = CONTROLLER_DIR.parent
DEPLOYED_APP_ROOT = SOURCE_APP_ROOT / ".runtime-macos" / "app"
APP_ROOT = (
    DEPLOYED_APP_ROOT
    if (DEPLOYED_APP_ROOT / "main.py").is_file()
    else SOURCE_APP_ROOT
)
SCRIPT_DIR = APP_ROOT / "scripts"
PYTHON_EXECUTABLE = APP_ROOT / ".venv" / "bin" / "python"
RUN_SCRIPT = SCRIPT_DIR / "run-macos.sh"
MAIN_SCRIPT = APP_ROOT / "main.py"
CONFIG_FILE = APP_ROOT / "config.json"
LOG_FILE = APP_ROOT / "logs" / "macos-launcher.log"
SERVICE_NAME = "dino-mutant-bot-status"


class ControlError(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code


def write_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def api_root(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def request_json(port: int, path: str, *, method: str = "GET") -> dict[str, Any]:
    request = Request(f"{api_root(port)}{path}", method=method)
    try:
        with urlopen(request, timeout=3) as response:  # noqa: S310
            payload = json.load(response)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise ControlError("status_api_unavailable", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ControlError("status_api_identity_mismatch", "API returned a non-object response")
    return payload


def listener_pids(port: int) -> set[int]:
    completed = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
            "-Fp",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise ControlError("port_identity_check_failed", completed.stderr.strip())
    return {
        int(line[1:])
        for line in completed.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }


def process_command(pid: int) -> list[str]:
    completed = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ControlError("status_api_process_identity_missing")
    try:
        return shlex.split(completed.stdout.strip())
    except ValueError as exc:
        raise ControlError("status_api_process_identity_mismatch", str(exc)) from exc


def assert_api_identity(port: int, *, require_process: bool = False) -> dict[str, Any]:
    health = request_json(port, "/health")
    if health.get("ok") is not True or health.get("service") != SERVICE_NAME:
        raise ControlError("status_api_identity_mismatch")
    if not require_process:
        return health

    raw_pid = health.get("process_id")
    if not isinstance(raw_pid, int):
        raise ControlError("status_api_process_identity_missing")
    if raw_pid not in listener_pids(port):
        raise ControlError("status_api_process_identity_mismatch")

    command = process_command(raw_pid)
    main_matches = any(Path(item).resolve() == MAIN_SCRIPT for item in command if "main.py" in item)
    port_matches = any(
        command[index : index + 2] == ["--status-port", str(port)]
        for index in range(max(0, len(command) - 1))
    )
    if not main_matches or "run" not in command or not port_matches:
        raise ControlError("status_api_process_identity_mismatch")
    return health


def require_runtime() -> None:
    if not PYTHON_EXECUTABLE.is_file():
        raise ControlError(
            "runtime_not_installed",
            f"Run {SCRIPT_DIR / 'install-macos-runtime.sh'} first",
        )


def require_confirmation(confirm: bool) -> None:
    if not confirm:
        raise ControlError(
            "confirmation_required",
            "start, stop, and restart require --confirm",
        )


def start_bot(speed: str, port: int) -> dict[str, Any]:
    require_runtime()
    occupants = listener_pids(port)
    if occupants:
        health = assert_api_identity(port, require_process=True)
        return {
            "ok": True,
            "action": "start",
            "result": "already_running",
            "process_id": health["process_id"],
            "status_port": port,
        }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(RUN_SCRIPT), speed, str(port)],
            cwd=APP_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    return {
        "ok": True,
        "action": "start",
        "result": "launcher_started",
        "launcher_pid": process.pid,
        "speed": speed,
        "status_port": port,
        "log": str(LOG_FILE),
    }


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_bot(port: int) -> tuple[dict[str, Any], int]:
    health = assert_api_identity(port, require_process=True)
    response = request_json(port, "/control/stop", method="POST")
    return response, int(health["process_id"])


def run_python_command(arguments: list[str], *, quiet: bool = False) -> int:
    require_runtime()
    return subprocess.run(
        [str(PYTHON_EXECUTABLE), str(MAIN_SCRIPT), "--config", str(CONFIG_FILE), *arguments],
        cwd=APP_ROOT,
        stdout=subprocess.DEVNULL if quiet else None,
        check=False,
    ).returncode


def port_value(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "status",
            "start",
            "stop",
            "restart",
            "doctor",
            "diagnostics",
            "snapshot",
        ),
    )
    parser.add_argument("--speed", choices=("fast", "safe"), default="fast")
    parser.add_argument("--status-port", type=port_value, default=8765)
    parser.add_argument("--confirm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if sys.platform != "darwin":
            raise ControlError("unsupported_platform", "control-macos.py requires macOS")
        if args.action == "status":
            assert_api_identity(args.status_port)
            status = request_json(args.status_port, "/status")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        if args.action == "start":
            require_confirmation(args.confirm)
            write_json(start_bot(args.speed, args.status_port))
            return 0
        if args.action == "stop":
            require_confirmation(args.confirm)
            response, _ = stop_bot(args.status_port)
            write_json(
                {
                    "ok": True,
                    "action": "stop",
                    "result": "graceful_stop_requested",
                    "response": response,
                }
            )
            return 0
        if args.action == "restart":
            require_confirmation(args.confirm)
            _, process_id = stop_bot(args.status_port)
            deadline = time.monotonic() + 20
            while process_exists(process_id) and time.monotonic() < deadline:
                time.sleep(0.25)
            if process_exists(process_id):
                raise ControlError("stop_timeout", "Bot did not stop within 20 seconds")
            result = start_bot(args.speed, args.status_port)
            result["action"] = "restart"
            write_json(result)
            return 0
        if args.action == "doctor":
            return run_python_command(["doctor"])
        if args.action == "diagnostics":
            output = (
                APP_ROOT
                / "diagnostics"
                / f"dino-diagnostic-{datetime.now():%Y%m%d-%H%M%S}.zip"
            )
            exit_code = run_python_command(
                ["diagnostics", "--output", str(output)],
                quiet=True,
            )
            if exit_code == 0:
                write_json(
                    {
                        "ok": True,
                        "action": "diagnostics",
                        "output": str(output),
                        "includes_screenshot": False,
                    }
                )
            return exit_code
        output = APP_ROOT / "debug" / f"macos-{datetime.now():%Y%m%d-%H%M%S}.png"
        exit_code = run_python_command(
            ["snapshot", "--backend", "adb", "--output", str(output)]
        )
        if exit_code == 0:
            write_json({"ok": True, "action": "snapshot", "output": str(output)})
        return exit_code
    except ControlError as exc:
        write_json(
            {
                "ok": False,
                "action": args.action,
                "error": exc.code,
                "message": str(exc),
                "url": f"{api_root(args.status_port)}/status",
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
