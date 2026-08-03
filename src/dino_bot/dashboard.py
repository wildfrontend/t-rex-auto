"""Loopback-only web dashboard for Bot statistics and safe controls."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .hatch_inventory import HatchBoostInventoryStore
from .metrics import MetricsStore

DASHBOARD_VERSION = 1
DEFAULT_BOT_PORTS = {"hatch-hunt": 8773, "hunt": 8765, "hatch-stage": 8774}
HATCH_STAGE_LABELS = {
    "hatch": "孵蛋一輪",
    "attack": "攻擊親代",
    "hp": "HP 親代",
    "top": "頂尖自動放置",
    "mass": "量產自動放置",
    "collect": "收集所有巢蛋",
    "cave": "洞穴容量與淘汰",
}
_ASSET_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_LOG_LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[^|]+) \| (?P<message>.*)$"
)
_PLANNING_TARGET = re.compile(r"^Planning \| (?P<target>\S+) at \(")
_COLLECT_TARGETS = frozenset(
    {
        "hatch_collect_eggs_button",
        "hatch_nest_mask_close",
    }
)
_CAVE_TARGETS = frozenset(
    {
        "hatch_cave_swipe",
        "hatch_cave_recenter",
        "cave",
        "hatch_cave_select_button",
        "hatch_cull_tag_header",
        "hatch_select_weakest_button",
        "hatch_select_choose_button",
        "hatch_cave_continuous_button",
        "hatch_cave_close_button",
    }
)


def _loopback_origin_allowed(origin: str) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


def _get_json(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback URL only
            return json.load(response)
    except (OSError, ValueError, URLError):
        return None


def _post_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(url, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback URL only
            return json.load(response)
    except (OSError, ValueError, URLError) as exc:
        raise RuntimeError(f"Bot control unavailable: {exc}") from exc


def _latest_log_lines(logs_dir: Path, limit: int = 300) -> list[str]:
    paths = sorted(logs_dir.glob("20*.log"))
    if not paths:
        return []
    try:
        with paths[-1].open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 2 * 1024 * 1024))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if " | DEBUG | " not in line]
    return lines[-max(1, limit) :]


def _workflow_status(logs_dir: Path, mode: str | None) -> dict[str, Any]:
    stage = "hunt" if mode == "hunt" else "hatch"
    label = "純狩獵" if mode == "hunt" else "檢查孵蛋"
    cooldown_remaining: int | None = None
    latest_date: str | None = None
    paths = sorted(logs_dir.glob("20*.log"))
    if paths:
        try:
            latest_date = datetime.strptime(paths[-1].stem, "%Y%m%d").date().isoformat()
        except ValueError:
            latest_date = None

    for line in _latest_log_lines(logs_dir):
        match = _LOG_LINE.match(line)
        if match is None:
            continue
        time_text = match.group("time")
        message = match.group("message")
        planning = _PLANNING_TARGET.match(message)
        planned_target = planning.group("target") if planning is not None else None
        if message.startswith("Feature | hatch-stage | stage="):
            stage_name = message.split("stage=", 1)[1].split(" ", 1)[0]
            stage = {
                "hatch": "hatch",
                "attack": "nest_attack",
                "hp": "nest_hp",
                "top": "nest_top",
                "mass": "nest_mass",
                "collect": "collect",
                "cave": "cave",
            }.get(stage_name, "hatch")
            label = HATCH_STAGE_LABELS.get(stage_name, "單階段孵化")
        elif "Hatch full | screening stage starting | stage=" in message:
            stage_name = message.split("stage=", 1)[1].split(" ", 1)[0]
            stage, label = {
                "attack": ("nest_attack", "篩選攻擊親代"),
                "hp": ("nest_hp", "篩選 HP 親代"),
                "top": ("nest_top", "頂尖自動放置"),
                "mass": ("nest_mass", "量產自動放置"),
            }.get(stage_name, ("nest_attack", "篩選攻擊親代"))
        elif "Hatch full | screening complete" in message:
            stage, label = "collect", "收集所有巢蛋"
        elif "Hatch full | phase A complete" in message:
            # 只有觸發篩選的批次才會進入親代篩選;一般循環接著收蛋。
            if "trigger=none" in message:
                stage, label = "collect", "收集所有巢蛋"
            else:
                stage, label = "nest_attack", "篩選攻擊親代"
        elif "Hatch 攻擊特化 |" in message:
            stage, label = "nest_attack", "篩選攻擊親代"
        elif "Hatch HP特化 |" in message:
            stage, label = "nest_hp", "篩選 HP 親代"
        elif "Hatch auto-place | tag=頂尖" in message:
            stage, label = "nest_top", "頂尖自動放置"
        elif "Hatch auto-place | tag=量產" in message:
            stage, label = "nest_mass", "量產自動放置"
        elif (
            "Hatch full | no ready incubator eggs" in message
            or planned_target in _COLLECT_TARGETS
        ):
            stage, label = "collect", "收集所有巢蛋"
        elif "Hatch cave |" in message or planned_target in _CAVE_TARGETS:
            stage, label = "cave", "洞穴容量與淘汰"
        elif "Hatch full | completed management cycle" in message:
            stage, label = "hatch", "檢查孵蛋"
        elif "Hatch+Hunt | cooldown handoff window" in message:
            stage, label = "handoff", "返回孵蛋首頁"
        elif "Hatch+Hunt | centered home confirmed" in message:
            stage, label = "hatch", "檢查孵蛋"
            cooldown_remaining = 0
        elif "Hatch+Hunt | egg cooldown" in message and "switching to hunt" in message:
            stage, label = "cooldown_hunt", "冷卻期間狩獵"
        elif "interim nest collection during hunt idle" in message:
            # 差事出發時帶著剩餘冷卻;收完會自動回狩獵。
            try:
                duration = int(float(message.split("resume=", 1)[1].split("s", 1)[0]))
                if latest_date:
                    started = datetime.fromisoformat(f"{latest_date}T{time_text}").astimezone()
                    elapsed = (datetime.now().astimezone() - started).total_seconds()
                    cooldown_remaining = max(0, round(duration - elapsed))
                    stage, label = "cooldown_hunt", "冷卻期間狩獵"
            except (ValueError, IndexError):
                pass
        elif "Hatch | wait " in message:
            # 兩種等待都算冷卻:關閉孵化器後、收完巢蛋後。
            try:
                duration = int(float(message.split("Hatch | wait ", 1)[1].split("s", 1)[0]))
                if latest_date:
                    started = datetime.fromisoformat(f"{latest_date}T{time_text}").astimezone()
                    elapsed = (datetime.now().astimezone() - started).total_seconds()
                    cooldown_remaining = max(0, round(duration - elapsed))
                    if cooldown_remaining > 30:
                        stage, label = "cooldown_hunt", "冷卻期間狩獵"
                    elif cooldown_remaining > 0:
                        stage, label = "handoff", "返回孵蛋首頁"
                    else:
                        # 冷卻已過:回孵化器檢查,不再殘留舊狀態。
                        stage, label = "hatch", "檢查孵蛋"
            except (ValueError, IndexError):
                pass
    return {
        "stage": stage,
        "label": label,
        "cooldown_remaining_seconds": cooldown_remaining,
    }


class DashboardController:
    """Discover a running Bot and expose only explicitly allowlisted actions."""

    def __init__(
        self,
        runtime_root: Path,
        logs_dir: Path,
        *,
        bot_ports: dict[str, int] | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        candidate = runtime_root / "app"
        self.app_root = candidate if candidate.is_dir() else runtime_root
        self.logs_dir = logs_dir
        self.config_path = config_path or self.app_root / "config.json"
        self.bot_ports = dict(bot_ports or DEFAULT_BOT_PORTS)
        self._start_lock = threading.Lock()
        self._last_start = 0.0
        self._explicit_stop_at = 0.0

    def discover(self) -> dict[str, Any]:
        for mode in ("hatch-hunt", "hunt", "hatch-stage"):
            port = self.bot_ports[mode]
            health = _get_json(f"http://127.0.0.1:{port}/health")
            if health and health.get("service") == "dino-mutant-bot-status":
                status = _get_json(f"http://127.0.0.1:{port}/status") or {}
                return {
                    "running": bool(status.get("running", True)),
                    "mode": mode,
                    "mode_label": (
                        "自動孵蛋＋狩獵"
                        if mode == "hatch-hunt"
                        else "純狩獵"
                        if mode == "hunt"
                        else "單階段孵化"
                    ),
                    "port": port,
                    "status": status,
                    "workflow": _workflow_status(self.logs_dir, mode),
                }
        return {
            "running": False,
            "mode": None,
            "mode_label": "未啟動",
            "port": None,
            "status": {},
            "workflow": _workflow_status(self.logs_dir, None),
        }

    def _bot_command(self, mode: str, *, stage: str | None = None) -> list[str]:
        """Direct Python launch used on non-Windows hosts (no PowerShell runner)."""

        command = [
            sys.executable,
            str(self.app_root / "main.py"),
            "--config",
            str(self.config_path),
        ]
        if mode == "hunt":
            command += [
                "run",
                "--mode",
                "runtime",
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--batch-size",
                "10",
                "--mail-after-hunts",
                "30",
                "--speed",
                "fast",
                "--status-port",
                str(self.bot_ports[mode]),
                "--verbose",
            ]
        elif mode == "hatch-hunt":
            command += [
                "run",
                "--feature",
                "hatch-hunt",
                "--mode",
                "debug",
                "--speed",
                "safe",
                "--status-port",
                str(self.bot_ports[mode]),
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--verbose",
            ]
        elif mode == "hatch-stage" and stage in HATCH_STAGE_LABELS:
            command += [
                "run",
                "--feature",
                f"hatch-stage-{stage}",
                "--mode",
                "debug",
                "--speed",
                "safe",
                "--status-port",
                str(self.bot_ports[mode]),
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--verbose",
            ]
        else:
            raise RuntimeError("Unsupported Bot mode")
        return command

    def _runner_command(self, mode: str, *, stage: str | None = None) -> list[str]:
        scripts = self.app_root / "scripts"
        if mode == "hunt":
            runner = scripts / "run-windows.ps1"
            arguments = [
                "-Mode",
                "runtime",
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
                "-BatchSize",
                "10",
                "-MailAfterHunts",
                "30",
                "-Speed",
                "fast",
                "-StatusPort",
                str(self.bot_ports[mode]),
            ]
        elif mode == "hatch-hunt":
            runner = scripts / "run-hatch-windows.ps1"
            arguments = [
                "-Feature",
                "hatch-hunt",
                "-Mode",
                "debug",
                "-Speed",
                "safe",
                "-StatusPort",
                str(self.bot_ports[mode]),
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
            ]
        elif mode == "hatch-stage" and stage in HATCH_STAGE_LABELS:
            runner = scripts / "run-hatch-windows.ps1"
            arguments = [
                "-Feature",
                f"hatch-stage-{stage}",
                "-Mode",
                "debug",
                "-Speed",
                "safe",
                "-StatusPort",
                str(self.bot_ports[mode]),
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
            ]
        else:
            raise RuntimeError("Unsupported Bot mode")
        runner = runner.resolve()
        if runner.parent != scripts.resolve() or not runner.is_file():
            raise RuntimeError(f"Runner not found: {runner}")
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            *arguments,
        ]

    def start(self, mode: str, *, stage: str | None = None) -> dict[str, Any]:
        if mode not in self.bot_ports:
            raise RuntimeError("Unsupported Bot mode")
        if mode == "hatch-stage" and stage not in HATCH_STAGE_LABELS:
            raise RuntimeError("Unsupported hatch stage")
        with self._start_lock:
            now = time.monotonic()
            if now - self._last_start < 5:
                raise RuntimeError("Bot start already requested")
            self._last_start = now
            active = self.discover()
            if not active["running"]:
                self._launch(mode, stage=stage)
                return {"accepted": True, "action": "start", "mode": mode, "stage": stage}
            previous_label = str(active["mode_label"])
            with suppress(RuntimeError):
                self.stop()

        def start_after_stop() -> None:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if not self.discover()["running"]:
                    time.sleep(1)
                    with suppress(RuntimeError), self._start_lock:
                        self._launch(mode, stage=stage)
                    return
                time.sleep(0.5)

        threading.Thread(target=start_after_stop, daemon=True).start()
        return {
            "accepted": True,
            "action": "switch",
            "mode": mode,
            "stage": stage,
            "message": f"正在停止{previous_label}，隨後自動啟動新模式",
        }

    def _launch(self, mode: str, *, stage: str | None = None) -> None:
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            subprocess.Popen(  # noqa: S603 - fixed local PowerShell runner and allowlist
                self._runner_command(mode, stage=stage),
                cwd=self.runtime_root,
                creationflags=creation_flags,
            )
        else:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            launch_log = self.logs_dir / f"dashboard-launch-{mode}.log"
            with launch_log.open("ab") as stream:
                subprocess.Popen(  # noqa: S603 - fixed local Python entrypoint and allowlist
                    self._bot_command(mode, stage=stage),
                    cwd=self.app_root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

    def _active_control(self, action: str) -> dict[str, Any]:
        active = self.discover()
        if not active["running"] or active["port"] is None:
            raise RuntimeError("No running Bot was found")
        return _post_json(f"http://127.0.0.1:{active['port']}/control/{action}")

    def stop(self) -> dict[str, Any]:
        self._explicit_stop_at = time.monotonic()
        return self._active_control("stop")

    def restart_game(self) -> dict[str, Any]:
        return self._active_control("restart-game")

    def restart_bot(self) -> dict[str, Any]:
        active = self.discover()
        if not active["running"] or not active["mode"]:
            raise RuntimeError("No running Bot was found")
        mode = str(active["mode"])
        if mode == "hatch-stage":
            raise RuntimeError("單階段工作不支援重啟；請停止後重新選擇階段")
        self.stop()

        def restart_after_stop() -> None:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if not self.discover()["running"]:
                    time.sleep(1)
                    with suppress(RuntimeError):
                        self.start(mode)
                    return
                time.sleep(0.5)

        threading.Thread(target=restart_after_stop, daemon=True).start()
        return {"accepted": True, "action": "restart", "mode": mode}

    def run_tool(self, action: str) -> dict[str, Any]:
        main_script = self.app_root / "main.py"
        config = self.config_path
        if action == "open-logs":
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                startfile(str(self.logs_dir))
            elif sys.platform == "darwin":
                subprocess.run(["/usr/bin/open", str(self.logs_dir)], check=False)
            else:
                raise RuntimeError("Opening folders is only available on Windows and macOS")
            return {"accepted": True, "action": action}
        if action == "diagnostics":
            command = [sys.executable, str(main_script), "--config", str(config), "diagnostics"]
        elif action == "snapshot":
            output = self.app_root / "debug" / (
                "dashboard-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
            )
            command = [
                sys.executable,
                str(main_script),
                "--config",
                str(config),
                "snapshot",
                "--backend",
                "adb",
                "--output",
                str(output),
            ]
        else:
            raise RuntimeError("Unsupported dashboard tool")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"{action} failed")
        return {
            "accepted": True,
            "action": action,
            "message": completed.stdout.strip(),
        }


class _DashboardHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        metrics: MetricsStore,
        hatch_inventory: HatchBoostInventoryStore,
        controller: DashboardController,
        assets: Path,
    ) -> None:
        self.metrics = metrics
        self.hatch_inventory = hatch_inventory
        self.controller = controller
        self.assets = assets
        super().__init__(address, _DashboardHandler)


class _DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHttpServer

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, name: str) -> None:
        path = (self.server.assets / name).resolve()
        if path.parent != self.server.assets.resolve() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _ASSET_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > 1024:
            raise ValueError("JSON body must be between 1 and 1024 bytes")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_asset("index.html")
        elif path == "/dashboard.css":
            self._send_asset("dashboard.css")
        elif path == "/dashboard.js":
            self._send_asset("dashboard.js")
        elif path == "/api/health":
            self._send_json(
                200,
                {"ok": True, "service": "dino-dashboard", "api_version": DASHBOARD_VERSION},
            )
        elif path == "/api/overview":
            self._send_json(
                200,
                {
                    "active": self.server.controller.discover(),
                    "metrics": self.server.metrics.snapshot(),
                    "hatch_boost_inventory": (
                        self.server.hatch_inventory.snapshot().as_dict()
                    ),
                },
            )
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin", "")
        if not _loopback_origin_allowed(origin):
            self._send_json(403, {"error": "forbidden_origin"})
            return
        if self.headers.get("X-Dino-Dashboard") != "1":
            self._send_json(403, {"error": "dashboard_header_required"})
            return
        action = urlparse(self.path).path.removeprefix("/api/control/")
        try:
            if action == "start-hunt":
                result = self.server.controller.start("hunt")
            elif action == "start-hatch-hunt":
                result = self.server.controller.start("hatch-hunt")
            elif action.startswith("start-stage-"):
                stage = action.removeprefix("start-stage-")
                result = self.server.controller.start("hatch-stage", stage=stage)
            elif action == "stop":
                result = self.server.controller.stop()
            elif action == "restart-game":
                result = self.server.controller.restart_game()
            elif action == "restart-bot":
                result = self.server.controller.restart_bot()
            elif action == "set-boost-stock":
                payload = self._read_json()
                remaining = payload.get("remaining")
                if isinstance(remaining, bool) or not isinstance(remaining, int):
                    raise ValueError("remaining must be an integer")
                inventory = self.server.hatch_inventory.set_remaining(remaining)
                result = {
                    "accepted": True,
                    "action": action,
                    "inventory": inventory.as_dict(),
                    "message": f"冷卻加速券庫存已更新為 {inventory.remaining}",
                }
            elif action == "set-boost-enabled":
                payload = self._read_json()
                enabled = payload.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be a boolean")
                inventory = self.server.hatch_inventory.set_enabled(enabled)
                result = {
                    "accepted": True,
                    "action": action,
                    "inventory": inventory.as_dict(),
                    "message": (
                        "下一輪孵化將使用冷卻加速券"
                        if enabled
                        else "已關閉冷卻加速券使用"
                    ),
                }
            elif action in {"diagnostics", "snapshot", "open-logs"}:
                result = self.server.controller.run_tool(action)
            else:
                self._send_json(404, {"error": "not_found"})
                return
        except (RuntimeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc), "action": action})
            return
        self._send_json(202, result)

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardServer:
    """Run the dashboard service on loopback."""

    def __init__(
        self,
        runtime_root: Path,
        logs_dir: Path,
        database: Path,
        *,
        port: int = 8780,
        assets: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.logs_dir = logs_dir
        self.database = database
        self.port = port
        self.assets = assets or Path(__file__).with_name("dashboard_assets")
        self.metrics = MetricsStore(database, logs_dir)
        self.hatch_inventory = HatchBoostInventoryStore(database)
        self.controller = DashboardController(runtime_root, logs_dir, config_path=config_path)
        self._server: _DashboardHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._auto_resume_thread: threading.Thread | None = None
        self._auto_resume_stop = threading.Event()

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server else self.port
        return f"http://127.0.0.1:{port}"

    AUTO_RESUME_MODE = "hatch-hunt"
    AUTO_RESUME_IDLE_SECONDS = 10.0

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _DashboardHttpServer(
            ("127.0.0.1", self.port),
            self.metrics,
            self.hatch_inventory,
            self.controller,
            self.assets,
        )
        if self._auto_resume_thread is None:
            self._auto_resume_thread = threading.Thread(
                target=self._watch_stage_completion, daemon=True
            )
            self._auto_resume_thread.start()

    def _watch_stage_completion(self) -> None:
        """Queue the continuous mode after a standalone stage finishes on its own.

        A stage that exits by itself leaves nothing running; if the user takes
        no action within the grace window, resume hatch-hunt. An explicit
        dashboard stop suppresses the resume - stopping means stopping.
        """

        last_mode: str | None = None
        idle_since: float | None = None
        while not self._auto_resume_stop.wait(3):
            try:
                active = self.controller.discover()
            except Exception:
                continue
            if active["running"]:
                last_mode = str(active["mode"])
                idle_since = None
                continue
            if last_mode != "hatch-stage":
                idle_since = None
                continue
            now = time.monotonic()
            if now - self.controller._explicit_stop_at < 60:
                # 使用者主動按停止:這次不自動接手。
                last_mode = None
                idle_since = None
                continue
            if now - self.controller._last_start < 30:
                # 剛有啟動請求(可能是切換的延遲啟動),不插手。
                idle_since = None
                continue
            if idle_since is None:
                idle_since = now
                continue
            if now - idle_since < self.AUTO_RESUME_IDLE_SECONDS:
                continue
            last_mode = None
            idle_since = None
            with suppress(RuntimeError):
                self.controller.start(self.AUTO_RESUME_MODE)

    def serve_forever(self, *, open_browser: bool = False) -> None:
        self.start()
        assert self._server is not None
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(self.url)).start()
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            self._server = None

    def start_background(self) -> None:
        self.start()
        assert self._server is not None
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dino-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._auto_resume_stop.set()
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> DashboardServer:
        self.start_background()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
