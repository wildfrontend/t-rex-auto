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

from .metrics import MetricsStore

DASHBOARD_VERSION = 1
DEFAULT_BOT_PORTS = {"hatch-hunt": 8773, "hunt": 8765}
_ASSET_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_LOG_LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[^|]+) \| (?P<message>.*)$"
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
        if (
            "Hatch full | phase A complete" in message
            or "Hatch 攻擊特化 |" in message
        ):
            stage, label = "nest_attack", "篩選攻擊親代"
        elif "Hatch HP特化 |" in message:
            stage, label = "nest_hp", "篩選 HP 親代"
        elif "Hatch auto-place | tag=頂尖" in message:
            stage, label = "nest_top", "頂尖自動放置"
        elif "Hatch auto-place | tag=量產" in message:
            stage, label = "nest_mass", "量產自動放置"
        elif "Hatch full | no ready incubator eggs" in message:
            stage, label = "collect", "收集所有巢蛋"
        elif "Hatch cave |" in message:
            stage, label = "cave", "洞穴容量與淘汰"
        elif "Hatch+Hunt | cooldown handoff window" in message:
            stage, label = "handoff", "返回孵蛋首頁"
        elif "Hatch+Hunt | centered home confirmed" in message:
            stage, label = "hatch", "檢查孵蛋"
            cooldown_remaining = 0
        elif "Hatch+Hunt | egg cooldown" in message and "switching to hunt" in message:
            stage, label = "cooldown_hunt", "冷卻期間狩獵"
        elif "Hatch | wait " in message and "collected all nest eggs" in message:
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
    ) -> None:
        self.runtime_root = runtime_root
        self.logs_dir = logs_dir
        self.bot_ports = dict(bot_ports or DEFAULT_BOT_PORTS)
        self._start_lock = threading.Lock()
        self._last_start = 0.0

    def discover(self) -> dict[str, Any]:
        for mode in ("hatch-hunt", "hunt"):
            port = self.bot_ports[mode]
            health = _get_json(f"http://127.0.0.1:{port}/health")
            if health and health.get("service") == "dino-mutant-bot-status":
                status = _get_json(f"http://127.0.0.1:{port}/status") or {}
                return {
                    "running": bool(status.get("running", True)),
                    "mode": mode,
                    "mode_label": "自動孵蛋＋狩獵" if mode == "hatch-hunt" else "純狩獵",
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

    def _runner_command(self, mode: str) -> list[str]:
        scripts = self.runtime_root / "app" / "scripts"
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
        else:
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

    def start(self, mode: str) -> dict[str, Any]:
        if mode not in self.bot_ports:
            raise RuntimeError("Unsupported Bot mode")
        with self._start_lock:
            active = self.discover()
            if active["running"]:
                raise RuntimeError(f"Bot already running in {active['mode_label']} mode")
            now = time.monotonic()
            if now - self._last_start < 5:
                raise RuntimeError("Bot start already requested")
            if os.name != "nt":
                raise RuntimeError("Bot launch is only available on Windows")
            creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            subprocess.Popen(  # noqa: S603 - fixed local PowerShell runner and allowlist
                self._runner_command(mode),
                cwd=self.runtime_root,
                creationflags=creation_flags,
            )
            self._last_start = now
        return {"accepted": True, "action": "start", "mode": mode}

    def _active_control(self, action: str) -> dict[str, Any]:
        active = self.discover()
        if not active["running"] or active["port"] is None:
            raise RuntimeError("No running Bot was found")
        return _post_json(f"http://127.0.0.1:{active['port']}/control/{action}")

    def stop(self) -> dict[str, Any]:
        return self._active_control("stop")

    def restart_game(self) -> dict[str, Any]:
        return self._active_control("restart-game")

    def restart_bot(self) -> dict[str, Any]:
        active = self.discover()
        if not active["running"] or not active["mode"]:
            raise RuntimeError("No running Bot was found")
        mode = str(active["mode"])
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
        main_script = self.runtime_root / "app" / "main.py"
        config = self.runtime_root / "app" / "config.json"
        if action == "open-logs":
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise RuntimeError("Opening folders is only available on Windows")
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            startfile(str(self.logs_dir))
            return {"accepted": True, "action": action}
        if action == "diagnostics":
            command = [sys.executable, str(main_script), "--config", str(config), "diagnostics"]
        elif action == "snapshot":
            output = self.runtime_root / "app" / "debug" / (
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
        controller: DashboardController,
        assets: Path,
    ) -> None:
        self.metrics = metrics
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
            elif action == "stop":
                result = self.server.controller.stop()
            elif action == "restart-game":
                result = self.server.controller.restart_game()
            elif action == "restart-bot":
                result = self.server.controller.restart_bot()
            elif action in {"diagnostics", "snapshot", "open-logs"}:
                result = self.server.controller.run_tool(action)
            else:
                self._send_json(404, {"error": "not_found"})
                return
        except RuntimeError as exc:
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
    ) -> None:
        self.runtime_root = runtime_root
        self.logs_dir = logs_dir
        self.database = database
        self.port = port
        self.assets = assets or Path(__file__).with_name("dashboard_assets")
        self.metrics = MetricsStore(database, logs_dir)
        self.controller = DashboardController(runtime_root, logs_dir)
        self._server: _DashboardHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server else self.port
        return f"http://127.0.0.1:{port}"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _DashboardHttpServer(
            ("127.0.0.1", self.port),
            self.metrics,
            self.controller,
            self.assets,
        )

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
