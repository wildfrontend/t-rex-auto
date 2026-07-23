"""Localhost status interface with a minimal allowlisted control surface."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .status import build_runtime_status


class _StatusHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        logs_dir: Path,
        control_handlers: dict[str, Callable[[], None]],
    ) -> None:
        self.logs_dir = logs_dir
        self.control_handlers = control_handlers
        super().__init__(address, _StatusHandler)


class _StatusHandler(BaseHTTPRequestHandler):
    server: _StatusHttpServer

    def _send_json(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "dino-mutant-bot-status",
                    "api_version": 1,
                    "process_id": os.getpid(),
                },
            )
            return
        status = build_runtime_status(self.server.logs_dir)
        if path == "/status":
            self._send_json(200, status)
        elif path == "/actions":
            self._send_json(
                200,
                {
                    "session_started": status["session_started"],
                    "actions": status["recent_actions"],
                },
            )
        elif path == "/settings":
            self._send_json(200, {"timing": status["timing"]})
        elif path == "/":
            self._send_json(
                200,
                {
                    "service": "Dino Mutant Bot local status and control API",
                    "read_endpoints": ["/health", "/status", "/actions", "/settings"],
                    "control_endpoints": ["POST /control/stop"],
                },
            )
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        origin = self.headers.get("Origin", "")
        if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
            self._send_json(403, {"error": "forbidden_origin"})
            return
        action = "stop" if path == "/control/stop" else None
        if action is None:
            self._send_json(404, {"error": "not_found"})
            return
        handler = self.server.control_handlers.get(action)
        if handler is None:
            self._send_json(503, {"error": "control_unavailable", "action": action})
            return
        handler()
        self._send_json(202, {"accepted": True, "action": action})

    def log_message(self, format: str, *args: object) -> None:
        return


class LocalStatusServer:
    """Serve status JSON and explicit control handlers on loopback."""

    def __init__(
        self,
        logs_dir: Path,
        port: int = 8765,
        *,
        control_handlers: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self.logs_dir = logs_dir
        self.port = port
        self.control_handlers = dict(control_handlers or {})
        self._server: _StatusHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server else self.port
        return f"http://127.0.0.1:{port}"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _StatusHttpServer(
            ("127.0.0.1", self.port),
            self.logs_dir,
            self.control_handlers,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dino-bot-status-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> LocalStatusServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
