"""Localhost status interface with a minimal allowlisted control surface."""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .http_security import is_loopback_origin
from .status import build_runtime_status

_CONTROL_PATHS = {
    "/control/stop": "stop",
    "/control/restart-game": "restart-game",
}


_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


class _StatusHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that hangs up mid-response (a dashboard poll that timed out,
        # a closed browser tab) is routine, not a fault worth a stack trace on
        # the Bot's console.  Anything else still surfaces normally.
        if isinstance(sys.exc_info()[1], _DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)

    def __init__(
        self,
        address: tuple[str, int],
        logs_dir: Path,
        control_handlers: dict[str, Callable[[], bool | None]],
        metadata: dict[str, Any] | None = None,
        workflow_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.logs_dir = logs_dir
        self.control_handlers = control_handlers
        self.metadata = dict(metadata or {})
        self.workflow_provider = workflow_provider
        super().__init__(address, _StatusHandler)


class _StatusHandler(BaseHTTPRequestHandler):
    server: _StatusHttpServer

    def _send_json(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except _DISCONNECT_ERRORS:
            # The caller went away before it read the reply.  Nothing left to
            # send, and no reason to unwind through the rest of the handler.
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            payload = {
                "ok": True,
                "service": "dino-mutant-bot-status",
                "api_version": 2,
                "process_id": os.getpid(),
            }
            payload.update(self.server.metadata)
            self._send_json(200, payload)
            return
        status = build_runtime_status(self.server.logs_dir)
        if path == "/status":
            if self.server.workflow_provider is not None:
                status["workflow"] = self.server.workflow_provider()
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
                    "service": "猛龍計畫本機狀態與控制 API",
                    "read_endpoints": ["/health", "/status", "/actions", "/settings"],
                    "control_endpoints": [
                        f"POST {path}"
                        for path, action in _CONTROL_PATHS.items()
                        if action in self.server.control_handlers
                    ],
                },
            )
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        origin = self.headers.get("Origin", "")
        if not is_loopback_origin(origin):
            self._send_json(403, {"error": "forbidden_origin"})
            return
        action = _CONTROL_PATHS.get(path)
        if action is None:
            self._send_json(404, {"error": "not_found"})
            return
        handler = self.server.control_handlers.get(action)
        if handler is None:
            self._send_json(503, {"error": "control_unavailable", "action": action})
            return
        accepted = handler()
        if accepted is False:
            self._send_json(409, {"error": "control_rejected", "action": action})
            return
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
        control_handlers: dict[str, Callable[[], bool | None]] | None = None,
        metadata: dict[str, Any] | None = None,
        workflow_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.logs_dir = logs_dir
        self.port = port
        self.control_handlers = dict(control_handlers or {})
        self.metadata = dict(metadata or {})
        self.workflow_provider = workflow_provider
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
            self.metadata,
            self.workflow_provider,
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
