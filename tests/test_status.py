from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dino_bot.status import build_runtime_status
from dino_bot.status_server import LocalStatusServer


def write_log(logs_dir: Path, content: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "20260722.log").write_text(content.strip() + "\n", encoding="utf-8")


def sample_log() -> str:
    return (
        """
19:00:00 | INFO | Bot started | Sense -> Think -> Act
19:00:01 | INFO | Bot stopped | actions=0 | cycles=0
19:01:00 | INFO | Timing | click=1000ms | dinosaur=1000ms | hunt=3000ms | confirm=2000ms"""
        " | idle=250ms\n"
        """
19:01:01 | INFO | Bot started | Sense -> Think -> Act
19:01:02 | INFO | Planning | dinosaur at (400,600) confidence=0.900
19:01:02 | INFO | Action | tap (400,600) | attempt=1
19:01:03 | INFO | Verify | Success | next UI detected: hunt_button
19:01:04 | INFO | Planning | hunt_confirm_button at (451,1412) confidence=1.000
19:01:04 | INFO | Action | tap (451,1412) | attempt=1
19:01:05 | WARNING | Verify | Failed | expected next UI not detected
19:01:06 | INFO | Planning | hunt_confirm_button at (451,1412) confidence=1.000
19:01:06 | INFO | Action | tap (451,1412) | attempt=2
19:01:07 | INFO | Verify | Success | next UI detected: map_exit_nest_button
19:01:08 | INFO | Workflow | completed cycle 1/unlimited
19:01:09 | WARNING | Recovery | black screen detected | mean=0.00 | waiting 45s
19:01:54 | ERROR | Recovery | black screen persisted 45s; restarting game app
19:01:55 | INFO | Recovery | game restarted; waiting 15s for launch
"""
    )


def test_status_parses_latest_session_counts_and_actions(tmp_path: Path) -> None:
    write_log(tmp_path, sample_log())

    status = build_runtime_status(tmp_path, recent_action_limit=2)

    assert status["running"] is True
    assert status["current_stage"] == "recovering"
    assert status["session_started"] == "2026-07-22T19:01:01"
    assert status["successful_hunts"] == 1
    assert status["mailbox_cycles"] == 1
    assert status["total_actions"] == 3
    assert status["verification_failures"] == 1
    assert status["black_screen_detections"] == 1
    assert status["black_screen_persisted"] == 1
    assert status["game_restarts"] == 1
    assert status["last_successful_hunt"] == "2026-07-22T19:01:07"
    assert status["timing"]["hunt_button_delay_ms"] == 3000
    assert [item["result"] for item in status["recent_actions"]] == [
        "failed",
        "success",
    ]


def test_status_reports_stopped_session(tmp_path: Path) -> None:
    write_log(tmp_path, sample_log() + "19:02:00 | INFO | Bot stopped | actions=3 | cycles=1\n")

    status = build_runtime_status(tmp_path)

    assert status["running"] is False
    assert status["current_stage"] == "stopped"


def test_local_status_server_exposes_read_only_json(tmp_path: Path) -> None:
    write_log(tmp_path, sample_log())

    with LocalStatusServer(tmp_path, port=0) as server:
        with urlopen(f"{server.url}/health", timeout=2) as response:  # noqa: S310
            health = json.load(response)
        with urlopen(f"{server.url}/status", timeout=2) as response:  # noqa: S310
            status = json.load(response)
        with urlopen(f"{server.url}/actions", timeout=2) as response:  # noqa: S310
            actions = json.load(response)
        with urlopen(f"{server.url}/settings", timeout=2) as response:  # noqa: S310
            settings = json.load(response)

    assert health["ok"] is True
    assert health["service"] == "dino-mutant-bot-status"
    assert health["api_version"] == 2
    assert health["process_id"] == os.getpid()
    assert status["successful_hunts"] == 1
    assert len(actions["actions"]) == 3
    assert settings["timing"]["idle_delay_ms"] == 250


def test_local_status_server_accepts_allowlisted_controls(tmp_path: Path) -> None:
    requested: list[str] = []
    with LocalStatusServer(
        tmp_path,
        port=0,
        control_handlers={
            "stop": lambda: requested.append("stop"),
            "restart-game": lambda: requested.append("restart-game"),
        },
    ) as server:
        stop_request = Request(f"{server.url}/control/stop", method="POST")
        with urlopen(stop_request, timeout=2) as response:  # noqa: S310
            stop_payload = json.load(response)
        restart_request = Request(f"{server.url}/control/restart-game", method="POST")
        with urlopen(restart_request, timeout=2) as response:  # noqa: S310
            restart_payload = json.load(response)

    assert stop_payload == {"accepted": True, "action": "stop"}
    assert restart_payload == {"accepted": True, "action": "restart-game"}
    assert requested == ["stop", "restart-game"]


def test_local_status_server_rejects_declined_control(tmp_path: Path) -> None:
    with LocalStatusServer(
        tmp_path,
        port=0,
        control_handlers={"restart-game": lambda: False},
    ) as server:
        request = Request(f"{server.url}/control/restart-game", method="POST")
        with pytest.raises(HTTPError) as rejected:
            urlopen(request, timeout=2)  # noqa: S310

    assert rejected.value.code == 409


def test_local_status_server_rejects_unknown_control_and_remote_origin(
    tmp_path: Path,
) -> None:
    with LocalStatusServer(tmp_path, port=0, control_handlers={"stop": lambda: None}) as server:
        unknown = Request(f"{server.url}/control/restart", method="POST")
        with pytest.raises(HTTPError) as unknown_error:
            urlopen(unknown, timeout=2)  # noqa: S310

        remote = Request(
            f"{server.url}/control/stop",
            method="POST",
            headers={"Origin": "https://example.com"},
        )
        with pytest.raises(HTTPError) as origin_error:
            urlopen(remote, timeout=2)  # noqa: S310

        lookalike = Request(
            f"{server.url}/control/stop",
            method="POST",
            headers={"Origin": "http://localhost.example.com"},
        )
        with pytest.raises(HTTPError) as lookalike_error:
            urlopen(lookalike, timeout=2)  # noqa: S310

    assert unknown_error.value.code == 404
    assert origin_error.value.code == 403
    assert lookalike_error.value.code == 403


def test_client_disconnect_does_not_raise_or_kill_the_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A client that hangs up mid-response must stay silent and non-fatal.

    Reproduces the real failure: a status poll that timed out client-side left
    the server writing into a closed socket, which printed a ConnectionAborted
    traceback to the Bot's console.
    """

    write_log(tmp_path, sample_log())

    with LocalStatusServer(tmp_path, port=0) as server:
        host, port = "127.0.0.1", int(server.url.rsplit(":", 1)[1])

        # Send a request, then close without reading the reply.
        for _ in range(5):
            sock = socket.create_connection((host, port), timeout=2)
            sock.sendall(b"GET /status HTTP/1.1\r\nHost: localhost\r\n\r\n")
            sock.close()

        # The server must still answer normally afterwards.
        with urlopen(f"{server.url}/status", timeout=5) as response:  # noqa: S310
            status = json.load(response)

    assert status["successful_hunts"] == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "ConnectionAborted" not in captured.err
