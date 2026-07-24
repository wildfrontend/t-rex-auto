from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CONTROL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "control-macos.py"
SPEC = importlib.util.spec_from_file_location("control_macos", CONTROL_PATH)
assert SPEC is not None and SPEC.loader is not None
control_macos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control_macos)


def test_mutating_control_requires_explicit_confirmation() -> None:
    with pytest.raises(control_macos.ControlError) as error:
        control_macos.require_confirmation(False)

    assert error.value.code == "confirmation_required"


def test_process_identity_matches_api_port_and_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        control_macos,
        "request_json",
        lambda *_args, **_kwargs: {
            "ok": True,
            "service": control_macos.SERVICE_NAME,
            "process_id": 123,
        },
    )
    monkeypatch.setattr(control_macos, "listener_pids", lambda _port: {123})
    monkeypatch.setattr(
        control_macos,
        "process_command",
        lambda _pid: [
            str(control_macos.PYTHON_EXECUTABLE),
            str(control_macos.MAIN_SCRIPT),
            "--config",
            str(control_macos.CONFIG_FILE),
            "run",
            "--status-port",
            "8765",
        ],
    )

    health = control_macos.assert_api_identity(8765, require_process=True)

    assert health["process_id"] == 123


def test_process_identity_rejects_wrong_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        control_macos,
        "request_json",
        lambda *_args, **_kwargs: {
            "ok": True,
            "service": control_macos.SERVICE_NAME,
            "process_id": 123,
        },
    )
    monkeypatch.setattr(control_macos, "listener_pids", lambda _port: {999})

    with pytest.raises(control_macos.ControlError) as error:
        control_macos.assert_api_identity(8765, require_process=True)

    assert error.value.code == "status_api_process_identity_mismatch"


def test_start_rejects_occupied_port_with_unknown_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control_macos, "require_runtime", lambda: None)
    monkeypatch.setattr(control_macos, "listener_pids", lambda _port: {999})
    monkeypatch.setattr(
        control_macos,
        "request_json",
        lambda *_args, **_kwargs: {"ok": True, "service": "another-service"},
    )

    with pytest.raises(control_macos.ControlError) as error:
        control_macos.start_bot("fast", 8765)

    assert error.value.code == "status_api_identity_mismatch"
