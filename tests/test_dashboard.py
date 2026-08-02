from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dino_bot.dashboard import DashboardController, DashboardServer


def write_assets(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
    (root / "dashboard.css").write_text("body{}", encoding="utf-8")
    (root / "dashboard.js").write_text("", encoding="utf-8")


def test_dashboard_serves_assets_overview_and_health(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_assets(assets)
    server = DashboardServer(
        tmp_path,
        tmp_path / "app" / "logs",
        tmp_path / "app" / "data" / "stats.sqlite3",
        port=0,
        assets=assets,
    )
    server.controller.discover = lambda: {  # type: ignore[method-assign]
        "running": False,
        "mode": None,
        "mode_label": "未啟動",
        "port": None,
        "status": {},
        "workflow": {"stage": "hatch", "label": "檢查孵蛋"},
    }

    with server:
        with urlopen(server.url, timeout=2) as response:  # noqa: S310
            html = response.read().decode()
        with urlopen(f"{server.url}/api/health", timeout=2) as response:  # noqa: S310
            health = json.load(response)
        with urlopen(f"{server.url}/api/overview", timeout=2) as response:  # noqa: S310
            overview = json.load(response)

    assert "dashboard" in html
    assert health == {"ok": True, "service": "dino-dashboard", "api_version": 1}
    assert overview["active"]["running"] is False
    assert overview["metrics"]["counters"]["hunt"]["total"] == 0


def test_dashboard_controls_require_same_origin_header(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_assets(assets)
    server = DashboardServer(
        tmp_path,
        tmp_path / "logs",
        tmp_path / "data" / "stats.sqlite3",
        port=0,
        assets=assets,
    )

    with server:
        missing_header = Request(
            f"{server.url}/api/control/stop",
            method="POST",
        )
        with pytest.raises(HTTPError) as rejected:
            urlopen(missing_header, timeout=2)  # noqa: S310

        remote = Request(
            f"{server.url}/api/control/stop",
            method="POST",
            headers={"Origin": "https://example.com", "X-Dino-Dashboard": "1"},
        )
        with pytest.raises(HTTPError) as forbidden:
            urlopen(remote, timeout=2)  # noqa: S310

        unknown = Request(
            f"{server.url}/api/control/unknown",
            method="POST",
            headers={"X-Dino-Dashboard": "1"},
        )
        with pytest.raises(HTTPError) as not_found:
            urlopen(unknown, timeout=2)  # noqa: S310

    assert rejected.value.code == 403
    assert forbidden.value.code == 403
    assert not_found.value.code == 404


def test_dashboard_builds_noninteractive_runner_commands(tmp_path: Path) -> None:
    scripts = tmp_path / "app" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run-windows.ps1").write_text("", encoding="utf-8")
    (scripts / "run-hatch-windows.ps1").write_text("", encoding="utf-8")
    controller = DashboardController(tmp_path, tmp_path / "app" / "logs")

    hunt = controller._runner_command("hunt")
    combined = controller._runner_command("hatch-hunt")

    assert hunt[:2] == ["powershell.exe", "-NoLogo"]
    assert "run-windows.ps1" in hunt[6]
    assert hunt[-2:] == ["-StatusPort", "8765"]
    assert "run-hatch-windows.ps1" in combined[6]
    assert combined[-4:] == ["-MaxActions", "0", "-MaxCycles", "0"]
