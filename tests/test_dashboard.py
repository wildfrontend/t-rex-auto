from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dino_bot.dashboard import DashboardController, DashboardServer, _workflow_status


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
    assert health == {
        "ok": True,
        "service": "dino-dashboard",
        "api_version": 1,
        "process_id": os.getpid(),
    }
    assert overview["active"]["running"] is False
    assert overview["metrics"]["counters"]["hunt"]["total"] == 0
    assert overview["hatch_boost_inventory"]["remaining"] == 100
    assert overview["hatch_boost_inventory"]["maximum"] == 100
    assert overview["hatch_boost_inventory"]["cost"] == 1
    assert overview["hatch_boost_inventory"]["enabled"] is False


def test_dashboard_updates_local_boost_inventory(tmp_path: Path) -> None:
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
        body = json.dumps({"remaining": 73}).encode()
        update = Request(
            f"{server.url}/api/control/set-boost-stock",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Dino-Dashboard": "1",
            },
        )
        with urlopen(update, timeout=2) as response:  # noqa: S310
            result = json.load(response)
        with urlopen(f"{server.url}/api/overview", timeout=2) as response:  # noqa: S310
            overview = json.load(response)

    assert result["inventory"]["remaining"] == 73
    assert overview["hatch_boost_inventory"]["remaining"] == 73


def test_dashboard_toggles_boost_for_next_hatch_cycle(tmp_path: Path) -> None:
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
        body = json.dumps({"enabled": True}).encode()
        update = Request(
            f"{server.url}/api/control/set-boost-enabled",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Dino-Dashboard": "1",
            },
        )
        with urlopen(update, timeout=2) as response:  # noqa: S310
            result = json.load(response)

    assert result["inventory"]["enabled"] is True


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
    cave = controller._runner_command("hatch-stage", stage="cave")
    cave_after_switch = controller._runner_command(
        "hatch-stage",
        stage="cave",
        wait_for_existing_seconds=20,
    )

    assert hunt[:2] == ["powershell.exe", "-NoLogo"]
    assert "run-windows.ps1" in hunt[6]
    assert hunt[-2:] == ["-StatusPort", "8765"]
    assert "run-hatch-windows.ps1" in combined[6]
    assert combined[-4:] == ["-MaxActions", "0", "-MaxCycles", "0"]
    assert "run-hatch-windows.ps1" in cave[6]
    assert "hatch-stage-cave" in cave
    assert "8765" in cave
    assert cave_after_switch[-2:] == ["-WaitForExistingSeconds", "20"]


def test_windows_launch_keeps_live_output_in_the_bot_console() -> None:
    source = inspect.getsource(DashboardController._launch)
    windows_branch, non_windows_branch = source.split("        else:\n", 1)

    assert 'if os.name == "nt"' in windows_branch
    assert "stdout=" not in windows_branch
    assert "stderr=" not in windows_branch
    assert "stdout=stream" in non_windows_branch
    assert "stderr=subprocess.STDOUT" in non_windows_branch


def test_dashboard_builds_commands_for_the_selected_instance(tmp_path: Path) -> None:
    app = tmp_path / "app"
    scripts = app / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run-windows.ps1").write_text("", encoding="utf-8")
    (scripts / "run-hatch-windows.ps1").write_text("", encoding="utf-8")
    primary = app / "config.json"
    primary.write_text(json.dumps({"adb": {"serial": "127.0.0.1:16384"}}), encoding="utf-8")
    second = tmp_path / "instances" / "second" / "config.json"
    second.parent.mkdir(parents=True)
    second.write_text(json.dumps({"adb": {"serial": "127.0.0.1:16385"}}), encoding="utf-8")
    (tmp_path / "instances.json").write_text(
        json.dumps(
            {
                "instances": [
                    {"id": "main", "name": "主力", "config": "app/config.json", "status_port": 8765},
                    {
                        "id": "second",
                        "name": "第二台",
                        "config": "instances/second/config.json",
                        "status_port": 8775,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    controller = DashboardController(tmp_path, app / "logs", config_path=primary)
    command = controller._runner_command("hunt", instance_id="second")

    assert "-ConfigPath" in command
    assert str(second) in command
    assert command[-2:] == ["-StatusPort", "8775"]
    assert [item.instance_id for item in controller.instances] == ["main", "second"]


def test_dashboard_can_create_an_isolated_instance(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "config.json").write_text(
        json.dumps({"adb": {"serial": "127.0.0.1:16384"}}),
        encoding="utf-8",
    )
    (app / "assets").mkdir()
    (app / "assets" / "manifest.json").write_text("{}", encoding="utf-8")
    controller = DashboardController(tmp_path, app / "logs", config_path=app / "config.json")

    result = controller.add_instance(
        name="第二台",
        serial="127.0.0.1:16385",
        status_port=8775,
    )

    assert result["instance_id"] == "instance-2"
    config = json.loads(
        (tmp_path / "instances" / "instance-2" / "config.json").read_text(encoding="utf-8")
    )
    assert config["adb"]["serial"] == "127.0.0.1:16385"
    assert (tmp_path / "instances" / "instance-2" / "assets" / "manifest.json").is_file()
    registry = json.loads((tmp_path / "instances.json").read_text(encoding="utf-8"))
    assert registry["instances"][-1]["status_port"] == 8775


def test_dashboard_can_update_instance_settings_without_manual_file_edits(
    tmp_path: Path,
) -> None:
    app = tmp_path / "app"
    app.mkdir()
    config_path = app / "config.json"
    config_path.write_text(
        json.dumps({"adb": {"serial": "127.0.0.1:16384"}}),
        encoding="utf-8",
    )
    controller = DashboardController(tmp_path, app / "logs", config_path=config_path)

    result = controller.update_instance(
        instance_id="main",
        name="主力新名稱",
        serial="127.0.0.1:16386",
        status_port=8786,
    )

    assert result["restarted"] is False
    assert result["status_port"] == 8786
    assert json.loads(config_path.read_text(encoding="utf-8"))["adb"]["serial"] == (
        "127.0.0.1:16386"
    )
    registry = json.loads((tmp_path / "instances.json").read_text(encoding="utf-8"))
    assert registry["instances"][0] == {
        "id": "main",
        "name": "主力新名稱",
        "config": "app/config.json",
        "status_port": 8786,
    }


def test_dashboard_rejects_unknown_standalone_stage(tmp_path: Path) -> None:
    controller = DashboardController(tmp_path, tmp_path / "logs")

    with pytest.raises(RuntimeError, match="Unsupported Bot mode"):
        controller._runner_command("hatch-stage", stage="unknown")


def test_dashboard_workflow_reads_standalone_stage_feature(tmp_path: Path) -> None:
    log = tmp_path / "20260802.log"
    log.write_text(
        "20:10:00 | INFO | Feature | hatch-stage | stage=attack | "
        "bounded preflight + run + return\n",
        encoding="utf-8",
    )

    workflow = _workflow_status(tmp_path, "hatch-stage")
    assert workflow["stage"] == "nest_attack"
    assert workflow["label"] == "攻擊親代"


def test_dashboard_workflow_reads_the_complete_log_message(tmp_path: Path) -> None:
    log = tmp_path / "20260802.log"
    log.write_text(
        "20:10:22 | INFO | Planning | hatch_label at (95, 671), score=0.994\n"
        "20:10:34 | INFO | Planning | hatch_egg_pile at (506, 952), score=0.806\n"
        "20:10:50 | INFO | Hatch cave | capacity=350/350 | threshold=350 | cull=True\n",
        encoding="utf-8",
    )

    workflow = _workflow_status(tmp_path, "hatch-hunt")

    assert workflow["stage"] == "cave"
    assert workflow["label"] == "洞穴容量與淘汰"


def test_dashboard_workflow_advances_from_mass_through_collect_and_cave(
    tmp_path: Path,
) -> None:
    log = tmp_path / "20260802.log"
    log.write_text(
        "20:10:00 | INFO | Hatch auto-place | tag=量產 | sort=等級 | completed\n"
        "20:10:10 | INFO | Planning | hatch_collect_eggs_button at (640,1315) "
        "confidence=1.000\n",
        encoding="utf-8",
    )

    assert _workflow_status(tmp_path, "hatch-hunt")["stage"] == "collect"

    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            "20:10:20 | INFO | Planning | hatch_cave_swipe at (450,1050) "
            "confidence=1.000\n"
        )

    workflow = _workflow_status(tmp_path, "hatch-hunt")
    assert workflow["stage"] == "cave"
    assert workflow["label"] == "洞穴容量與淘汰"


def test_dashboard_workflow_returns_to_hatch_after_management_cycle(
    tmp_path: Path,
) -> None:
    log = tmp_path / "20260802.log"
    log.write_text(
        "20:10:20 | INFO | Planning | hatch_cave_recenter at (842,1497) "
        "confidence=1.000\n"
        "20:10:30 | INFO | Hatch full | completed management cycle 1 | "
        "restarting Phase A\n",
        encoding="utf-8",
    )

    workflow = _workflow_status(tmp_path, "hatch-hunt")
    assert workflow["stage"] == "hatch"
    assert workflow["label"] == "檢查孵蛋"
