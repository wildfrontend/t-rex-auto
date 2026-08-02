from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_hatch_entrypoint_is_separate_and_fixed_to_hatch() -> None:
    command = (REPO / "scripts/start-hatch-bot.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert "run-hatch-windows.ps1" in command
    assert '[string]$Feature = "hatch"' in runner
    assert '"--feature", $Feature' in runner
    assert "$StatusPort = 8766" in runner
    assert "Another Bot is already running" in runner
    assert '-MaxCycles "%hatch_max_cycles%"' in command


def test_windows_deploy_exposes_exactly_three_user_entrypoints() -> None:
    deploy = (REPO / "scripts/deploy-windows.sh").read_text(encoding="utf-8")

    assert 'scripts/run-hatch-windows.ps1"' in deploy
    assert 'scripts/run-dashboard-windows.ps1"' in deploy
    assert 'scripts/watch-dashboard-windows.ps1"' in deploy
    assert 'scripts/start-bot.cmd"' in deploy
    assert 'scripts/start-dashboard.cmd"' in deploy
    assert 'scripts/start-hatch-hunt.cmd"' in deploy
    assert 'legacy_launchers="${runtime_app}/scripts/legacy-launchers"' in deploy


def test_dashboard_entrypoint_uses_loopback_web_service() -> None:
    command = (REPO / "scripts/start-dashboard.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/run-dashboard-windows.ps1").read_text(encoding="utf-8")
    watcher = (REPO / "scripts/watch-dashboard-windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "run-dashboard-windows.ps1" in command
    assert 'set "dashboard_port=8780"' in command
    assert '"--server-only"' in command.lower()
    assert "Install-LoginStartup" in runner
    assert "Dino Dashboard Server.cmd" in runner
    assert "WindowStyle Hidden" in runner
    assert '"dashboard"' in watcher
    assert "DinoMutantBotDashboard-$Port" in watcher


def test_filter_test_entrypoint_is_isolated_and_cycle_limited() -> None:
    command = (REPO / "scripts/start-hatch-filter-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-filter-test"' in command
    assert '-StatusPort "8767"' in command
    assert '-MaxActions "0"' in command
    assert '-MaxCycles "1"' in command


def test_sort_test_entrypoint_is_isolated_and_read_only() -> None:
    command = (REPO / "scripts/start-hatch-sort-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-sort-test"' in command
    assert '-StatusPort "8768"' in command
    assert '-MaxActions "0"' in command
    assert '-MaxCycles "0"' in command


def test_parent_test_entrypoint_is_isolated_and_one_tap_limited() -> None:
    command = (REPO / "scripts/start-hatch-parent-test.cmd").read_text(
        encoding="utf-8"
    )

    assert '-Feature "hatch-parent-test"' in command
    assert '-StatusPort "8769"' in command
    assert '-MaxActions "1"' in command
    assert '-MaxCycles "1"' in command
    assert "will not select or replace any dinosaur" in command


def test_attack_entrypoint_is_isolated_and_strictly_upgrades() -> None:
    command = (REPO / "scripts/start-hatch-attack-test.cmd").read_text(
        encoding="utf-8"
    )

    assert '-Feature "hatch-attack-test"' in command
    assert '-StatusPort "8770"' in command
    assert '-MaxActions "20"' in command
    assert '-MaxCycles "2"' in command
    assert "ONLY when its attack is strictly higher" in command


def test_hp_entrypoint_is_isolated_and_strictly_upgrades() -> None:
    command = (REPO / "scripts/start-hatch-hp-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-hp-test"' in command
    assert '-StatusPort "8771"' in command
    assert '-MaxActions "20"' in command
    assert '-MaxCycles "2"' in command
    assert "ONLY when its HP is strictly higher" in command


def test_full_hatch_entrypoint_is_separate_unbounded_and_not_hunt() -> None:
    command = (REPO / "scripts/start-hatch-full.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert '-Feature "hatch-full"' in command
    assert '-StatusPort "%hatch_status_port%"' in command
    assert 'set "hatch_status_port=8772"' in command
    assert 'set "hatch_max_actions=0"' in command
    assert '-MaxCycles "0"' in command
    assert '"hatch-full"' in runner
    assert "Full Auto Hatch" in command


def test_hatch_hunt_entrypoint_is_combined_unbounded_and_separate() -> None:
    command = (REPO / "scripts/start-hatch-hunt.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert '-Feature "hatch-hunt"' in command
    assert '-StatusPort "%combined_status_port%"' in command
    assert 'set "combined_status_port=8773"' in command
    assert 'set "combined_max_actions=0"' in command
    assert '-MaxCycles "0"' in command
    assert '"hatch-hunt"' in runner
    assert "Auto Hatch + Hunt" in command
