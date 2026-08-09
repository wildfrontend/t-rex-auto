from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_hatch_entrypoint_is_separate_and_fixed_to_hatch() -> None:
    command = (REPO / "scripts/windows/start-hatch-bot.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/windows/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert "run-hatch-windows.ps1" in command
    assert '[string]$Feature = "hatch"' in runner
    assert '"--feature", $Feature' in runner
    assert "$StatusPort = 8766" in runner
    assert "$ConfigPath = \"\"" in runner
    assert "$ConfigToken = [Regex]::Escape($ConfigPath)" in runner
    assert "Another Bot instance using this config is already running" in runner
    assert "Stop-BundledAdbWhenIdle" in runner
    assert '-MaxCycles "%hatch_max_cycles%"' in command


def test_windows_deploy_exposes_only_dashboard_entrypoint() -> None:
    deploy = (REPO / "scripts/windows/deploy-windows.sh").read_text(encoding="utf-8")
    package = (REPO / "scripts/windows/package-windows-lite.sh").read_text(
        encoding="utf-8"
    )

    assert 'scripts/windows/run-hatch-windows.ps1"' in deploy
    assert 'scripts/windows/run-dashboard-windows.ps1"' in deploy
    assert 'scripts/windows/watch-dashboard-windows.ps1"' not in deploy
    assert 'cp -a "${project_root}/scripts/windows/start-hunt.cmd"' not in deploy
    assert 'scripts/windows/start-dashboard.cmd"' in deploy
    assert 'scripts/windows/start-hatch-hunt.cmd"' not in deploy
    assert 'legacy_launchers="${runtime_app}/scripts/legacy-launchers"' in deploy
    assert "start-bot.cmd" in deploy
    assert 'scripts/windows/start-dashboard.cmd"' in package
    assert 'scripts/windows/start-hunt.cmd"' not in package
    assert 'scripts/windows/start-hatch-hunt.cmd"' not in package
    assert 'package_name="DinoMutantBot-v${version}-Windows-Lite"' in package


def test_hatch_runner_allows_dashboard_standalone_stages() -> None:
    runner = (REPO / "scripts/windows/run-hatch-windows.ps1").read_text(encoding="utf-8")

    for stage in ("hatch", "attack", "hp", "top", "mass", "collect", "cave"):
        assert f'"hatch-stage-{stage}"' in runner


def test_windows_runners_wait_for_a_previous_bot_during_mode_switch() -> None:
    for script in ("run-windows.ps1", "run-hatch-windows.ps1"):
        runner = (REPO / "scripts/windows" / script).read_text(encoding="utf-8")

        assert "$WaitForExistingSeconds = 0" in runner
        assert "function Get-ExistingBots" in runner
        assert "Waiting up to $WaitForExistingSeconds seconds" in runner
        assert "$WaitTimer.Elapsed.TotalSeconds" in runner


def test_hunt_entrypoint_name_matches_its_feature() -> None:
    command = (REPO / "scripts/windows/start-hunt.cmd").read_text(encoding="utf-8")

    assert "launcher-windows.ps1" in command
    assert "猛龍計畫 - Hunt" in command
    assert not (REPO / "scripts/windows/start-bot.cmd").exists()


def test_dashboard_entrypoint_uses_loopback_web_service() -> None:
    command = (REPO / "scripts/windows/start-dashboard.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/windows/run-dashboard-windows.ps1").read_text(encoding="utf-8")
    uninstaller = (REPO / "scripts/windows/uninstall-windows.ps1").read_text(
        encoding="utf-8-sig"
    )
    cleanup = (REPO / "scripts/windows/cleanup-runtime-windows.ps1").read_text(
        encoding="utf-8-sig"
    )

    assert "run-dashboard-windows.ps1" in command
    assert 'set "dashboard_port=8780"' in command
    assert '"--server-only"' not in command.lower()
    assert "Remove-LegacyLoginStartup" in runner
    assert "Dino Dashboard Server.cmd" in runner
    assert "WindowStyle Hidden" not in runner
    assert "--open-browser" in runner
    assert "& $PythonExecutable @DashboardArguments" in runner
    assert not (REPO / "scripts/windows/watch-dashboard-windows.ps1").exists()
    assert '"%~1"=="uninstall"' in command
    assert "dino-mutant-bot-status" in uninstaller
    assert "shutdown-dashboard" in uninstaller
    assert "instances.json" in uninstaller
    assert "Stopped bundled ADB process" in uninstaller
    assert 'ArgumentList @("kill-server")' in uninstaller
    assert "DinoPendingDelete" in cleanup


def test_filter_test_entrypoint_is_isolated_and_cycle_limited() -> None:
    command = (REPO / "scripts/windows/start-hatch-filter-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-filter-test"' in command
    assert '-StatusPort "8767"' in command
    assert '-MaxActions "0"' in command
    assert '-MaxCycles "1"' in command


def test_sort_test_entrypoint_is_isolated_and_read_only() -> None:
    command = (REPO / "scripts/windows/start-hatch-sort-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-sort-test"' in command
    assert '-StatusPort "8768"' in command
    assert '-MaxActions "0"' in command
    assert '-MaxCycles "0"' in command


def test_parent_test_entrypoint_is_isolated_and_one_tap_limited() -> None:
    command = (REPO / "scripts/windows/start-hatch-parent-test.cmd").read_text(
        encoding="utf-8"
    )

    assert '-Feature "hatch-parent-test"' in command
    assert '-StatusPort "8769"' in command
    assert '-MaxActions "1"' in command
    assert '-MaxCycles "1"' in command
    assert "will not select or replace any dinosaur" in command


def test_attack_entrypoint_is_isolated_and_strictly_upgrades() -> None:
    command = (REPO / "scripts/windows/start-hatch-attack-test.cmd").read_text(
        encoding="utf-8"
    )

    assert '-Feature "hatch-attack-test"' in command
    assert '-StatusPort "8770"' in command
    assert '-MaxActions "20"' in command
    assert '-MaxCycles "2"' in command
    assert "ONLY when its attack is strictly higher" in command


def test_hp_entrypoint_is_isolated_and_strictly_upgrades() -> None:
    command = (REPO / "scripts/windows/start-hatch-hp-test.cmd").read_text(encoding="utf-8")

    assert '-Feature "hatch-hp-test"' in command
    assert '-StatusPort "8771"' in command
    assert '-MaxActions "20"' in command
    assert '-MaxCycles "2"' in command
    assert "ONLY when its HP is strictly higher" in command


def test_full_hatch_entrypoint_is_separate_unbounded_and_not_hunt() -> None:
    command = (REPO / "scripts/windows/start-hatch-full.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/windows/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert '-Feature "hatch-full"' in command
    assert '-StatusPort "%hatch_status_port%"' in command
    assert 'set "hatch_status_port=8772"' in command
    assert 'set "hatch_max_actions=0"' in command
    assert '-MaxCycles "0"' in command
    assert '"hatch-full"' in runner
    assert "Full Auto Hatch" in command


def test_hatch_hunt_entrypoint_is_combined_unbounded_and_separate() -> None:
    command = (REPO / "scripts/windows/start-hatch-hunt.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/windows/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert '-Feature "hatch-hunt"' in command
    assert '-StatusPort "%combined_status_port%"' in command
    assert 'set "combined_status_port=8773"' in command
    assert 'set "combined_max_actions=0"' in command
    assert 'set "combined_speed=fast"' in command
    assert '-MaxCycles "0"' in command
    assert '"hatch-hunt"' in runner
    assert "Auto Hatch + Hunt" in command
