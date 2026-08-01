from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_hatch_entrypoint_is_separate_and_fixed_to_hatch() -> None:
    command = (REPO / "scripts/start-hatch-bot.cmd").read_text(encoding="utf-8")
    runner = (REPO / "scripts/run-hatch-windows.ps1").read_text(encoding="utf-8")

    assert "run-hatch-windows.ps1" in command
    assert '"--feature", "hatch"' in runner
    assert "$StatusPort = 8766" in runner
    assert "Another Bot is already running" in runner
    assert '-MaxCycles "%hatch_max_cycles%"' in command


def test_windows_deploy_includes_hatch_entrypoint() -> None:
    deploy = (REPO / "scripts/deploy-windows.sh").read_text(encoding="utf-8")

    assert 'scripts/run-hatch-windows.ps1"' in deploy
    assert 'scripts/start-hatch-bot.cmd"' in deploy
