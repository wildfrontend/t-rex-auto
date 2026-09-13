"""The game's modal server-error dialog must be recognised and dismissed.

A blocking dialog the detector cannot see is the worst possible failure: the
bot stays "alive", never recovers, and only a human returning to the home
screen clears it.  One instance lost 1h51m to exactly this.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from dino_bot.application import _build_hunt_planner
from dino_bot.config import load_config
from dino_bot.detection import OpenCvDetector
from dino_bot.models import Frame

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "recovery" / "server-error-dialog.png"


def _frame(path: Path) -> Frame:
    image = cv2.imread(str(path))
    assert image is not None, f"unreadable fixture: {path}"
    return Frame(image=image, source="test", sequence=1)


def test_server_error_dialog_is_detected() -> None:
    detector = OpenCvDetector(REPO / "assets" / "manifest.json")

    found = detector.detect_types(_frame(FIXTURE), {"server_error_restart_button"})

    assert len(found) == 1
    # The RESTART button sits at (450, 1030) on the 900x1600 reference frame.
    assert abs(found[0].x - 450) <= 4
    assert abs(found[0].y - 1030) <= 4


def test_planner_taps_restart_before_anything_else() -> None:
    """The dialog is modal, so it must win over every other candidate."""

    config = load_config(REPO / "config.json")
    planner = _build_hunt_planner(config)
    detector = OpenCvDetector(REPO / "assets" / "manifest.json")
    frame = _frame(FIXTURE)

    target = planner.choose(frame, detector.detect(frame))

    assert target is not None, "planner ignored a blocking dialog"
    assert target.type == "server_error_restart_button"
    assert abs(target.x - 450) <= 4
    assert abs(target.y - 1030) <= 4


def test_server_error_is_an_interrupt_in_every_mode() -> None:
    """It appears mid-session, so it must not be treated as launch-only."""

    from dino_bot import full_hatch
    from dino_bot.planning import HuntPlanner
    from dino_bot.recovery import HuntProgressWatchdog

    planner = HuntPlanner()
    assert "server_error_restart_button" in planner.interrupt_button_types
    assert "server_error_restart_button" not in planner.launch_only_types
    # Scoped scans must keep looking for it, not only the widened ones.
    assert "server_error_restart_button" in (
        planner.interrupt_button_types - planner.launch_only_types
    )
    assert "server_error_restart_button" in full_hatch.STARTUP_SIMPLE_INTERRUPTS
    # The stall timer must not count time spent blocked by the dialog.
    assert "server_error_restart_button" in HuntProgressWatchdog._SUSPENDED_TYPES
