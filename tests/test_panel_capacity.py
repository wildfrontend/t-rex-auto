"""Reading the population count off the My Dinosaurs panel."""

from __future__ import annotations

import pathlib
import sys

import cv2
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dino_bot.cull import (  # noqa: E402
    CAPACITY_REGION,
    locate_panel_capacity,
    PANEL_CAPACITY_REGION,
    probe_dino_count,
)
from dino_bot.digits import DigitReader  # noqa: E402

GLYPHS = REPO / "assets" / "hatch" / "digits"
SAMPLE = REPO / "tests" / "data" / "my-dino-panel.png"
TEMPLATE = REPO / "assets" / "hatch" / "templates" / "hatch-my-dino-title.png"


def panel_frame():
    if not SAMPLE.exists():
        pytest.skip("panel sample not captured")
    return cv2.imread(str(SAMPLE), cv2.IMREAD_GRAYSCALE)


def test_reads_the_population_off_the_panel() -> None:
    result = probe_dino_count(
        panel_frame(),
        DigitReader(GLYPHS),
        expected_capacity=370,
        capacity_region=PANEL_CAPACITY_REGION,
    )
    assert result.reason == "ok"
    assert result.fraction == (237, 370)


def test_the_cave_hud_region_cannot_read_this_frame() -> None:
    # The point of the panel: the same figure is unreadable where the bot used
    # to look. On s9 the cave HUD lost the slash to the scenery behind it and
    # returned "32110" for 321/370, and read nothing at all over dark forest.
    result = probe_dino_count(
        panel_frame(),
        DigitReader(GLYPHS),
        expected_capacity=370,
        capacity_region=CAPACITY_REGION,
    )
    assert result.reason != "ok"


def test_the_locator_finds_the_panel_and_reads_it() -> None:
    """Locating by ink is what makes this work at all.

    The title renders slightly differently between openings (188x43 in one
    live capture, 183x40 in another), so a grayscale match tops out near 0.6
    and cannot be told from a miss.
    """

    if not TEMPLATE.exists():
        pytest.skip("panel template not built")
    template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
    region = locate_panel_capacity(panel_frame(), template)

    assert region is not None
    result = probe_dino_count(
        panel_frame(),
        DigitReader(GLYPHS),
        expected_capacity=370,
        capacity_region=region,
    )
    assert result.fraction == (237, 370)
