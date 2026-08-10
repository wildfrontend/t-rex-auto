from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_bot.digits import (
    GLYPH_SIZE,
    DigitReader,
    DigitReadError,
    _count_enclosed_holes,
    _prefer_matching_topology,
    _prefer_narrow_one,
    _prefer_tall_hole_zero,
    segment_glyphs,
)

REPO = Path(__file__).resolve().parent.parent
GLYPH_DIR = REPO / "assets" / "hatch" / "digits"
FIXTURES = REPO / "tests" / "fixtures" / "hatch"


def load_fixture(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"missing fixture {name}"
    return image


# -- segmentation (synthetic) -------------------------------------------------


def test_segment_glyphs_orders_left_to_right() -> None:
    image = np.full((40, 90), 255, dtype=np.uint8)
    for x in (60, 10, 35):
        cv2.rectangle(image, (x, 10), (x + 12, 30), 0, thickness=-1)
    boxes = [box for box, _ in segment_glyphs(image)]
    assert [x for x, _, _, _ in boxes] == [10, 35, 60]


def test_segment_glyphs_merges_stacked_dots() -> None:
    # A colon is two dots sharing a column; they must come back as one glyph.
    image = np.full((40, 60), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 8), (16, 14), 0, thickness=-1)
    cv2.rectangle(image, (10, 24), (16, 30), 0, thickness=-1)
    cv2.rectangle(image, (35, 8), (45, 30), 0, thickness=-1)
    boxes = [box for box, _ in segment_glyphs(image)]
    assert len(boxes) == 2
    assert boxes[0] == (10, 8, 7, 23)


def test_segment_glyphs_ignores_specks() -> None:
    image = np.full((40, 60), 255, dtype=np.uint8)
    image[20, 30] = 0
    assert segment_glyphs(image) == []


# -- reader construction (synthetic) ------------------------------------------


def test_reader_rejects_empty_dir(tmp_path) -> None:
    with pytest.raises(DigitReadError):
        DigitReader(tmp_path)


def test_reader_rejects_multichar_name(tmp_path) -> None:
    cv2.imwrite(str(tmp_path / "42.png"), np.full(GLYPH_SIZE[::-1], 255, np.uint8))
    with pytest.raises(DigitReadError):
        DigitReader(tmp_path)


def test_reader_marks_low_scores_unknown(tmp_path) -> None:
    # Template: solid block. Candidate: thin frame, mostly background — the
    # equality score stays far below MIN_MATCH_SCORE and must yield "?".
    cv2.imwrite(str(tmp_path / "0.png"), np.full(GLYPH_SIZE[::-1], 255, np.uint8))
    reader = DigitReader(tmp_path)
    image = np.full((40, 40), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 4), (31, 35), 0, thickness=2)
    assert reader.read(image) == "?"
    assert reader.read_int(image) is None


def test_close_narrow_1_7_match_prefers_one() -> None:
    assert _prefer_narrow_one(
        "7",
        0.730,
        {"1": 0.717, "7": 0.730},
        (29, 4, 5, 14),
    ) == ("1", 0.717)


def test_wide_or_unconvincing_7_match_stays_seven() -> None:
    assert _prefer_narrow_one(
        "7",
        0.800,
        {"1": 0.780, "7": 0.800},
        (10, 4, 10, 14),
    ) == ("7", 0.800)
    assert _prefer_narrow_one(
        "7",
        0.700,
        {"1": 0.590, "7": 0.700},
        (0, 6, 4, 10),
    ) == ("7", 0.700)


def test_counts_only_fully_enclosed_glyph_holes() -> None:
    closed = np.zeros(GLYPH_SIZE[::-1], dtype=np.uint8)
    cv2.rectangle(closed, (4, 4), (19, 27), 255, thickness=3)
    opened = closed.copy()
    opened[:9, 10:14] = 0

    assert _count_enclosed_holes(closed) == 1
    assert _count_enclosed_holes(opened) == 0


def test_close_5_6_8_match_prefers_matching_topology() -> None:
    scores = {"5": 0.763, "6": 0.738, "8": 0.742}
    holes = {"5": 0, "6": 1, "8": 2}

    assert _prefer_matching_topology("5", 0.763, scores, holes, 1) == (
        "6",
        0.738,
    )
    assert _prefer_matching_topology("5", 0.900, scores, holes, 1) == (
        "5",
        0.900,
    )


def test_close_six_match_with_tall_hole_prefers_zero() -> None:
    raster = np.zeros(GLYPH_SIZE[::-1], dtype=np.uint8)
    cv2.rectangle(raster, (4, 3), (20, 29), 255, thickness=3)

    assert _prefer_tall_hole_zero(
        "6",
        0.770,
        {"0": 0.747, "6": 0.770},
        raster,
    ) == ("0", 0.747)


def test_short_six_bowl_is_not_changed_to_zero() -> None:
    raster = np.zeros(GLYPH_SIZE[::-1], dtype=np.uint8)
    cv2.rectangle(raster, (8, 16), (19, 28), 255, thickness=3)

    assert _prefer_tall_hole_zero(
        "6",
        0.770,
        {"0": 0.747, "6": 0.770},
        raster,
    ) == ("6", 0.770)


def test_live_hp_zero_with_antialiased_tail_is_not_read_as_six(
    reader: DigitReader,
) -> None:
    # 2026-08-10 live right-parent HP ended in a real zero, but its short
    # bottom tail scored 0.770 as ``6`` versus 0.747 as ``0``.  Preserve the
    # exact thresholded 8x10 glyph here without retaining account screenshots.
    rows = (
        "..#####.",
        ".###.###",
        "##....##",
        "##.....#",
        "##.....#",
        "##.....#",
        "##.....#",
        ".##...##",
        ".######.",
        "...###..",
    )
    image = np.full((14, 12), 255, dtype=np.uint8)
    for y, row in enumerate(rows, 2):
        for x, pixel in enumerate(row, 2):
            if pixel == "#":
                image[y, x] = 0

    assert reader.read(image) == "0"


# -- regression against shipped glyphs (offline T6) ---------------------------


@pytest.fixture(scope="module")
def reader() -> DigitReader:
    return DigitReader(GLYPH_DIR)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("hud_282_350.png", "282/350"),
        ("c1l_atk.png", "276"),
        ("c1l_spd.png", "1"),
        ("c2l_hp.png", "2230"),
        ("c3l_atk.png", "274"),
        ("c3l_spd.png", "150"),
        ("lv429.png", "429"),
    ],
)
def test_reads_known_screenshot_regions(reader: DigitReader, fixture: str, expected: str) -> None:
    assert reader.read(load_fixture(fixture)) == expected


def test_read_fraction_parses_capacity(reader: DigitReader) -> None:
    assert reader.read_fraction(load_fixture("hud_282_350.png")) == (282, 350)


def test_read_fraction_rejects_plain_number(reader: DigitReader) -> None:
    assert reader.read_fraction(load_fixture("c1l_atk.png")) is None


def test_read_int_rejects_fraction(reader: DigitReader) -> None:
    assert reader.read_int(load_fixture("hud_282_350.png")) is None


def test_read_int_parses_stat(reader: DigitReader) -> None:
    assert reader.read_int(load_fixture("c2l_hp.png")) == 2230
