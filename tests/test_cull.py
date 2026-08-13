from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_bot.config import ConfigError, load_config
from dino_bot.cull import (
    CAPACITY_REGION,
    probe_dino_count,
    read_dino_count,
    should_cull,
)
from dino_bot.digits import DigitReader

REPO = Path(__file__).resolve().parent.parent
GLYPH_DIR = REPO / "assets" / "hatch" / "digits"
FIXTURES = REPO / "tests" / "fixtures" / "hatch"


@pytest.fixture(scope="module")
def reader() -> DigitReader:
    return DigitReader(GLYPH_DIR)


def frame_with_hud(fixture: str = "hud_282_350.png") -> np.ndarray:
    """Blank 900x1600 frame with a real HUD crop pasted at its true position."""

    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    crop = cv2.imread(str(FIXTURES / fixture))
    assert crop is not None
    x0, y0 = int(CAPACITY_REGION[0]), int(CAPACITY_REGION[1])
    frame[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return frame


def test_reads_capacity_at_reference_size(reader: DigitReader) -> None:
    assert read_dino_count(frame_with_hud(), reader) == 282


def test_capacity_crop_ignores_particles_above_the_number_row(reader: DigitReader) -> None:
    frame = frame_with_hud()
    # A live cave-view frame had confetti in the old crop's top-right corner.
    # The trusted region starts below it and must still parse the counter.
    frame[236:239, 100:110] = 0
    assert read_dino_count(frame, reader) == 282


def test_reads_capacity_at_two_thirds_scale(reader: DigitReader) -> None:
    small = cv2.resize(frame_with_hud(), (600, 1067), interpolation=cv2.INTER_AREA)
    assert read_dino_count(small, reader) == 282


def test_tiny_frame_fails_safe(reader: DigitReader) -> None:
    # At 450 wide the glyphs are beyond recovery; the reject-don't-guess
    # guard must return None rather than a wrong count.
    small = cv2.resize(frame_with_hud(), (450, 800), interpolation=cv2.INTER_AREA)
    assert read_dino_count(small, reader) is None


def test_rejects_wrong_denominator(reader: DigitReader) -> None:
    # A stat row ("276") in the HUD position must not pass the /350 guard.
    assert read_dino_count(frame_with_hud("c1l_atk.png"), reader) is None


def test_rejects_blank_frame(reader: DigitReader) -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    assert read_dino_count(frame, reader) is None


class StubReader:
    """Stands in for glyph matching so each reject branch can be reached."""

    def __init__(self, text: str) -> None:
        self.text = text

    def read(self, image: np.ndarray) -> str:
        return self.text


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        # All four fail the planner identically. An absent HUD is a navigation
        # problem, a wrong denominator is a crop problem, and a '?' is a glyph
        # problem - opposite fixes, so the bundle has to tell them apart.
        ("282/350", "ok"),
        ("28?/350", "unparsed"),
        ("276", "unparsed"),
        ("282/100", "unexpected_capacity"),
    ],
)
def test_probe_names_the_reject_branch(text: str, reason: str) -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(frame, StubReader(text))
    assert read.reason == reason
    assert read.ok is (reason == "ok")
    assert read.text == text


def test_probe_reads_a_narrow_hud_clipped_by_the_next_icon() -> None:
    # S13 的 174/200 比主力的 280/350 窄，裁切框右緣會切到隔壁圖示，多讀出一個
    # 斜線。數字本身完整，不該因此整串作廢並讓孵蛋停機。
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(frame, StubReader("174/200/"), expected_capacity=200)
    assert read.ok
    assert read.fraction == (174, 200)


@pytest.mark.parametrize(
    ("text", "reason", "count"),
    [
        # 洞穴視角下 HUD 壓在石階上,分母最後一位會和場景描邊連成一塊而讀不出來。
        ("188/20?", "ok", 188),
        ("188/?00", "ok", 188),
        # 分子不完整、可讀位數對不上設定、或壞掉不只一位,都必須照舊拒絕。
        ("18?/200", "unparsed", None),
        ("188/3?0", "unparsed", None),
        ("188/2??", "unparsed", None),
        # 讀成別的數字(而不是讀不出來)仍走原本的分母不符路徑。
        ("188/208", "unexpected_capacity", None),
    ],
)
def test_probe_waives_one_unreadable_denominator_digit(
    text: str, reason: str, count: int | None
) -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(frame, StubReader(text), expected_capacity=200)
    assert read.reason == reason
    assert read.count == count


def test_probe_still_rejects_a_stray_digit_in_the_readout() -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(frame, StubReader("174/200/1"), expected_capacity=200)
    assert read.reason == "unparsed"


def test_probe_accepts_configured_capacity_limit() -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(
        frame,
        StubReader("369/380"),
        expected_capacity=380,
    )
    assert read.ok
    assert read.count == 369
    assert read.fraction == (369, 380)


def test_probe_rejects_a_count_above_its_own_capacity() -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    read = probe_dino_count(frame, StubReader("999/350"))
    assert read.reason == "count_out_of_range"
    assert read.fraction == (999, 350)


def test_probe_reports_an_absent_hud_on_a_real_blank_frame(reader: DigitReader) -> None:
    blank = probe_dino_count(np.full((1600, 900, 3), 255, dtype=np.uint8), reader)
    assert blank.reason == "unparsed"
    assert blank.count is None and blank.fraction is None


def test_probe_keeps_the_glyphs_and_crop_it_worked_from(reader: DigitReader) -> None:
    read = probe_dino_count(frame_with_hud(), reader)
    assert read.ok and read.count == 282
    assert read.text == "282/350"
    assert read.region == tuple(int(value) for value in CAPACITY_REGION)


def test_probe_rejects_a_frame_too_small_to_hold_the_hud(reader: DigitReader) -> None:
    # A 200px-tall frame cannot contain a crop that ends at y=258; without the
    # bounds check this silently reads an empty array as an ordinary miss.
    read = probe_dino_count(np.full((200, 900, 3), 255, dtype=np.uint8), reader)
    assert read.reason == "region_outside_frame"
    assert read.count is None


def test_should_cull_boundary() -> None:
    assert not should_cull(299, 300)
    assert should_cull(300, 300)
    assert should_cull(301, 300)


def test_cull_threshold_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    assert load_config(config_path).hatch.cull_threshold == 330
    assert load_config(config_path).hatch.capacity_limit == 350
    config_path.write_text('{"hatch": {"cull_threshold": 325}}', encoding="utf-8")
    assert load_config(config_path).hatch.cull_threshold == 325
    config_path.write_text('{"hatch": {"capacity_limit": 380}}', encoding="utf-8")
    assert load_config(config_path).hatch.capacity_limit == 380
    config_path.write_text('{"hatch": {"capacity_limit": 0}}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)
    config_path.write_text('{"hatch": {"cull_threshold": -1}}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)
