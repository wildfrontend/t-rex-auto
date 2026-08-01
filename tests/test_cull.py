from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_bot.config import ConfigError, load_config
from dino_bot.cull import CAPACITY_REGION, read_dino_count, should_cull
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


def test_should_cull_boundary() -> None:
    assert not should_cull(299, 300)
    assert not should_cull(300, 300)
    assert should_cull(301, 300)


def test_cull_threshold_config(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    assert load_config(config_path).hatch.cull_threshold == 300
    config_path.write_text('{"hatch": {"cull_threshold": 320}}', encoding="utf-8")
    assert load_config(config_path).hatch.cull_threshold == 320
    config_path.write_text('{"hatch": {"cull_threshold": -1}}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)
