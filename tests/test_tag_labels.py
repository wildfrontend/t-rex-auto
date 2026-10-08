from __future__ import annotations

import cv2
import numpy as np
import pytest

from dino_bot.config import ConfigError, _tag_names
from dino_bot.models import Frame
from dino_bot.nest_filter import (
    TAG_ATTACK,
    TAG_HDR_ATTACK,
    TAG_HDR_HP,
    TAG_HP,
    TAG_TOP,
)
from dino_bot.tag_labels import (
    OCR_REGION,
    TagLabelOcrDetector,
    custom_tag_names,
    match_role,
    normalize_tag_name,
    ocr_available,
)


class FakeReader:
    """Return canned OCR lines in crop coordinates and count the calls."""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int, int, int]]]) -> None:
        self.lines = lines
        self.calls = 0

    def read(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        self.calls += 1
        return list(self.lines)


def screen() -> Frame:
    return Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))


def crop_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Convert a reference-space box into OCR crop coordinates."""

    ox, oy = OCR_REGION[0], OCR_REGION[1]
    return (x0 - ox, y0 - oy, x1 - ox, y1 - oy)


def test_names_fold_width_case_and_spacing() -> None:
    assert normalize_tag_name("ＨＰ 特化") == normalize_tag_name("hp特化")
    assert normalize_tag_name(" Attack ") == "attack"


def test_only_renamed_roles_need_ocr() -> None:
    # A role still on the game's default keeps its proven screenshot match.
    assert custom_tag_names({"attack": "攻擊特化", "hp": "hp"}) == {"hp": "hp"}
    assert custom_tag_names({"attack": "  ", "top": "頂尖"}) == {}
    assert custom_tag_names(None) == {}


def test_short_names_must_match_exactly() -> None:
    names = {"attack": "attack", "hp": "hp"}
    assert match_role("Attack", names) == "attack"
    # One misread character in a long name is still that name...
    assert match_role("attaek", names) == "attack"
    # ...but in a two-letter name it is a different word.
    assert match_role("hb", names) is None
    assert match_role("HP", names) == "hp"
    assert match_role("所有", names) is None


def test_renamed_tags_report_the_template_types() -> None:
    reader = FakeReader(
        [
            ("attack", 0.98, crop_box(170, 152, 250, 182)),  # collapsed header
            ("所有", 0.99, crop_box(170, 197, 215, 225)),
            ("hp", 0.97, crop_box(170, 326, 200, 354)),
            ("attack", 0.96, crop_box(170, 369, 250, 397)),
        ]
    )
    detector = TagLabelOcrDetector({"attack": "attack", "hp": "hp"}, reader=reader)

    found = {(item.type, item.x, item.y) for item in detector.detect(screen())}

    assert found == {
        (TAG_HDR_ATTACK, 210, 167),
        (TAG_HP, 185, 340),
        (TAG_ATTACK, 210, 383),
    }


def test_ocr_runs_only_when_a_renamed_tag_is_wanted() -> None:
    reader = FakeReader([("hp", 0.97, crop_box(170, 152, 200, 182))])
    detector = TagLabelOcrDetector({"hp": "hp"}, reader=reader)

    # Hunting and every non-tag hatch step ask for other types: no OCR at all.
    assert detector.detect_types(screen(), {"dinosaur", "hatch_nest_title"}) == []
    # A default-named tag is the template detector's job.
    assert detector.detect_types(screen(), {TAG_TOP}) == []
    assert reader.calls == 0

    found = detector.detect_types(screen(), {TAG_HDR_HP, TAG_HP})
    assert [item.type for item in found] == [TAG_HDR_HP]
    assert reader.calls == 1


def test_low_confidence_reads_are_ignored() -> None:
    reader = FakeReader([("hp", 0.2, crop_box(170, 326, 200, 354))])
    detector = TagLabelOcrDetector({"hp": "hp"}, reader=reader)

    assert detector.detect(screen()) == []


def test_config_rejects_unknown_roles_and_non_strings() -> None:
    assert _tag_names({"attack": " attack ", "hp": ""}) == {"attack": "attack"}
    with pytest.raises(ConfigError):
        _tag_names({"speed": "spd"})
    with pytest.raises(ConfigError):
        _tag_names({"hp": 3})
    with pytest.raises(ConfigError):
        _tag_names(["hp"])


@pytest.mark.skipif(not ocr_available(), reason="rapidocr is not installed")
def test_real_ocr_reads_a_renamed_dropdown() -> None:
    """End to end: the bundled OCR model reads English tag names off the menu."""

    image = np.full((1600, 900, 3), 245, dtype=np.uint8)
    # Collapsed header, then the open dropdown: 所有 stays built in.
    rows = [("ATTACK", 166), ("All", 211), ("HP", 340), ("ATTACK", 383)]
    for text, y in rows:
        cv2.putText(image, text, (170, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    detector = TagLabelOcrDetector({"attack": "attack", "hp": "hp"})

    found = {item.type: (item.x, item.y) for item in detector.detect(Frame(image))}

    assert set(found) == {TAG_HDR_ATTACK, TAG_HP, TAG_ATTACK}
    assert abs(found[TAG_HP][1] - 340) <= 15
    assert abs(found[TAG_ATTACK][1] - 383) <= 15


def test_renamed_tags_without_ocr_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silently falling back to the default-name screenshots would leave the
    # hatch side unable to find a single renamed tag, with nothing in the log
    # saying why.
    import logging

    from dino_bot import application

    monkeypatch.setattr(application, "ocr_available", lambda: False)
    logger = logging.getLogger("test")

    assert application._tag_label_detectors({"attack": "攻擊特化"}, logger) == []
    with pytest.raises(RuntimeError, match="OCR"):
        application._tag_label_detectors({"attack": "attack"}, logger)
