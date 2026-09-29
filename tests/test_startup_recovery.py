from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_bot.detection import (
    DetectorAssetError,
    OpenCvDetector,
    StartupAutoBattleDialogDetector,
    StartupGrowthResultDetector,
    StartupOfferDismissDetector,
)
from dino_bot.models import Frame

ASSET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "templates"
    / "device-history-confirm-button.png"
)


def _manifest(tmp_path: Path, scales: list[float]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "reference_size": [900, 1600],
                "templates": [
                    {
                        "type": "device_history_confirm_button",
                        "file": str(ASSET),
                        "threshold": 0.82,
                        "scales": scales,
                        "click_offset": [86, 41],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("scale", [0.88, 0.95])
def test_scaled_template_uses_scaled_click_offset(tmp_path: Path, scale: float) -> None:
    template = cv2.imread(str(ASSET), cv2.IMREAD_COLOR)
    assert template is not None
    height, width = template.shape[:2]
    resized = cv2.resize(
        template,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    x, y = 290, 820
    scaled_height, scaled_width = resized.shape[:2]
    image[y : y + scaled_height, x : x + scaled_width] = resized
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    detections = OpenCvDetector(_manifest(tmp_path, [scale, 1.0])).detect(frame)

    match = max(detections, key=lambda item: item.confidence)
    assert match.type == "device_history_confirm_button"
    assert match.x == x + round(86 * scale)
    assert match.y == y + round(41 * scale)
    assert match.metadata["template_scale"] == scale


@pytest.mark.parametrize("scales", [[], [0], [-1]])
def test_template_scales_must_be_positive(tmp_path: Path, scales: list[float]) -> None:
    with pytest.raises(DetectorAssetError, match="scales must be positive"):
        OpenCvDetector(_manifest(tmp_path, scales))


def test_startup_growth_result_layout_is_detected() -> None:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[180:1350, 125:775] = 245
    image[1190:1340, 190:425] = (220, 190, 90)
    image[1190:1340, 475:710] = (110, 190, 100)
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    detections = StartupGrowthResultDetector().detect(frame)

    assert len(detections) == 1
    assert detections[0].type == "startup_growth_result_back"
    assert (detections[0].x, detections[0].y) == (307, 1265)


def test_startup_growth_result_requires_both_shortcut_buttons() -> None:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[180:1350, 125:775] = 245
    image[1190:1340, 190:425] = (220, 190, 90)
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    assert StartupGrowthResultDetector().detect(frame) == []


def test_startup_growth_result_detects_new_centred_nest_shortcut() -> None:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[180:1350, 125:775] = 245
    # Newer builds show one centred green nest-management shortcut, with the
    # egg artwork overlapping the button's top edge.
    image[1210:1330, 345:555] = (105, 185, 125)
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    detections = StartupGrowthResultDetector().detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.type == "startup_growth_result_back"
    assert (detection.x, detection.y) == (450, 1270)
    assert detection.metadata["shortcut_layout"] == "centered_nest"
    assert detection.metadata["shortcut_point"] == (450.0, 1270.0)


def test_startup_auto_battle_dialog_taps_outside() -> None:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[355:1230, 175:725] = 245
    image[850:980, 345:550] = (220, 190, 90)
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    detections = StartupAutoBattleDialogDetector().detect(frame)

    assert len(detections) == 1
    assert detections[0].type == "startup_auto_battle_close"
    assert (detections[0].x, detections[0].y) == (50, 800)


def test_startup_auto_battle_dialog_requires_cyan_button() -> None:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[355:1230, 175:725] = 245
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    assert StartupAutoBattleDialogDetector().detect(frame) == []


def _offer_card_image() -> np.ndarray:
    """Build the timed promotion card that stalled S9 on 2026-08-31.

    Ratios match the captured frame: orange ribbon 0.75, cream body 0.69,
    side margins 0.99 dimmed by the modal backdrop.
    """

    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    image[265:365, 165:740] = (40, 160, 230)
    image[800:1250, 175:725] = (160, 205, 232)
    return image


def test_startup_offer_card_is_detected_and_taps_the_backdrop() -> None:
    frame = Frame(image=_offer_card_image(), captured_at=datetime.now(UTC), source="test")

    detections = StartupOfferDismissDetector().detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.type == "startup_offer_dismiss"
    # The card has no close button; the tap must land on the dimmed backdrop.
    assert (detection.x, detection.y) == (50, 800)


def test_startup_offer_card_scales_to_the_device_resolution() -> None:
    image = cv2.resize(_offer_card_image(), (1080, 1920), interpolation=cv2.INTER_LINEAR)
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    detections = StartupOfferDismissDetector().detect(frame)

    assert len(detections) == 1
    assert (detections[0].x, detections[0].y) == (60, 960)


def test_startup_offer_card_ignores_a_bright_undimmed_screen() -> None:
    """The dimmed margins are what prove a modal is covering the map.

    Without them an ordinary bright screen carrying orange and cream artwork
    would be dismissed by a tap on live map controls.
    """

    image = _offer_card_image()
    image[:, 0:120] = 255
    image[:, 780:900] = 255
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    assert StartupOfferDismissDetector().detect(frame) == []


def test_startup_offer_card_requires_its_title_ribbon() -> None:
    image = _offer_card_image()
    image[265:365, 165:740] = 0
    frame = Frame(image=image, captured_at=datetime.now(UTC), source="test")

    assert StartupOfferDismissDetector().detect(frame) == []
