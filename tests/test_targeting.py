from __future__ import annotations

from dino_bot.models import Detection
from dino_bot.targeting import (
    best_detection,
    detection_target,
    swipe_target,
    synthetic_target,
)


def test_best_detection_handles_missing_empty_and_ranked_inputs() -> None:
    low = Detection("button", 10, 20, 0.7)
    high = Detection("button", 30, 40, 0.9)

    assert best_detection(None) is None
    assert best_detection([]) is None
    assert best_detection([low, high]) is high


def test_detection_target_preserves_evidence_when_action_name_changes() -> None:
    detection = Detection("confirm_yes", 30, 40, 0.9, metadata={"source": "template"})

    target = detection_target(detection, target_type="boost_confirm")

    assert target.type == "boost_confirm"
    assert (target.x, target.y, target.confidence) == (30, 40, 0.9)
    assert target.detection is detection


def test_synthetic_and_swipe_targets_share_provenance_contract() -> None:
    tap = synthetic_target("fixed_tap", 10, 20, metadata={"reason": "screen-gated"})
    swipe = swipe_target("drag", 1, 2, 3, 4, duration_ms=500)

    assert tap.detection.metadata == {"synthetic": True, "reason": "screen-gated"}
    assert swipe.detection.metadata == {
        "synthetic": True,
        "swipe": {"x2": 3, "y2": 4, "duration_ms": 500},
    }
