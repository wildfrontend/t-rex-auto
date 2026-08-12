from dino_bot.models import Detection
from dino_bot.overlays import (
    AUTOPLACE_NOTICE,
    AUTOPLACE_UNAVAILABLE,
    BLOCKED,
    CONFIRM_NO,
    CONFIRM_YES,
    DISMISS,
    INCUBATOR_FULL_TOAST,
    INFO,
    NESTED_PARENT_WARNING,
    NONE,
    SELECT_CONFIRM_PROMPT,
    next_safe_step,
)


def detection(kind: str, x: int = 10, y: int = 20, confidence: float = 1.0) -> Detection:
    return Detection(kind, x, y, confidence)


def test_known_confirmation_is_cancelled_only_through_detected_no_button() -> None:
    step = next_safe_step(
        [
            detection(SELECT_CONFIRM_PROMPT),
            detection(CONFIRM_YES, 100, 200),
            detection(CONFIRM_NO, 300, 400),
        ]
    )
    assert step.kind == DISMISS
    assert step.point == (300, 400)


def test_nested_parent_warning_is_safely_cancelled_by_generic_overlay_policy() -> None:
    step = next_safe_step(
        [detection(NESTED_PARENT_WARNING), detection(CONFIRM_NO, 300, 400)]
    )
    assert step.kind == DISMISS
    assert step.point == (300, 400)


def test_prompt_without_no_button_fails_closed() -> None:
    step = next_safe_step([detection(AUTOPLACE_NOTICE), detection(CONFIRM_YES)])
    assert step.kind == BLOCKED
    assert step.point is None


def test_incubator_full_toast_is_informational_not_a_wait_state() -> None:
    step = next_safe_step([detection(INCUBATOR_FULL_TOAST)])
    assert step.kind == INFO
    assert step.point is None


def test_autoplace_unavailable_toast_is_informational() -> None:
    step = next_safe_step([detection(AUTOPLACE_UNAVAILABLE)])
    assert step.kind == INFO
    assert step.point is None


def test_confirmation_colours_without_known_prompt_are_inert() -> None:
    step = next_safe_step([detection(CONFIRM_YES), detection(CONFIRM_NO)])
    assert step.kind == NONE
    assert step.point is None
