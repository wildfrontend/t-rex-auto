"""Safe handling rules for hatch confirmation dialogs and transient notices.

T8 deliberately exercises dialogs without authorizing a nest change.  A
confirmation is therefore only actionable when both its known prompt and the
red ``No`` button are visible.  The cyan ``Yes`` button is recognized for
diagnostics, but is never returned as a target by this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import Detection

SELECT_CONFIRM_PROMPT = "hatch_select_confirm_prompt"
NESTED_PARENT_WARNING = "hatch_nested_parent_warning"
AUTOPLACE_NOTICE = "hatch_autoplace_notice"
CONFIRM_YES = "hatch_confirm_yes"
CONFIRM_NO = "hatch_confirm_no"
INCUBATOR_FULL_TOAST = "hatch_incubator_full_toast"

DISMISS = "dismiss"
INFO = "info"
NONE = "none"
BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class OverlayStep:
    kind: str
    point: tuple[int, int] | None
    reason: str


def next_safe_step(detections: Sequence[Detection]) -> OverlayStep:
    """Choose a non-destructive T8 response from current detections.

    Known confirmations are cancelled through their detected ``No`` button.
    A prompt without that button fails closed instead of guessing a fixed
    coordinate.  The incubator-full toast is informational and must not make
    the workflow wait for a button that will never appear.
    """

    by_type: dict[str, list[Detection]] = {}
    for detection in detections:
        by_type.setdefault(detection.type, []).append(detection)

    prompt_type = next(
        (
            kind
            for kind in (
                SELECT_CONFIRM_PROMPT,
                NESTED_PARENT_WARNING,
                AUTOPLACE_NOTICE,
            )
            if kind in by_type
        ),
        None,
    )
    if prompt_type is not None:
        no_buttons = by_type.get(CONFIRM_NO, [])
        if not no_buttons:
            return OverlayStep(BLOCKED, None, f"{prompt_type} visible without No button")
        button = max(no_buttons, key=lambda item: item.confidence)
        return OverlayStep(DISMISS, (button.x, button.y), f"cancel {prompt_type}")

    if INCUBATOR_FULL_TOAST in by_type:
        return OverlayStep(INFO, None, "incubator full toast is transient")

    # Yes/No-looking controls without a known prompt are intentionally inert:
    # the game reuses their colours and shapes for destructive actions.
    return OverlayStep(NONE, None, "no known hatch overlay")
