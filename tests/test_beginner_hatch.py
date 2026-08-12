from __future__ import annotations

import numpy as np

from dino_bot import hatch, nest_filter
from dino_bot.beginner_hatch import (
    AUTOPLACE_YES,
    AUTOPLACE_BUTTON,
    COLLECT_EGGS_BUTTON,
    DEFAULT_SUCCESS_TRANSITIONS,
    HATCH_DETECTION_TYPES,
    NEST_DETECTION_TYPES,
    BeginnerHatchPlanner,
)
from dino_bot.full_hatch import (
    AUTOPLACE_PROMPT,
    NEST_MASK_CLOSE,
    OPEN_NEST,
    STARTUP_AUTO_BATTLE_CLOSE,
    STARTUP_GROWTH_RESULT,
    STARTUP_NEST_SHORTCUT,
)
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.overlays import CONFIRM_YES, INCUBATOR_FULL_TOAST
from dino_bot.parent_open import NEST_TITLE


def frame() -> Frame:
    return Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))


def detection(
    target_type: str,
    x: int = 100,
    y: int = 100,
    *,
    metadata: dict[str, object] | None = None,
) -> Detection:
    return Detection(
        target_type,
        x,
        y,
        0.99,
        BoundingBox(x - 5, y - 5, 10, 10),
        metadata or {},
    )


def planner() -> BeginnerHatchPlanner:
    return BeginnerHatchPlanner(
        egg_pile_point=(450.0, 1330.0),
        max_scrolls=0,
    )


def nest_screen(*extra: Detection) -> list[Detection]:
    return [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_ALL, 218, 168),
        *extra,
    ]


def reach_autoplace(current: BeginnerHatchPlanner) -> None:
    current.on_action_success(hatch.CLOSE_BUTTON)
    target = current.choose(frame(), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert target is not None and target.type == OPEN_NEST
    target = current.choose(
        frame(), nest_screen(detection(AUTOPLACE_BUTTON, 450, 1315))
    )
    assert target is not None and target.type == AUTOPLACE_BUTTON


def test_beginner_hatches_then_switches_to_all_and_autoplaces_once() -> None:
    current = planner()
    current.on_action_success(hatch.CLAIM_BUTTON)
    reach_autoplace(current)

    # The target is armed before the tap so a repeated planning frame cannot
    # emit a second auto-place action.
    assert current.choose(frame(), nest_screen(detection(AUTOPLACE_BUTTON, 450, 1315))) is None


def test_beginner_confirms_autoplace_then_collects_once() -> None:
    current = planner()
    reach_autoplace(current)
    current.on_action_success(AUTOPLACE_BUTTON)

    yes = current.choose(
        frame(),
        nest_screen(
            detection(AUTOPLACE_PROMPT, 450, 760),
            detection(CONFIRM_YES, 450, 1040),
        ),
    )
    assert yes is not None and yes.type == AUTOPLACE_YES
    current.on_action_success(yes.type)

    collect = current.choose(
        frame(), nest_screen(detection(COLLECT_EGGS_BUTTON, 640, 1315))
    )
    assert collect is not None and collect.type == COLLECT_EGGS_BUTTON
    assert current.choose(
        frame(), nest_screen(detection(COLLECT_EGGS_BUTTON, 640, 1315))
    ) is None


def test_beginner_full_incubator_toast_closes_nest_without_recollecting() -> None:
    current = planner()
    reach_autoplace(current)
    current.on_action_success(AUTOPLACE_BUTTON)
    collect = current.choose(
        frame(), nest_screen(detection(COLLECT_EGGS_BUTTON, 640, 1315))
    )
    assert collect is not None and collect.type == COLLECT_EGGS_BUTTON
    assert INCUBATOR_FULL_TOAST in DEFAULT_SUCCESS_TRANSITIONS[COLLECT_EGGS_BUTTON]

    current.on_action_success(COLLECT_EGGS_BUTTON)
    close = current.choose(
        frame(), nest_screen(detection(INCUBATOR_FULL_TOAST, 450, 245))
    )
    assert close is not None and close.type == NEST_MASK_CLOSE
    current.on_action_success(close.type)

    assert current.choose(frame(), [detection(hatch.HOME_ANCHOR, 59, 561)]) is None
    assert current.next_ready_delay_ms() > 0
    assert current.management_rounds == 1


def test_beginner_exposes_only_the_incubator_wait_for_hunting() -> None:
    now = [0.0]
    current = BeginnerHatchPlanner(
        egg_pile_point=(450.0, 1330.0),
        clock=lambda: now[0],
    )
    current._hatch_child.begin_rescan_wait("test", seconds=60)

    assert current.is_hunt_cooldown_active()
    assert current.begin_interim_collection() is False

    now[0] = 61.0
    assert not current.is_hunt_cooldown_active()


def test_beginner_stops_instead_of_repeating_an_unverified_mutation() -> None:
    current = planner()
    reach_autoplace(current)
    current.on_action_failure(AUTOPLACE_BUTTON)

    assert current.is_complete()
    assert current.is_hatch_blocked()
    assert current.choose(frame(), nest_screen(detection(AUTOPLACE_BUTTON, 450, 1315))) is None


def test_beginner_uses_centred_startup_nest_shortcut() -> None:
    current = planner()
    target = current.choose(
        frame(),
        [
            detection(
                STARTUP_GROWTH_RESULT,
                450,
                1270,
                metadata={"shortcut_layout": "centered_nest"},
            )
        ],
    )

    assert target is not None and target.type == STARTUP_NEST_SHORTCUT
    assert (target.x, target.y) == (450, 1270)


def test_beginner_hatch_result_beats_auto_battle_layout_false_positive() -> None:
    current = planner()
    target = current.choose(
        frame(),
        [
            detection(hatch.CLAIM_BUTTON, 330, 1242),
            detection(hatch.EXPEL_BUTTON, 570, 1242),
            detection(STARTUP_AUTO_BATTLE_CLOSE, 50, 800),
        ],
    )

    assert target is not None and target.type == hatch.CLAIM_BUTTON


def test_beginner_detection_scope_excludes_parent_and_cave_controls() -> None:
    assert hatch.HATCH_BUTTON in HATCH_DETECTION_TYPES
    assert hatch.INCUBATOR_TITLE in HATCH_DETECTION_TYPES
    assert AUTOPLACE_BUTTON in NEST_DETECTION_TYPES
    assert "hatch_cave" not in HATCH_DETECTION_TYPES | NEST_DETECTION_TYPES
    assert "hatch_parent_left" not in HATCH_DETECTION_TYPES | NEST_DETECTION_TYPES
