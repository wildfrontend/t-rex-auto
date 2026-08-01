from __future__ import annotations

import numpy as np

from dino_bot import nest_filter
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.nest_filter import NestTagFilterTestPlanner


def frame(width: int = 900, height: int = 1600) -> Frame:
    return Frame(np.zeros((height, width, 3), dtype=np.uint8))


def detection(target_type: str, x: int, y: int) -> Detection:
    return Detection.from_bbox(target_type, BoundingBox(x - 5, y - 5, 10, 10), 0.99)


def nest(*items: Detection) -> list[Detection]:
    return [detection(nest_filter.NEST_TITLE, 451, 261), *items]


def test_wrong_screen_never_taps() -> None:
    planner = NestTagFilterTestPlanner()
    assert planner.choose(frame(), [detection(nest_filter.TAG_HDR_ALL, 228, 168)]) is None
    assert planner.last_stage() == "waiting_for_nest"


def test_known_header_is_opened() -> None:
    planner = NestTagFilterTestPlanner()
    target = planner.choose(frame(), nest(detection(nest_filter.TAG_HDR_ALL, 228, 168)))
    assert target is not None
    assert target.type == nest_filter.FILTER_HEADER
    assert (target.x, target.y) == (228, 168)


def test_attack_header_is_reselected_for_on_device_verification() -> None:
    planner = NestTagFilterTestPlanner()
    target = planner.choose(frame(), nest(detection(nest_filter.TAG_HDR_ATTACK, 218, 168)))
    assert target is not None and target.type == nest_filter.FILTER_HEADER


def test_unrecognized_remembered_header_uses_scaled_safe_point() -> None:
    planner = NestTagFilterTestPlanner()
    target = planner.choose(frame(width=450, height=800), nest())
    assert target is not None
    assert (target.x, target.y) == (114, 84)


def test_expanded_menu_selects_attack() -> None:
    planner = NestTagFilterTestPlanner()
    target = planner.choose(
        frame(),
        nest(
            detection(nest_filter.TAG_ALL, 227, 211),
            detection(nest_filter.TAG_HP, 228, 340),
            detection(nest_filter.TAG_ATTACK, 229, 383),
        ),
    )
    assert target is not None
    assert target.type == nest_filter.TAG_ATTACK
    assert (target.x, target.y) == (229, 383)


def test_successful_attack_selection_completes_planner() -> None:
    planner = NestTagFilterTestPlanner()
    planner.on_action_success(nest_filter.TAG_ATTACK)
    assert planner.choose(frame(), nest(detection(nest_filter.TAG_HDR_ATTACK, 218, 168))) is None
    assert planner.last_stage() == "filter_done"


def test_filter_cycle_and_transitions_are_nondestructive() -> None:
    assert nest_filter.DEFAULT_CYCLE_COMPLETE_TARGETS == (nest_filter.TAG_ATTACK,)
    assert set(nest_filter.DEFAULT_TARGET_ACTIONS) == {
        nest_filter.FILTER_HEADER,
        nest_filter.TAG_ATTACK,
    }
    assert nest_filter.DEFAULT_SUCCESS_TRANSITIONS[nest_filter.TAG_ATTACK] == (
        nest_filter.TAG_HDR_ATTACK,
    )
