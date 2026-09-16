from __future__ import annotations

import numpy as np

from dino_bot import parent_open
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.nest_readout import ATTACK_PARENT_REGIONS
from dino_bot.parent_open import ParentOpenTestPlanner


class EncodedReader:
    VALUES = {1: 30, 2: 282, 3: 1, 4: 2230, 5: 276}

    def read_int(self, image: np.ndarray) -> int | None:
        if not image.size:
            return None
        return self.VALUES.get(int(image[0, 0, 0]))


def make_frame(*, readable: bool = True, width: int = 900) -> Frame:
    image = np.zeros((round(1600 * width / 900), width, 3), dtype=np.uint8)
    if readable:
        scale = width / 900
        for regions, values in zip(
            ATTACK_PARENT_REGIONS,
            ((1, 2, 3), (4, 5, 3)),
            strict=True,
        ):
            for region, value in zip(regions, values, strict=True):
                x0, y0, x1, y1 = (round(coordinate * scale) for coordinate in region)
                image[y0:y1, x0:x1] = value
    return Frame(image)


def detection(target_type: str, x: int, y: int) -> Detection:
    return Detection.from_bbox(target_type, BoundingBox(x - 5, y - 5, 10, 10), 0.99)


def attack_nest(*items: Detection) -> list[Detection]:
    return [
        detection(parent_open.NEST_TITLE, 451, 261),
        detection(parent_open.TAG_HDR_ATTACK, 217, 166),
        *items,
    ]


def planner() -> ParentOpenTestPlanner:
    return ParentOpenTestPlanner(EncodedReader())  # type: ignore[arg-type]


def test_reads_stats_then_targets_only_left_parent_body() -> None:
    test_planner = planner()
    target = test_planner.choose(make_frame(), attack_nest())

    assert target is not None
    assert target.type == parent_open.PARENT_LEFT
    assert (target.x, target.y) == (264, 407)
    assert test_planner.choose(make_frame(), attack_nest()) is None


def test_wrong_screen_never_taps() -> None:
    test_planner = planner()
    assert test_planner.choose(
        make_frame(),
        [detection(parent_open.TAG_HDR_ATTACK, 217, 166)],
    ) is None
    assert test_planner.last_stage() == "waiting_for_nest"


def test_requires_attack_header_and_readable_six_stats() -> None:
    test_planner = planner()
    assert test_planner.choose(
        make_frame(),
        [detection(parent_open.NEST_TITLE, 451, 261)],
    ) is None
    assert test_planner.last_stage() == "attack_filter_required"

    assert planner().choose(make_frame(readable=False), attack_nest()) is None


def test_open_tag_menu_blocks_parent_tap() -> None:
    test_planner = planner()
    assert test_planner.choose(
        make_frame(),
        attack_nest(detection("hatch_tag_all", 227, 211)),
    ) is None
    assert test_planner.last_stage() == "tag_menu_open"


def test_select_dino_screen_completes_without_tap() -> None:
    test_planner = planner()
    detections = [
        detection(parent_open.SELECT_TITLE, 451, 299),
        detection(parent_open.TAG_HDR_ATTACK, 217, 166),
    ]
    assert test_planner.choose(make_frame(), detections) is None
    assert test_planner.is_complete()


def test_action_success_or_failure_ends_bounded_workflow() -> None:
    successful = planner()
    successful.on_action_success(parent_open.PARENT_LEFT)
    assert successful.is_complete()

    failed = planner()
    failed.on_action_failure(parent_open.PARENT_LEFT)
    assert failed.is_complete()


def test_coordinate_scales_and_action_vocabulary_is_closed() -> None:
    test_planner = planner()
    detections = [
        detection(parent_open.NEST_TITLE, 226, 131),
        detection(parent_open.TAG_HDR_ATTACK, 109, 83),
    ]
    target = test_planner.choose(make_frame(width=450), detections)
    assert target is not None and (target.x, target.y) == (132, 204)
    assert parent_open.DEFAULT_TARGET_ACTIONS == {parent_open.PARENT_LEFT: "tap"}
    assert parent_open.DEFAULT_SUCCESS_TRANSITIONS == {
        parent_open.PARENT_LEFT: (parent_open.SELECT_TITLE,),
    }
