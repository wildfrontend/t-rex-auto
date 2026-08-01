from __future__ import annotations

import numpy as np

from dino_bot import attack_replacement, nest_filter, select_sort
from dino_bot.attack_replacement import AttackReplacementTestPlanner
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.nest_readout import (
    ATTACK_PARENT_REGIONS,
    SELECT_FIRST_ROW_REGIONS,
    SELECT_ROW_PITCH,
)
from dino_bot.nests import HP_RULE, Stats
from dino_bot.overlays import CONFIRM_YES, NESTED_PARENT_WARNING, SELECT_CONFIRM_PROMPT


class EncodedReader:
    def __init__(self) -> None:
        self._next_code = 1
        self._codes: dict[int, int] = {}
        self._values: dict[int, int] = {}

    def encode(self, value: int) -> int:
        if value not in self._codes:
            code = self._next_code
            self._next_code += 1
            self._codes[value] = code
            self._values[code] = value
        return self._codes[value]

    def read_int(self, image: np.ndarray) -> int | None:
        if not image.size:
            return None
        return self._values.get(int(image[0, 0, 0]))


def detection(target_type: str, x: int, y: int) -> Detection:
    return Detection.from_bbox(target_type, BoundingBox(x - 5, y - 5, 10, 10), 0.99)


def nest_detections(*items: Detection) -> list[Detection]:
    return [
        detection(attack_replacement.NEST_TITLE, 451, 261),
        detection(nest_filter.TAG_HDR_ATTACK, 217, 166),
        *items,
    ]


def select_detections(*items: Detection) -> list[Detection]:
    return [
        detection(attack_replacement.SELECT_TITLE, 451, 299),
        detection(select_sort.TAG_HDR_ALL, 228, 204),
        detection(select_sort.SORT_HDR_ATTACK, 637, 355),
        *items,
    ]


def hp_nest_detections(*items: Detection) -> list[Detection]:
    return [
        detection(attack_replacement.NEST_TITLE, 451, 261),
        detection(nest_filter.TAG_HDR_HP, 217, 166),
        *items,
    ]


def hp_select_detections(*items: Detection) -> list[Detection]:
    return [
        detection(attack_replacement.SELECT_TITLE, 451, 299),
        detection(select_sort.TAG_HDR_ALL, 228, 204),
        detection(select_sort.SORT_HP, 650, 355),
        *items,
    ]


def nest_frame(reader: EncodedReader, left: Stats, right: Stats) -> Frame:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    for regions, stats in zip(ATTACK_PARENT_REGIONS, (left, right), strict=True):
        for region, value in zip(
            regions,
            (stats.hp, stats.attack, stats.speed),
            strict=True,
        ):
            x0, y0, x1, y1 = map(int, region)
            image[y0:y1, x0:x1] = reader.encode(value)
    return Frame(image)


def select_frame(reader: EncodedReader, rows: list[Stats]) -> Frame:
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    for index, stats in enumerate(rows):
        regions = tuple(
            (x0, y0 + index * SELECT_ROW_PITCH, x1, y1 + index * SELECT_ROW_PITCH)
            for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
        )
        for region, value in zip(
            regions,
            (stats.hp, stats.attack, stats.speed),
            strict=True,
        ):
            x0, y0, x1, y1 = map(int, region)
            image[y0:y1, x0:x1] = reader.encode(value)
    return Frame(image)


def finish_main_filter(planner: AttackReplacementTestPlanner) -> None:
    planner.on_action_success(nest_filter.TAG_ATTACK)


def hp_planner(reader: EncodedReader) -> AttackReplacementTestPlanner:
    return AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        rule=HP_RULE,
        nest_filter_option=nest_filter.TAG_HP,
        nest_filter_header=nest_filter.TAG_HDR_HP,
        select_sort_option=select_sort.SORT_HP,
        select_sort_header=select_sort.SORT_HP,
        select_sort_menu_point=(650.0, 501.0),
    )


def test_starts_by_converging_main_nest_filter_to_attack() -> None:
    reader = EncodedReader()
    frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]

    open_filter = planner.choose(frame, nest_detections())
    assert open_filter is not None and open_filter.type == nest_filter.FILTER_HEADER
    planner.on_action_success(open_filter.type)

    select_attack = planner.choose(
        frame,
        nest_detections(detection(nest_filter.TAG_ATTACK, 229, 383)),
    )
    assert select_attack is not None and select_attack.type == nest_filter.TAG_ATTACK
    planner.on_action_success(select_attack.type)

    left = planner.choose(frame, nest_detections())
    assert left is not None and left.type == attack_replacement.PARENT_LEFT


def test_equal_attack_keeps_both_parents_and_closes_each_list() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    candidates = select_frame(reader, [Stats(30, 282, 1), Stats(2230, 276, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    left = planner.choose(parent_frame, nest_detections())
    assert left is not None and left.type == attack_replacement.PARENT_LEFT
    assert (left.x, left.y) == (264, 407)
    planner.on_action_success(left.type)

    close_left = planner.choose(candidates, select_detections())
    assert close_left is not None and close_left.type == attack_replacement.SELECT_MASK_CLOSE
    assert (close_left.x, close_left.y) == (50, 800)
    planner.on_action_success(close_left.type)

    right = planner.choose(parent_frame, nest_detections())
    assert right is not None and right.type == attack_replacement.PARENT_RIGHT
    assert (right.x, right.y) == (523, 407)
    planner.on_action_success(right.type)

    close_right = planner.choose(candidates, select_detections())
    assert close_right is not None and close_right.type == attack_replacement.SELECT_MASK_CLOSE
    planner.on_action_success(close_right.type)
    assert planner.is_complete()


def test_stronger_tied_candidate_selects_lowest_secondary_then_confirms() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(
        reader,
        [Stats(200, 283, 10), Stats(30, 283, 1), Stats(30, 282, 1)],
    )
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    planner.on_action_success(
        planner.choose(parent_frame, nest_detections()).type  # type: ignore[union-attr]
    )

    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 520)
    planner.on_action_success(candidate.type)

    yes = planner.choose(
        candidates,
        [
            detection(SELECT_CONFIRM_PROMPT, 450, 660),
            detection(CONFIRM_YES, 350, 850),
        ],
    )
    assert yes is not None and yes.type == CONFIRM_YES
    assert (yes.x, yes.y) == (350, 850)


def test_confirmation_yes_is_never_guessed_without_known_prompt() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 283, 1), Stats(30, 282, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None
    planner.on_action_success(candidate.type)

    assert planner.choose(candidates, [detection(CONFIRM_YES, 350, 850)]) is None
    assert planner.is_complete()


def test_nested_parent_warning_is_confirmed_then_normal_prompt_is_confirmed() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    candidates = select_frame(reader, [Stats(30, 283, 1), Stats(30, 282, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None
    planner.on_action_success(candidate.type)

    remove = planner.choose(
        candidates,
        [
            detection(NESTED_PARENT_WARNING, 450, 724),
            detection(CONFIRM_YES, 366, 864),
        ],
    )
    assert remove is not None
    assert remove.type == attack_replacement.NESTED_PARENT_YES
    assert (remove.x, remove.y) == (366, 864)
    planner.on_action_success(remove.type)

    confirm = planner.choose(
        candidates,
        [
            detection(SELECT_CONFIRM_PROMPT, 450, 724),
            detection(CONFIRM_YES, 366, 828),
        ],
    )
    assert confirm is not None and confirm.type == CONFIRM_YES


def test_nested_parent_warning_can_replace_directly_and_advance() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    candidates = select_frame(reader, [Stats(30, 283, 1), Stats(30, 282, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None
    planner.on_action_success(candidate.type)
    remove = planner.choose(
        candidates,
        [
            detection(NESTED_PARENT_WARNING, 450, 724),
            detection(CONFIRM_YES, 366, 864),
        ],
    )
    assert remove is not None
    planner.on_action_success(remove.type)

    assert planner.choose(parent_frame, nest_detections()) is None
    right = planner.choose(parent_frame, nest_detections())
    assert right is not None and right.type == attack_replacement.PARENT_RIGHT


def test_equal_parent_plateau_stops_without_searching_or_selecting() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    candidates = select_frame(reader, [Stats(30 + index, 282, 1) for index in range(9)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    close = planner.choose(candidates, select_detections())
    assert close is not None and close.type == attack_replacement.SELECT_MASK_CLOSE


def test_hp_workflow_selects_highest_hp_with_lowest_secondary_load() -> None:
    reader = EncodedReader()
    parents = nest_frame(reader, Stats(2230, 2, 1), Stats(2230, 2, 1))
    candidates = select_frame(
        reader,
        [Stats(2300, 10, 10), Stats(2300, 2, 1), Stats(2250, 1, 1)],
    )
    planner = hp_planner(reader)
    planner.on_action_success(nest_filter.TAG_HP)
    parent = planner.choose(parents, hp_nest_detections())
    assert parent is not None and parent.type == attack_replacement.PARENT_LEFT
    planner.on_action_success(parent.type)

    candidate = planner.choose(candidates, hp_select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 520)


def test_hp_equal_plateau_keeps_parent_without_searching() -> None:
    reader = EncodedReader()
    parents = nest_frame(reader, Stats(2300, 2, 1), Stats(2300, 2, 1))
    candidates = select_frame(
        reader,
        [Stats(2300, index + 1, 1) for index in range(9)],
    )
    planner = hp_planner(reader)
    planner.on_action_success(nest_filter.TAG_HP)
    parent = planner.choose(parents, hp_nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    close = planner.choose(candidates, hp_select_detections())
    assert close is not None and close.type == attack_replacement.SELECT_MASK_CLOSE


def test_select_sort_setup_actions_are_reused_but_rows_are_rule_gated() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 283, 1), Stats(30, 282, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    open_tag = planner.choose(
        candidates,
        [detection(attack_replacement.SELECT_TITLE, 451, 299)],
    )
    assert open_tag is not None and open_tag.type == select_sort.TAG_HEADER


def test_open_nest_tag_menu_and_unreadable_parents_fail_closed() -> None:
    reader = EncodedReader()
    frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    assert planner.choose(
        frame,
        nest_detections(detection("hatch_tag_all", 227, 211)),
    ) is None

    blank = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    unreadable = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(unreadable)
    assert unreadable.choose(
        blank,
        nest_detections(),
    ) is None


def test_action_vocabulary_contains_only_bounded_workflow_controls() -> None:
    assert attack_replacement.DEFAULT_TARGET_ACTIONS[attack_replacement.CANDIDATE_ROW] == "tap"
    assert attack_replacement.DEFAULT_TARGET_ACTIONS[CONFIRM_YES] == "tap"
    assert attack_replacement.DEFAULT_SUCCESS_TRANSITIONS[
        attack_replacement.CANDIDATE_ROW
    ] == (SELECT_CONFIRM_PROMPT, NESTED_PARENT_WARNING)
    assert attack_replacement.DEFAULT_SUCCESS_TRANSITIONS[
        attack_replacement.NESTED_PARENT_YES
    ] == (SELECT_CONFIRM_PROMPT, attack_replacement.NEST_TITLE)
    assert attack_replacement.DEFAULT_SUCCESS_TRANSITIONS[CONFIRM_YES] == (
        attack_replacement.NEST_TITLE,
    )
