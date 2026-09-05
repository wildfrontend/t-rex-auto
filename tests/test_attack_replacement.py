from __future__ import annotations

import numpy as np

from dino_bot import attack_replacement, nest_filter, nest_readout, select_sort
from dino_bot.attack_replacement import AttackReplacementTestPlanner
from dino_bot.models import BoundingBox, Detection, Frame, VerificationResult
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


class SnapshotCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def capture(self, frame, reader, regions, *, stage, side, attempts):
        self.calls.append((stage, side, attempts))
        return None


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


def hp_planner(
    reader: EncodedReader,
    *,
    allow_extreme_specialization_parent: bool = False,
    prefer_specialization_purity: bool = False,
) -> AttackReplacementTestPlanner:
    return AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        rule=HP_RULE,
        nest_filter_option=nest_filter.TAG_HP,
        nest_filter_header=nest_filter.TAG_HDR_HP,
        select_sort_option=select_sort.SORT_HP,
        select_sort_header=select_sort.SORT_HP,
        select_sort_menu_point=(650.0, 501.0),
        allow_extreme_specialization_parent=allow_extreme_specialization_parent,
        prefer_specialization_purity=prefer_specialization_purity,
    )


def test_starts_by_converging_main_nest_filter_to_attack() -> None:
    reader = EncodedReader()
    frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]

    # 表頭已是「攻擊」:標籤收斂直接跳過,馬上進入親代讀取。
    left = planner.choose(frame, nest_detections())
    assert left is not None and left.type == attack_replacement.PARENT_LEFT


def test_unreadable_parent_stats_collects_rate_limited_evidence() -> None:
    reader = EncodedReader()
    evidence = SnapshotCollector()
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        parent_stats_snapshots=evidence,  # type: ignore[arg-type]
    )

    assert (
        planner.choose(
            Frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
            nest_detections(),
        )
        is None
    )
    assert evidence.calls == [("parent_stats_unreadable", "left", 1)]


def test_parent_stats_require_two_matching_frames_before_a_tap() -> None:
    reader = EncodedReader()
    frame = nest_frame(reader, Stats(2920, 3, 1), Stats(2920, 3, 1))
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        minimum_consistent_stat_reads=2,
        stat_read_retries=3,
    )
    finish_main_filter(planner)

    assert planner.choose(frame, nest_detections()) is None
    target = planner.choose(frame, nest_detections())

    assert target is not None and target.type == attack_replacement.PARENT_LEFT


def test_parent_stats_recalibrate_when_consecutive_frames_disagree() -> None:
    reader = EncodedReader()
    first = nest_frame(reader, Stats(2920, 3, 1), Stats(2920, 3, 1))
    corrected = nest_frame(reader, Stats(2930, 3, 1), Stats(2930, 3, 1))
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        minimum_consistent_stat_reads=2,
        stat_read_retries=3,
    )
    finish_main_filter(planner)

    assert planner.choose(first, nest_detections()) is None
    assert planner.choose(corrected, nest_detections()) is None
    target = planner.choose(corrected, nest_detections())

    assert target is not None and target.type == attack_replacement.PARENT_LEFT


def test_invalid_numeric_hp_does_not_exhaust_parent_calibration() -> None:
    reader = EncodedReader()
    evidence = SnapshotCollector()
    invalid = nest_frame(reader, Stats(2926, 3, 1), Stats(2920, 3, 1))
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        minimum_consistent_stat_reads=2,
        stat_read_retries=3,
        parent_stats_snapshots=evidence,  # type: ignore[arg-type]
    )
    finish_main_filter(planner)

    assert planner.choose(invalid, nest_detections()) is None
    target = planner.choose(invalid, nest_detections())

    assert target is not None and target.type == attack_replacement.PARENT_LEFT
    assert planner.last_stage() == "open_left"
    assert evidence.calls == []


def test_candidate_stats_require_two_matching_frames_before_a_tap() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 279, 1), Stats(30, 278, 1)])
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        minimum_consistent_stat_reads=2,
        stat_read_retries=3,
    )
    finish_main_filter(planner)
    assert planner.choose(parent_frame, nest_detections()) is None
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    assert planner.choose(candidates, select_detections()) is None
    candidate = planner.choose(candidates, select_detections())

    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW


def test_suspicious_parent_candidate_ocr_is_reread_then_fails_closed() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(970, 716, 115), Stats(970, 716, 115))
    candidates = select_frame(
        reader,
        [Stats(970, 116, 115), Stats(970, 116, 115), Stats(970, 116, 115)],
    )
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None and parent.type == attack_replacement.PARENT_LEFT
    planner.on_action_success(parent.type)

    # The first conflict requests a fresh candidate read and never taps.
    assert planner.choose(candidates, select_detections()) is None
    assert planner.last_stage() == "select_left"
    assert not planner.is_complete()

    # A stable repeat of the same conflict is treated as unsafe, not as an
    # instruction to replace the parent.
    assert planner.choose(candidates, select_detections()) is None
    assert planner.last_stage() == "suspicious_ocr_left"
    assert planner.is_complete()


def test_extreme_hp_parent_requires_one_high_and_two_low_candidate() -> None:
    reader = EncodedReader()
    parents = nest_frame(reader, Stats(10, 1, 1), Stats(10, 1, 1))
    ordinary = select_frame(reader, [Stats(1300, 68, 20), Stats(1200, 68, 20)])
    specialized = select_frame(reader, [Stats(1300, 1, 1), Stats(1200, 1, 1)])
    planner = hp_planner(reader, allow_extreme_specialization_parent=True)
    planner.on_action_success(nest_filter.TAG_HP)

    parent = planner.choose(parents, hp_nest_detections())
    assert parent is not None and parent.type == attack_replacement.PARENT_LEFT
    planner.on_action_success(parent.type)
    close = planner.choose(ordinary, hp_select_detections())
    assert close is not None and close.type == attack_replacement.SELECT_MASK_CLOSE
    planner.on_action_success(close.type)

    right = planner.choose(parents, hp_nest_detections())
    assert right is not None and right.type == attack_replacement.PARENT_RIGHT
    planner.on_action_success(right.type)
    candidate = planner.choose(specialized, hp_select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW


def test_extreme_attack_parent_accepts_attack_specialized_candidate() -> None:
    reader = EncodedReader()
    parents = nest_frame(reader, Stats(10, 1, 1), Stats(10, 1, 1))
    candidates = select_frame(reader, [Stats(10, 72, 1), Stats(10, 68, 1)])
    planner = AttackReplacementTestPlanner(
        reader,  # type: ignore[arg-type]
        allow_extreme_specialization_parent=True,
    )
    finish_main_filter(planner)

    parent = planner.choose(parents, nest_detections())
    assert parent is not None and parent.type == attack_replacement.PARENT_LEFT
    planner.on_action_success(parent.type)
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW


def test_extreme_condition_unlocks_normal_flow_after_specialized_candidate() -> None:
    reader = EncodedReader()
    parents = nest_frame(reader, Stats(10, 1, 1), Stats(10, 1, 1))
    specialized = select_frame(reader, [Stats(1300, 1, 1)])
    normal = select_frame(reader, [Stats(900, 68, 20)])
    planner = hp_planner(reader, allow_extreme_specialization_parent=True)
    planner.on_action_success(nest_filter.TAG_HP)

    parent = planner.choose(parents, hp_nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    candidate = planner.choose(specialized, hp_select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW
    planner.on_action_success(candidate.type)

    prompt = planner.choose(
        specialized,
        [
            detection(SELECT_CONFIRM_PROMPT, 450, 660),
            detection(CONFIRM_YES, 350, 850),
        ],
    )
    assert prompt is not None and prompt.type == CONFIRM_YES
    planner.on_action_success(prompt.type)
    right = planner.choose(parents, hp_nest_detections())
    assert right is not None and right.type == attack_replacement.PARENT_RIGHT
    planner.on_action_success(right.type)
    normal_candidate = planner.choose(normal, hp_select_detections())
    assert normal_candidate is not None
    assert normal_candidate.type == attack_replacement.CANDIDATE_ROW


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
        [Stats(200, 279, 10), Stats(30, 279, 1), Stats(30, 278, 1)],
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


def test_purity_repair_selects_clean_attack_line_and_ignores_speed() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(
        reader,
        Stats(2230, 282, 1),
        Stats(2230, 282, 150),
    )
    candidates = select_frame(
        reader,
        [Stats(2240, 290, 1), Stats(30, 276, 150)],
    )
    planner = AttackReplacementTestPlanner(  # type: ignore[arg-type]
        reader,
        prefer_specialization_purity=True,
    )
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    candidate = planner.choose(candidates, select_detections())

    assert candidate is not None
    assert candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 520)


def test_purity_repair_selects_clean_hp_line_below_mixed_high_stat() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(
        reader,
        Stats(2340, 276, 150),
        Stats(2340, 276, 1),
    )
    candidates = select_frame(
        reader,
        [Stats(2400, 280, 1), Stats(2230, 2, 150)],
    )
    planner = hp_planner(reader, prefer_specialization_purity=True)
    planner.on_action_success(nest_filter.TAG_HP)
    parent = planner.choose(parent_frame, hp_nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)

    candidate = planner.choose(candidates, hp_select_detections())

    assert candidate is not None
    assert candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 520)


def test_confirmation_yes_is_never_guessed_without_known_prompt() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 279, 1), Stats(30, 278, 1)])
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


def test_candidate_can_apply_directly_and_return_to_nest() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 279, 1), Stats(30, 278, 1)])
    planner = AttackReplacementTestPlanner(  # type: ignore[arg-type]
        reader,
        minimum_consistent_stat_reads=2,
        stat_read_retries=3,
    )
    finish_main_filter(planner)
    assert planner.choose(parent_frame, nest_detections()) is None
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    assert planner.choose(candidates, select_detections()) is None
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None
    planner.on_action_success_context(
        candidate,
        parent_frame,
        [],
        VerificationResult(True, "previous UI disappeared: hatch_select_title"),
    )
    assert planner.choose(parent_frame, []) is None
    right = planner.choose(parent_frame, [])

    assert right is not None and right.type == attack_replacement.PARENT_RIGHT


def test_parent_panel_without_nest_evidence_is_rejected_without_direct_return() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    assert planner.choose(parent_frame, []) is None
    assert planner.last_stage() == "nest_left"


def test_candidate_prompt_still_uses_legacy_confirmation_flow() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 279, 1), Stats(30, 278, 1)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)
    parent = planner.choose(parent_frame, nest_detections())
    assert parent is not None
    planner.on_action_success(parent.type)
    candidate = planner.choose(candidates, select_detections())
    assert candidate is not None
    prompt = detection(SELECT_CONFIRM_PROMPT, 450, 660)
    yes_button = detection(CONFIRM_YES, 350, 850)

    planner.on_action_success_context(
        candidate,
        candidates,
        [prompt],
        VerificationResult(True, "next UI detected: hatch_select_confirm_prompt"),
    )
    yes = planner.choose(candidates, [prompt, yes_button])

    assert yes is not None and yes.type == CONFIRM_YES


def test_nested_parent_warning_is_confirmed_then_normal_prompt_is_confirmed() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 282, 1), Stats(30, 282, 1))
    candidates = select_frame(reader, [Stats(30, 285, 1), Stats(30, 282, 1)])
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
    candidates = select_frame(reader, [Stats(30, 285, 1), Stats(30, 282, 1)])
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
        [Stats(2260, 10, 10), Stats(2260, 2, 1), Stats(2250, 1, 1)],
    )
    planner = hp_planner(reader)
    planner.on_action_success(nest_filter.TAG_HP)
    parent = planner.choose(parents, hp_nest_detections())
    assert parent is not None and parent.type == attack_replacement.PARENT_LEFT
    planner.on_action_success(parent.type)

    candidate = planner.choose(candidates, hp_select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 520)


def test_hp_equal_plateau_selects_row_with_lower_secondary_load() -> None:
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

    candidate = planner.choose(candidates, hp_select_detections())
    assert candidate is not None and candidate.type == attack_replacement.CANDIDATE_ROW
    assert (candidate.x, candidate.y) == (350, 435)


def test_select_sort_setup_actions_are_reused_but_rows_are_rule_gated() -> None:
    reader = EncodedReader()
    parent_frame = nest_frame(reader, Stats(30, 276, 1), Stats(30, 276, 1))
    candidates = select_frame(reader, [Stats(30, 279, 1), Stats(30, 278, 1)])
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
    assert attack_replacement.DEFAULT_SUCCESS_DISAPPEARANCES[
        attack_replacement.CANDIDATE_ROW
    ] == (attack_replacement.SELECT_TITLE,)
    assert attack_replacement.DEFAULT_SUCCESS_TRANSITIONS[
        attack_replacement.NESTED_PARENT_YES
    ] == (SELECT_CONFIRM_PROMPT, attack_replacement.NEST_TITLE)
    assert attack_replacement.DEFAULT_SUCCESS_TRANSITIONS[CONFIRM_YES] == (
        attack_replacement.NEST_TITLE,
    )


def multi_nest_frame(
    reader: EncodedReader, nests: list[tuple[Stats, Stats]]
) -> Frame:
    """A My Nest list with one green marker bar and stat block per nest."""

    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    for index, (left, right) in enumerate(nests):
        offset = index * nest_readout.NEST_CARD_PITCH
        image[347 + offset : 603 + offset, 181:191] = (105, 211, 115)
        regions = nest_readout.shift_parent_regions(ATTACK_PARENT_REGIONS, index)
        for side_regions, stats in zip(regions, (left, right), strict=True):
            for region, value in zip(
                side_regions,
                (stats.hp, stats.attack, stats.speed),
                strict=True,
            ):
                x0, y0, x1, y1 = map(int, region)
                image[y0:y1, x0:x1] = reader.encode(value)
    return Frame(image)


def test_every_visible_nest_is_screened_in_turn() -> None:
    # S9's live layout: three attack nests, each with the same 10/1/150 left
    # parent and a near-identical right parent. Screening must walk all three
    # rather than repeating the first card.
    reader = EncodedReader()
    nests = [
        (Stats(10, 1, 150), Stats(40, 626, 150)),
        (Stats(10, 1, 150), Stats(40, 620, 150)),
        (Stats(10, 1, 150), Stats(40, 627, 150)),
    ]
    frames = multi_nest_frame(reader, nests)
    no_upgrade = select_frame(reader, [Stats(10, 1, 150)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    tapped: list[tuple[str, float]] = []
    for _ in range(6):
        assert not planner.is_complete()
        target = planner.choose(frames, nest_detections())
        assert target is not None
        tapped.append((target.type, target.y))
        planner.on_action_success(target.type)
        close = planner.choose(no_upgrade, select_detections())
        assert close is not None
        assert close.type == attack_replacement.SELECT_MASK_CLOSE
        planner.on_action_success(close.type)

    assert planner.is_complete()
    # Left/right alternates, and each pair sits one card pitch further down.
    pitch = nest_readout.NEST_CARD_PITCH
    assert tapped == [
        (attack_replacement.PARENT_LEFT, 407.0),
        (attack_replacement.PARENT_RIGHT, 407.0),
        (attack_replacement.PARENT_LEFT, 407.0 + pitch),
        (attack_replacement.PARENT_RIGHT, 407.0 + pitch),
        (attack_replacement.PARENT_LEFT, 407.0 + 2 * pitch),
        (attack_replacement.PARENT_RIGHT, 407.0 + 2 * pitch),
    ]


def test_single_visible_nest_keeps_the_original_one_nest_flow() -> None:
    reader = EncodedReader()
    frames = multi_nest_frame(reader, [(Stats(10, 1, 150), Stats(40, 626, 150))])
    no_upgrade = select_frame(reader, [Stats(10, 1, 150)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    for _ in range(2):
        target = planner.choose(frames, nest_detections())
        assert target is not None and target.y == 407.0
        planner.on_action_success(target.type)
        close = planner.choose(no_upgrade, select_detections())
        assert close is not None
        planner.on_action_success(close.type)
    assert planner.is_complete()
    # With one nest the stage names stay unprefixed, so existing log parsing
    # and diagnostics keep working unchanged.
    assert planner.last_stage() == "replacement_done"


def test_unreadable_nest_markers_screen_only_the_anchor_nest() -> None:
    # No green bars: the list may be mid-animation or covered. Screening the
    # first nest is the old behaviour; tapping cards that may not be there is
    # strictly worse than doing less.
    reader = EncodedReader()
    frames = nest_frame(reader, Stats(10, 1, 150), Stats(40, 626, 150))
    no_upgrade = select_frame(reader, [Stats(10, 1, 150)])
    planner = AttackReplacementTestPlanner(reader)  # type: ignore[arg-type]
    finish_main_filter(planner)

    for _ in range(2):
        target = planner.choose(frames, nest_detections())
        assert target is not None and target.y == 407.0
        planner.on_action_success(target.type)
        close = planner.choose(no_upgrade, select_detections())
        assert close is not None
        planner.on_action_success(close.type)
    assert planner.is_complete()
