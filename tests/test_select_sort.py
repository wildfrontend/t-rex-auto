from __future__ import annotations

import numpy as np

from dino_bot import select_sort
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.nest_readout import SELECT_FIRST_ROW_REGIONS, SELECT_ROW_PITCH
from dino_bot.select_sort import SelectSortTestPlanner


class EncodedReader:
    def __init__(self, values: dict[int, int]) -> None:
        self.values = values

    def read_int(self, image: np.ndarray) -> int | None:
        return self.values.get(int(image[0, 0, 0])) if image.size else None


def make_frame(attacks: tuple[int, ...] = ()) -> tuple[Frame, EncodedReader]:
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    codes = {value: index + 10 for index, value in enumerate(dict.fromkeys(attacks))}
    for index, attack in enumerate(attacks):
        regions = tuple(
            (x0, y0 + index * SELECT_ROW_PITCH, x1, y1 + index * SELECT_ROW_PITCH)
            for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
        )
        for stat_index, (region, value) in enumerate(
            zip(regions, (10, attack, 1), strict=True)
        ):
            code = 2 if stat_index == 0 else codes.get(value, 1)
            x0, y0, x1, y1 = map(int, region)
            image[y0:y1, x0:x1] = code
    reader_values = {1: 1, 2: 10, **{code: value for value, code in codes.items()}}
    return Frame(image), EncodedReader(reader_values)


def detection(target_type: str, x: int, y: int) -> Detection:
    return Detection.from_bbox(target_type, BoundingBox(x - 5, y - 5, 10, 10), 0.99)


def select_screen(*items: Detection) -> list[Detection]:
    return [detection(select_sort.SELECT_TITLE, 451, 299), *items]


def planner_for(attacks: tuple[int, ...]) -> tuple[SelectSortTestPlanner, Frame]:
    frame, reader = make_frame(attacks)
    return SelectSortTestPlanner(reader), frame  # type: ignore[arg-type]


def hp_planner_for(values: tuple[int, ...]) -> tuple[SelectSortTestPlanner, Frame]:
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    codes = {value: index + 10 for index, value in enumerate(dict.fromkeys(values))}
    for index, hp in enumerate(values):
        regions = tuple(
            (x0, y0 + index * SELECT_ROW_PITCH, x1, y1 + index * SELECT_ROW_PITCH)
            for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
        )
        for region, value in zip(regions, (hp, 1, 1), strict=True):
            code = codes.get(value, 1)
            x0, y0, x1, y1 = map(int, region)
            image[y0:y1, x0:x1] = code
    reader = EncodedReader({1: 1, **{code: value for value, code in codes.items()}})
    planner = SelectSortTestPlanner(
        reader,  # type: ignore[arg-type]
        sort_option_type=select_sort.SORT_HP,
        sort_header_type=select_sort.SORT_HP,
        sort_menu_point=(650.0, 501.0),
        primary_attr="hp",
        sort_label="HP",
    )
    return planner, Frame(image)


def test_wrong_screen_never_taps() -> None:
    planner, frame = planner_for((276, 270))
    assert planner.choose(frame, [detection(select_sort.TAG_HDR_ALL, 228, 204)]) is None
    assert planner.last_stage() == "waiting_for_select_dino"


def test_foreground_all_header_ignores_background_tag_header() -> None:
    planner, frame = planner_for((276, 270))
    detections = select_screen(
        detection("hatch_tag_hdr_attack", 218, 166),
        detection(select_sort.TAG_HDR_ALL, 228, 204),
        detection(select_sort.SORT_HDR_ATTACK, 637, 355),
    )
    assert planner.choose(frame, detections) is None
    assert planner.is_complete()


def test_unknown_tag_header_opens_foreground_filter() -> None:
    planner, frame = planner_for((276, 270))
    target = planner.choose(frame, select_screen())
    assert target is not None
    assert target.type == select_sort.TAG_HEADER
    assert (target.x, target.y) == (228, 204)


def test_expanded_tag_menu_selects_all() -> None:
    planner, frame = planner_for((276, 270))
    target = planner.choose(
        frame,
        select_screen(detection(select_sort.TAG_ALL, 228, 249)),
    )
    assert target is not None and target.type == select_sort.TAG_ALL


def test_unknown_sort_header_opens_sort_menu_after_all_tag() -> None:
    planner, frame = planner_for((276, 270))
    target = planner.choose(
        frame,
        select_screen(detection(select_sort.TAG_HDR_ALL, 228, 204)),
    )
    assert target is not None
    assert target.type == select_sort.SORT_HEADER
    assert (target.x, target.y) == (649, 355)


def test_expanded_sort_menu_selects_attack() -> None:
    planner, frame = planner_for((276, 270))
    target = planner.choose(
        frame,
        select_screen(
            detection(select_sort.TAG_HDR_ALL, 228, 204),
            detection(select_sort.SORT_ATTACK, 649, 550),
        ),
    )
    assert target is not None and target.type == select_sort.SORT_ATTACK


def test_ascending_attack_values_toggle_direction_once() -> None:
    planner, frame = planner_for((2, 5, 270))
    detections = select_screen(
        detection(select_sort.TAG_HDR_ALL, 228, 204),
        detection(select_sort.SORT_HDR_ATTACK, 637, 355),
    )
    target = planner.choose(frame, detections)
    assert target is not None
    assert target.type == select_sort.SORT_DIRECTION
    planner.on_action_success(select_sort.SORT_DIRECTION)
    target = planner.choose(frame, detections)
    assert target is None
    assert planner.last_stage() == "direction_failed"


def test_equal_values_pass_as_already_sorted() -> None:
    # 同值高原上任何順序都成立;舊的 fail-closed 會讓 full-hatch
    # 篩選在恢復與重試之間無限循環(2026-08-03 生產實例:攻擊全
    # 296)。同值中任選皆不會降低主屬性；替換階段會再比較副屬性。
    planner, frame = planner_for((276, 276, 276))
    target = planner.choose(
        frame,
        select_screen(
            detection(select_sort.TAG_HDR_ALL, 228, 204),
            detection(select_sort.SORT_HDR_ATTACK, 637, 355),
        ),
    )
    assert target is None
    assert planner.is_complete()


def test_non_monotonic_ocr_tail_does_not_change_descending_direction() -> None:
    planner, frame = hp_planner_for((2340, 2320, 2310, 2370, 2300))
    planner._tag_ready = True
    planner._sort_ready = True
    assert planner.choose(frame, select_screen()) is None
    assert planner.is_complete()


def test_single_hp_ocr_inversion_does_not_reverse_descending_list() -> None:
    # S13 can consistently read the second visible 860 as 890. The remaining
    # rows still provide stronger evidence that the on-screen list descends.
    planner, frame = hp_planner_for((880, 890, 860, 840, 840, 830, 810, 810))
    planner._tag_ready = True
    planner._sort_ready = True

    assert planner.choose(frame, select_screen()) is None
    assert planner.is_complete()


def test_tied_direction_vote_accepts_the_list_instead_of_stalling() -> None:
    # Observed on s9: the second 5760 read as 5790, giving one ascending and
    # one descending vote. A tie used to return "unknown", and because the
    # screen is static every later frame reproduced it - the round only
    # escaped through a 30s stall recovery that re-entered the same nest.
    planner, frame = hp_planner_for(
        (5770, 5790, 5790, 5710, 5710, 5710, 5710, 5710, 5710)
    )
    planner._tag_ready = True
    planner._sort_ready = True

    assert planner.choose(frame, select_screen()) is None
    assert planner.is_complete()


def test_genuinely_ascending_list_still_flips_the_arrow() -> None:
    planner, frame = hp_planner_for((5710, 5760, 5770, 5790))
    planner._tag_ready = True
    planner._sort_ready = True

    target = planner.choose(frame, select_screen())
    assert target is not None
    assert target.type == select_sort.SORT_DIRECTION
    assert not planner.is_complete()


def test_hp_sort_uses_menu_option_then_proves_result_from_hp_values() -> None:
    planner, frame = hp_planner_for((2300, 2250, 2200))
    option = planner.choose(
        frame,
        select_screen(
            detection(select_sort.TAG_HDR_ALL, 228, 204),
            detection(select_sort.SORT_HP, 650, 501),
        ),
    )
    assert option is not None and option.type == select_sort.SORT_HP
    assert (option.x, option.y) == (650, 501)
    planner.on_action_success(option.type)

    assert planner.choose(
        frame,
        select_screen(
            detection(select_sort.TAG_HDR_ALL, 228, 204),
        ),
    ) is None
    assert planner.is_complete()


def test_hp_sort_verification_returns_to_select_screen_before_value_proof() -> None:
    assert select_sort.DEFAULT_SUCCESS_TRANSITIONS[select_sort.SORT_HP] == (
        select_sort.SELECT_TITLE,
    )


def test_no_dinosaur_row_can_become_an_action() -> None:
    assert set(select_sort.DEFAULT_TARGET_ACTIONS) == {
        select_sort.TAG_HEADER,
        select_sort.TAG_ALL,
        select_sort.SORT_HEADER,
        select_sort.SORT_ATTACK,
        select_sort.SORT_HP,
        select_sort.SORT_DIRECTION,
    }
