import numpy as np

from dino_bot.nest_readout import (
    ATTACK_PARENT_REGIONS,
    SELECT_FIRST_ROW_REGIONS,
    SELECT_ROW_PITCH,
    read_attack_parents,
    read_candidate_rows,
    rehearse_attack_replacement,
)
from dino_bot.nests import Stats


class EncodedReader:
    def __init__(self, values: dict[int, int]) -> None:
        self.values = values

    def read_int(self, image: np.ndarray) -> int | None:
        return self.values.get(int(image[0, 0, 0])) if image.size else None


def fill_regions(
    frame: np.ndarray,
    regions: tuple[tuple[float, float, float, float], ...],
    codes: tuple[int, int, int],
) -> None:
    for region, code in zip(regions, codes, strict=True):
        x0, y0, x1, y1 = map(int, region)
        frame[y0:y1, x0:x1] = code


def test_reads_both_parent_sides_and_candidate_rows() -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    reader = EncodedReader({10: 30, 20: 282, 30: 1, 40: 2230, 50: 276, 60: 150})
    fill_regions(frame, ATTACK_PARENT_REGIONS[0], (10, 20, 30))
    fill_regions(frame, ATTACK_PARENT_REGIONS[1], (40, 50, 60))
    fill_regions(frame, SELECT_FIRST_ROW_REGIONS, (10, 20, 30))
    second = tuple(
        (x0, y0 + SELECT_ROW_PITCH, x1, y1 + SELECT_ROW_PITCH)
        for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
    )
    fill_regions(frame, second, (40, 50, 60))

    assert read_attack_parents(frame, reader) == (Stats(30, 282, 1), Stats(2230, 276, 150))
    assert read_candidate_rows(frame, reader) == [Stats(30, 282, 1), Stats(2230, 276, 150)]


def test_rejects_speed_reading_outside_game_range() -> None:
    for speed in (0, 151):
        frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
        reader = EncodedReader(
            {10: 30, 20: 282, 30: speed, 40: 2230, 50: 276, 60: 1}
        )
        fill_regions(frame, ATTACK_PARENT_REGIONS[0], (10, 20, 30))
        fill_regions(frame, ATTACK_PARENT_REGIONS[1], (40, 50, 60))

        assert read_attack_parents(frame, reader) is None


def test_rejects_hp_that_is_not_a_multiple_of_ten() -> None:
    frame = np.full((1600, 900, 3), 255, dtype=np.uint8)
    reader = EncodedReader({10: 2920, 20: 3, 30: 1, 40: 2926})
    fill_regions(frame, ATTACK_PARENT_REGIONS[0], (10, 20, 30))
    fill_regions(frame, ATTACK_PARENT_REGIONS[1], (40, 20, 30))

    assert read_attack_parents(frame, reader) is None


def test_left_parent_hp_crop_keeps_the_complete_leading_digit() -> None:
    # Regression for the live 2320 -> 7320 misread: the old x0=252 cut the
    # first digit down to four pixels, while the glyph starts around x=249.
    assert ATTACK_PARENT_REGIONS[0][0] == (244, 478, 302, 500)


def test_equal_top_attack_recommends_no_replacement() -> None:
    parent = np.full((1600, 900, 3), 255, dtype=np.uint8)
    candidates = parent.copy()
    reader = EncodedReader({10: 30, 20: 282, 30: 1})
    for regions in ATTACK_PARENT_REGIONS:
        fill_regions(parent, regions, (10, 20, 30))
    fill_regions(candidates, SELECT_FIRST_ROW_REGIONS, (10, 20, 30))

    suggestion = rehearse_attack_replacement(parent, candidates, reader)
    assert suggestion is not None
    assert suggestion.parent == Stats(30, 282, 1)
    assert suggestion.replacement_index is None


def test_higher_attack_recommends_first_row_without_a_screen_coordinate() -> None:
    parent = np.full((1600, 900, 3), 255, dtype=np.uint8)
    candidates = parent.copy()
    reader = EncodedReader({10: 30, 20: 282, 21: 285, 30: 1})
    for regions in ATTACK_PARENT_REGIONS:
        fill_regions(parent, regions, (10, 20, 30))
    fill_regions(candidates, SELECT_FIRST_ROW_REGIONS, (10, 21, 30))

    suggestion = rehearse_attack_replacement(parent, candidates, reader)
    assert suggestion is not None
    assert suggestion.replacement_index == 0
