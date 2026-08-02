"""Coordinate-anchored stat readout for the read-only T10 rehearsal.

The My Nest card and Select Dino rows are fixed panels in 900-wide reference
space.  This module crops only the numeric ink beside each stat icon, feeds it
to :class:`DigitReader`, and combines the result with the replacement rules
without ever returning a screen tap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .digits import DigitReader
from .models import Image
from .nests import ATTACK_RULE, Stats, pick_replacement

Region = tuple[float, float, float, float]
StatRegions = tuple[Region, Region, Region]

ATTACK_PARENT_REGIONS: tuple[StatRegions, StatRegions] = (
    (
        # The left HP number begins as far left as x=249.  Starting at x=252
        # clipped the leading curve of 2, making 2320 look like 7320.
        (244, 478, 302, 500),
        (252, 506, 302, 528),
        (252, 533, 290, 553),
    ),
    (
        (510, 478, 565, 500),
        (510, 506, 565, 528),
        (510, 533, 548, 553),
    ),
)

SELECT_FIRST_ROW_REGIONS: StatRegions = (
    (304, 441, 365, 463),
    (399, 441, 449, 463),
    (476, 441, 520, 463),
)
SELECT_ROW_PITCH = 85
DEFAULT_VISIBLE_ROWS = 9


@dataclass(frozen=True, slots=True)
class ReplacementSuggestion:
    parent: Stats
    rows: tuple[Stats, ...]
    replacement_index: int | None


def read_stats(
    image: Image,
    reader: DigitReader,
    regions: StatRegions,
    *,
    reference_width: float = 900.0,
) -> Stats | None:
    """Read HP/attack/speed regions, failing closed if any one is unclear."""

    if reference_width <= 0:
        raise ValueError("reference_width must be greater than zero")
    scale = image.shape[1] / reference_width
    values: list[int] = []
    for region in regions:
        x0, y0, x1, y1 = (round(value * scale) for value in region)
        value = reader.read_int(image[y0:y1, x0:x1])
        if value is None:
            return None
        values.append(value)
    return Stats(hp=values[0], attack=values[1], speed=values[2])


def read_attack_parents(image: Image, reader: DigitReader) -> tuple[Stats, Stats] | None:
    parents = tuple(read_stats(image, reader, regions) for regions in ATTACK_PARENT_REGIONS)
    if any(parent is None for parent in parents):
        return None
    return parents  # type: ignore[return-value]


def read_candidate_rows(
    image: Image,
    reader: DigitReader,
    *,
    max_rows: int = DEFAULT_VISIBLE_ROWS,
) -> list[Stats]:
    """Read consecutive visible rows; stop at the first blank/unclear row."""

    rows: list[Stats] = []
    for index in range(max(0, max_rows)):
        shifted = tuple(
            (x0, y0 + index * SELECT_ROW_PITCH, x1, y1 + index * SELECT_ROW_PITCH)
            for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
        )
        stats = read_stats(image, reader, shifted)  # type: ignore[arg-type]
        if stats is None:
            break
        rows.append(stats)
    return rows


def rehearse_attack_replacement(
    parent_image: Image,
    candidate_image: Image,
    reader: DigitReader,
    *,
    parent_side: int = 0,
) -> ReplacementSuggestion | None:
    """Return the T10 recommendation, never an actionable screen coordinate."""

    if parent_side not in (0, 1):
        raise ValueError("parent_side must be 0 (left) or 1 (right)")
    parents = read_attack_parents(parent_image, reader)
    rows = read_candidate_rows(candidate_image, reader)
    if parents is None or not rows:
        return None
    parent = parents[parent_side]
    return ReplacementSuggestion(
        parent=parent,
        rows=tuple(rows),
        replacement_index=pick_replacement(parent, rows, ATTACK_RULE),
    )
