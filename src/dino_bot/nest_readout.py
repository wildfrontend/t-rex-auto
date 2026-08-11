"""Coordinate-anchored stat readout for the read-only T10 rehearsal.

The My Nest card and Select Dino rows are fixed panels in 900-wide reference
space.  This module crops only the numeric ink beside each stat icon, feeds it
to :class:`DigitReader`, and combines the result with the replacement rules
without ever returning a screen tap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .digits import DigitReader
from .models import Image
from .nests import (
    ATTACK_RULE,
    DEFAULT_STAT_UPGRADE_GUARDS,
    Stats,
    StatUpgradeGuard,
    pick_replacement,
    stat_value_is_valid,
)

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


def _normalize_speed_overflow(value: int, guard: StatUpgradeGuard | None) -> int:
    """Cap impossible speed OCR instead of blocking the screening round.

    Speed is never the primary stat in the manual Attack/HP replacement
    rounds.  The game also caps its base value, so an OCR result above the
    configured maximum is evidence of a digit mismatch (for example the live
    ``150`` -> ``750`` case), not a useful reason to stop parent screening.
    Keeping the value at the configured ceiling also prevents the bad OCR
    from dominating equal-primary secondary-stat comparisons.
    """

    if guard is not None and guard.max_value is not None and value > guard.max_value:
        return guard.max_value
    return value


class ConsecutiveReadConsensus[Readout]:
    """Accept a readout only after it repeats across consecutive frames."""

    def __init__(self, minimum_reads: int = 1) -> None:
        if minimum_reads <= 0:
            raise ValueError("minimum_reads must be greater than zero")
        self.minimum_reads = minimum_reads
        self._pending: Readout | None = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def observe(self, value: Readout | None) -> Readout | None:
        if value is None:
            self.reset()
            return None
        if value != self._pending:
            self._pending = value
            self._count = 1
        else:
            self._count += 1
        return value if self._count >= self.minimum_reads else None

    def reset(self) -> None:
        self._pending = None
        self._count = 0


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
    stat_guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> Stats | None:
    """Read numeric stats without turning guard anomalies into workflow stalls.

    A non-numeric OCR result is still unreadable.  Numeric values outside the
    configured guards are returned so the caller can finish the current
    parent/candidate pass; replacement selection separately rejects any
    untrusted values.
    """

    if reference_width <= 0:
        raise ValueError("reference_width must be greater than zero")
    scale = image.shape[1] / reference_width
    values: list[int] = []
    for name, region in zip(("hp", "attack", "speed"), regions, strict=True):
        x0, y0, x1, y1 = (round(value * scale) for value in region)
        value = reader.read_int(image[y0:y1, x0:x1])
        if value is None:
            return None
        if name == "speed":
            normalized = _normalize_speed_overflow(value, stat_guards.get(name))
            if normalized != value:
                logger = getattr(reader, "logger", None)
                if logger is not None:
                    logger.warning(
                        "Hatch OCR | speed reading %d exceeds configured max %d"
                        " | using max instead of blocking",
                        value,
                        normalized,
                    )
                value = normalized
        values.append(value)
    stats = Stats(hp=values[0], attack=values[1], speed=values[2])
    if not stat_value_is_valid(stats, stat_guards):
        logger = getattr(reader, "logger", None)
        if logger is not None:
            logger.warning(
                "Hatch OCR | numeric stats outside configured guards"
                " | hp=%d attack=%d speed=%d | continuing without unsafe replacement",
                stats.hp,
                stats.attack,
                stats.speed,
            )
    return stats


def read_attack_parents(
    image: Image,
    reader: DigitReader,
    *,
    stat_guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> tuple[Stats, Stats] | None:
    parents = tuple(
        read_stats(image, reader, regions, stat_guards=stat_guards)
        for regions in ATTACK_PARENT_REGIONS
    )
    if any(parent is None for parent in parents):
        return None
    return parents  # type: ignore[return-value]


def read_candidate_rows(
    image: Image,
    reader: DigitReader,
    *,
    max_rows: int = DEFAULT_VISIBLE_ROWS,
    stat_guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> list[Stats]:
    """Read consecutive visible rows; stop at the first blank/unclear row."""

    rows: list[Stats] = []
    for index in range(max(0, max_rows)):
        shifted = tuple(
            (x0, y0 + index * SELECT_ROW_PITCH, x1, y1 + index * SELECT_ROW_PITCH)
            for x0, y0, x1, y1 in SELECT_FIRST_ROW_REGIONS
        )
        stats = read_stats(
            image,
            reader,
            shifted,  # type: ignore[arg-type]
            stat_guards=stat_guards,
        )
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
    stat_guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> ReplacementSuggestion | None:
    """Return the T10 recommendation, never an actionable screen coordinate."""

    if parent_side not in (0, 1):
        raise ValueError("parent_side must be 0 (left) or 1 (right)")
    parents = read_attack_parents(parent_image, reader, stat_guards=stat_guards)
    rows = read_candidate_rows(candidate_image, reader, stat_guards=stat_guards)
    if parents is None or not rows:
        return None
    parent = parents[parent_side]
    return ReplacementSuggestion(
        parent=parent,
        rows=tuple(rows),
        replacement_index=pick_replacement(
            parent,
            rows,
            ATTACK_RULE,
            guards=stat_guards,
        ),
    )
