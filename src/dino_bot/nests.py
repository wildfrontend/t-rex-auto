"""Nest management (Phase B) decision rules.

Pure logic for docs/auto-hatch-plan.md §4.4: in which order the tag rounds
run, which sort option each round needs, and which dinosaur (if any) should
replace a nest parent. The screen-driving planner arrives once its templates
are captured; keeping the rules separate lets them be verified offline.

Replacement doctrine (confirmed 2026-07-30): the round's primary stat is an
absolute priority — a candidate with a higher primary always beats one with a
lower primary regardless of the other stats. The other two stats are only a
soft tiebreaker among equal primaries (lower total preferred, no cutoff).
Cooldowns and star markers never disqualify a candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Stats:
    """The three values shown on cards and list rows, in display order."""

    hp: int
    attack: int
    speed: int


@dataclass(frozen=True, slots=True)
class StatUpgradeGuard:
    """Bounds used to reject impossible OCR readings and upgrades.

    ``min_delta``/``max_delta`` apply only when the stat is the round's
    primary stat. ``min_value``/``max_value`` and ``multiple_of`` apply to
    every readout.
    """

    min_delta: int | None = None
    max_delta: int | None = None
    min_value: int | None = None
    max_value: int | None = None
    multiple_of: int | None = None


def default_stat_upgrade_guards() -> dict[str, StatUpgradeGuard]:
    """Return the conservative stat rules used by the game today."""

    return {
        "hp": StatUpgradeGuard(min_delta=10, max_delta=30, multiple_of=10),
        "attack": StatUpgradeGuard(min_delta=1, max_delta=3),
        "speed": StatUpgradeGuard(min_value=1, max_value=150),
    }


DEFAULT_STAT_UPGRADE_GUARDS = default_stat_upgrade_guards()


def stat_value_is_valid(
    stats: Stats,
    guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> bool:
    """Whether every configured absolute stat bound accepts this readout."""

    for name, guard in guards.items():
        value = getattr(stats, name, None)
        if value is None:
            return False
        if guard.min_value is not None and value < guard.min_value:
            return False
        if guard.max_value is not None and value > guard.max_value:
            return False
        if guard.multiple_of is not None:
            if guard.multiple_of <= 0 or value % guard.multiple_of != 0:
                return False
    return True


def upgrade_is_valid(
    parent: Stats,
    candidate: Stats,
    rule: ReplacementRule,
    guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> bool:
    """Whether a candidate's primary-stat increase fits the configured guard."""

    guard = guards.get(rule.primary)
    if guard is None:
        return True
    delta = primary_of(candidate, rule) - primary_of(parent, rule)
    if guard.min_delta is not None and delta < guard.min_delta:
        return False
    return guard.max_delta is None or delta <= guard.max_delta


@dataclass(frozen=True, slots=True)
class ReplacementRule:
    """A tag round that walks every nest and swaps parents by hand."""

    tag: str
    # Exact label of the sort-dropdown option the list must be set to.
    sort_option: str
    primary: str


@dataclass(frozen=True, slots=True)
class AutoPlaceRule:
    """A tag round that only sets the auto-place sort and taps the button."""

    tag: str
    sort_option: str


ATTACK_RULE = ReplacementRule(tag="攻擊特化", sort_option="攻擊力", primary="attack")
HP_RULE = ReplacementRule(tag="HP特化", sort_option="HP", primary="hp")
TOP_RULE = AutoPlaceRule(tag="頂尖", sort_option="最佳屬性組合")
MASS_RULE = AutoPlaceRule(tag="量產", sort_option="等級")

# 攻擊特化 → HP特化 → 頂尖 → 量產, then reset the filter and collect eggs.
ROUND_ORDER: tuple[ReplacementRule | AutoPlaceRule, ...] = (
    ATTACK_RULE,
    HP_RULE,
    TOP_RULE,
    MASS_RULE,
)
FINAL_TAG = "所有"


def primary_of(stats: Stats, rule: ReplacementRule) -> int:
    return int(getattr(stats, rule.primary))


def secondary_load(stats: Stats, rule: ReplacementRule) -> int:
    """Sum of the two non-primary stats; lower is preferred among ties."""

    return stats.hp + stats.attack + stats.speed - primary_of(stats, rule)


def is_descending(first: int, second: int) -> bool:
    """Whether the top two list rows are consistent with high-to-low sorting.

    The sort arrow's direction cannot be judged from its looks, so the caller
    reads the first two rows and flips the arrow when this returns False.
    """

    return first >= second


def descending_prefix(rows: list[Stats], rule: ReplacementRule) -> list[Stats]:
    """Keep the reliable high-to-low prefix of a sorted candidate list.

    A single digit can occasionally be confused by the screenshot reader
    (for example, 2310 as 2370).  Once a later row appears stronger than the
    row above it, the remaining OCR output is unsafe for replacement logic.
    """

    if not rows:
        return []
    result = [rows[0]]
    for row in rows[1:]:
        if primary_of(row, rule) > primary_of(result[-1], rule):
            break
        result.append(row)
    return result


def pick_replacement(
    parent: Stats,
    rows: list[Stats],
    rule: ReplacementRule,
    *,
    guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
) -> int | None:
    """Pick which list row should replace ``parent``, or None to keep it.

    ``rows`` are the visible list rows top-down, already sorted descending by
    the rule's primary stat (the caller verifies via ``is_descending``). Only
    a strictly higher primary justifies a swap; among rows tied at the top
    value the lowest secondary load wins.
    """

    if not rows:
        return None
    parent_primary = primary_of(parent, rule)
    valid_indices = [
        index
        for index, row in enumerate(rows)
        if primary_of(row, rule) > parent_primary
        and upgrade_is_valid(parent, row, rule, guards)
    ]
    if not valid_indices:
        return None
    best_index = valid_indices[0]
    top = primary_of(rows[best_index], rule)
    for index in valid_indices[1:]:
        if primary_of(rows[index], rule) != top:
            break
        if secondary_load(rows[index], rule) < secondary_load(rows[best_index], rule):
            best_index = index
    return best_index
