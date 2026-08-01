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

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Stats:
    """The three values shown on cards and list rows, in display order."""

    hp: int
    attack: int
    speed: int


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
# 排序選項的實際名稱待截圖確認(推測「生命」),見計畫 §8 2026-07-30 HP特化條目。
HP_RULE = ReplacementRule(tag="HP特化", sort_option="生命", primary="hp")
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


def pick_replacement(parent: Stats, rows: list[Stats], rule: ReplacementRule) -> int | None:
    """Pick which list row should replace ``parent``, or None to keep it.

    ``rows`` are the visible list rows top-down, already sorted descending by
    the rule's primary stat (the caller verifies via ``is_descending``). Only
    a strictly higher primary justifies a swap; among rows tied at the top
    value the lowest secondary load wins.
    """

    if not rows:
        return None
    top = primary_of(rows[0], rule)
    if top <= primary_of(parent, rule):
        return None
    best_index = 0
    for index in range(1, len(rows)):
        if primary_of(rows[index], rule) != top:
            break
        if secondary_load(rows[index], rule) < secondary_load(rows[best_index], rule):
            best_index = index
    return best_index
