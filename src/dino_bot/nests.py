"""Nest management (Phase B) decision rules.

Pure logic for docs/auto-hatch-plan.md §4.4: in which order the tag rounds
run, which sort option each round needs, and which dinosaur (if any) should
replace a nest parent. The screen-driving planner arrives once its templates
are captured; keeping the rules separate lets them be verified offline.

Replacement doctrine: the round's primary stat is an absolute priority — a
candidate with a higher primary always beats one with a lower primary
regardless of the other stats. When a candidate ties the parent's primary,
it is still an improvement if the other two stats have a lower total.
Cooldowns and star markers never disqualify a candidate.
"""

from __future__ import annotations

from collections import Counter
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
    """Bounds used to reject impossible OCR readings.

    ``min_value``/``max_value`` and ``multiple_of`` apply to every readout.

    ``min_delta``/``max_delta`` are accepted for backward compatibility and no
    longer decide anything. They once required a replacement candidate to beat
    its parent by a specific margin, which meant a parent on 389 attack
    declined the 390 sitting next to it, and a candidate far ahead of the
    parent was refused as an implausible reading. Any real increase is now a
    reason to swap; see ``pick_replacement``.
    """

    min_delta: int | None = None
    max_delta: int | None = None
    min_value: int | None = None
    max_value: int | None = None
    multiple_of: int | None = None

    @property
    def has_delta_bounds(self) -> bool:
        return self.min_delta is not None or self.max_delta is not None


def default_stat_upgrade_guards() -> dict[str, StatUpgradeGuard]:
    """Return the conservative stat rules used by the game today."""

    return {
        "hp": StatUpgradeGuard(multiple_of=10),
        "attack": StatUpgradeGuard(),
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
        if guard.multiple_of is not None and (
            guard.multiple_of <= 0 or value % guard.multiple_of != 0
        ):
            return False
    return True


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
EXTREME_SPECIALIZATION_PARENT = Stats(10, 1, 1)
TOP_RULE = AutoPlaceRule(tag="頂尖", sort_option="最佳屬性組合")
MASS_RULE = AutoPlaceRule(tag="量產", sort_option="等級")

# Main-account screening replaces Attack/HP parents, then uses the game's
# guarded auto-place flow for Top/Mass before collecting all eggs.
ROUND_ORDER: tuple[ReplacementRule | AutoPlaceRule, ...] = (
    ATTACK_RULE,
    HP_RULE,
    TOP_RULE,
    MASS_RULE,
)
FINAL_TAG = "所有"


def is_intentional_extreme_specialization_parent(
    parent: Stats,
    rule: ReplacementRule,
    *,
    enabled: bool,
) -> bool:
    """Whether the configured 10/1/1 parent is intentional for this round."""

    return bool(
        enabled
        and rule in (ATTACK_RULE, HP_RULE)
        and parent == EXTREME_SPECIALIZATION_PARENT
    )


def is_extreme_specialization_candidate(
    parent: Stats,
    candidate: Stats,
    rule: ReplacementRule,
) -> bool:
    """Whether a candidate preserves the two low stats of a 10/1/1 line."""

    if parent != EXTREME_SPECIALIZATION_PARENT:
        return False
    if primary_of(candidate, rule) <= primary_of(parent, rule):
        return False
    if rule == HP_RULE:
        return candidate.attack == parent.attack and candidate.speed == parent.speed
    if rule == ATTACK_RULE:
        return candidate.hp == parent.hp and candidate.speed == parent.speed
    return False


def primary_of(stats: Stats, rule: ReplacementRule) -> int:
    return int(getattr(stats, rule.primary))


def find_primary_ocr_conflict(
    parent: Stats,
    rows: list[Stats],
    rule: ReplacementRule,
    *,
    minimum_repeats: int = 2,
) -> tuple[int, int] | None:
    """Find a repeated candidate value that looks like a 1/7 OCR flip.

    This is deliberately not a growth-range check. Parent values can start at
    different levels and rise over multiple generations, so an absolute
    ceiling would reject legitimate parents. Instead, flag only the narrow
    evidence pattern seen in live logs: a parent primary value and a repeated
    candidate primary value differ in exactly one digit, and that digit is
    ``1`` versus ``7``. The caller must re-read or fail closed.
    """

    if minimum_repeats <= 0 or not rows:
        return None
    parent_value = primary_of(parent, rule)
    parent_text = str(parent_value)
    counts = Counter(primary_of(row, rule) for row in rows)
    for candidate_value, count in counts.items():
        if count < minimum_repeats:
            continue
        candidate_text = str(candidate_value)
        if len(parent_text) != len(candidate_text):
            continue
        differences = [
            (left, right)
            for left, right in zip(parent_text, candidate_text, strict=True)
            if left != right
        ]
        if len(differences) == 1 and set(differences[0]) == {"1", "7"}:
            return parent_value, candidate_value
    return None


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
    partner: Stats | None = None,
) -> int | None:
    """Pick which list row should replace ``parent``, or None to keep it.

    ``rows`` are the visible list rows top-down, already sorted descending by
    the rule's primary stat (the caller verifies via ``is_descending``). Any
    higher primary justifies a swap, by however little: a round whose whole
    purpose is to raise one stat has no reason to decline a rise in it. An
    equal primary also justifies one when its secondary load is lower than the
    parent's. Among all candidates at the best primary value, the lowest
    secondary load wins.

    ``guards`` still rejects readouts that cannot be real (``min_value``,
    ``max_value``, ``multiple_of``); it no longer requires the increase itself
    to fall in a band.

    ``partner`` is the other parent already standing in this nest. The game
    will not let one dinosaur hold both sides, so picking it produces a tap
    that does nothing at all - no confirmation, no warning - and the round
    stalls. Rows are only known by their three numbers here, so a different
    dinosaur with identical stats is skipped too: losing one swap is cheaper
    than a stall.
    """

    if not rows or not stat_value_is_valid(parent, guards):
        return None
    parent_primary = primary_of(parent, rule)
    valid_indices = [
        index
        for index, row in enumerate(rows)
        if stat_value_is_valid(row, guards)
        and (partner is None or row != partner)
        and (
            primary_of(row, rule) > parent_primary
            or (
                primary_of(row, rule) == parent_primary
                and secondary_load(row, rule) < secondary_load(parent, rule)
            )
        )
    ]
    if not valid_indices:
        return None
    return min(
        valid_indices,
        key=lambda index: (
            -primary_of(rows[index], rule),
            secondary_load(rows[index], rule),
            index,
        ),
    )
