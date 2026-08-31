from __future__ import annotations

from dino_bot import nests
from dino_bot.nests import (
    ATTACK_RULE,
    EXTREME_SPECIALIZATION_PARENT,
    HP_RULE,
    MASS_RULE,
    TOP_RULE,
    Stats,
    descending_prefix,
    find_primary_ocr_conflict,
    is_descending,
    is_extreme_specialization_candidate,
    is_intentional_extreme_specialization_parent,
    pick_replacement,
    secondary_load,
)


def test_round_order_matches_plan() -> None:
    assert [rule.tag for rule in nests.ROUND_ORDER] == [
        "攻擊特化",
        "HP特化",
        "頂尖",
        "量產",
    ]
    assert nests.FINAL_TAG == "所有"


def test_autoplace_sort_options() -> None:
    assert TOP_RULE.sort_option == "最佳屬性組合"
    assert MASS_RULE.sort_option == "等級"


def test_replacement_sort_options() -> None:
    assert ATTACK_RULE.sort_option == "攻擊力"
    assert ATTACK_RULE.primary == "attack"
    assert HP_RULE.sort_option == "HP"
    assert HP_RULE.primary == "hp"


def test_primary_ocr_conflict_flags_repeated_one_seven_flip() -> None:
    parent = Stats(970, 716, 115)
    rows = [
        Stats(970, 116, 115),
        Stats(970, 116, 115),
        Stats(970, 115, 115),
    ]

    assert find_primary_ocr_conflict(parent, rows, ATTACK_RULE) == (716, 116)


def test_primary_ocr_conflict_does_not_apply_growth_or_other_digit_rules() -> None:
    parent = Stats(970, 716, 115)

    assert find_primary_ocr_conflict(
        parent,
        [Stats(970, 616, 115), Stats(970, 616, 115)],
        ATTACK_RULE,
    ) is None


def test_extreme_specialization_parent_is_opt_in() -> None:
    assert Stats(10, 1, 1) == EXTREME_SPECIALIZATION_PARENT
    assert is_intentional_extreme_specialization_parent(
        EXTREME_SPECIALIZATION_PARENT, HP_RULE, enabled=True
    )
    assert is_intentional_extreme_specialization_parent(
        EXTREME_SPECIALIZATION_PARENT, ATTACK_RULE, enabled=True
    )
    assert not is_intentional_extreme_specialization_parent(
        EXTREME_SPECIALIZATION_PARENT, HP_RULE, enabled=False
    )


def test_extreme_specialization_candidate_has_one_high_and_two_low_stats() -> None:
    parent = EXTREME_SPECIALIZATION_PARENT
    assert is_extreme_specialization_candidate(parent, Stats(1300, 1, 1), HP_RULE)
    assert not is_extreme_specialization_candidate(parent, Stats(1300, 68, 20), HP_RULE)
    assert is_extreme_specialization_candidate(parent, Stats(10, 68, 1), ATTACK_RULE)
    assert not is_extreme_specialization_candidate(parent, Stats(20, 68, 1), ATTACK_RULE)
    assert find_primary_ocr_conflict(
        parent,
        [Stats(970, 116, 115)],
        ATTACK_RULE,
    ) is None


def test_secondary_load_excludes_primary() -> None:
    assert secondary_load(Stats(30, 276, 1), ATTACK_RULE) == 31
    assert secondary_load(Stats(2230, 2, 1), HP_RULE) == 3


def test_is_descending() -> None:
    assert is_descending(276, 270)
    assert is_descending(276, 276)
    assert not is_descending(268, 276)


def test_empty_list_keeps_parent() -> None:
    assert pick_replacement(Stats(30, 276, 1), [], ATTACK_RULE) is None


def test_the_other_parent_in_the_nest_is_never_offered_as_a_replacement() -> None:
    # 巢中另一隻親代出現在候選清單時,遊戲不會讓牠同時佔兩個位置:點下去毫無
    # 回應,沒有確認框也沒有警告,整輪就卡在驗證失敗。
    parent = Stats(760, 77, 85)
    partner = Stats(670, 89, 79)
    rows = [partner, Stats(670, 87, 79)]

    assert pick_replacement(parent, rows, ATTACK_RULE) == 0
    assert pick_replacement(parent, rows, ATTACK_RULE, partner=partner) == 1


def test_partner_exclusion_can_leave_no_candidate_at_all() -> None:
    parent = Stats(760, 77, 85)
    partner = Stats(670, 89, 79)

    assert pick_replacement(parent, [partner], ATTACK_RULE, partner=partner) is None


def test_equal_primary_without_lower_secondaries_keeps_parent() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(2160, 276, 1), Stats(50, 270, 150)]
    assert pick_replacement(parent, rows, ATTACK_RULE) is None


def test_equal_hp_prefers_lower_other_stats() -> None:
    parent = Stats(1000, 3, 1)
    rows = [Stats(1000, 1, 1)]

    assert pick_replacement(parent, rows, HP_RULE) == 0


def test_equal_attack_prefers_lower_other_stats() -> None:
    parent = Stats(30, 1000, 3)
    rows = [Stats(10, 1000, 1)]

    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_higher_primary_wins_despite_high_secondaries() -> None:
    # 攻擊絕對優先:50/279/150 勝過持有 2160/268/1 的想像替代品。
    parent = Stats(30, 276, 1)
    rows = [Stats(50, 279, 150), Stats(2160, 268, 1)]
    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_tie_at_top_prefers_lowest_secondary_load() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(2160, 279, 5), Stats(30, 279, 2), Stats(50, 278, 1)]
    assert pick_replacement(parent, rows, ATTACK_RULE) == 1


def test_tie_scan_stops_at_first_lower_primary() -> None:
    # 279 之後的列不再檢查——即使 secondary 更低也不能贏過較高攻擊。
    parent = Stats(30, 276, 1)
    rows = [Stats(100, 279, 50), Stats(10, 278, 1)]
    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_hp_rule_swaps_on_hp() -> None:
    parent = Stats(2230, 2, 1)
    rows = [Stats(2300, 500, 50), Stats(2300, 2, 1), Stats(2260, 1, 1)]
    assert pick_replacement(parent, rows, HP_RULE) == 1


def test_smallest_possible_increase_still_replaces_the_parent() -> None:
    # Observed 2026-08-12: a parent on 389 attack sat next to a 390 and kept
    # itself, because a replacement had to win by at least the mutation-sized
    # margin. A round that exists to raise one stat cannot decline a rise.
    parent = Stats(10, 389, 1)
    rows = [
        Stats(10, 390, 1),
        Stats(10, 389, 1),
        Stats(10, 389, 1),
        Stats(10, 384, 4),
        Stats(3360, 377, 1),
    ]

    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_a_large_increase_is_not_treated_as_an_impossible_reading() -> None:
    # The same band had an upper edge, so the better a candidate was the more
    # likely it was skipped. Absolute bounds catch unreadable rows; the size
    # of the improvement is not evidence against it.
    parent = Stats(30, 276, 1)

    assert pick_replacement(parent, [Stats(30, 284, 1)], ATTACK_RULE) == 0
    assert pick_replacement(parent, [Stats(30, 400, 1)], ATTACK_RULE) == 0
    assert pick_replacement(Stats(2230, 2, 1), [Stats(9000, 2, 1)], HP_RULE) == 0


def test_best_increase_wins_over_a_smaller_one() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(30, 284, 1), Stats(30, 283, 1)]

    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_equal_best_primary_falls_back_to_row_order() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(30, 283, 1), Stats(30, 283, 1)]

    assert pick_replacement(parent, rows, ATTACK_RULE) == 0


def test_parent_outside_absolute_guard_keeps_parent_without_blocking_readout() -> None:
    parent = Stats(2926, 3, 1)
    rows = [Stats(2960, 3, 1)]

    assert pick_replacement(parent, rows, HP_RULE) is None


def test_candidate_outside_absolute_guard_is_skipped_for_valid_lower_row() -> None:
    parent = Stats(2920, 3, 1)
    rows = [Stats(2990, 3, 0), Stats(2960, 3, 1)]

    assert pick_replacement(parent, rows, HP_RULE) == 1


def test_descending_prefix_rejects_late_ocr_value_that_rises() -> None:
    rows = [
        Stats(2340, 3, 1),
        Stats(2320, 2, 1),
        Stats(2310, 282, 1),
        Stats(2370, 282, 1),
        Stats(2300, 288, 1),
    ]
    assert descending_prefix(rows, HP_RULE) == rows[:3]
