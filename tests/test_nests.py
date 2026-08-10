from __future__ import annotations

from dino_bot import nests
from dino_bot.nests import (
    ATTACK_RULE,
    HP_RULE,
    MASS_RULE,
    TOP_RULE,
    Stats,
    descending_prefix,
    is_descending,
    pick_replacement,
    secondary_load,
)


def test_round_order_matches_plan() -> None:
    assert [rule.tag for rule in nests.ROUND_ORDER] == ["攻擊特化", "HP特化", "頂尖", "量產"]
    assert nests.FINAL_TAG == "所有"


def test_autoplace_sort_options() -> None:
    assert TOP_RULE.sort_option == "最佳屬性組合"
    assert MASS_RULE.sort_option == "等級"


def test_replacement_sort_options() -> None:
    assert ATTACK_RULE.sort_option == "攻擊力"
    assert ATTACK_RULE.primary == "attack"
    assert HP_RULE.sort_option == "HP"
    assert HP_RULE.primary == "hp"


def test_secondary_load_excludes_primary() -> None:
    assert secondary_load(Stats(30, 276, 1), ATTACK_RULE) == 31
    assert secondary_load(Stats(2230, 2, 1), HP_RULE) == 3


def test_is_descending() -> None:
    assert is_descending(276, 270)
    assert is_descending(276, 276)
    assert not is_descending(268, 276)


def test_empty_list_keeps_parent() -> None:
    assert pick_replacement(Stats(30, 276, 1), [], ATTACK_RULE) is None


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
    # 攻擊絕對優先:50/277/150 勝過持有 2160/268/1 的想像替代品。
    parent = Stats(30, 276, 1)
    rows = [Stats(50, 277, 150), Stats(2160, 268, 1)]
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
    rows = [Stats(2250, 500, 50), Stats(2250, 2, 1), Stats(2240, 1, 1)]
    assert pick_replacement(parent, rows, HP_RULE) == 1


def test_stat_upgrade_guard_rejects_attack_jump_above_three() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(30, 280, 1)]
    assert pick_replacement(parent, rows, ATTACK_RULE) is None


def test_stat_upgrade_guard_skips_invalid_top_row_for_valid_lower_row() -> None:
    parent = Stats(30, 276, 1)
    rows = [Stats(30, 280, 1), Stats(30, 279, 1)]
    assert pick_replacement(parent, rows, ATTACK_RULE) == 1


def test_stat_upgrade_guard_accepts_hp_boundaries() -> None:
    parent = Stats(2230, 2, 1)
    rows = [Stats(2260, 2, 1), Stats(2240, 2, 1)]
    assert pick_replacement(parent, rows, HP_RULE) == 0


def test_descending_prefix_rejects_late_ocr_value_that_rises() -> None:
    rows = [
        Stats(2340, 3, 1),
        Stats(2320, 2, 1),
        Stats(2310, 282, 1),
        Stats(2370, 282, 1),
        Stats(2300, 288, 1),
    ]
    assert descending_prefix(rows, HP_RULE) == rows[:3]
