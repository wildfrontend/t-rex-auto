from __future__ import annotations

from dino_bot.dropdowns import DONE, OPEN, SELECT, STUCK, Sighting, Step, converged, next_step

HEADER = (120, 240)


def test_collapsed_on_target_is_done() -> None:
    sightings = [Sighting("攻擊特化", 118, 242, in_header=True)]
    assert next_step("攻擊特化", sightings, HEADER) == Step(DONE)
    assert converged("攻擊特化", sightings)


def test_collapsed_on_other_label_opens_menu() -> None:
    sightings = [Sighting("所有", 118, 242, in_header=True)]
    step = next_step("攻擊特化", sightings, HEADER)
    assert step.kind == OPEN
    assert step.point == HEADER


def test_unrecognized_header_opens_menu() -> None:
    # 上次停在「速度特化」之類沒有模板的標籤:表頭認不出來,唯一正解是打開選單。
    step = next_step("攻擊特化", [], HEADER)
    assert step.kind == OPEN
    assert step.point == HEADER


def test_expanded_menu_selects_target() -> None:
    sightings = [
        Sighting("所有", 118, 242, in_header=True),
        Sighting("所有", 130, 350, in_header=False),
        Sighting("攻擊特化", 130, 610, in_header=False),
    ]
    step = next_step("攻擊特化", sightings, HEADER)
    assert step == Step(SELECT, (130, 610))


def test_expanded_menu_without_target_is_stuck() -> None:
    sightings = [Sighting("所有", 130, 350, in_header=False)]
    step = next_step("攻擊特化", sightings, HEADER)
    assert step.kind == STUCK
    assert "所有" in step.reason


def test_expanded_menu_is_not_converged_even_with_header_hit() -> None:
    sightings = [
        Sighting("攻擊特化", 118, 242, in_header=True),
        Sighting("所有", 130, 350, in_header=False),
    ]
    # 選單開著就不算收斂:必須回到收合狀態才能繼續讀畫面。
    assert not converged("攻擊特化", sightings)
    assert next_step("攻擊特化", sightings, HEADER).kind == STUCK
