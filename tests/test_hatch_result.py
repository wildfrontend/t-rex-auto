"""Judging the newborn on the hatch-result screen."""

from __future__ import annotations

import pathlib
import sys

import cv2
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dino_bot.digits import DigitReader  # noqa: E402
from dino_bot.hatch_result import (  # noqa: E402
    judge_newborn,
    read_hatch_result_stats,
)
from dino_bot.models import Frame  # noqa: E402
from dino_bot.nest_readout import Stats  # noqa: E402

GLYPHS = REPO / "assets" / "hatch" / "digits"
FLOORS = {"hp_floor": 5000, "attack_floor": 600}


def stats(hp: int, attack: int, speed: int = 150) -> Stats:
    return Stats(hp=hp, attack=attack, speed=speed)


def test_a_dinosaur_that_reaches_neither_target_is_expelled() -> None:
    # A purity line that drifted: too weak for HP, too weak for attack.
    assert judge_newborn(stats(1000, 20), **FLOORS).expel is True
    assert judge_newborn(stats(20, 444), **FLOORS).expel is True


def test_clearing_either_floor_keeps_the_dinosaur() -> None:
    # Specialization deliberately floors the opposing stat, so a lopsided
    # result is a success, not a failure.
    assert judge_newborn(stats(5880, 1), **FLOORS).expel is False
    assert judge_newborn(stats(10, 854), **FLOORS).expel is False
    # And the double-high result everyone actually wants.
    assert judge_newborn(stats(5780, 648), **FLOORS).expel is False


def test_an_unreadable_screen_is_never_expelled() -> None:
    verdict = judge_newborn(None, **FLOORS)
    assert verdict.expel is False
    assert verdict.readable is False


def test_reads_the_three_stats_off_a_real_hatch_result_frame() -> None:
    # HP and attack are drawn in blue, speed in black; a plain grey threshold
    # sees only the black row, so this guards the colour-aware ink mask.
    sample = REPO / "tests" / "data" / "hatch-result.png"
    if not sample.exists():
        pytest.skip("hatch-result sample not captured")
    image = cv2.imread(str(sample))
    result = read_hatch_result_stats(Frame(image=image), DigitReader(GLYPHS))
    assert result == stats(5780, 648)


def planner_with(**kwargs):
    from dino_bot.hatch import HatchPlanner

    return HatchPlanner(
        egg_pile_point=(450, 1330),
        reader=DigitReader(GLYPHS),
        **kwargs,
    )


def hatch_result_frame() -> Frame:
    sample = REPO / "tests" / "data" / "hatch-result.png"
    return Frame(image=cv2.imread(str(sample)))


def result_detections():
    from dino_bot.hatch import CLAIM_BUTTON, EXPEL_BUTTON
    from dino_bot.models import Detection

    return [
        Detection(type=CLAIM_BUTTON, x=330, y=1242, confidence=1.0),
        Detection(type=EXPEL_BUTTON, x=570, y=1242, confidence=1.0),
    ]


def test_dry_run_logs_the_verdict_but_still_claims() -> None:
    from dino_bot.hatch import CLAIM_BUTTON

    # Floors set high enough that the 5780/648 sample fails both.
    planner = planner_with(
        expel_below_hp=6000,
        expel_below_attack=700,
        expel_dry_run=True,
    )
    target = planner.choose(hatch_result_frame(), result_detections())
    assert target is not None
    assert target.type == CLAIM_BUTTON


def test_expelling_taps_expel_once_the_dry_run_is_off() -> None:
    from dino_bot.hatch import EXPEL_BUTTON

    planner = planner_with(
        expel_below_hp=6000,
        expel_below_attack=700,
        expel_dry_run=False,
    )
    target = planner.choose(hatch_result_frame(), result_detections())
    assert target is not None
    assert target.type == EXPEL_BUTTON


def test_a_keeper_is_claimed_even_with_the_dry_run_off() -> None:
    from dino_bot.hatch import CLAIM_BUTTON

    planner = planner_with(
        expel_below_hp=5000,
        expel_below_attack=600,
        expel_dry_run=False,
    )
    target = planner.choose(hatch_result_frame(), result_detections())
    assert target is not None
    assert target.type == CLAIM_BUTTON


def test_unconfigured_floors_never_expel() -> None:
    from dino_bot.hatch import CLAIM_BUTTON

    # The default instance must behave exactly as it did before this feature.
    planner = planner_with(expel_dry_run=False)
    target = planner.choose(hatch_result_frame(), result_detections())
    assert target is not None
    assert target.type == CLAIM_BUTTON


def test_best_stats_are_logged_only_when_a_record_improves(caplog) -> None:
    """One line per record, not one per hatch.

    Manual screening used to print every nest parent's stats; that pass is
    gone, so the hatch-result screen is the only full stat block the bot still
    sees. Logging every hatch would bury it.
    """

    import logging

    from dino_bot.hatch_result import HatchVerdict

    planner = planner_with(expel_below_hp=5000, expel_below_attack=600)

    def verdict(hp: int, attack: int) -> HatchVerdict:
        return HatchVerdict(stats(hp, attack), False, "keeping")

    with caplog.at_level(logging.INFO, logger="dino_bot"):
        planner._record_best_stats(verdict(5800, 640))
        planner._record_best_stats(verdict(5700, 630))  # neither improves
        planner._record_best_stats(verdict(5900, 620))  # hp only

    records = [r for r in caplog.records if "Hatch best" in r.getMessage()]
    assert len(records) == 2
    assert planner._best_hp == 5900
    assert planner._best_attack == 640


def test_an_unreadable_hatch_never_moves_the_record() -> None:
    planner = planner_with(expel_below_hp=5000, expel_below_attack=600)
    planner._best_hp = 5900

    from dino_bot.hatch_result import HatchVerdict

    planner._record_best_stats(HatchVerdict(None, False, "stats unreadable"))
    planner._record_best_stats(None)

    assert planner._best_hp == 5900
