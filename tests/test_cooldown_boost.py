from __future__ import annotations

import numpy as np
import pytest

from dino_bot import hatch
from dino_bot.cooldown_boost import (
    BOOST_BUTTON,
    BOOST_CANCEL,
    BOOST_CLOSE,
    BOOST_CONFIRM,
    BOOST_OPEN,
    CooldownBoostVisit,
)
from dino_bot.hatch_inventory import HatchBoostInventoryStore
from dino_bot.models import Detection, Frame
from dino_bot.overlays import CONFIRM_NO, CONFIRM_YES


def detections(*names):
    return [Detection(name, 450, 800, 1.0) for name in names]


PANEL = detections(hatch.INCUBATOR_TITLE, hatch.CLOSE_BUTTON)
PROMPT = PANEL + detections(CONFIRM_YES, CONFIRM_NO)


def choose(visit, items=PANEL, *, ready=True, home=False):
    return visit.choose(
        Frame(np.zeros((1, 1, 3), dtype=np.uint8)), items,
        home_centered=home, home_point=(450, 1330) if home else None,
        button_ready=ready, button_point=(450, 1420),
    )


@pytest.fixture
def setup_visit(tmp_path):
    now = [1000.0]
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: now[0])
    inventory.set_enabled(True)
    visit = CooldownBoostVisit(inventory, clock=lambda: now[0])
    return visit, inventory, now


def test_visit_spends_once_only_after_confirmation_and_active_bar(setup_visit):
    visit, inventory, _ = setup_visit
    assert choose(visit, [], home=True).type == BOOST_OPEN
    visit.on_action_success(BOOST_OPEN)
    assert choose(visit).type == BOOST_BUTTON
    visit.on_action_success(BOOST_BUTTON)
    assert choose(visit, PROMPT).type == BOOST_CONFIRM
    visit.on_action_success(BOOST_CONFIRM)
    # Background title/gray bar must not count while the dialog remains.
    assert choose(visit, PROMPT, ready=False) is None
    assert inventory.snapshot().remaining == 100
    assert choose(visit, ready=False).type == BOOST_CLOSE
    assert inventory.snapshot().remaining == 99
    assert inventory.ready_delay_seconds() == 1800
    choose(visit, ready=False)
    assert inventory.snapshot().remaining == 99
    visit.on_action_success(BOOST_CLOSE)
    choose(visit, [], home=True)
    assert visit.complete and not visit.failed


def test_live_disable_cancels_open_confirmation(setup_visit):
    visit, inventory, _ = setup_visit
    choose(visit)
    visit.on_action_success(BOOST_BUTTON)
    inventory.set_enabled(False)
    assert choose(visit, PROMPT).type == BOOST_CANCEL
    assert not visit.confirm_sent
    assert inventory.snapshot().remaining == 100


@pytest.mark.parametrize("active", [True, False])
def test_uncertain_confirmation_is_observed_without_resubmission(setup_visit, active):
    visit, inventory, _ = setup_visit
    choose(visit)
    visit.on_action_success(BOOST_BUTTON)
    assert choose(visit, PROMPT).type == BOOST_CONFIRM
    visit.on_action_failure(BOOST_CONFIRM)
    assert choose(visit, PROMPT) is None
    assert choose(visit, ready=not active).type == BOOST_CLOSE
    assert inventory.snapshot().remaining == (99 if active else 100)
    assert inventory.ready_delay_seconds() == (1800 if active else 60)


def test_gray_bar_without_own_confirmation_defers_without_consuming(setup_visit):
    visit, inventory, now = setup_visit
    assert choose(visit, ready=False).type == BOOST_CLOSE
    assert inventory.snapshot().remaining == 100
    assert inventory.ready_delay_seconds() == 60
    now[0] += 60
    assert inventory.ready_delay_seconds() == 0


def test_visit_and_return_timeouts_are_bounded(setup_visit):
    visit, inventory, now = setup_visit
    now[0] += 46
    assert choose(visit, []) is None
    assert inventory.ready_delay_seconds() == 60
    now[0] += 31
    assert choose(visit, []) is None
    assert visit.complete and visit.failed
    assert inventory.snapshot().remaining == 100


@pytest.mark.parametrize("ready", [False, True])
def test_inline_check_returns_open_incubator_without_navigation(setup_visit, ready):
    _, inventory, now = setup_visit
    visit = CooldownBoostVisit(
        inventory, clock=lambda: now[0], keep_incubator_open=True,
    )
    if ready:
        assert choose(visit).type == BOOST_BUTTON
        visit.on_action_success(BOOST_BUTTON)
        assert choose(visit, PROMPT).type == BOOST_CONFIRM
        visit.on_action_success(BOOST_CONFIRM)
    assert choose(visit, ready=False) is None
    assert visit.complete and not visit.failed
    assert inventory.snapshot().remaining == (99 if ready else 100)
    choose(visit, ready=False)
    assert inventory.snapshot().remaining == (99 if ready else 100)
