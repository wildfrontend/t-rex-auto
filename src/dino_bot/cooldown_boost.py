"""One scheduled incubator visit, independent of hatch/management cycles.

The caller supplies measured navigation and button geometry. This planner owns
only the visit and its confirmation; the persistent inventory owns the clock.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from . import hatch
from .hatch_inventory import HatchBoostInventoryStore
from .models import Detection, Frame, Target
from .overlays import CONFIRM_NO, CONFIRM_YES
from .targeting import best_detection, synthetic_target

BOOST_BUTTON = "hatch_cooldown_boost_button"
BOOST_CONFIRM = "hatch_cooldown_boost_confirm_yes"
BOOST_OPEN = "hatch_cooldown_boost_open"
BOOST_CLOSE = "hatch_cooldown_boost_close"
BOOST_CANCEL = "hatch_cooldown_boost_cancel"
BOOST_ACTIONS = frozenset({BOOST_BUTTON, BOOST_CONFIRM, BOOST_OPEN, BOOST_CLOSE, BOOST_CANCEL})


class CooldownBoostVisit:
    def __init__(
        self, inventory: HatchBoostInventoryStore, *,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
        keep_incubator_open: bool = False,
    ) -> None:
        self.inventory = inventory
        self.clock = clock
        self.logger = logger or logging.getLogger("dino_bot")
        self.phase = "open"
        self.deadline = clock() + 45.0
        self.complete = False
        self.failed = False
        self.used = False
        self.confirm_sent = False
        self.keep_incubator_open = keep_incubator_open
        self.logger.info(
            "Cooldown boost | %s",
            "incubator check starting" if keep_incubator_open else "scheduled visit starting",
        )

    def finish(self, reason: str) -> None:
        if self.phase == "close":
            return
        if not self.used:
            self.inventory.defer(reason)
        self.logger.info("Cooldown boost | %s", reason)
        self.phase = "close"
        self.deadline = self.clock() + 30.0

    def choose(
        self, frame: Frame, detections: Sequence[Detection], *,
        home_centered: bool, home_point: tuple[int, int] | None,
        button_ready: bool, button_point: tuple[int, int],
    ) -> Target | None:
        del frame
        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)
        title = hatch.INCUBATOR_TITLE in by_type
        yes = best_detection(by_type.get(CONFIRM_YES))
        no = best_detection(by_type.get(CONFIRM_NO))
        close = best_detection(by_type.get(hatch.CLOSE_BUTTON))
        panel = title and close is not None and yes is None and no is None

        if self.clock() >= self.deadline:
            if self.phase == "close":
                self.failed = True
                self.complete = True
                return None
            self.finish("visit timed out; retry later")

        # Confirmation disappearing alone is insufficient: require a fresh
        # unobscured incubator and an active gray bar before accounting for use.
        if self.phase == "verify" and panel:
            if not button_ready and self.confirm_sent:
                consumed = self.inventory.consume_one(only_if_due=True)
                self.used = consumed is not None
                if consumed is not None:
                    self.logger.info(
                        "Cooldown boost | used 1 ticket | remaining=%d | next use in 1800s",
                        consumed.remaining,
                    )
                self.finish("activation verified; returning to previous workflow")
            else:
                self.finish("activation not verified; retry later")

        delay = self.inventory.ready_delay_seconds()
        if delay is None and self.phase not in {"verify", "close"}:
            self.finish("disabled or empty stock")

        if self.phase == "close":
            if no is not None:
                return synthetic_target(BOOST_CANCEL, no.x, no.y)
            if panel and self.keep_incubator_open:
                self.complete = True
                return None
            if panel and close is not None:
                return synthetic_target(BOOST_CLOSE, close.x, close.y)
            if home_centered:
                self.complete = True
            return None

        if self.phase == "confirm":
            if yes is not None and no is not None:
                # Recheck the live switch immediately before spending a ticket.
                if delay != 0:
                    self.finish("permission changed before confirmation")
                    return synthetic_target(BOOST_CANCEL, no.x, no.y)
                self.confirm_sent = True
                return synthetic_target(BOOST_CONFIRM, yes.x, yes.y)
            return None
        if self.phase == "verify":
            # An uncertain confirmation must never be blindly submitted twice.
            return None
        if panel:
            if delay != 0:
                self.finish("use is no longer due")
                if self.keep_incubator_open:
                    self.complete = True
                    return None
                return synthetic_target(BOOST_CLOSE, close.x, close.y)
            if not button_ready:
                self.finish("加速仍生效或按鈕不可用，下次正常開啟孵化器時複查")
                if self.keep_incubator_open:
                    self.complete = True
                    return None
                return synthetic_target(BOOST_CLOSE, close.x, close.y)
            return synthetic_target(BOOST_BUTTON, *button_point)
        if home_centered and home_point is not None:
            return synthetic_target(BOOST_OPEN, *home_point)
        return None

    def on_action_success(self, target_type: str) -> None:
        if target_type == BOOST_BUTTON:
            self.phase = "confirm"
        elif target_type == BOOST_CONFIRM:
            self.phase = "verify"

    def on_action_failure(self, target_type: str) -> None:
        if target_type == BOOST_CONFIRM:
            # A missed transition can still have spent the ticket. Observe the
            # next frame before deciding; do not immediately try another Yes.
            self.phase = "verify"
        else:
            self.finish(f"action not verified: {target_type}; retry later")
