"""Auto Hatch feature: planner and defaults for the egg incubator loop.

Phase A of docs/auto-hatch-plan.md: home page -> egg pile (fixed coordinate)
-> incubator grid -> per-egg detail loop (hatch -> claim) -> close -> wait.
Phases B (nest management) and C (culling) will reuse the same target-type
vocabulary once their assets and digit reading exist.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from .models import Detection, Frame, Target

# Target types. The hatch_ prefix keeps the vocabulary disjoint from hunt so
# both features can coexist in one config without colliding.
HOME_ANCHOR = "hatch_home_anchor"
EGG_PILE = "hatch_egg_pile"
INCUBATOR_TITLE = "hatch_incubator_title"
HATCH_LABEL = "hatch_label"
HATCH_BUTTON = "hatch_button"
CLAIM_BUTTON = "hatch_claim_button"
EXPEL_BUTTON = "hatch_expel_button"
CLOSE_BUTTON = "hatch_close_button"
SCROLL = "hatch_scroll"

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    EGG_PILE: "tap",
    HATCH_LABEL: "tap",
    HATCH_BUTTON: "tap",
    CLAIM_BUTTON: "tap",
    CLOSE_BUTTON: "tap",
    SCROLL: "swipe",
}

# Post-action delays double as verification timeouts. The hatch button plays a
# full hatching cutscene before the claim button exists, so its window must
# outlast the animation.
DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    EGG_PILE: 3000,
    HATCH_LABEL: 2500,
    HATCH_BUTTON: 20000,
    CLAIM_BUTTON: 4000,
    CLOSE_BUTTON: 2500,
    SCROLL: 1500,
}

# What must be visible after each tap for it to count as done. Claiming may
# land on the next egg's detail page or back on the grid, so it accepts both.
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    EGG_PILE: (INCUBATOR_TITLE,),
    HATCH_LABEL: (HATCH_BUTTON,),
    HATCH_BUTTON: (CLAIM_BUTTON,),
    CLAIM_BUTTON: (HATCH_BUTTON, INCUBATOR_TITLE, HATCH_LABEL, CLAIM_BUTTON),
    CLOSE_BUTTON: (HOME_ANCHOR,),
}

# One successfully claimed dinosaur is one hatch workflow cycle. This stays
# feature-local because the shared config's cycle target normally belongs to
# hunt (mail_reward_collect_button).
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (CLAIM_BUTTON,)


class HatchPlanner:
    """Reactive planner for the hatch loop.

    Priority rules over the current detections replace an explicit state
    machine: whatever screen the game actually shows decides the next tap, so
    animations, dropped taps, and app restarts all converge onto the loop
    without special recovery cases. ``EXPEL_BUTTON`` is never a target under
    any rule.
    """

    def __init__(
        self,
        *,
        egg_pile_point: tuple[float, float],
        reference_width: float = 900.0,
        scroll_vector: tuple[float, float, float, float] = (450, 1100, 450, 500),
        scroll_duration_ms: int = 400,
        max_scrolls: int = 4,
        rescan_interval_seconds: float = 600.0,
        require_home_anchor: bool = True,
        home_failure_limit: int = 3,
        home_backoff_seconds: float = 30.0,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.egg_pile_point = egg_pile_point
        self.reference_width = reference_width
        self.scroll_vector = scroll_vector
        self.scroll_duration_ms = max(1, scroll_duration_ms)
        self.max_scrolls = max(0, max_scrolls)
        self.rescan_interval_seconds = max(0.0, rescan_interval_seconds)
        self.require_home_anchor = require_home_anchor
        self.home_failure_limit = max(1, home_failure_limit)
        self.home_backoff_seconds = max(0.0, home_backoff_seconds)
        self.logger = logger or logging.getLogger("dino_bot")
        self.clock = clock
        self._wait_until: float | None = None
        self._scrolls_done = 0
        self._home_failures = 0
        self._stage = "start"
        self.hatched = 0

    # -- engine hooks ------------------------------------------------------

    @property
    def last_stage(self) -> str:
        return self._stage

    def next_ready_delay_ms(self) -> int:
        if self._wait_until is None:
            return 0
        remaining = self._wait_until - self.clock()
        return max(0, int(remaining * 1000))

    def on_action_success(self, target_type: str) -> None:
        if target_type == EGG_PILE:
            self._home_failures = 0
            self._scrolls_done = 0
        elif target_type == SCROLL:
            self._scrolls_done += 1
        elif target_type == CLAIM_BUTTON:
            self.hatched += 1
            self._scrolls_done = 0
        elif target_type == CLOSE_BUTTON:
            self._begin_wait("closed incubator")

    def on_action_failure(self, target_type: str) -> None:
        if target_type == EGG_PILE:
            self._home_failures += 1
            if self._home_failures >= self.home_failure_limit:
                # The pile coordinate is not opening the incubator; hammering
                # it will not change that. Back off and let the screen settle.
                self._home_failures = 0
                self._begin_wait(
                    "egg pile tap not reaching incubator",
                    seconds=self.home_backoff_seconds,
                )

    # -- planning ----------------------------------------------------------

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._wait_until is not None:
            if self.clock() < self._wait_until:
                self._stage = "waiting"
                return None
            self._wait_until = None
        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)

        claim = self._best(by_type.get(CLAIM_BUTTON))
        if claim is not None:
            self._stage = "claim"
            return self._target(claim)
        hatch_button = self._best(by_type.get(HATCH_BUTTON))
        if hatch_button is not None:
            self._stage = "detail"
            return self._target(hatch_button)
        if INCUBATOR_TITLE in by_type:
            labels = by_type.get(HATCH_LABEL)
            if labels:
                self._stage = "grid"
                # Template hits in one visual row can differ by a few pixels
                # vertically. Lock onto the top row first, then choose its
                # leftmost egg so processing is deterministic row-major.
                return self._target(self._top_left(labels, frame))
            if self._scrolls_done < self.max_scrolls:
                self._stage = "scroll"
                return self._scroll_target(frame)
            close = self._best(by_type.get(CLOSE_BUTTON))
            if close is not None:
                self._stage = "close"
                return self._target(close)
            self._stage = "grid_no_close"
            return None
        if HOME_ANCHOR in by_type or not self.require_home_anchor:
            self._stage = "home"
            return self._egg_pile_target(frame)
        self._stage = "unknown_screen"
        return None

    # -- helpers -----------------------------------------------------------

    def _begin_wait(self, reason: str, *, seconds: float | None = None) -> None:
        duration = self.rescan_interval_seconds if seconds is None else seconds
        self._wait_until = self.clock() + duration
        self._scrolls_done = 0
        self.logger.info(
            "Hatch | wait %.0fs | %s | hatched=%d",
            duration,
            reason,
            self.hatched,
        )

    @staticmethod
    def _best(items: list[Detection] | None) -> Detection | None:
        if not items:
            return None
        return max(items, key=lambda item: item.confidence)

    @staticmethod
    def _top_left(items: list[Detection], frame: Frame) -> Detection:
        top_y = min(item.y for item in items)
        row_tolerance = max(12, int(frame.width * 0.06))
        top_row = [item for item in items if item.y <= top_y + row_tolerance]
        return min(top_row, key=lambda item: item.x)

    @staticmethod
    def _target(detection: Detection) -> Target:
        return Target(
            type=detection.type,
            x=detection.x,
            y=detection.y,
            confidence=detection.confidence,
            detection=detection,
        )

    def _scale(self, frame: Frame) -> float:
        return frame.width / self.reference_width

    def _egg_pile_target(self, frame: Frame) -> Target:
        scale = self._scale(frame)
        x = int(self.egg_pile_point[0] * scale)
        y = int(self.egg_pile_point[1] * scale)
        detection = Detection(
            type=EGG_PILE,
            x=x,
            y=y,
            confidence=1.0,
            metadata={"synthetic": True},
        )
        return self._target(detection)

    def _scroll_target(self, frame: Frame) -> Target:
        scale = self._scale(frame)
        x0, y0, x1, y1 = (value * scale for value in self.scroll_vector)
        detection = Detection(
            type=SCROLL,
            x=int(x0),
            y=int(y0),
            confidence=1.0,
            metadata={
                "synthetic": True,
                "swipe": {
                    "x2": int(x1),
                    "y2": int(y1),
                    "duration_ms": self.scroll_duration_ms,
                },
            },
        )
        return self._target(detection)
