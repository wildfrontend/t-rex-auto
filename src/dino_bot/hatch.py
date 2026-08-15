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

import cv2
import numpy as np

from .digits import DigitReader
from .models import Detection, Frame, Image, Target

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

# Timer text positions in the 900x1600 incubator grid. The visible grid has
# three columns and three rows; each timer is read without the clock icon or
# progress bar. The reader accepts both ``HHMMSS`` (the colon dots are too
# small to survive segmentation) and ``HH?MM?SS``.
HATCH_TIMER_REGIONS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x0, y0, x1, y1)
    for y0, y1 in ((608.0, 638.0), (873.0, 903.0), (1138.0, 1168.0))
    for x0, x1 in ((200.0, 350.0), (380.0, 530.0), (560.0, 710.0))
)

# The home egg pile is placed at a calibrated, stable map position.  Its
# artwork changes between accounts, so this gate intentionally measures only
# the large dark/coloured structure in a fixed lower-map region.  It does not
# depend on a particular nest image or colour.
HOME_PILE_STRUCTURE_REGION: tuple[float, float, float, float] = (
    250.0,
    1080.0,
    650.0,
    1510.0,
)
HOME_PILE_MIN_STRUCTURE_AREA = 1_000.0
HOME_PILE_MIN_STRUCTURE_WIDTH = 150.0
HOME_PILE_MIN_STRUCTURE_ROWS = 4
HOME_PILE_MIN_STRUCTURE_ROW_WIDTH = 110


def has_home_pile_structure(
    frame: Frame,
    *,
    egg_pile_point: tuple[float, float],
    reference_width: float = 900.0,
) -> bool:
    """Return whether the fixed home-pile region contains a nest structure.

    This is deliberately a screen gate only.  It uses the configured
    reference coordinates and a broad structure mask, so a small icon or a
    particular nest skin cannot authorize the egg-pile tap.
    """

    if frame.image.size == 0 or frame.width <= 0 or reference_width <= 0:
        return False
    scale = frame.width / reference_width
    rx0, ry0, rx1, ry1 = (
        round(value * scale) for value in HOME_PILE_STRUCTURE_REGION
    )
    x0 = max(rx0, round((egg_pile_point[0] - 220.0) * scale))
    y0 = max(ry0, round((egg_pile_point[1] - 250.0) * scale))
    x1 = min(rx1, round((egg_pile_point[0] + 220.0) * scale))
    y1 = min(ry1, round((egg_pile_point[1] + 140.0) * scale))
    roi = frame.image[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Snow is bright and low-saturation.  The nest base, regardless of skin,
    # contains either dark outlines or saturated material.  Closing joins the
    # separate painted pieces without turning the whole map into one blob.
    mask = np.where((gray <= 205) | (hsv[:, :, 1] >= 45), 255, 0).astype(np.uint8)
    kernel_size = max(3, round(9 * scale) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = HOME_PILE_MIN_STRUCTURE_AREA * scale * scale
    min_width = HOME_PILE_MIN_STRUCTURE_WIDTH * scale
    max_area = roi.shape[0] * roi.shape[1] * 0.70
    expected_x = egg_pile_point[0] * scale - x0
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        center_x = x + width / 2.0
        if (
            area >= min_area
            and area <= max_area
            and width >= min_width
            and abs(center_x - expected_x) <= 180.0 * scale
        ):
            component = mask[y : y + height, x : x + width]
            occupied_rows = np.count_nonzero(component, axis=1)
            structure_rows = int(
                np.count_nonzero(
                    occupied_rows >= HOME_PILE_MIN_STRUCTURE_ROW_WIDTH * scale
                )
            )
            if structure_rows >= HOME_PILE_MIN_STRUCTURE_ROWS:
                return True
    return False


def parse_hatch_timer_text(text: str) -> int | None:
    """Parse a six-digit incubator timer returned by :class:`DigitReader`."""

    compact = text.replace("?", "").replace(":", "")
    if len(compact) != 6 or not compact.isdigit():
        return None
    hours, minutes, seconds = (
        int(compact[0:2]),
        int(compact[2:4]),
        int(compact[4:6]),
    )
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def read_hatch_cooldown_seconds(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
) -> int | None:
    """Return the longest visible incubator countdown, or ``None``.

    A ready egg has the ``孵化`` label and no timer. Full hatch uses the
    longest readable timer so a group of eggs can mature together before the
    next batch is collected and compared. A zero is returned when the timers
    are visible but already due; unreadable regions are ignored and fail safe
    to ``None`` when none can be parsed.
    """

    if image.ndim < 2 or image.shape[1] <= 0 or reference_width <= 0:
        return None
    scale = image.shape[1] / reference_width
    values: list[int] = []
    for x0, y0, x1, y1 in HATCH_TIMER_REGIONS:
        left, top, right, bottom = (
            round(value * scale) for value in (x0, y0, x1, y1)
        )
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        value = parse_hatch_timer_text(reader.read(crop))
        if value is not None:
            values.append(value)
    return max(values) if values else None


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
        max_scrolls: int = 0,
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

    def begin_rescan_wait(self, reason: str, *, seconds: float | None = None) -> None:
        """Pause on home until the next configured incubator rescan."""

        self._begin_wait(reason, seconds=seconds)

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
        if (
            HOME_ANCHOR in by_type
            or not self.require_home_anchor
            or has_home_pile_structure(
                frame,
                egg_pile_point=self.egg_pile_point,
                reference_width=self.reference_width,
            )
        ):
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
