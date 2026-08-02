"""End-to-end Auto Hatch workflow.

This module closes the loop described in ``docs/auto-hatch-plan.md``:

* hatch every ready egg;
* optimize Attack and HP parents;
* auto-place Top and Mass nests;
* collect every egg and close My Nest;
* navigate to the cave, cull only above the configured threshold, and return;
* when no egg was ready, collect completed nest eggs and then keep the
  incubator rescan cooldown without running the mutation/cave stages;
* start the next hatch pass.

The planner is deliberately screen-gated.  Fixed coordinates are used only
inside a screen whose title/anchor has already been detected, and every
destructive-looking affirmative action also requires its known prompt.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from . import attack_replacement as replacement_feature
from . import hatch as hatch_feature
from . import nest_filter as nest_filter_feature
from . import select_sort as select_sort_feature
from .attack_replacement import AttackReplacementTestPlanner
from .cave_navigation import (
    CAVE,
    DEFAULT_SWIPE_VECTORS,
    DONE,
    RESCAN,
    STUCK,
    SWIPE,
    CaveNavigator,
)
from .cull import read_dino_count, should_cull
from .digits import DigitReader
from .models import Detection, Frame, Target
from .nests import ATTACK_RULE, HP_RULE, MASS_RULE, TOP_RULE, AutoPlaceRule
from .overlays import (
    AUTOPLACE_NOTICE,
    CONFIRM_NO,
    CONFIRM_YES,
    INCUBATOR_FULL_TOAST,
    NESTED_PARENT_WARNING,
    SELECT_CONFIRM_PROMPT,
)
from .parent_open import NEST_TITLE, SELECT_TITLE

# Full-workflow synthetic actions and newly cropped screen anchors.
OPEN_NEST = "hatch_full_open_nest"
NEST_GEAR = "hatch_nest_gear"
AUTOPLACE_TITLE = "hatch_autoplace_title"
AUTOPLACE_PROMPT = "hatch_autoplace_prompt"
AUTOPLACE_SORT_HEADER = "hatch_autoplace_sort_header"
AUTOPLACE_MASK_CLOSE = "hatch_autoplace_mask_close"
AUTOPLACE_BUTTON = "hatch_autoplace_button"
AUTOPLACE_YES = "hatch_autoplace_yes"
COLLECT_EGGS_BUTTON = "hatch_collect_eggs_button"
NEST_MASK_CLOSE = "hatch_nest_mask_close"
HATCH_DETAIL_CLOSE = "hatch_unready_detail_close"

CAVE_SWIPE = "hatch_cave_swipe"
CAVE_RECENTER = "hatch_cave_recenter"
CAVE_SELECT_BUTTON = "hatch_cave_select_button"
CAVE_CONTINUOUS_BUTTON = "hatch_cave_continuous_button"
CAVE_CLOSE_BUTTON = "hatch_cave_close_button"
SELECT_TAG_HEADER = "hatch_cull_tag_header"
SELECT_WEAKEST_BUTTON = "hatch_select_weakest_button"
SELECT_CHOOSE_BUTTON = "hatch_select_choose_button"

RECOVERY_NO = "hatch_recovery_no"
RECOVERY_MASK_CLOSE = "hatch_recovery_mask_close"
RECOVERY_CLOSE = "hatch_recovery_close"
RECOVERY_CLAIM = "hatch_recovery_claim"
RECOVERY_MAP_EXIT = "hatch_recovery_map_exit"
RECOVERY_RECENTER = "hatch_recovery_recenter"
RECOVERY_BACK = "hatch_recovery_back"
HUNT_MAP_EXIT = "map_exit_nest_button"

PLACE_SORT_BEST = "hatch_place_sort_best"
PLACE_SORT_LEVEL = "hatch_place_sort_level"
PLACE_HDR_BEST = "hatch_place_hdr_best"
PLACE_HDR_LEVEL = "hatch_place_hdr_level"

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    **hatch_feature.DEFAULT_TARGET_ACTIONS,
    **replacement_feature.DEFAULT_TARGET_ACTIONS,
    OPEN_NEST: "tap",
    NEST_GEAR: "tap",
    AUTOPLACE_SORT_HEADER: "tap",
    PLACE_SORT_BEST: "tap",
    PLACE_SORT_LEVEL: "tap",
    AUTOPLACE_MASK_CLOSE: "tap",
    AUTOPLACE_BUTTON: "tap",
    AUTOPLACE_YES: "tap",
    COLLECT_EGGS_BUTTON: "tap",
    NEST_MASK_CLOSE: "tap",
    HATCH_DETAIL_CLOSE: "tap",
    CAVE_SWIPE: "swipe",
    CAVE_RECENTER: "swipe",
    CAVE: "tap",
    CAVE_SELECT_BUTTON: "tap",
    SELECT_TAG_HEADER: "tap",
    nest_filter_feature.TAG_ALL: "tap",
    SELECT_WEAKEST_BUTTON: "tap",
    SELECT_CHOOSE_BUTTON: "tap",
    CAVE_CONTINUOUS_BUTTON: "tap",
    RECOVERY_NO: "tap",
    RECOVERY_MASK_CLOSE: "tap",
    RECOVERY_CLOSE: "tap",
    RECOVERY_CLAIM: "tap",
    RECOVERY_MAP_EXIT: "tap",
    RECOVERY_RECENTER: "swipe",
    RECOVERY_BACK: "back",
}

DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    **hatch_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    **replacement_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    OPEN_NEST: 3500,
    NEST_GEAR: 3000,
    AUTOPLACE_SORT_HEADER: 2500,
    PLACE_SORT_BEST: 2500,
    PLACE_SORT_LEVEL: 2500,
    AUTOPLACE_MASK_CLOSE: 2500,
    AUTOPLACE_BUTTON: 4000,
    AUTOPLACE_YES: 5000,
    COLLECT_EGGS_BUTTON: 5000,
    NEST_MASK_CLOSE: 3000,
    HATCH_DETAIL_CLOSE: 3000,
    CAVE_SWIPE: 3000,
    CAVE_RECENTER: 3500,
    CAVE: 4000,
    CAVE_SELECT_BUTTON: 3500,
    SELECT_TAG_HEADER: 2500,
    SELECT_WEAKEST_BUTTON: 3500,
    SELECT_CHOOSE_BUTTON: 4000,
    # Continuous battle can run through several waves before the result page.
    CAVE_CONTINUOUS_BUTTON: 120_000,
    RECOVERY_NO: 3000,
    RECOVERY_MASK_CLOSE: 3000,
    RECOVERY_CLOSE: 4000,
    RECOVERY_CLAIM: 5000,
    RECOVERY_MAP_EXIT: 4000,
    RECOVERY_RECENTER: 4000,
    RECOVERY_BACK: 4000,
}

DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    **hatch_feature.DEFAULT_SUCCESS_TRANSITIONS,
    **replacement_feature.DEFAULT_SUCCESS_TRANSITIONS,
    OPEN_NEST: (NEST_TITLE,),
    NEST_GEAR: (AUTOPLACE_TITLE,),
    AUTOPLACE_SORT_HEADER: (PLACE_SORT_BEST, PLACE_SORT_LEVEL),
    PLACE_SORT_BEST: (PLACE_HDR_BEST,),
    PLACE_SORT_LEVEL: (PLACE_HDR_LEVEL,),
    # NEST_TITLE is visible behind the settings overlay.  The planner performs
    # an additional foreground check and repeats the outside-mask tap if the
    # overlay title is still present.
    AUTOPLACE_MASK_CLOSE: (NEST_TITLE,),
    AUTOPLACE_BUTTON: (AUTOPLACE_PROMPT, AUTOPLACE_NOTICE, NEST_TITLE),
    AUTOPLACE_YES: (NEST_TITLE,),
    COLLECT_EGGS_BUTTON: (INCUBATOR_FULL_TOAST, NEST_TITLE),
    HATCH_DETAIL_CLOSE: (hatch_feature.INCUBATOR_TITLE,),
    # The home anchor also remains visible behind My Nest; choose() verifies
    # that NEST_TITLE disappeared before advancing to cave navigation.
    NEST_MASK_CLOSE: (hatch_feature.HOME_ANCHOR,),
    CAVE: (CAVE_SELECT_BUTTON,),
    CAVE_SELECT_BUTTON: (SELECT_TITLE,),
    SELECT_TAG_HEADER: tuple(nest_filter_feature.OPTION_LABELS),
    nest_filter_feature.TAG_ALL: (nest_filter_feature.TAG_HDR_ALL,),
    SELECT_WEAKEST_BUTTON: (SELECT_CHOOSE_BUTTON,),
    SELECT_CHOOSE_BUTTON: (CAVE_CONTINUOUS_BUTTON,),
    CAVE_CONTINUOUS_BUTTON: (hatch_feature.CLAIM_BUTTON,),
    hatch_feature.CLAIM_BUTTON: (
        hatch_feature.HATCH_BUTTON,
        hatch_feature.INCUBATOR_TITLE,
        hatch_feature.HATCH_LABEL,
        hatch_feature.CLAIM_BUTTON,
        hatch_feature.HOME_ANCHOR,
    ),
}

# Full mode is intentionally unbounded.  It reports its own completed
# management cycles and does not let a cave result's reused "獲取" button be
# confused with a hatched-dinosaur cycle counter.
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = ()

# These controls only open/close a dropdown or select an idempotent view
# option.  A dropped tap can be retried in place without changing a parent,
# applying auto-place, collecting eggs, or starting a cave battle.
RETRYABLE_NAVIGATION_TARGETS: frozenset[str] = frozenset(
    {
        nest_filter_feature.FILTER_HEADER,
        *nest_filter_feature.OPTION_LABELS,
        select_sort_feature.TAG_HEADER,
        select_sort_feature.SORT_HEADER,
        select_sort_feature.SORT_ATTACK,
        select_sort_feature.SORT_HP,
        AUTOPLACE_SORT_HEADER,
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        AUTOPLACE_MASK_CLOSE,
    }
)
MAX_NAVIGATION_RETRIES = 2


HOME_FOREGROUND_TYPES: frozenset[str] = frozenset(
    {
        hatch_feature.INCUBATOR_TITLE,
        hatch_feature.HATCH_BUTTON,
        hatch_feature.CLAIM_BUTTON,
        NEST_TITLE,
        SELECT_TITLE,
        AUTOPLACE_TITLE,
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        SELECT_CONFIRM_PROMPT,
        NESTED_PARENT_WARNING,
        CAVE_SELECT_BUTTON,
        CAVE_CONTINUOUS_BUTTON,
    }
)


def is_home_screen(frame: Frame, detections: Sequence[Detection]) -> bool:
    """Return whether the unobscured outdoor map is currently foreground.

    ``HOME_ANCHOR`` alone is insufficient: the left HUD remains visible and
    template-matchable behind My Nest and most confirmation dialogs.  Known
    foreground anchors reject those screens, while the brightness of two
    outside strips rejects dimmed/unknown overlays for which no template
    exists.  The snowy outdoor map keeps those strips bright in both the
    default and calibrated cave views.
    """

    types = {item.type for item in detections}
    if hatch_feature.HOME_ANCHOR not in types or types & HOME_FOREGROUND_TYPES:
        return False
    image = frame.image
    if image.size == 0:
        return False
    scale = frame.width / 900.0
    y0 = min(frame.height, max(0, round(650 * scale)))
    y1 = min(frame.height, max(y0 + 1, round(1000 * scale)))
    strip = max(1, round(120 * scale))
    left = image[y0:y1, :strip]
    right = image[y0:y1, max(0, frame.width - strip) :]
    if not left.size or not right.size:
        return False
    gray = np.concatenate((left.mean(axis=2).ravel(), right.mean(axis=2).ravel()))
    return bool(np.mean(gray > 100) >= 0.45)


def is_centered_home_screen(frame: Frame, detections: Sequence[Detection]) -> bool:
    """Return whether the normal home map (not the shifted cave view) is ready."""

    if not is_home_screen(frame, detections) or any(
        item.type == CAVE for item in detections
    ):
        return False
    pile = _egg_pile_base_center(frame)
    if pile is None:
        return False
    scale = frame.width / 900.0
    expected_x, expected_y = 450 * scale, 1455 * scale
    return abs(pile[0] - expected_x) <= 100 * scale and abs(
        pile[1] - expected_y
    ) <= 100 * scale


def _egg_pile_base_center(frame: Frame) -> tuple[float, float] | None:
    """Locate the stable cyan base of the variable-looking home egg pile."""

    if frame.image.size == 0:
        return None
    scale = frame.width / 900.0
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, (75, 70, 70), (105, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(cyan)
    candidates: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, width, _height, area = stats[index]
        center_x, center_y = centers[index]
        if (
            area >= 350 * scale * scale
            and width >= 170 * scale
            and y >= 850 * scale
            and 150 * scale <= center_x <= 750 * scale
        ):
            candidates.append((float(center_x), float(center_y)))
    return max(candidates, key=lambda center: center[1]) if candidates else None


def _egg_pile_safe_tap(frame: Frame) -> tuple[int, int] | None:
    """Choose a point on the eggs from the pile's stable cyan base.

    Cave-return swipes do not always move the map by exactly the requested
    distance.  A fixed home coordinate can consequently land just above the
    basket, where roaming dinosaurs open their detail card instead.  The cyan
    base moves with the basket; its centre minus a small vertical offset stays
    on the large middle egg across the observed centred positions.
    """

    pile = _egg_pile_base_center(frame)
    if pile is None:
        return None
    scale = frame.width / 900.0
    return round(pile[0]), round(pile[1] - 100 * scale)


def _unready_egg_detail_close(frame: Frame) -> tuple[int, int] | None:
    """Locate the alternate red X shown on an unready egg detail page."""

    if frame.image.size == 0:
        return None
    scale = frame.width / 900.0
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 100), (12, 255, 255))
    yellow = cv2.inRange(hsv, (12, 80, 100), (40, 255, 255))

    def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, float, float]]:
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        return [
            (*map(int, stats[index]), float(centers[index][0]), float(centers[index][1]))
            for index in range(1, count)
        ]

    # Require both the square red close control and the broad yellow
    # "immediate hatch" control.  The pair avoids treating a generic red game
    # button as permission to tap a fixed coordinate on an unknown screen.
    has_hatch_button = any(
        area >= 10_000 * scale * scale
        and 280 * scale <= x <= 620 * scale
        and 1050 * scale <= y <= 1250 * scale
        for x, y, _width, _height, area, _cx, _cy in components(yellow)
    )
    if not has_hatch_button:
        return None
    close_candidates = [
        (cx, cy)
        for x, y, width, height, area, cx, cy in components(red)
        if area >= 2500 * scale * scale
        and 50 * scale <= width <= 100 * scale
        and 50 * scale <= height <= 100 * scale
        and 620 * scale <= x <= 760 * scale
        and 1080 * scale <= y <= 1280 * scale
    ]
    if not close_candidates:
        return None
    cx, cy = max(close_candidates, key=lambda center: center[0])
    return round(cx), round(cy)


class HatchHomeRecoveryPlanner:
    """Unwind known/unknown foregrounds and prove the map is centered again."""

    def __init__(
        self,
        *,
        reference_width: float = 900.0,
        logger: logging.Logger | None = None,
        max_back_attempts: int = 2,
        required_home_frames: int = 2,
    ) -> None:
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self.max_back_attempts = max(0, max_back_attempts)
        self.required_home_frames = max(1, required_home_frames)
        self._stage = "inspect"
        self._back_attempts = 0
        self._home_frames = 0
        self._recenter_swipes = 0
        self._cave_recovery_required = False
        self._recenter_end = 0
        self._complete = False
        self._failed = False

    def last_stage(self) -> str:
        return f"recover_home_{self._stage}"

    def is_complete(self) -> bool:
        return self._complete

    def is_failed(self) -> bool:
        return self._failed

    def on_action_success(self, target_type: str) -> None:
        if target_type == RECOVERY_RECENTER:
            self._recenter_swipes += 1
        self._stage = f"verify_{target_type}"

    def on_action_failure(self, target_type: str) -> None:
        self.logger.error("Hatch recovery | action failed | target=%s", target_type)
        self._stage = f"failed_{target_type}"
        self._failed = True

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete or self._failed:
            return None
        by_type = _group(detections)
        vectors = _inverse_swipe_vectors(DEFAULT_SWIPE_VECTORS)
        # Once a shifted cave view is observed, replay the complete inverse
        # route.  The cave leaves the viewport after the first (horizontal)
        # swipe, but the egg pile is still roughly 330 px above its fixed tap
        # coordinate until the second (vertical) swipe finishes.
        if (
            self._cave_recovery_required
            and self._recenter_swipes < self._recenter_end
        ):
            x1, y1, x2, y2 = _scaled_swipe(
                frame,
                vectors[self._recenter_swipes],
                self.reference_width,
            )
            self._stage = "recenter_cave_view"
            return _swipe_target(RECOVERY_RECENTER, x1, y1, x2, y2)
        if is_centered_home_screen(frame, detections):
            self._home_frames += 1
            self._stage = f"confirm_home_{self._home_frames}/{self.required_home_frames}"
            if self._home_frames >= self.required_home_frames:
                self._complete = True
                self._stage = "done"
            return None
        self._home_frames = 0

        known_prompt = any(
            target_type in by_type
            for target_type in (
                AUTOPLACE_PROMPT,
                AUTOPLACE_NOTICE,
                SELECT_CONFIRM_PROMPT,
                NESTED_PARENT_WARNING,
            )
        )
        if known_prompt:
            no = _best(by_type.get(CONFIRM_NO))
            if no is None:
                self.logger.error("Hatch recovery | known prompt has no No button")
                self._stage = "prompt_without_no"
                self._failed = True
                return None
            self._stage = "cancel_prompt"
            return _synthetic(RECOVERY_NO, no.x, no.y)

        if AUTOPLACE_TITLE in by_type or SELECT_TITLE in by_type or NEST_TITLE in by_type:
            self._stage = "close_mask_layer"
            return _synthetic(
                RECOVERY_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )

        claim = _best(by_type.get(hatch_feature.CLAIM_BUTTON))
        if claim is not None:
            # Collecting a completed hatch/battle result is safer than Back:
            # it preserves the result and leads to another named screen.
            self._stage = "collect_result"
            return _synthetic(RECOVERY_CLAIM, claim.x, claim.y)

        close = _best(
            [
                *by_type.get(hatch_feature.CLOSE_BUTTON, ()),
                *by_type.get(CAVE_CLOSE_BUTTON, ()),
            ]
        )
        if close is not None:
            self._stage = "close_named_screen"
            return _synthetic(RECOVERY_CLOSE, close.x, close.y)

        map_exit = _best(by_type.get(HUNT_MAP_EXIT))
        if map_exit is not None:
            # A combined hatch+hunt process can be restarted during the hunt
            # cooldown. Android Back does not leave this map, but its explicit
            # exit control is stable and already template-gated.
            self._stage = "leave_hunt_map"
            return _synthetic(RECOVERY_MAP_EXIT, map_exit.x, map_exit.y)

        if is_home_screen(frame, detections) and CAVE in by_type:
            if self._recenter_swipes >= len(vectors):
                self.logger.error(
                    "Hatch recovery | cave view remains after %d safe return swipes",
                    self._recenter_swipes,
                )
            else:
                self._cave_recovery_required = True
                self._recenter_end = len(vectors)
                return self.choose(frame, detections)

        if is_home_screen(frame, detections):
            pile = _egg_pile_base_center(frame)
            if pile is not None:
                scale = frame.width / 900.0
                expected_x, expected_y = 450 * scale, 1455 * scale
                x_shifted = abs(pile[0] - expected_x) > 100 * scale
                y_too_high = pile[1] < expected_y - 100 * scale
                if x_shifted or y_too_high:
                    # A process may restart after the horizontal cave-return
                    # swipe. Resume at the vertical leg when the pile is
                    # already horizontally centred instead of replaying the
                    # first leg and pushing it past centre again.
                    self._recenter_swipes = 0 if x_shifted else 1
                    self._recenter_end = 2 if y_too_high else 1
                    self._cave_recovery_required = True
                    return self.choose(frame, detections)

        if self._back_attempts < self.max_back_attempts:
            self._back_attempts += 1
            self._stage = f"back_{self._back_attempts}/{self.max_back_attempts}"
            return _synthetic(RECOVERY_BACK, frame.width // 2, frame.height // 2)

        self.logger.error("Hatch recovery | unable to prove centered home after bounded escape")
        self._stage = "exhausted"
        self._failed = True
        return None


class AutoPlaceRoundPlanner:
    """Converge one Top/Mass tag, set auto-place sorting, and apply it."""

    def __init__(
        self,
        rule: AutoPlaceRule,
        *,
        reference_width: float = 900.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        mapping = {
            TOP_RULE.tag: (
                nest_filter_feature.TAG_TOP,
                nest_filter_feature.TAG_HDR_TOP,
                PLACE_SORT_BEST,
                PLACE_HDR_BEST,
                (450.0, 702.0),
            ),
            MASS_RULE.tag: (
                nest_filter_feature.TAG_MASS,
                nest_filter_feature.TAG_HDR_MASS,
                PLACE_SORT_LEVEL,
                PLACE_HDR_LEVEL,
                (450.0, 749.0),
            ),
        }
        if rule.tag not in mapping:
            raise ValueError(f"unsupported auto-place rule: {rule.tag}")
        (
            self.tag_option,
            self.tag_header,
            self.sort_option,
            self.sort_header,
            self.sort_option_point,
        ) = mapping[rule.tag]
        self.rule = rule
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self._filter = nest_filter_feature.NestTagFilterTestPlanner(
            reference_width=reference_width,
            target_label=rule.tag,
            target_option_type=self.tag_option,
            target_header_type=self.tag_header,
        )
        self._stage = "filter"
        self._complete = False

    def last_stage(self) -> str:
        return f"autoplace_{self.rule.tag}_{self._stage}"

    def is_complete(self) -> bool:
        return self._complete

    def on_action_success(self, target_type: str) -> None:
        if self._stage == "filter":
            self._filter.on_action_success(target_type)
            if self._filter.is_complete():
                self._stage = "open_settings"
            return
        if target_type == NEST_GEAR:
            self._stage = "settings"
        elif target_type in (PLACE_SORT_BEST, PLACE_SORT_LEVEL):
            self._stage = "verify_sort"
        elif target_type == AUTOPLACE_MASK_CLOSE:
            self._stage = "verify_settings_closed"
        elif target_type == AUTOPLACE_BUTTON:
            self._stage = "after_autoplace"
        elif target_type == AUTOPLACE_YES:
            self.logger.info(
                "Hatch auto-place | tag=%s | sort=%s | completed with confirmation",
                self.rule.tag,
                self.rule.sort_option,
            )
            self._stage = "done"
            self._complete = True

    def on_action_failure(self, target_type: str) -> None:
        self._stage = f"failed_{target_type}"

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            return None
        by_type = _group(detections)
        if self._stage == "filter":
            target = self._filter.choose(frame, detections)
            if target is None and self._filter.is_complete():
                self._stage = "open_settings"
                return self.choose(frame, detections)
            return target
        if self._stage == "open_settings":
            if NEST_TITLE not in by_type:
                return None
            if not _near(
                by_type.get(self.tag_header),
                _scaled(frame, (228.0, 168.0), self.reference_width),
                45,
            ):
                self._filter = nest_filter_feature.NestTagFilterTestPlanner(
                    reference_width=self.reference_width,
                    target_label=self.rule.tag,
                    target_option_type=self.tag_option,
                    target_header_type=self.tag_header,
                )
                self._stage = "filter"
                return self.choose(frame, detections)
            gear = _best(by_type.get(NEST_GEAR))
            if gear is None:
                return None
            return _target(gear)
        if self._stage == "settings":
            if AUTOPLACE_TITLE not in by_type:
                return None
            desired_header = _near(
                by_type.get(self.sort_header),
                _scaled(frame, (450.0, 648.0), self.reference_width),
                25,
            )
            if desired_header is not None:
                self._stage = "close_settings"
                return self.choose(frame, detections)
            desired_option = _near(
                by_type.get(self.sort_option),
                _scaled(frame, self.sort_option_point, self.reference_width),
                50,
            )
            if desired_option is not None:
                return _target(desired_option)
            return _synthetic(
                AUTOPLACE_SORT_HEADER,
                *_scaled(frame, (450.0, 648.0), self.reference_width),
            )
        if self._stage == "verify_sort":
            if AUTOPLACE_TITLE not in by_type:
                return None
            desired_header = _near(
                by_type.get(self.sort_header),
                _scaled(frame, (450.0, 648.0), self.reference_width),
                25,
            )
            if desired_header is not None:
                self._stage = "close_settings"
                return self.choose(frame, detections)
            self._stage = "settings"
            return self.choose(frame, detections)
        if self._stage == "close_settings":
            if AUTOPLACE_TITLE not in by_type:
                self._stage = "verify_settings_closed"
                return self.choose(frame, detections)
            return _synthetic(
                AUTOPLACE_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )
        if self._stage == "verify_settings_closed":
            if AUTOPLACE_TITLE in by_type:
                self._stage = "close_settings"
                return self.choose(frame, detections)
            if NEST_TITLE not in by_type:
                return None
            button = _best(by_type.get(AUTOPLACE_BUTTON))
            if button is None:
                return None
            return _target(button)
        if self._stage == "after_autoplace":
            if AUTOPLACE_PROMPT in by_type or AUTOPLACE_NOTICE in by_type:
                yes = _best(by_type.get(CONFIRM_YES))
                if yes is None:
                    self._stage = "blocked_notice_without_yes"
                    return None
                return _synthetic(AUTOPLACE_YES, yes.x, yes.y)
            if NEST_TITLE in by_type:
                self.logger.info(
                    "Hatch auto-place | tag=%s | sort=%s | completed without prompt",
                    self.rule.tag,
                    self.rule.sort_option,
                )
                self._stage = "done"
                self._complete = True
            return None
        return None


class CaveCullPlanner:
    """Navigate, make the threshold decision, and run one bounded cull."""

    def __init__(
        self,
        reader: DigitReader,
        *,
        threshold: int,
        reference_width: float = 900.0,
        safe_margin: int = 80,
        bottom_exclusion_px: int = 180,
        logger: logging.Logger | None = None,
    ) -> None:
        self.reader = reader
        self.threshold = max(0, threshold)
        self.reference_width = reference_width
        self.safe_margin = max(0, safe_margin)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.logger = logger or logging.getLogger("dino_bot")
        self.navigator = CaveNavigator(reference_width=reference_width)
        self._stage = "navigate"
        self._capacity_failures = 0
        self._navigation_swipes = 0
        self._return_swipes = 0
        self._recenter_checks = 0
        self._home_frames = 0
        self._complete = False

    def last_stage(self) -> str:
        return f"cave_{self._stage}"

    def is_complete(self) -> bool:
        return self._complete

    def on_action_success(self, target_type: str) -> None:
        if target_type == CAVE_SWIPE:
            self.navigator.on_swipe_result(moved=True)
            self._navigation_swipes += 1
        elif target_type == CAVE:
            self._stage = "cave_screen"
        elif target_type == CAVE_SELECT_BUTTON:
            self._stage = "select_dino"
        elif target_type == nest_filter_feature.TAG_ALL:
            self._stage = "select_weakest"
        elif target_type == SELECT_WEAKEST_BUTTON:
            self._stage = "confirm_selection"
        elif target_type == SELECT_CHOOSE_BUTTON:
            self._stage = "start_battle"
        elif target_type == CAVE_CONTINUOUS_BUTTON:
            self._stage = "battle_result"
        elif target_type == hatch_feature.CLAIM_BUTTON and self._stage == "battle_result":
            self._stage = "recenter"
        elif target_type == CAVE_RECENTER:
            self._return_swipes += 1
            self._stage = "recenter"

    def on_action_failure(self, target_type: str) -> None:
        if target_type == CAVE_SWIPE:
            self.navigator.on_swipe_result(moved=False)
            return
        self._stage = f"failed_{target_type}"

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            return None
        by_type = _group(detections)
        if self._stage == "navigate":
            cave = self._safe_cave(frame, by_type.get(CAVE))
            step = self.navigator.next_step(cave_visible=cave is not None, frame_width=frame.width)
            if step.kind == SWIPE:
                assert step.vector is not None
                x1, y1, x2, y2 = step.vector
                return _swipe_target(CAVE_SWIPE, x1, y1, x2, y2)
            if step.kind == RESCAN:
                return None
            if step.kind == STUCK:
                self.logger.warning("Hatch cave | navigation failed; recentering safely")
                self._stage = "recenter"
                return self.choose(frame, detections)
            assert step.kind == DONE and cave is not None
            count = read_dino_count(
                frame.image,
                self.reader,
                reference_width=self.reference_width,
            )
            if count is None:
                self._capacity_failures += 1
                if self._capacity_failures <= 2:
                    self.logger.warning(
                        "Hatch cave | capacity unreadable | retry=%d/2",
                        self._capacity_failures,
                    )
                    return None
                self.logger.error("Hatch cave | capacity unreadable; skipping cull")
                self._stage = "recenter"
                return self.choose(frame, detections)
            self.logger.info(
                "Hatch cave | capacity=%d/350 | threshold=%d | cull=%s",
                count,
                self.threshold,
                should_cull(count, self.threshold),
            )
            if not should_cull(count, self.threshold):
                self._stage = "recenter"
                return self.choose(frame, detections)
            self._stage = "open_cave"
            return _target(cave)
        if self._stage == "open_cave":
            cave = self._safe_cave(frame, by_type.get(CAVE))
            return _target(cave) if cave is not None else None
        if self._stage == "cave_screen":
            button = _best(by_type.get(CAVE_SELECT_BUTTON))
            return _target(button) if button is not None else None
        if self._stage == "select_dino":
            if SELECT_TITLE not in by_type:
                return None
            header = _near(
                by_type.get(nest_filter_feature.TAG_HDR_ALL),
                _scaled(frame, (228.0, 204.0), self.reference_width),
                45,
            )
            if header is not None:
                self._stage = "select_weakest"
                return self.choose(frame, detections)
            option = _near(
                by_type.get(nest_filter_feature.TAG_ALL),
                _scaled(frame, (228.0, 249.0), self.reference_width),
                50,
            )
            if option is not None:
                return _target(option)
            return _synthetic(
                SELECT_TAG_HEADER,
                *_scaled(frame, (228.0, 204.0), self.reference_width),
            )
        if self._stage == "select_weakest":
            if SELECT_TITLE not in by_type:
                return None
            button = _best(by_type.get(SELECT_WEAKEST_BUTTON))
            return _target(button) if button is not None else None
        if self._stage == "confirm_selection":
            if SELECT_TITLE not in by_type:
                return None
            button = _best(by_type.get(SELECT_CHOOSE_BUTTON))
            return _target(button) if button is not None else None
        if self._stage == "start_battle":
            button = _best(by_type.get(CAVE_CONTINUOUS_BUTTON))
            return _target(button) if button is not None else None
        if self._stage == "battle_result":
            claim = _best(by_type.get(hatch_feature.CLAIM_BUTTON))
            return _target(claim) if claim is not None else None
        if self._stage == "recenter":
            if is_centered_home_screen(frame, detections):
                self._stage = "verify_recenter"
                return self.choose(frame, detections)
            used_vectors = self.navigator.swipe_vectors[: self._navigation_swipes]
            # A resumed run can begin in the cave view without having observed
            # the outbound gestures. In that case, use the complete calibrated
            # return path; every gesture stays outside the protected UI bands.
            if not used_vectors:
                used_vectors = self.navigator.swipe_vectors
            vectors = _inverse_swipe_vectors(used_vectors)
            if self._return_swipes < len(vectors):
                x1, y1, x2, y2 = _scaled_swipe(
                    frame,
                    vectors[self._return_swipes],
                    self.reference_width,
                )
                return _swipe_target(CAVE_RECENTER, x1, y1, x2, y2)
            self._stage = "verify_recenter"
            return self.choose(frame, detections)
        if self._stage == "verify_recenter":
            if is_centered_home_screen(frame, detections):
                self._home_frames += 1
                if self._home_frames >= 2:
                    self._stage = "done"
                    self._complete = True
            else:
                self._home_frames = 0
                self._recenter_checks += 1
                if self._recenter_checks >= 3:
                    self.logger.error(
                        "Hatch cave | safe return swipes did not prove centered home"
                    )
                    self._stage = "recenter_failed"
            return None
        return None

    def _safe_cave(
        self,
        frame: Frame,
        items: list[Detection] | None,
    ) -> Detection | None:
        """Return the best cave whose tap point is outside UI exclusion bands."""

        if not items:
            return None
        scale = frame.width / self.reference_width
        margin = round(self.safe_margin * scale)
        bottom = round(self.bottom_exclusion_px * scale)
        safe = [
            item
            for item in items
            if margin <= item.x <= frame.width - margin
            and margin <= item.y <= frame.height - bottom
        ]
        rejected = len(items) - len(safe)
        if rejected:
            self.logger.warning(
                "Hatch cave | rejected %d target(s) inside protected screen edge",
                rejected,
            )
        return _best(safe)


class FullHatchPlanner:
    """Orchestrate Phase A, all Phase B rounds, and optional Phase C."""

    def __init__(
        self,
        reader: DigitReader,
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
        cull_threshold: int = 320,
        cave_safe_margin: int = 80,
        cave_bottom_exclusion_px: int = 180,
        recovery_timeout_seconds: float = 15.0,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.reader = reader
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self.cull_threshold = cull_threshold
        self.cave_safe_margin = max(0, cave_safe_margin)
        self.cave_bottom_exclusion_px = max(0, cave_bottom_exclusion_px)
        self.recovery_timeout_seconds = max(1.0, recovery_timeout_seconds)
        self.clock = clock or time.monotonic
        self._hatch_kwargs = dict(
            egg_pile_point=egg_pile_point,
            reference_width=reference_width,
            scroll_vector=scroll_vector,
            scroll_duration_ms=scroll_duration_ms,
            max_scrolls=max_scrolls,
            rescan_interval_seconds=rescan_interval_seconds,
            require_home_anchor=require_home_anchor,
            home_failure_limit=home_failure_limit,
            home_backoff_seconds=home_backoff_seconds,
            logger=self.logger,
        )
        if clock is not None:
            self._hatch_kwargs["clock"] = clock
        self._stage = "hatch"
        self._child: object = self._new_hatch()
        self._hatch_baseline = 0
        self._complete = False
        self._no_target_since: float | None = None
        self._recovery_reason: str | None = None
        self._collect_only_after_empty = False
        self._empty_rescan_wait = False
        self._navigation_failures: dict[tuple[str, str], int] = {}
        self._pending_claim_verification = False
        self.completed_management_cycles = 0

    def last_stage(self) -> str:
        child_stage = getattr(self._child, "last_stage", None)
        detail = child_stage() if callable(child_stage) else child_stage
        return f"full_{self._stage}:{detail or '-'}"

    def is_complete(self) -> bool:
        return self._complete

    def next_ready_delay_ms(self) -> int:
        method = getattr(self._child, "next_ready_delay_ms", None)
        return int(method()) if callable(method) else 0

    def is_hunt_cooldown_active(self) -> bool:
        """Whether the no-ready-egg cooldown can be spent hunting."""

        return self._empty_rescan_wait and self.next_ready_delay_ms() > 0

    def reset_workflow(self) -> None:
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._complete = False
        self._no_target_since = None
        self._recovery_reason = None
        self._collect_only_after_empty = False
        self._empty_rescan_wait = False
        self._navigation_failures.clear()
        self._pending_claim_verification = False

    def on_action_success(self, target_type: str) -> None:
        self._navigation_failures.pop((self._stage, target_type), None)
        if self._stage == "recover_home":
            self._recovery_child.on_action_success(target_type)
            return
        if self._stage == "hatch":
            hatch = self._hatch_child
            hatch.on_action_success(target_type)
            if target_type == hatch_feature.CLAIM_BUTTON:
                self._pending_claim_verification = False
            if target_type == hatch_feature.CLOSE_BUTTON:
                hatched = hatch.hatched - self._hatch_baseline
                if hatched > 0:
                    self.logger.info(
                        "Hatch full | phase A complete | hatched=%d | entering My Nest",
                        hatched,
                    )
                    self._enter_open_nest(collect_only=False)
                else:
                    self.logger.info(
                        "Hatch full | no ready incubator eggs | "
                        "collect all nest eggs before cooldown"
                    )
                    self._enter_open_nest(collect_only=True)
            return
        if self._stage == "open_nest" and target_type == OPEN_NEST:
            if self._collect_only_after_empty:
                self._start_collect()
            else:
                self._start_replacement("attack")
            return
        if self._stage in ("attack", "hp"):
            child = self._replacement_child
            child.on_action_success(target_type)
            self._advance_replacement_if_done()
            return
        if self._stage in ("top", "mass"):
            child = self._autoplace_child
            child.on_action_success(target_type)
            self._advance_autoplace_if_done()
            return
        if self._stage == "collect":
            if target_type == nest_filter_feature.TAG_ALL:
                self._stage = "collect_button"
            return
        if self._stage == "collect_button" and target_type == COLLECT_EGGS_BUTTON:
            self._stage = "close_nest"
            return
        if self._stage == "close_nest" and target_type == NEST_MASK_CLOSE:
            self._stage = "verify_nest_closed"
            return
        if self._stage == "cave":
            self._cave_child.on_action_success(target_type)

    def on_action_failure(self, target_type: str) -> None:
        if self._stage == "recover_home":
            self._recovery_child.on_action_failure(target_type)
            self._complete = True
            return
        if self._stage == "hatch":
            self._hatch_child.on_action_failure(target_type)
            if target_type == hatch_feature.CLAIM_BUTTON:
                self._pending_claim_verification = True
            if target_type == hatch_feature.EGG_PILE:
                # A roaming dinosaur can cover the old fixed pile point.  Its
                # detail card leaves HOME_ANCHOR visible behind a dark modal,
                # so blindly retrying the same coordinate could press one of
                # that card's action buttons.  Unwind it before retrying at a
                # newly measured point on the basket.
                self._begin_home_recovery("egg pile tap opened an unexpected screen")
            return
        retry_key = (self._stage, target_type)
        retry_count = self._navigation_failures.get(retry_key, 0)
        if (
            target_type in RETRYABLE_NAVIGATION_TARGETS
            and retry_count < MAX_NAVIGATION_RETRIES
        ):
            retry_count += 1
            self._navigation_failures[retry_key] = retry_count
            self.logger.warning(
                "Hatch full | retrying safe navigation in place | "
                "stage=%s target=%s retry=%d/%d",
                self._stage,
                target_type,
                retry_count,
                MAX_NAVIGATION_RETRIES,
            )
            return
        child = self._child
        method = getattr(child, "on_action_failure", None)
        if callable(method):
            method(target_type)
        # A failed mutation may have reached the game even when its expected
        # transition was missed. Unwind to a proven home screen instead of
        # guessing whether it is safe to continue the current stage.
        if self._stage in (
            "open_nest",
            "attack",
            "hp",
            "top",
            "mass",
            "collect",
            "collect_button",
            "close_nest",
            "cave",
        ):
            self._begin_home_recovery(
                f"verification failed at stage={self._stage} target={target_type}"
            )

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            return None
        if self._stage == "recover_home":
            target = self._recovery_child.choose(frame, detections)
            if self._recovery_child.is_failed():
                self.logger.error(
                    "Hatch full | recovery failed | reason=%s",
                    self._recovery_reason or "unknown",
                )
                self._complete = True
                return None
            if self._recovery_child.is_complete():
                self.logger.info(
                    "Hatch full | centered home confirmed | recovered=%s",
                    self._recovery_reason or "unknown",
                )
                self.reset_workflow()
                return self._choose_current(frame, detections)
            return target

        # The hatch HUD anchor remains visible when the outdoor map is still
        # shifted to the cave view.  HatchPlanner intentionally treats that
        # anchor as permission to tap the fixed egg-pile coordinate, but the
        # coordinate is only valid on the centred home map.  Route a resumed
        # or freshly started workflow through the existing bounded recovery
        # before the child can issue that unsafe/misplaced tap.
        if (
            self._stage == "hatch"
            and is_home_screen(frame, detections)
            and not is_centered_home_screen(frame, detections)
        ):
            self._begin_home_recovery("hatch started from shifted cave view")
            return self.choose(frame, detections)

        target = self._choose_current(frame, detections)
        if target is not None:
            self._no_target_since = None
            return target
        if self._stage == "recover_home":
            return self.choose(frame, detections)
        if self.next_ready_delay_ms() > 0:
            self._no_target_since = None
            return None
        if self._stage != "hatch" and is_centered_home_screen(frame, detections):
            self._begin_home_recovery("unexpected return to home")
            return self.choose(frame, detections)
        now = self.clock()
        if self._no_target_since is None:
            self._no_target_since = now
            return None
        stalled_seconds = now - self._no_target_since
        if stalled_seconds >= self.recovery_timeout_seconds:
            self._begin_home_recovery(
                f"no actionable target for {stalled_seconds:.0f}s at {self.last_stage()}"
            )
            return self.choose(frame, detections)
        return None

    def _choose_current(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        by_type = _group(detections)
        if self._stage == "hatch":
            detail_close = _unready_egg_detail_close(frame)
            if detail_close is not None:
                if self._pending_claim_verification:
                    self._hatch_child.hatched += 1
                    self._pending_claim_verification = False
                    self.logger.info(
                        "Hatch | claim confirmed by next unready egg detail | hatched=%d",
                        self._hatch_child.hatched,
                    )
                return _synthetic(HATCH_DETAIL_CLOSE, *detail_close)
            target = self._hatch_child.choose(frame, detections)
            if target is not None and target.type == hatch_feature.EGG_PILE:
                safe_point = _egg_pile_safe_tap(frame)
                if safe_point is not None:
                    return _synthetic(hatch_feature.EGG_PILE, *safe_point)
            return target
        if self._stage == "open_nest":
            if NEST_TITLE in by_type:
                if self._collect_only_after_empty:
                    self._start_collect()
                else:
                    self._start_replacement("attack")
                return self._choose_current(frame, detections)
            anchor = _best(by_type.get(hatch_feature.HOME_ANCHOR))
            return _synthetic(OPEN_NEST, anchor.x, anchor.y) if anchor is not None else None
        if self._stage in ("attack", "hp"):
            target = self._replacement_child.choose(frame, detections)
            if target is None:
                self._advance_replacement_if_done()
                if self._stage not in ("attack", "hp"):
                    return self._choose_current(frame, detections)
            return target
        if self._stage in ("top", "mass"):
            target = self._autoplace_child.choose(frame, detections)
            if target is None:
                self._advance_autoplace_if_done()
                if self._stage not in ("top", "mass"):
                    return self._choose_current(frame, detections)
            return target
        if self._stage == "collect":
            filter_planner = self._child
            assert isinstance(filter_planner, nest_filter_feature.NestTagFilterTestPlanner)
            target = filter_planner.choose(frame, detections)
            if target is None and filter_planner.is_complete():
                self._stage = "collect_button"
                return self._choose_current(frame, detections)
            return target
        if self._stage == "collect_button":
            if NEST_TITLE not in by_type:
                return None
            button = _best(by_type.get(COLLECT_EGGS_BUTTON))
            return _target(button) if button is not None else None
        if self._stage == "close_nest":
            if NEST_TITLE not in by_type:
                self._stage = "verify_nest_closed"
                return self._choose_current(frame, detections)
            return _synthetic(
                NEST_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )
        if self._stage == "verify_nest_closed":
            if NEST_TITLE in by_type:
                self._stage = "close_nest"
                return self._choose_current(frame, detections)
            if is_home_screen(frame, detections):
                if self._collect_only_after_empty:
                    self._start_empty_rescan_wait()
                    return self._choose_current(frame, detections)
                self._stage = "cave"
                self._child = CaveCullPlanner(
                    self.reader,
                    threshold=self.cull_threshold,
                    reference_width=self.reference_width,
                    safe_margin=self.cave_safe_margin,
                    bottom_exclusion_px=self.cave_bottom_exclusion_px,
                    logger=self.logger,
                )
                return self._choose_current(frame, detections)
            return None
        if self._stage == "cave":
            target = self._cave_child.choose(frame, detections)
            if target is None and self._cave_child.is_complete():
                self.completed_management_cycles += 1
                self.logger.info(
                    "Hatch full | completed management cycle %d | restarting Phase A",
                    self.completed_management_cycles,
                )
                self._stage = "hatch"
                self._child = self._new_hatch()
                self._hatch_baseline = 0
                return self._choose_current(frame, detections)
            return target
        return None

    def _begin_home_recovery(self, reason: str) -> None:
        self.logger.warning("Hatch full | recovering to centered home | %s", reason)
        self._stage = "recover_home"
        self._child = HatchHomeRecoveryPlanner(
            reference_width=self.reference_width,
            logger=self.logger,
        )
        self._no_target_since = None
        self._recovery_reason = reason

    def _new_hatch(self) -> hatch_feature.HatchPlanner:
        return hatch_feature.HatchPlanner(**self._hatch_kwargs)

    def _enter_open_nest(self, *, collect_only: bool) -> None:
        self._stage = "open_nest"
        # Drop HatchPlanner's own close-triggered wait while My Nest is being
        # opened. Otherwise that cooldown could mask recovery if the home
        # frame takes longer than expected to settle.
        self._child = object()
        self._collect_only_after_empty = collect_only
        self._empty_rescan_wait = False

    def _start_collect(self) -> None:
        self._stage = "collect"
        self._child = nest_filter_feature.NestTagFilterTestPlanner(
            reference_width=self.reference_width,
            target_label="所有",
            target_option_type=nest_filter_feature.TAG_ALL,
            target_header_type=nest_filter_feature.TAG_HDR_ALL,
        )

    def _start_empty_rescan_wait(self) -> None:
        self.logger.info(
            "Hatch full | nest eggs collected after empty incubator | starting rescan cooldown"
        )
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._collect_only_after_empty = False
        self._empty_rescan_wait = True
        self._hatch_child.begin_rescan_wait(
            "collected all nest eggs after no ready incubator eggs"
        )

    def _start_replacement(self, kind: str) -> None:
        if kind == "attack":
            self._stage = "attack"
            self._child = AttackReplacementTestPlanner(
                self.reader,
                reference_width=self.reference_width,
                rule=ATTACK_RULE,
                logger=self.logger,
            )
            return
        self._stage = "hp"
        self._child = AttackReplacementTestPlanner(
            self.reader,
            reference_width=self.reference_width,
            rule=HP_RULE,
            nest_filter_option=nest_filter_feature.TAG_HP,
            nest_filter_header=nest_filter_feature.TAG_HDR_HP,
            select_sort_option=select_sort_feature.SORT_HP,
            select_sort_header=select_sort_feature.SORT_HP,
            select_sort_menu_point=(650.0, 501.0),
            logger=self.logger,
        )

    def _advance_replacement_if_done(self) -> None:
        child = self._replacement_child
        if not child.is_complete():
            return
        if child.last_stage() != "replacement_done":
            self.logger.error(
                "Hatch full | replacement round needs recovery | %s",
                child.last_stage(),
            )
            self._begin_home_recovery(
                f"replacement round stopped at {child.last_stage()}"
            )
            return
        if self._stage == "attack":
            self._start_replacement("hp")
        else:
            self._stage = "top"
            self._child = AutoPlaceRoundPlanner(
                TOP_RULE,
                reference_width=self.reference_width,
                logger=self.logger,
            )

    def _advance_autoplace_if_done(self) -> None:
        child = self._autoplace_child
        if not child.is_complete():
            return
        if self._stage == "top":
            self._stage = "mass"
            self._child = AutoPlaceRoundPlanner(
                MASS_RULE,
                reference_width=self.reference_width,
                logger=self.logger,
            )
        else:
            self._start_collect()

    @property
    def _hatch_child(self) -> hatch_feature.HatchPlanner:
        assert isinstance(self._child, hatch_feature.HatchPlanner)
        return self._child

    @property
    def _replacement_child(self) -> AttackReplacementTestPlanner:
        assert isinstance(self._child, AttackReplacementTestPlanner)
        return self._child

    @property
    def _autoplace_child(self) -> AutoPlaceRoundPlanner:
        assert isinstance(self._child, AutoPlaceRoundPlanner)
        return self._child

    @property
    def _cave_child(self) -> CaveCullPlanner:
        assert isinstance(self._child, CaveCullPlanner)
        return self._child

    @property
    def _recovery_child(self) -> HatchHomeRecoveryPlanner:
        assert isinstance(self._child, HatchHomeRecoveryPlanner)
        return self._child


def _group(detections: Sequence[Detection]) -> dict[str, list[Detection]]:
    result: dict[str, list[Detection]] = {}
    for item in detections:
        result.setdefault(item.type, []).append(item)
    return result


def _scaled(
    frame: Frame,
    point: tuple[float, float],
    reference_width: float,
) -> tuple[int, int]:
    scale = frame.width / reference_width
    return (round(point[0] * scale), round(point[1] * scale))


def _scaled_swipe(
    frame: Frame,
    vector: tuple[int, int, int, int],
    reference_width: float,
) -> tuple[int, int, int, int]:
    scale = frame.width / reference_width
    return tuple(round(value * scale) for value in vector)  # type: ignore[return-value]


def _inverse_swipe_vectors(
    vectors: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Undo calibrated camera gestures in reverse order."""

    return tuple((x2, y2, x1, y1) for x1, y1, x2, y2 in reversed(vectors))


def _best(items: list[Detection] | None) -> Detection | None:
    return max(items, key=lambda item: item.confidence) if items else None


def _near(
    items: list[Detection] | None,
    point: tuple[int, int],
    radius_at_900: float,
) -> Detection | None:
    if not items:
        return None
    item = min(items, key=lambda hit: (hit.x - point[0]) ** 2 + (hit.y - point[1]) ** 2)
    # Production uses the calibrated 900-wide layout.  Keep the argument name
    # explicit so a future multi-resolution pass can scale it at the callers.
    if (item.x - point[0]) ** 2 + (item.y - point[1]) ** 2 > radius_at_900**2:
        return None
    return item


def _target(detection: Detection) -> Target:
    return Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )


def _synthetic(target_type: str, x: int, y: int) -> Target:
    detection = Detection(
        type=target_type,
        x=x,
        y=y,
        confidence=1.0,
        metadata={"synthetic": True},
    )
    return _target(detection)


def _swipe_target(
    target_type: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration_ms: int = 400,
) -> Target:
    detection = Detection(
        type=target_type,
        x=x1,
        y=y1,
        confidence=1.0,
        metadata={
            "synthetic": True,
            "swipe": {"x2": x2, "y2": y2, "duration_ms": duration_ms},
        },
    )
    return _target(detection)
