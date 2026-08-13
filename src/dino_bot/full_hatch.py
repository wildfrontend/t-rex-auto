"""End-to-end Auto Hatch workflow.

This module closes the loop described in ``docs/auto-hatch-plan.md``:

* hatch every ready egg;
* optimize Attack and HP parents;
* auto-place Top by best attributes and Mass by level;
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
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

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
from .cull import EXPECTED_CAPACITY, CapacityRead, probe_dino_count, should_cull
from .digits import DigitReader
from .hatch_inventory import HatchBoostInventoryStore
from .models import Detection, Frame, Target
from .nests import (
    ATTACK_RULE,
    DEFAULT_STAT_UPGRADE_GUARDS,
    HP_RULE,
    MASS_RULE,
    TOP_RULE,
    AutoPlaceRule,
    StatUpgradeGuard,
)
from .overlays import (
    AUTOPLACE_NOTICE,
    CONFIRM_NO,
    CONFIRM_YES,
    INCUBATOR_FULL_TOAST,
    NESTED_PARENT_WARNING,
    SELECT_CONFIRM_PROMPT,
)
from .parent_open import NEST_TITLE, SELECT_TITLE
from .stalls import EggPileSnapshot, HomeRecoverySnapshot, ParentStatsSnapshot

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
# 「我的巢」面板左外側的遮罩,900 寬座標。點它就關閉面板回主畫面。
NEST_MASK_POINT = (50.0, 800.0)
HATCH_DETAIL_CLOSE = "hatch_unready_detail_close"
HATCH_BOOST_BUTTON = "hatch_cooldown_boost_button"
HATCH_BOOST_CONFIRM = "hatch_cooldown_boost_confirm_yes"
HATCH_BOOST_POINT = (450.0, 1380.0)
# 按鈕帶中段(避開左側 50% 圖示與右側票券圖示)的取樣框,900 寬座標。
_BOOST_BAR_SAMPLE = (380, 1355, 520, 1405)
_BOOST_BAR_MIN_SATURATION = 80.0

CAVE_SWIPE = "hatch_cave_swipe"
CAVE_RECENTER = "hatch_cave_recenter"
CAVE_SELECT_BUTTON = "hatch_cave_select_button"
CAVE_CONTINUOUS_BUTTON = "hatch_cave_continuous_button"
CAVE_CLOSE_BUTTON = "hatch_cave_close_button"
SELECT_TAG_HEADER = "hatch_cull_tag_header"
SELECT_WEAKEST_BUTTON = "hatch_select_weakest_button"
SELECT_CHOOSE_BUTTON = "hatch_select_choose_button"
DEFAULT_CULL_BATCH_SIZE = 40
DEFAULT_CAPACITY_CONSISTENT_READS = 2

RECOVERY_NO = "hatch_recovery_no"
RECOVERY_MASK_CLOSE = "hatch_recovery_mask_close"
RECOVERY_CLOSE = "hatch_recovery_close"
RECOVERY_CLAIM = "hatch_recovery_claim"
RECOVERY_MAP_EXIT = "hatch_recovery_map_exit"
RECOVERY_FOREST = "hatch_recovery_forest_recenter"
RECOVERY_RECENTER = "hatch_recovery_recenter"
RECOVERY_UNDO = "hatch_recovery_undo_recenter"
RECOVERY_BACK = "hatch_recovery_back"
RECOVERY_HUNT_DIALOG_CLOSE = "hatch_recovery_hunt_dialog_close"
RECOVERY_HUNT_DIALOG_DISMISS = "hatch_recovery_hunt_dialog_dismiss"
# 收掉狩獵氣泡框用的地圖空點,900 寬座標。左側中下最不容易壓到 HUD;真的
# 壓到別隻恐龍也只是換一個氣泡,下一輪再收,代價與現況相同。
HUNT_DIALOG_DISMISS_POINT = (110.0, 1200.0)
HUNT_DIALOG_CLOSE = "hunt_dialog_close_button"
HUNT_MAP_EXIT = "map_exit_nest_button"
FOREST_RECENTER = "forest_recenter_button"
STARTUP_GROWTH_RESULT = "startup_growth_result_back"
STARTUP_AUTO_BATTLE_CLOSE = "startup_auto_battle_close"
STARTUP_NEST_SHORTCUT = "hatch_startup_nest_shortcut"
STARTUP_SIMPLE_INTERRUPTS: tuple[str, ...] = (
    "duplicate_login_close_button",
    "device_history_confirm_button",
    "startup_offer_dismiss",
)
STARTUP_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_SIMPLE_INTERRUPTS,
        STARTUP_GROWTH_RESULT,
        STARTUP_AUTO_BATTLE_CLOSE,
    }
)
STARTUP_INTERRUPTS: frozenset[str] = frozenset(
    {*STARTUP_DETECTION_TYPES, STARTUP_NEST_SHORTCUT}
)
HUNT_ACTIVE_TYPES: frozenset[str] = frozenset(
    {
        HUNT_MAP_EXIT,
        "hunt_button",
        "hunt_max_group_button",
        "hunt_confirm_button",
        "hunt_dialog_close_button",
    }
)
STANDALONE_STAGES: frozenset[str] = frozenset(
    {"hatch", "attack", "hp", "collect", "cave"}
)
SCREENING_STAGES: tuple[str, ...] = ("attack", "hp", "top", "mass")

PLACE_SORT_BEST = "hatch_place_sort_best"
PLACE_SORT_LEVEL = "hatch_place_sort_level"
PLACE_HDR_BEST = "hatch_place_hdr_best"
PLACE_HDR_LEVEL = "hatch_place_hdr_level"
# The redesigned auto-place dialog moved upward, but the dropdown order is
# unchanged. Coordinates are in the 900-wide reference layout.
AUTOPLACE_SORT_HEADER_POINT = (450.0, 576.0)
AUTOPLACE_SORT_BEST_POINT = (450.0, 630.0)
AUTOPLACE_SORT_LEVEL_POINT = (450.0, 678.0)

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
    HATCH_BOOST_BUTTON: "tap",
    HATCH_BOOST_CONFIRM: "tap",
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
    RECOVERY_FOREST: "tap",
    RECOVERY_RECENTER: "swipe",
    RECOVERY_UNDO: "swipe",
    RECOVERY_BACK: "back",
    RECOVERY_HUNT_DIALOG_CLOSE: "tap",
    RECOVERY_HUNT_DIALOG_DISMISS: "tap",
    STARTUP_GROWTH_RESULT: "tap",
    STARTUP_AUTO_BATTLE_CLOSE: "tap",
    STARTUP_NEST_SHORTCUT: "tap",
    **{target_type: "tap" for target_type in STARTUP_SIMPLE_INTERRUPTS},
}

DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    **hatch_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    **replacement_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    OPEN_NEST: 3500,
    NEST_GEAR: 3000,
    AUTOPLACE_SORT_HEADER: 5000,
    PLACE_SORT_BEST: 2500,
    PLACE_SORT_LEVEL: 2500,
    AUTOPLACE_MASK_CLOSE: 2500,
    AUTOPLACE_BUTTON: 4000,
    AUTOPLACE_YES: 5000,
    COLLECT_EGGS_BUTTON: 5000,
    NEST_MASK_CLOSE: 3000,
    HATCH_DETAIL_CLOSE: 3000,
    HATCH_BOOST_BUTTON: 3000,
    HATCH_BOOST_CONFIRM: 3000,
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
    RECOVERY_FOREST: 4000,
    RECOVERY_RECENTER: 4000,
    RECOVERY_UNDO: 4000,
    RECOVERY_BACK: 4000,
    RECOVERY_HUNT_DIALOG_CLOSE: 3000,
    RECOVERY_HUNT_DIALOG_DISMISS: 3000,
    STARTUP_GROWTH_RESULT: 3000,
    STARTUP_AUTO_BATTLE_CLOSE: 3000,
    STARTUP_NEST_SHORTCUT: 4000,
    **{target_type: 5000 for target_type in STARTUP_SIMPLE_INTERRUPTS},
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
    HATCH_BOOST_BUTTON: (CONFIRM_YES, CONFIRM_NO),
    HATCH_BOOST_CONFIRM: (hatch_feature.INCUBATOR_TITLE,),
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
    RECOVERY_FOREST: (HUNT_MAP_EXIT,),
    # 氣泡收掉的證據就是它蓋住的那些地圖控制項重新露出來。這個檢查同時
    # 擋掉 Back 那種「合成目標當然不見了」的假成功。
    RECOVERY_HUNT_DIALOG_CLOSE: (HUNT_MAP_EXIT, FOREST_RECENTER),
    RECOVERY_HUNT_DIALOG_DISMISS: (HUNT_MAP_EXIT, FOREST_RECENTER),
    hatch_feature.CLAIM_BUTTON: (
        hatch_feature.HATCH_BUTTON,
        hatch_feature.INCUBATOR_TITLE,
        hatch_feature.HATCH_LABEL,
        hatch_feature.CLAIM_BUTTON,
        hatch_feature.HOME_ANCHOR,
    ),
    # The growth-result shortcut opens the auto-battle overlay before both
    # layers return to home.  HOME_ANCHOR can remain visible behind either
    # modal, so application.py also requires the acted-on startup target to
    # disappear before accepting either transition.
    STARTUP_GROWTH_RESULT: (STARTUP_AUTO_BATTLE_CLOSE, hatch_feature.HOME_ANCHOR),
    STARTUP_AUTO_BATTLE_CLOSE: (hatch_feature.HOME_ANCHOR,),
    STARTUP_NEST_SHORTCUT: (NEST_TITLE,),
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
MAX_SCREENING_RECOVERY_FAILURES = 3

# The centred home map is proven by the cyan egg-pile base rather than a
# template, because the pile artwork changes with its contents while the base
# does not.  Reference and tolerance live together so the "is it centred" test
# and the correction that follows it cannot drift apart.
HOME_PILE_BASE: tuple[float, float] = (450.0, 1455.0)
HOME_PILE_TOLERANCE = 100.0

# One observed run spent 111 seconds and 25 actions bouncing between home and
# the hunt map before its attempt counters ran out.  Recovery is a detour, not
# the work: bound the whole episode, retries included, by wall clock too.
MAX_HOME_RECOVERY_SECONDS = 45.0


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
        *HUNT_ACTIVE_TYPES,
    }
)


# Keep the full-hatch workflow scoped even when it is combined with hunting.
# The hatch manifest contains controls for every management phase, while the
# hunt manifest adds the dinosaur label and the HSV hunt path.  Scanning both
# manifests on every hatch cycle is the expensive path seen in production
# logs (roughly 5-10 seconds versus sub-second scoped scans).
HATCH_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        *hatch_feature.DEFAULT_TARGET_ACTIONS,
        # Required state evidence for HatchPlanner; without it a grid can
        # show ready "孵化" labels but still have no actionable target.
        hatch_feature.INCUBATOR_TITLE,
        hatch_feature.HOME_ANCHOR,
        hatch_feature.EXPEL_BUTTON,
        CONFIRM_YES,
        CONFIRM_NO,
        *HOME_FOREGROUND_TYPES,
        "duplicate_hunt_alert",
    }
)

NEST_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        hatch_feature.HOME_ANCHOR,
        NEST_TITLE,
        COLLECT_EGGS_BUTTON,
        SELECT_TITLE,
        *nest_filter_feature.DEFAULT_TARGET_ACTIONS,
        # The option tap is verified against a collapsed tag header, but the
        # replacement planners must prove that same header again on
        # the following planning frame before touching parents or settings.
        # Keeping only actionable controls in this scoped set made every
        # verified Attack/HP/Top/Mass selection disappear one frame later and
        # parked the workflow at ``target_filter_required``.
        *nest_filter_feature.HEADER_LABELS,
        *select_sort_feature.DEFAULT_TARGET_ACTIONS,
        # 排序表頭狀態:看得見「已是攻擊力排序」才能跳過重選。
        select_sort_feature.SORT_HDR_ATTACK,
        *replacement_feature.DEFAULT_TARGET_ACTIONS,
        NEST_GEAR,
        AUTOPLACE_TITLE,
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        # 排序設定完成後要按的主按鈕;漏掃會讓 top/mass 階段
        # 在 verify_settings_closed 空轉直到恢復假完成。
        AUTOPLACE_BUTTON,
        PLACE_HDR_BEST,
        PLACE_HDR_LEVEL,
        # 展開的排序選單選項也要掃:規劃看不見它們時會重按表頭,
        # 把剛打開的選單又關上,top/mass 階段就此死循環。
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        COLLECT_EGGS_BUTTON,
        INCUBATOR_FULL_TOAST,
        CONFIRM_NO,
        CONFIRM_YES,
        NESTED_PARENT_WARNING,
        SELECT_CONFIRM_PROMPT,
        *HOME_FOREGROUND_TYPES,
    }
)

# My Nest used to scan every management template on every frame (55 types in
# the live log).  These stage sets retain every screen proof and interruption
# each branch can consume while excluding controls from unreachable sibling
# workflows.
NEST_BASE_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        hatch_feature.HOME_ANCHOR,
        NEST_TITLE,
        COLLECT_EGGS_BUTTON,
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        CONFIRM_NO,
    }
)
NEST_REPLACEMENT_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *NEST_BASE_DETECTION_TYPES,
        SELECT_TITLE,
        *nest_filter_feature.DEFAULT_TARGET_ACTIONS,
        *nest_filter_feature.HEADER_LABELS,
        *select_sort_feature.DEFAULT_TARGET_ACTIONS,
        select_sort_feature.SORT_HDR_ATTACK,
        *replacement_feature.DEFAULT_TARGET_ACTIONS,
        CONFIRM_NO,
        CONFIRM_YES,
        NESTED_PARENT_WARNING,
        SELECT_CONFIRM_PROMPT,
    }
)
NEST_AUTOPLACE_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *NEST_BASE_DETECTION_TYPES,
        *nest_filter_feature.DEFAULT_TARGET_ACTIONS,
        *nest_filter_feature.HEADER_LABELS,
        NEST_GEAR,
        AUTOPLACE_TITLE,
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        AUTOPLACE_BUTTON,
        PLACE_HDR_BEST,
        PLACE_HDR_LEVEL,
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        CONFIRM_NO,
        CONFIRM_YES,
    }
)
NEST_COLLECT_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *NEST_BASE_DETECTION_TYPES,
        *nest_filter_feature.DEFAULT_TARGET_ACTIONS,
        *nest_filter_feature.HEADER_LABELS,
        COLLECT_EGGS_BUTTON,
        INCUBATOR_FULL_TOAST,
    }
)

CAVE_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        hatch_feature.HOME_ANCHOR,
        NEST_TITLE,
        SELECT_TITLE,
        CAVE,
        CAVE_CLOSE_BUTTON,
        CAVE_SELECT_BUTTON,
        CAVE_CONTINUOUS_BUTTON,
        SELECT_WEAKEST_BUTTON,
        SELECT_CHOOSE_BUTTON,
        # select_dino 階段的「所有」標籤:表頭已對就跳過、選單開著
        # 就點選項 — 兩個防呆都要看得見這兩型才會生效。
        nest_filter_feature.TAG_HDR_ALL,
        nest_filter_feature.TAG_ALL,
        hatch_feature.CLAIM_BUTTON,
        CONFIRM_NO,
        CONFIRM_YES,
        NESTED_PARENT_WARNING,
        SELECT_CONFIRM_PROMPT,
        *HOME_FOREGROUND_TYPES,
    }
)

# Recovery runs before the workflow knows which foreground is open.  It still
# does not need every action inside every foreground: it only needs enough
# anchors to cancel/close known overlays, leave an active hunt map, and prove
# that the centred home screen is visible.  Keeping the phase-specific action
# sets here made every recovery frame pay for all hatch, nest, and cave
# controls, which is especially expensive on slower devices.
RECOVERY_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        *HOME_FOREGROUND_TYPES,
        hatch_feature.HOME_ANCHOR,
        hatch_feature.CLOSE_BUTTON,
        CAVE,
        CAVE_CLOSE_BUTTON,
        CONFIRM_YES,
        CONFIRM_NO,
        HUNT_MAP_EXIT,
        FOREST_RECENTER,
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
    return _is_bright_outdoor_map(frame)


def _is_bright_outdoor_map(frame: Frame) -> bool:
    """Reject dimmed/unknown foregrounds without requiring a template."""

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


def home_pile_offset(frame: Frame) -> tuple[float, float] | None:
    """Return how far the egg-pile base sits from its centred position.

    Positive values are the distance the map content must still travel, so a
    correction gesture can use the pair directly as its drag vector.
    """

    pile = _egg_pile_base_center(frame)
    if pile is None:
        return None
    scale = frame.width / 900.0
    return (
        HOME_PILE_BASE[0] * scale - pile[0],
        HOME_PILE_BASE[1] * scale - pile[1],
    )


def is_centered_home_screen(frame: Frame, detections: Sequence[Detection]) -> bool:
    """Return whether the normal home map (not the shifted cave view) is ready."""

    types = {item.type for item in detections}
    if any(item.type == CAVE for item in detections) or types & HOME_FOREGROUND_TYPES:
        return False
    # The home anchor can be clipped after a successful map return even when
    # the stable egg-pile base is exactly centred. The Forest control is a
    # second named outdoor-map landmark; require it for this narrow fallback
    # instead of accepting an arbitrary bright screen with cyan pixels.
    if hatch_feature.HOME_ANCHOR not in types and FOREST_RECENTER not in types:
        return False
    if not _is_bright_outdoor_map(frame):
        return False
    offset = home_pile_offset(frame)
    if offset is None:
        return False
    scale = frame.width / 900.0
    return max(abs(offset[0]), abs(offset[1])) <= HOME_PILE_TOLERANCE * scale


def _hatch_boost_ready(frame: Frame) -> bool:
    """Return whether the incubator cooldown-boost bar is pressable.

    The bar keeps its template shape while a boost is running, but the game
    desaturates it to gray for the countdown; normalized template matching is
    brightness-invariant, so color saturation is the only reliable signal.
    """

    if frame.image.size == 0:
        return False
    scale = frame.width / 900.0
    x0, y0, x1, y1 = (round(value * scale) for value in _BOOST_BAR_SAMPLE)
    roi = frame.image[y0:y1, x0:x1]
    if not roi.size:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= _BOOST_BAR_MIN_SATURATION


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


def is_unready_egg_detail(frame: Frame) -> bool:
    """Whether the post-claim frame is the structurally proven unready detail."""

    return _unready_egg_detail_close(frame) is not None


class HatchHomeRecoveryPlanner:
    """Unwind known/unknown foregrounds and prove the map is centered again."""

    def __init__(
        self,
        *,
        reference_width: float = 900.0,
        logger: logging.Logger | None = None,
        max_back_attempts: int = 2,
        required_home_frames: int = 2,
        max_forest_trips: int = 1,
        max_measured_corrections: int = 2,
        max_hunt_dialog_dismissals: int = 3,
    ) -> None:
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self.max_back_attempts = max(0, max_back_attempts)
        self.required_home_frames = max(1, required_home_frames)
        self.max_forest_trips = max(0, max_forest_trips)
        self.max_hunt_dialog_dismissals = max(0, max_hunt_dialog_dismissals)
        self.max_measured_corrections = max(0, max_measured_corrections)
        self._stage = "inspect"
        self._back_attempts = 0
        self._home_frames = 0
        self._recenter_swipes = 0
        self._cave_recovery_required = False
        self._recenter_end = 0
        self._forest_trips = 0
        self._forest_refused = False
        self._hunt_dialog_dismissals = 0
        self._measured_corrections = 0
        self._last_offset: tuple[float, float] | None = None
        self._applied_swipes: list[tuple[int, int, int, int]] = []
        self._undone_swipes = 0
        self._pending_swipe: tuple[int, int, int, int] | None = None
        self._pending_cave_leg = False
        self._complete = False
        self._failed = False

    def last_stage(self) -> str:
        return f"recover_home_{self._stage}"

    def forest_trips(self) -> int:
        """How many times this planner entered the hunt map to recentre."""

        return self._forest_trips

    def is_complete(self) -> bool:
        return self._complete

    def is_failed(self) -> bool:
        return self._failed

    def on_action_success(self, target_type: str) -> None:
        if target_type == RECOVERY_RECENTER:
            if self._pending_cave_leg:
                # Only the calibrated route is indexed by this counter.  A
                # measured nudge that advanced it would make a later cave
                # return skip its first leg and stop halfway home.
                self._recenter_swipes += 1
            if self._pending_swipe is not None:
                # Remember the gesture, not just the count: a camera move that
                # loses the home reference can only be undone by replaying the
                # exact vector backwards.
                self._applied_swipes.append(self._pending_swipe)
        elif target_type == RECOVERY_UNDO:
            self._undone_swipes += 1
        elif target_type == RECOVERY_FOREST:
            self._forest_trips += 1
        self._pending_swipe = None
        self._pending_cave_leg = False
        self._stage = f"verify_{target_type}"

    def on_action_failure(self, target_type: str) -> None:
        self.logger.error("Hatch recovery | action failed | target=%s", target_type)
        self._pending_swipe = None
        self._pending_cave_leg = False
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
            return self._swipe(RECOVERY_RECENTER, x1, y1, x2, y2, cave_leg=True)
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

        # A hunt prompt covers the map controls the rung above needs, and Back
        # does not close it: six presses against one left the button's box and
        # confidence identical every frame at pixel_change=0.000, and the
        # escape ladder spent its whole budget on them before failing. Clear
        # the prompt first so the named exit can be seen at all.
        hunt_prompt = _best(
            [
                *by_type.get("hunt_button", ()),
                *by_type.get("hunt_max_group_button", ()),
            ]
        )
        if hunt_prompt is not None:
            dialog_close = _best(by_type.get(HUNT_DIALOG_CLOSE))
            if dialog_close is not None:
                self._stage = "close_hunt_dialog"
                return _synthetic(
                    RECOVERY_HUNT_DIALOG_CLOSE,
                    dialog_close.x,
                    dialog_close.y,
                )
            if self._hunt_dialog_dismissals < self.max_hunt_dialog_dismissals:
                # The level-up bubble carries no close button of its own; a tap
                # on empty map is what dismisses it.
                self._hunt_dialog_dismissals += 1
                self._stage = (
                    f"dismiss_hunt_dialog_{self._hunt_dialog_dismissals}"
                    f"/{self.max_hunt_dialog_dismissals}"
                )
                return _synthetic(
                    RECOVERY_HUNT_DIALOG_DISMISS,
                    *_scaled(frame, HUNT_DIALOG_DISMISS_POINT, self.reference_width),
                )

        # Correcting the camera against a landmark that is still on screen
        # outranks any blind escape below: those only open and close screens,
        # and none of them can put the map back where it belongs.
        recenter = self._recenter_target(frame, detections, by_type, vectors)
        if recenter is not None:
            return recenter

        forest = _best(by_type.get(FOREST_RECENTER))
        if forest is not None and self._forest_trips < self.max_forest_trips:
            # The bottom-right Forest button survives positions where the
            # hatch home anchor is clipped off-screen. Entering Forest and
            # immediately using its named map-exit control has been observed
            # to restore a clipped anchor, but it returns to the *previous*
            # home camera position, so it cannot fix a panned map.  One trip
            # proves which case this is; repeating it only burns wall clock.
            self._stage = "enter_forest_for_recenter"
            return _synthetic(RECOVERY_FOREST, forest.x, forest.y)
        if forest is not None and not self._forest_refused:
            self._forest_refused = True
            self.logger.error(
                "Hatch recovery | home still unproven and the forest round trip"
                " is spent (trips=%d, budget=%d); it does not move the camera"
                " on this screen",
                self._forest_trips,
                self.max_forest_trips,
            )

        if self._back_attempts < self.max_back_attempts:
            self._back_attempts += 1
            self._stage = f"back_{self._back_attempts}/{self.max_back_attempts}"
            return _synthetic(RECOVERY_BACK, frame.width // 2, frame.height // 2)

        # Last resort, deliberately after the escape ladder: the frame captured
        # straight after any action is often still mid-transition and matches
        # nothing, and undoing a correction that actually worked because of one
        # such frame would be worse than the loop this repairs.  Reaching here
        # means the settled screen really has no landmark left.
        undo = self._undo_target()
        if undo is not None:
            return undo

        self.logger.error("Hatch recovery | unable to prove centered home after bounded escape")
        self._stage = "exhausted"
        self._failed = True
        return None

    def _swipe(
        self,
        target_type: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        cave_leg: bool = False,
    ) -> Target:
        """Issue a camera gesture, holding it until the action is confirmed."""

        if target_type == RECOVERY_RECENTER:
            self._pending_swipe = (x1, y1, x2, y2)
            self._pending_cave_leg = cave_leg
        return _swipe_target(target_type, x1, y1, x2, y2)

    def _recenter_target(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        by_type: dict[str, list[Detection]],
        vectors: Sequence[tuple[int, int, int, int]],
    ) -> Target | None:
        """Move the camera while a home landmark is still measurable."""

        if not is_home_screen(frame, detections):
            return None
        if CAVE in by_type:
            # The cave view is a calibrated displacement, so replaying the
            # inverse route is exact.  A measured nudge is not available here:
            # the egg pile is outside the viewport at the cave position.
            if self._recenter_swipes >= len(vectors):
                self.logger.error(
                    "Hatch recovery | cave view remains after %d safe return swipes",
                    self._recenter_swipes,
                )
                return None
            self._cave_recovery_required = True
            self._recenter_end = len(vectors)
            x1, y1, x2, y2 = _scaled_swipe(
                frame,
                vectors[self._recenter_swipes],
                self.reference_width,
            )
            self._stage = "recenter_cave_view"
            return self._swipe(RECOVERY_RECENTER, x1, y1, x2, y2, cave_leg=True)

        offset = home_pile_offset(frame)
        if offset is None:
            return None
        scale = frame.width / 900.0
        if max(abs(offset[0]), abs(offset[1])) <= HOME_PILE_TOLERANCE * scale:
            return None
        if (
            self._last_offset is not None
            and _hypot(offset) >= _hypot(self._last_offset)
        ):
            # Two corrections that do not converge mean the map is not
            # responding to the gesture the way the measurement assumes.
            # Repeating it walks the camera further from home, not closer.
            self.logger.error(
                "Hatch recovery | measured correction did not reduce the offset"
                " | before=(%.0f,%.0f) | after=(%.0f,%.0f)",
                self._last_offset[0],
                self._last_offset[1],
                offset[0],
                offset[1],
            )
            return None
        if self._measured_corrections >= self.max_measured_corrections:
            self.logger.error(
                "Hatch recovery | egg pile still off by (%.0f,%.0f)px"
                " after %d measured corrections",
                offset[0],
                offset[1],
                self._measured_corrections,
            )
            return None
        # Drag by exactly the measured offset instead of replaying a
        # calibrated leg.  The trigger tolerance is 100px while a calibrated
        # leg is 450px, so a full leg turns a small offset into a larger one
        # in the opposite direction - and once the pile and the anchor leave
        # the viewport, nothing on screen can measure the mistake.
        self._measured_corrections += 1
        self._last_offset = offset
        x1, y1, x2, y2 = self._nudge(frame, offset)
        self._stage = (
            f"recenter_pile_{self._measured_corrections}"
            f"/{self.max_measured_corrections}"
        )
        self.logger.info(
            "Hatch recovery | nudging map by (%.0f,%.0f)px | attempt=%d/%d",
            offset[0],
            offset[1],
            self._measured_corrections,
            self.max_measured_corrections,
        )
        return self._swipe(RECOVERY_RECENTER, x1, y1, x2, y2)

    def _nudge(
        self,
        frame: Frame,
        offset: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        """Build a drag that moves map content by ``offset``.

        The gesture is centred on the travel so both ends stay clear of the
        edges; a clamped end only shortens the correction, which the next
        measurement can finish, while an off-screen end would be dropped.
        """

        margin = round(80 * frame.width / 900.0)
        dx, dy = round(offset[0]), round(offset[1])

        def leg(span: int, travel: int) -> tuple[int, int]:
            start = round(span / 2 - travel / 2)
            start = max(margin, min(span - margin, start))
            end = max(margin, min(span - margin, start + travel))
            return start, end

        x1, x2 = leg(frame.width, dx)
        y1, y2 = leg(frame.height, dy)
        return x1, y1, x2, y2

    def _undo_target(self) -> Target | None:
        """Reverse this planner's own camera move once the landmarks are gone.

        Every branch that can prove or measure the home position needs either
        the home anchor or the egg pile.  A gesture that pushes both out of
        the viewport therefore blinds the planner to its own mistake, and no
        later branch can repair it: the escape ladder below only opens and
        closes screens.  Replaying the gesture backwards is the one move that
        restores something to measure against.
        """

        if self._undone_swipes >= len(self._applied_swipes):
            return None
        x1, y1, x2, y2 = self._applied_swipes[-1 - self._undone_swipes]
        self._stage = (
            f"undo_recenter_{self._undone_swipes + 1}/{len(self._applied_swipes)}"
        )
        self.logger.warning(
            "Hatch recovery | no home landmark after own camera move"
            " | replaying (%d,%d)->(%d,%d) backwards",
            x1,
            y1,
            x2,
            y2,
        )
        return self._swipe(RECOVERY_UNDO, x2, y2, x1, y1)


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
                AUTOPLACE_SORT_BEST_POINT,
            ),
            MASS_RULE.tag: (
                nest_filter_feature.TAG_MASS,
                nest_filter_feature.TAG_HDR_MASS,
                PLACE_SORT_LEVEL,
                PLACE_HDR_LEVEL,
                AUTOPLACE_SORT_LEVEL_POINT,
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
                _scaled(frame, AUTOPLACE_SORT_HEADER_POINT, self.reference_width),
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
                *_scaled(frame, AUTOPLACE_SORT_HEADER_POINT, self.reference_width),
            )
        if self._stage == "verify_sort":
            if AUTOPLACE_TITLE not in by_type:
                return None
            desired_header = _near(
                by_type.get(self.sort_header),
                _scaled(frame, AUTOPLACE_SORT_HEADER_POINT, self.reference_width),
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


class CapacitySnapshot(Protocol):
    """Writes the frame behind an unreadable capacity HUD.

    Structural, so the planner stays independent of where evidence lands and
    tests can assert on a list instead of a temp directory.
    """

    def capture(
        self,
        frame: Frame,
        read: CapacityRead,
        *,
        stage: str,
        attempts: int,
    ) -> Path | None: ...


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
        selection_size: int = DEFAULT_CULL_BATCH_SIZE,
        allow_cull: bool = True,
        capacity_limit: int = EXPECTED_CAPACITY,
        capacity_read_retries: int = 2,
        capacity_consistent_reads: int = DEFAULT_CAPACITY_CONSISTENT_READS,
        cave_recenter_checks: int = 3,
        capacity_snapshots: CapacitySnapshot | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.reader = reader
        self.threshold = max(0, threshold)
        self.reference_width = reference_width
        self.safe_margin = max(0, safe_margin)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.selection_size = max(1, selection_size)
        self.allow_cull = bool(allow_cull)
        if capacity_limit <= 0:
            raise ValueError("capacity_limit must be greater than zero")
        self.capacity_limit = capacity_limit
        self.capacity_read_retries = max(1, capacity_read_retries)
        self.capacity_consistent_reads = max(1, capacity_consistent_reads)
        self.cave_recenter_checks = max(1, cave_recenter_checks)
        self.capacity_snapshots = capacity_snapshots
        self.logger = logger or logging.getLogger("dino_bot")
        self.navigator = CaveNavigator(reference_width=reference_width)
        self._stage = "navigate"
        self._capacity_failures = 0
        self._navigation_swipes = 0
        self._return_swipes = 0
        self._recenter_checks = 0
        self._home_frames = 0
        self._complete = False
        self._capacity_before: int | None = None
        self._capacity_readable: bool | None = None
        self._cull_required = False
        self._capacity_candidate: int | None = None
        self._capacity_confirmations = 0
        self._selected_count = 0

    def last_stage(self) -> str:
        return f"cave_{self._stage}"

    def is_complete(self) -> bool:
        return self._complete

    def _read_capacity(self, frame: Frame) -> CapacityRead:
        return probe_dino_count(
            frame.image,
            self.reader,
            reference_width=self.reference_width,
            expected_capacity=self.capacity_limit,
        )

    def _snapshot_capacity(self, frame: Frame, read: CapacityRead) -> None:
        if self.capacity_snapshots is None:
            return
        self.capacity_snapshots.capture(
            frame,
            read,
            stage=self.last_stage(),
            attempts=self._capacity_failures,
        )

    @property
    def capacity_readable(self) -> bool:
        """Whether this run proved the configured population readout."""

        return self._capacity_readable is True

    @property
    def cull_required(self) -> bool:
        """Whether the readable capacity crossed the configured threshold."""

        return self._cull_required

    @property
    def last_capacity(self) -> int | None:
        """最後一次成功讀到的洞穴數量。"""

        return self._capacity_before

    @property
    def expected_population(self) -> int | None:
        """淘汰(若有)完成後的預期洞穴數量。"""

        if self._capacity_before is None:
            return None
        return max(0, self._capacity_before - self._selected_count)

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
            if self._capacity_before is not None:
                self._selected_count = min(self.selection_size, self._capacity_before)
            self._stage = "confirm_selection"
        elif target_type == SELECT_CHOOSE_BUTTON:
            self._stage = "start_battle"
        elif target_type == CAVE_CONTINUOUS_BUTTON:
            self._stage = "battle_result"
        elif target_type == hatch_feature.CLAIM_BUTTON and self._stage == "battle_result":
            if self._capacity_before is not None and self._selected_count:
                self.logger.info(
                    "Hatch cave | cull completed | before=%d/%d | selected=%d"
                    " | expected_after=%d | result=claim_verified",
                    self._capacity_before,
                    self.capacity_limit,
                    self._selected_count,
                    max(0, self._capacity_before - self._selected_count),
                )
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
                # The cave can be partially clipped at the left edge after a
                # valid calibrated move. Its template then cannot match, but
                # the structurally validated N/M HUD remains readable. A
                # below-threshold value is enough to safely skip entering the
                # cave and return home.
                count = self._read_capacity(frame).count
                if count is not None:
                    if not self._confirm_capacity(count):
                        return None
                    cull = self._cull_required
                    self.logger.info(
                        "Hatch cave | capacity=%d/%d | threshold=%d | cull=%s"
                        " | cave_visible=False",
                        count,
                        self.capacity_limit,
                        self.threshold,
                        cull,
                    )
                    if not cull:
                        self._capacity_readable = True
                        self._stage = "recenter"
                        return self.choose(frame, detections)
                    if not self.allow_cull:
                        self._capacity_readable = True
                        self._stage = "recenter"
                        return self.choose(frame, detections)
                else:
                    self._reset_capacity_confirmation()
                return None
            if step.kind == STUCK:
                read = self._read_capacity(frame)
                count = read.count
                if count is not None:
                    if not self._confirm_capacity(count):
                        return None
                    cull = self._cull_required
                    self._capacity_readable = not cull or not self.allow_cull
                    if cull:
                        if self.allow_cull:
                            self.logger.error(
                                "Hatch cave | capacity=%d/%d requires cull but cave"
                                " target is unavailable; recentering without hatching",
                                count,
                                self.capacity_limit,
                            )
                        else:
                            self.logger.info(
                                "Hatch cave | capacity=%d/%d requires screening"
                                " before cull | cave_visible=False",
                                count,
                                self.capacity_limit,
                            )
                    else:
                        self.logger.info(
                            "Hatch cave | capacity=%d/%d | threshold=%d | cull=False"
                            " | cave_visible=False",
                            count,
                            self.capacity_limit,
                            self.threshold,
                        )
                else:
                    self._reset_capacity_confirmation()
                    # Navigation is only *presumed* failed: the cave template
                    # missed and the HUD did not stand in for it. Which of the
                    # two actually broke is not in the event stream.
                    self._snapshot_capacity(frame, read)
                    self.logger.warning(
                        "Hatch cave | navigation failed; recentering safely"
                    )
                self._stage = "recenter"
                return self.choose(frame, detections)
            assert step.kind == DONE and cave is not None
            read = self._read_capacity(frame)
            count = read.count
            if count is None:
                self._reset_capacity_confirmation()
                self._capacity_failures += 1
                if self._capacity_failures <= self.capacity_read_retries:
                    self.logger.warning(
                        "Hatch cave | capacity unreadable | retry=%d/%d",
                        self._capacity_failures,
                        self.capacity_read_retries,
                    )
                    return None
                self._capacity_readable = False
                # Every retry saw the same still frame, so the log line below
                # is the same whether the HUD is absent, covered, or merely
                # too small for the glyph templates. Save the pixels.
                self._snapshot_capacity(frame, read)
                self.logger.error("Hatch cave | capacity unreadable; skipping cull")
                self._stage = "recenter"
                return self.choose(frame, detections)
            if not self._confirm_capacity(count):
                return None
            self._capacity_readable = True
            self.logger.info(
                "Hatch cave | capacity=%d/%d | threshold=%d | cull=%s",
                count,
                self.capacity_limit,
                self.threshold,
                should_cull(count, self.threshold),
            )
            if not self._cull_required or not self.allow_cull:
                self._stage = "recenter"
                return self.choose(frame, detections)
            self._capacity_before = count
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
                if self._recenter_checks >= self.cave_recenter_checks:
                    self.logger.error(
                        "Hatch cave | safe return swipes did not prove centered home"
                    )
                    self._stage = "recenter_failed"
            return None
        return None

    def _confirm_capacity(self, count: int) -> bool:
        """Require consecutive identical reads before using the value."""

        if self._capacity_candidate != count:
            if self._capacity_candidate is not None:
                self.logger.warning(
                    "Hatch cave | capacity confirmation changed | previous=%d/%d"
                    " | current=%d/%d",
                    self._capacity_candidate,
                    self.capacity_limit,
                    count,
                    self.capacity_limit,
                )
            self._capacity_candidate = count
            self._capacity_confirmations = 1
        else:
            self._capacity_confirmations += 1
        if self._capacity_confirmations < self.capacity_consistent_reads:
            self.logger.info(
                "Hatch cave | capacity confirmation | value=%d/%d | sample=%d/%d",
                count,
                self.capacity_limit,
                self._capacity_confirmations,
                self.capacity_consistent_reads,
            )
            return False
        self._capacity_before = count
        self._cull_required = should_cull(count, self.threshold)
        return True

    def _reset_capacity_confirmation(self) -> None:
        self._capacity_candidate = None
        self._capacity_confirmations = 0

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
        stat_upgrade_guards: Mapping[str, StatUpgradeGuard] = DEFAULT_STAT_UPGRADE_GUARDS,
        minimum_consistent_stat_reads: int = 1,
        stat_read_retries: int = 1,
        boost_inventory: HatchBoostInventoryStore | None = None,
        require_home_anchor: bool = True,
        home_failure_limit: int = 3,
        home_backoff_seconds: float = 30.0,
        capacity_limit: int = EXPECTED_CAPACITY,
        cull_threshold: int = 330,
        screening_growth_interval: int = 20,
        cave_safe_margin: int = 80,
        cave_bottom_exclusion_px: int = 180,
        recovery_timeout_seconds: float = 15.0,
        capacity_read_retries: int = 2,
        cave_recenter_checks: int = 3,
        capacity_snapshots: CapacitySnapshot | None = None,
        parent_stats_snapshots: ParentStatsSnapshot | None = None,
        egg_pile_snapshots: EggPileSnapshot | None = None,
        home_recovery_snapshots: HomeRecoverySnapshot | None = None,
        stage_scoped_scan: bool = True,
        standalone_stage: str | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.reader = reader
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        if capacity_limit <= 0:
            raise ValueError("capacity_limit must be greater than zero")
        self.capacity_limit = capacity_limit
        self.cull_threshold = cull_threshold
        self.stat_upgrade_guards = dict(stat_upgrade_guards)
        self.minimum_consistent_stat_reads = minimum_consistent_stat_reads
        self.stat_read_retries = stat_read_retries
        # Nest management is driven only by estimated total dinosaur growth.
        # The cull threshold leaves headroom below the configured capacity HUD.
        self._capacity_management_trigger = self.cull_threshold
        self.screening_growth_interval = max(1, screening_growth_interval)
        self.boost_inventory = boost_inventory
        self.cave_safe_margin = max(0, cave_safe_margin)
        self.cave_bottom_exclusion_px = max(0, cave_bottom_exclusion_px)
        self.recovery_timeout_seconds = max(1.0, recovery_timeout_seconds)
        self.capacity_read_retries = max(1, capacity_read_retries)
        self.cave_recenter_checks = max(1, cave_recenter_checks)
        self.capacity_snapshots = capacity_snapshots
        self.parent_stats_snapshots = parent_stats_snapshots
        self.egg_pile_snapshots = egg_pile_snapshots
        self.home_recovery_snapshots = home_recovery_snapshots
        self.stage_scoped_scan = bool(stage_scoped_scan)
        if standalone_stage is not None and standalone_stage not in STANDALONE_STAGES:
            raise ValueError(f"unsupported standalone hatch stage: {standalone_stage}")
        self.standalone_stage = standalone_stage
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
        self._recovery_rounds = 0
        self._recovery_started_at: float | None = None
        self._recovery_forest_exhausted = False
        self._collect_only_after_empty = False
        self._empty_rescan_wait = False
        # A fresh planner instance must prove that the dinosaur capacity is
        # safe before it opens the incubator. Later management cycles already
        # pass through CaveCullPlanner, so this guard is one preflight per
        # planner lifetime rather than once per hatch cycle.
        self._capacity_checked = False
        # The first verified count performs one baseline screening. Afterwards
        # every screening records its total as the new growth baseline.
        self._screening_baseline_population: int | None = None
        self._pending_screening_population: int | None = None
        self._cave_cleanup_after_management = False
        # Cleanup is destructive: a capacity-triggered management pass must
        # retain proof that every screening stage completed, including across
        # bounded home recovery.
        self._management_pending = False
        self._screening_completed: set[str] = set()
        self._observed_cooldown_until: float | None = None
        self._boost_enabled_for_cycle = False
        self._boost_attempted = False
        self._boost_confirmation_pending = False
        self._boost_revisit_pending = False
        self._collect_done_for_cycle = False
        # 洞穴容量估算：上次實讀 + 之後累積孵化；每增長門檻或達清理
        # 門檻時篩選。
        self._cave_population: int | None = None
        self._hatched_since_cave_read = 0
        self._navigation_failures: dict[tuple[str, str], int] = {}
        self._pending_claim_verification = False
        self._standalone_started = False
        self._standalone_returning = False
        self._egg_pile_failures = 0
        self._egg_pile_blocked = False
        # The first inert egg-pile tap can mean the cave reached its capacity,
        # not that the calibrated coordinate is wrong.  Recover home, run a
        # fresh two-frame capacity preflight, and only retry the pile after a
        # below-limit reading has been proved.
        self._egg_pile_capacity_check_pending = False
        self._egg_pile_capacity_rechecked = False
        self._egg_pile_retry_pending = False
        self._screening_blocked = False
        self._screening_recovery_failures: dict[str, int] = {}
        self.completed_management_cycles = 0
        self._start_hatch_cycle()
        if self.standalone_stage is not None:
            self._stage = "recover_home"
            self._child = HatchHomeRecoveryPlanner(
                reference_width=self.reference_width,
                logger=self.logger,
            )
            self._recovery_reason = (
                f"standalone {self.standalone_stage} preflight"
            )

    def last_stage(self) -> str:
        child_stage = getattr(self._child, "last_stage", None)
        detail = child_stage() if callable(child_stage) else child_stage
        return f"full_{self._stage}:{detail or '-'}"

    def is_complete(self) -> bool:
        # A standalone/full-only run has no hunt owner to fall back to. Stop
        # safely after calibration is blocked instead of spinning forever.
        return self._complete or self._egg_pile_blocked or self._screening_blocked

    def is_hatch_blocked(self) -> bool:
        """Whether egg-pile recovery exhausted its safe retry budget."""

        return self._egg_pile_blocked or self._screening_blocked

    def next_ready_delay_ms(self) -> int:
        method = getattr(self._child, "next_ready_delay_ms", None)
        return int(method()) if callable(method) else 0

    def planning_detection_types(self) -> frozenset[str] | None:
        """Return the detector types needed by the current hatch phase.

        Full-hatch runs use a separate manifest from hunting, but the combined
        mode still has both detectors installed.  Returning a phase-scoped set
        prevents the hunt manifest (especially the full-resolution dinosaur
        template and HSV path detector) from being evaluated while hatch or
        nest management is active.
        """

        if not self.stage_scoped_scan:
            return None
        if self._stage == "hatch":
            return HATCH_DETECTION_TYPES
        if self._stage == "open_nest":
            return NEST_BASE_DETECTION_TYPES
        if self._stage in {"attack", "hp"}:
            return NEST_REPLACEMENT_DETECTION_TYPES
        if self._stage in {"top", "mass"}:
            return NEST_AUTOPLACE_DETECTION_TYPES
        if self._stage in {
            "collect",
            "collect_button",
            "close_nest",
            "verify_nest_closed",
        }:
            return NEST_COLLECT_DETECTION_TYPES
        if self._stage == "cave":
            return CAVE_DETECTION_TYPES
        if self._stage == "capacity_preflight":
            return CAVE_DETECTION_TYPES
        return RECOVERY_DETECTION_TYPES

    def is_hunt_cooldown_active(self) -> bool:
        """Whether the no-ready-egg cooldown can be spent hunting."""

        return (
            self._empty_rescan_wait
            and self.next_ready_delay_ms() > 0
        )

    def begin_interim_collection(self) -> bool:
        """Spend a hunt idle window on one collect-only nest round.

        把剩餘冷卻釘進 observed deadline:收蛋結束後的等待會接續原本的
        倒數,而不是從設定值重新起算(`_start_empty_rescan_wait` 會在
        沒有 observed 值時退回整段設定時間)。
        """

        if not self.is_hunt_cooldown_active():
            return False
        if self._observed_cooldown_until is None:
            self._observed_cooldown_until = (
                self.clock() + self.next_ready_delay_ms() / 1000
            )
        self.logger.info(
            "Hatch full | interim nest collection during hunt idle | resume=%.0fs",
            max(0.0, self._observed_cooldown_until - self.clock()),
        )
        self._enter_open_nest(collect_only=True)
        return True

    def reset_workflow(self) -> None:
        self._egg_pile_failures = 0
        self._egg_pile_blocked = False
        self._egg_pile_capacity_check_pending = False
        self._egg_pile_capacity_rechecked = False
        self._egg_pile_retry_pending = False
        self._screening_blocked = False
        self._screening_recovery_failures.clear()
        if self.standalone_stage is not None:
            self._stage = "recover_home"
            self._child = HatchHomeRecoveryPlanner(
                reference_width=self.reference_width,
                logger=self.logger,
            )
            self._hatch_baseline = 0
            self._complete = False
            self._no_target_since = None
            self._recovery_reason = (
                f"standalone {self.standalone_stage} preflight"
            )
            self._collect_only_after_empty = False
            self._empty_rescan_wait = False
            self._observed_cooldown_until = None
            self._navigation_failures.clear()
            self._pending_claim_verification = False
            self._standalone_started = False
            self._standalone_returning = False
            return
        if self._management_pending:
            # A game/startup interruption must not erase an unfinished
            # screening checklist and later allow cleanup to proceed. Recover
            # home, reopen My Nest, and resume the first incomplete stage.
            self._stage = "recover_home"
            self._child = HatchHomeRecoveryPlanner(
                reference_width=self.reference_width,
                logger=self.logger,
            )
            self._complete = False
            self._no_target_since = None
            self._recovery_reason = "resume incomplete screening after workflow reset"
            self._collect_only_after_empty = False
            self._empty_rescan_wait = False
            self._observed_cooldown_until = None
            self._navigation_failures.clear()
            self._pending_claim_verification = False
            return
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._complete = False
        self._no_target_since = None
        self._recovery_reason = None
        self._collect_only_after_empty = False
        self._empty_rescan_wait = False
        self._management_pending = False
        self._screening_completed.clear()
        self._observed_cooldown_until = None
        self._navigation_failures.clear()
        self._pending_claim_verification = False
        self._egg_pile_failures = 0
        self._egg_pile_blocked = False
        self._start_hatch_cycle()

    def on_action_success(self, target_type: str) -> None:
        if target_type == RECOVERY_NO:
            self.logger.warning(
                "Hatch full | cancelled unexpected auto-place confirmation"
                " | outside top/mass stage"
            )
            self._begin_home_recovery("cancelled unexpected auto-place confirmation")
            return
        if target_type == STARTUP_NEST_SHORTCUT:
            self.logger.info(
                "Hatch full | startup nest shortcut opened | returning to hatch home"
            )
            self._begin_home_recovery("opened My Nest from startup growth results")
            return
        if target_type in STARTUP_INTERRUPTS:
            self.logger.info(
                "Hatch full | startup overlay cleared | target=%s",
                target_type,
            )
            self._no_target_since = None
            return
        self._navigation_failures.pop((self._stage, target_type), None)
        if self._stage == "recover_home":
            self._recovery_child.on_action_success(target_type)
            return
        if self._stage == "capacity_preflight":
            self._capacity_child.on_action_success(target_type)
            return
        if self._stage == "hatch":
            hatch = self._hatch_child
            hatch.on_action_success(target_type)
            if target_type == hatch_feature.EGG_PILE:
                self._egg_pile_failures = 0
                self._egg_pile_capacity_check_pending = False
                self._egg_pile_capacity_rechecked = False
                self._egg_pile_retry_pending = False
            if target_type == HATCH_BOOST_BUTTON:
                self._boost_attempted = True
                self._boost_confirmation_pending = True
                return
            if target_type == HATCH_BOOST_CONFIRM and self._boost_confirmation_pending:
                self._boost_confirmation_pending = False
                consumed = (
                    self.boost_inventory.consume_one()
                    if self.boost_inventory is not None
                    else None
                )
                if consumed is None:
                    self.logger.warning(
                        "Hatch boost | confirmation succeeded but Bot budget was empty"
                    )
                else:
                    self.logger.info(
                        "Hatch boost | used 1 ticket | remaining budget=%d",
                        consumed.remaining,
                    )
                return
            if target_type == hatch_feature.CLAIM_BUTTON:
                self._pending_claim_verification = False
            if target_type == hatch_feature.CLOSE_BUTTON:
                if self.standalone_stage == "hatch":
                    self._begin_home_recovery("standalone hatch completed")
                    return
                hatched = hatch.hatched - self._hatch_baseline
                if hatched > 0:
                    self._hatched_since_cave_read += hatched
                    estimate = (
                        None
                        if self._cave_population is None
                        else self._cave_population + self._hatched_since_cave_read
                    )
                    capacity_trigger = (
                        estimate is not None
                        and estimate >= self._capacity_management_trigger
                    )
                    growth = (
                        None
                        if estimate is None
                        or self._screening_baseline_population is None
                        else estimate - self._screening_baseline_population
                    )
                    growth_trigger = (
                        growth is not None
                        and growth >= self.screening_growth_interval
                    )
                    management_trigger = capacity_trigger or growth_trigger
                    self.logger.info(
                        "Hatch full | phase A complete | hatched=%d | cave≈%s"
                        " | growth=%s | trigger=%s",
                        hatched,
                        "?" if estimate is None else f"{estimate}/{self.capacity_limit}",
                        "?" if growth is None else growth,
                        "capacity"
                        if capacity_trigger
                        else ("growth" if growth_trigger else "none"),
                    )
                    if management_trigger and not self._management_pending:
                        self._queue_management(
                            population=estimate,
                            cave_cleanup_after=capacity_trigger,
                        )
                    self._enter_open_nest(collect_only=not management_trigger)
                else:
                    if self._collect_done_for_cycle:
                        # 加速回訪的收尾:本週期已收過蛋,直接進入等待。
                        self._collect_done_for_cycle = False
                        self._start_empty_rescan_wait()
                        return
                    self.logger.info(
                        "Hatch full | no ready incubator eggs | "
                        "collect all nest eggs before cooldown"
                    )
                    self._enter_open_nest(collect_only=True)
            return
        if self._stage == "open_nest" and target_type == OPEN_NEST:
            if self.standalone_stage is not None:
                self._start_standalone_nest_phase()
                return
            if self._collect_only_after_empty:
                self._start_collect()
            else:
                self._start_next_screening_stage()
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

    def on_action_failure_context(
        self,
        target: Target,
        frame: Frame | None,
        detections: Sequence[Detection],
        attempts: int,
    ) -> None:
        """Capture the post-action frame before recovery changes the screen."""

        if target.type != hatch_feature.EGG_PILE or frame is None:
            return
        self._egg_pile_failures += 1
        measured_base = _egg_pile_base_center(frame)
        proposed_point = _egg_pile_safe_tap(frame)
        if self.egg_pile_snapshots is not None:
            self.egg_pile_snapshots.capture(
                frame,
                detections,
                target_x=target.x,
                target_y=target.y,
                measured_base=measured_base,
                proposed_point=proposed_point,
                stage=self.last_stage(),
                failures=self._egg_pile_failures,
                attempts=attempts,
            )
        self.logger.warning(
            "Hatch calibration | egg pile failure=%d | target=(%d,%d)"
            " | measured_base=%s | proposed=%s",
            self._egg_pile_failures,
            target.x,
            target.y,
            measured_base,
            proposed_point,
        )

    def on_retry_exhausted(self, target: Target) -> None:
        """Trip a workflow fuse instead of allowing synthetic target reuse."""

        if target.type == OPEN_NEST:
            self._screening_blocked = True
            self._stage = "screening_blocked"
            self._no_target_since = None
            self.logger.error(
                "Hatch screening | My Nest failed after bounded retries"
                " | blocking hatch instead of reopening forever"
            )
            return
        if target.type != hatch_feature.EGG_PILE:
            return
        if (
            self._egg_pile_capacity_check_pending
            or self._egg_pile_retry_pending
        ):
            self.logger.warning(
                "Hatch calibration | egg pile retries exhausted"
                " | bounded capacity/recovery check already scheduled"
            )
            return
        self._egg_pile_blocked = True
        self._stage = "hatch_blocked"
        self._no_target_since = None
        self.logger.error(
            "Hatch calibration | egg pile retries exhausted"
            " | blocking hatch until explicit workflow reset"
        )

    def recover_from_action_failures(
        self,
        target: Target,
        stage: str,
        episodes: int,
        frame: Frame | None,
        detections: Sequence[Detection],
    ) -> bool:
        """Return uncertain hatch mutations to a proven centred home screen."""

        del frame, detections
        if self.is_hatch_blocked():
            return False
        self.logger.warning(
            "Hatch recovery | repeated action failure | stage=%s target=%s"
            " episodes=%d | returning to centered home",
            stage,
            target.type,
            episodes,
        )
        self._begin_home_recovery(
            f"repeated action failure at {stage}: {target.type} ({episodes})"
        )
        return True

    @staticmethod
    def is_recovery_progress(target_type: str) -> bool:
        """A verified hatch claim is the workflow's productive milestone."""

        return target_type == hatch_feature.CLAIM_BUTTON

    def on_action_failure(self, target_type: str) -> None:
        if target_type in STARTUP_INTERRUPTS:
            # Leave the workflow stage intact. If the modal remains visible,
            # the next frame will retry it before any hatch/home action; if it
            # disappeared despite a missed transition, normal planning resumes.
            self.logger.warning(
                "Hatch full | startup overlay transition not verified | target=%s",
                target_type,
            )
            self._no_target_since = None
            return
        if self._stage == "recover_home":
            self._recovery_child.on_action_failure(target_type)
            # Do not mark the full workflow complete here.  The recovery
            # child records this failed round, and choose() owns the bounded
            # retry policy for continuous runs.  Completing immediately made
            # one missed Forest/map transition stop an otherwise healthy
            # hatch+hunt session before either recovery retry could run.
            return
        if self._stage == "hatch":
            self._hatch_child.on_action_failure(target_type)
            if target_type == hatch_feature.CLAIM_BUTTON:
                self._pending_claim_verification = True
            if target_type == hatch_feature.EGG_PILE:
                if not self._egg_pile_capacity_rechecked:
                    self._capacity_checked = False
                    self._egg_pile_capacity_check_pending = True
                    self.logger.warning(
                        "Hatch capacity | egg pile tap had no verified response"
                        " | checking configured capacity before retry"
                    )
                    self._begin_home_recovery(
                        "egg pile tap failed; capacity recheck required"
                    )
                else:
                    # Capacity was already proved below the limit.  Treat a
                    # further miss as calibration/occlusion, but still unwind
                    # the unexpected foreground before a bounded retry.
                    self._egg_pile_retry_pending = True
                    self._begin_home_recovery(
                        "egg pile tap failed after safe capacity recheck"
                    )
            return
        if self._stage == "capacity_preflight":
            self._capacity_child.on_action_failure(target_type)
            # A failed cave navigation/read must never fall through to the
            # incubator.  Recover to a proven home screen and retry the guard
            # on the next workflow pass.
            self._begin_home_recovery(
                f"capacity preflight failed at target={target_type}"
            )
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
        if self.is_complete():
            return None

        # Login overlays sit above a still-detectable outdoor HUD. Handle them
        # before home/cave recovery or HatchPlanner can interpret that
        # background anchor as permission to tap the egg pile. A full hatch
        # run deliberately uses the right-hand Nest shortcut instead of the
        # detector's left-hand auto-battle shortcut; hunting is only allowed
        # after the no-ready-egg cooldown has actually begun.
        by_type = _group(detections)
        # Exact startup/login button templates outrank the intentionally broad
        # auto-place dialog layout detector. The device-history prompt also
        # has cyan Yes and red No buttons, so its layout can look like an
        # auto-place notice even though this explicit button proves which
        # startup dialog is actually in front.
        for target_type in STARTUP_SIMPLE_INTERRUPTS:
            interruption = _best(by_type.get(target_type))
            if interruption is not None:
                self._no_target_since = None
                return _target(interruption)
        # Same rule one step further: the parent-replacement confirmation also
        # carries cyan Yes and red No buttons, so the broad auto-place layout
        # matches it too. Its own exact templates prove which dialog is really
        # in front, and cancelling one aborts the swap the screening stage just
        # asked for - then the stage restarts and asks again, forever.
        replacing_parent = (
            SELECT_CONFIRM_PROMPT in by_type or NESTED_PARENT_WARNING in by_type
        )
        if (
            self._stage not in {"top", "mass"}
            and not replacing_parent
            and (AUTOPLACE_PROMPT in by_type or AUTOPLACE_NOTICE in by_type)
        ):
            no = _best(by_type.get(CONFIRM_NO))
            if no is None:
                self.logger.error(
                    "Hatch full | unexpected auto-place confirmation has no No button"
                )
                return None
            return _synthetic(RECOVERY_NO, no.x, no.y)
        hatch_result_visible = bool(
            by_type.get(hatch_feature.CLAIM_BUTTON)
            or by_type.get(hatch_feature.EXPEL_BUTTON)
        )
        auto_battle = (
            None
            if hatch_result_visible
            else _best(by_type.get(STARTUP_AUTO_BATTLE_CLOSE))
        )
        if auto_battle is not None:
            self._no_target_since = None
            return _target(auto_battle)
        if (
            not hatch_result_visible
            # The My Nest card is also a tall white panel with a centred
            # green button.  Its auto-place control therefore satisfies the
            # launch growth-result colour heuristic.  Never use the startup
            # shortcut while the nest title proves that this is already the
            # nest screen, or the synthetic tap lands on auto-place.
            and NEST_TITLE not in by_type
            and (growth_result := _best(by_type.get(STARTUP_GROWTH_RESULT))) is not None
        ):
            self._no_target_since = None
            # The current game build moved the only remaining nest shortcut
            # to the centre of the result card. Keep the old right-hand point
            # for screenshots from builds that still show two shortcuts.
            shortcut_point = (
                (450.0, 1270.0)
                if growth_result.metadata.get("shortcut_layout") == "centered_nest"
                else (592.0, 1265.0)
            )
            return _synthetic(
                STARTUP_NEST_SHORTCUT,
                *_scaled(frame, shortcut_point, self.reference_width),
            )

        if self._stage == "hatch" and any(
            item.type in HUNT_ACTIVE_TYPES for item in detections
        ):
            self._begin_home_recovery("hatch started from active hunt map")
            return self.choose(frame, detections)

        if self._stage == "recover_home":
            target = self._recovery_child.choose(frame, detections)
            if self._recovery_child.is_failed():
                self._capture_recovery_evidence(frame, detections)
                elapsed = self.clock() - (self._recovery_started_at or self.clock())
                if (
                    self.standalone_stage is None
                    and self._recovery_rounds < 2
                    and elapsed < MAX_HOME_RECOVERY_SECONDS
                ):
                    # 畫面轉場常只比恢復預算慢幾秒;連續模式先重試,
                    # 不要一次失敗就把整個流程標記結束。
                    self._recovery_rounds += 1
                    self._begin_home_recovery(
                        f"retry {self._recovery_rounds}/2 after failed recovery"
                        f" ({self._recovery_reason or 'unknown'})"
                    )
                    return None
                self.logger.error(
                    "Hatch full | recovery failed | reason=%s"
                    " | rounds=%d | elapsed=%.0fs",
                    self._recovery_reason or "unknown",
                    self._recovery_rounds,
                    elapsed,
                )
                self._complete = True
                return None
            if self._recovery_child.is_complete():
                if self.standalone_stage is not None:
                    if self._standalone_returning:
                        self.logger.info(
                            "Hatch stage | completed | stage=%s | centered home verified",
                            self.standalone_stage,
                        )
                        self._complete = True
                        return None
                    if not self._standalone_started:
                        self._start_standalone()
                        return self._choose_current(frame, detections)
                self._recovery_rounds = 0
                self._recovery_started_at = None
                self._recovery_forest_exhausted = False
                self.logger.info(
                    "Hatch full | centered home confirmed | recovered=%s",
                    self._recovery_reason or "unknown",
                )
                if self._egg_pile_capacity_check_pending:
                    self._begin_capacity_preflight(
                        "egg pile tap had no response"
                    )
                    return self._choose_current(frame, detections)
                if self._egg_pile_retry_pending:
                    self._egg_pile_retry_pending = False
                    failure_limit = int(
                        self._hatch_kwargs["home_failure_limit"]
                    )
                    if self._egg_pile_failures >= failure_limit:
                        self._egg_pile_blocked = True
                        self._stage = "hatch_blocked"
                        self._no_target_since = None
                        self.logger.error(
                            "Hatch calibration | egg pile failed %d times"
                            " after safe capacity check | blocking hatch",
                            self._egg_pile_failures,
                        )
                        return None
                    self.logger.warning(
                        "Hatch calibration | retrying egg pile after safe"
                        " capacity check | failure=%d/%d",
                        self._egg_pile_failures,
                        failure_limit,
                    )
                    self._stage = "hatch"
                    self._child = self._new_hatch()
                    self._hatch_baseline = 0
                    self._no_target_since = None
                    self._recovery_reason = None
                    self._start_hatch_cycle()
                    return self._choose_current(frame, detections)
                if self._management_pending:
                    missing = self._missing_screening_stages()
                    self.logger.info(
                        "Hatch full | resuming screening after recovery"
                        " | completed=%s | missing=%s",
                        self._format_screening_stages(self._screening_completed),
                        self._format_screening_stages(missing),
                    )
                    self._stage = "open_nest"
                    self._child = object()
                    self._collect_only_after_empty = False
                    self._empty_rescan_wait = False
                    self._no_target_since = None
                    self._recovery_reason = None
                    return self._choose_current(frame, detections)
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
        # CaveCullPlanner owns its own two-frame return-to-home confirmation.
        # Do not let the outer recovery guard interrupt that confirmation;
        # capacity preflight uses the same planner before the first hatch.
        if (
            self._stage not in {"hatch", "cave", "capacity_preflight"}
            and is_centered_home_screen(frame, detections)
        ):
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
            if self._boost_confirmation_pending:
                yes = _best(by_type.get(CONFIRM_YES))
                no = _best(by_type.get(CONFIRM_NO))
                if yes is not None and no is not None:
                    return _synthetic(HATCH_BOOST_CONFIRM, yes.x, yes.y)
                return None
            self._observe_hatch_cooldown(frame, by_type)
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
            if self._boost_allowed(by_type):
                if not self._eggs_cooling():
                    # 孵化器目前是空的:先收蛋讓新蛋開始冷卻,收完
                    # 回訪再按,加速期才不會空燒在沒有蛋的時段上。
                    if not self._boost_revisit_pending:
                        self._boost_revisit_pending = True
                        self.logger.info(
                            "Hatch boost | incubator empty | defer until eggs collected"
                        )
                elif not _hatch_boost_ready(frame):
                    # 加速已在生效倒數(按鈕帶轉灰);本週期不再嘗試。
                    self._boost_attempted = True
                    self.logger.info(
                        "Hatch boost | already active (gray countdown bar) | skip this cycle"
                    )
                else:
                    return _synthetic(
                        HATCH_BOOST_BUTTON,
                        *_scaled(frame, HATCH_BOOST_POINT, self.reference_width),
                    )
            target = self._hatch_child.choose(frame, detections)
            if target is not None and target.type == hatch_feature.EGG_PILE:
                if not self._capacity_checked:
                    self._begin_capacity_preflight(
                        "before first incubator access"
                    )
                    return self._choose_current(frame, detections)
                safe_point = _egg_pile_safe_tap(frame)
                if safe_point is not None:
                    return _synthetic(hatch_feature.EGG_PILE, *safe_point)
            return target
        if self._stage == "capacity_preflight":
            target = self._capacity_child.choose(frame, detections)
            self._remember_capacity_preflight_result()
            if target is None and self._capacity_child.is_complete():
                if not self._capacity_child.capacity_readable:
                    self._egg_pile_capacity_check_pending = False
                    self.logger.error(
                        "Hatch capacity | preflight failed; configured capacity unreadable; "
                        "hatching stopped safely"
                    )
                    self._complete = True
                    return None
                if self._capacity_child.cull_required:
                    self._egg_pile_capacity_check_pending = False
                    self._egg_pile_capacity_rechecked = False
                    self._egg_pile_retry_pending = False
                    self._stage = "open_nest"
                    self._child = object()
                    self._collect_only_after_empty = False
                    self._no_target_since = None
                    return self._choose_current(frame, detections)
                self._capacity_checked = True
                if self._screening_baseline_population is None:
                    self.logger.info(
                        "Hatch capacity | starting initial screening | capacity=%d/%d",
                        self._cave_population,
                        self.capacity_limit,
                    )
                    self._queue_management(
                        population=self._cave_population,
                        cave_cleanup_after=False,
                    )
                    self._stage = "open_nest"
                    self._child = object()
                    self._collect_only_after_empty = False
                    self._no_target_since = None
                    return self._choose_current(frame, detections)
                if self._egg_pile_capacity_check_pending:
                    self._egg_pile_capacity_check_pending = False
                    self._egg_pile_capacity_rechecked = True
                self.logger.info(
                    "Hatch capacity | preflight complete | safe to hatch"
                )
                self._stage = "hatch"
                self._child = self._new_hatch()
                self._hatch_baseline = 0
                self._start_hatch_cycle()
                return self._choose_current(frame, detections)
            return target
        if self._stage == "open_nest":
            if NEST_TITLE in by_type:
                if self.standalone_stage is not None:
                    self._start_standalone_nest_phase()
                    return self._choose_current(frame, detections)
                if self._collect_only_after_empty:
                    self._start_collect()
                else:
                    self._start_next_screening_stage()
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
            return nest_mask_close_target(frame, self.reference_width)
        if self._stage == "verify_nest_closed":
            if NEST_TITLE in by_type:
                self._stage = "close_nest"
                return self._choose_current(frame, detections)
            if is_home_screen(frame, detections):
                if self.standalone_stage is not None:
                    self._begin_home_recovery(
                        f"standalone {self.standalone_stage} completed"
                    )
                    return self._choose_current(frame, detections)
                if self._collect_only_after_empty:
                    if self._boost_revisit_pending:
                        # 蛋剛收進孵化器開始冷卻;回訪按加速,讓加速期
                        # 從冷卻第一秒就生效,同時讀到砍半後的精確倒數。
                        self._boost_revisit_pending = False
                        self._collect_done_for_cycle = True
                        self.logger.info(
                            "Hatch boost | revisiting incubator to boost collected eggs"
                        )
                        self._stage = "hatch"
                        self._child = self._new_hatch()
                        self._hatch_baseline = 0
                        return self._choose_current(frame, detections)
                    self._start_empty_rescan_wait()
                    return self._choose_current(frame, detections)
                missing = self._missing_screening_stages()
                if missing:
                    self.logger.error(
                        "Hatch full | cleanup blocked; screening incomplete"
                        " | completed=%s | missing=%s",
                        self._format_screening_stages(self._screening_completed),
                        self._format_screening_stages(missing),
                    )
                    self._management_pending = True
                    self._stage = "open_nest"
                    self._child = object()
                    return self._choose_current(frame, detections)
                self._finish_management_after_collection()
                return self._choose_current(frame, detections)
            return None
        if self._stage == "cave":
            if self.standalone_stage is None:
                missing = self._missing_screening_stages()
                if not self._management_pending or missing:
                    self.logger.error(
                        "Hatch full | cave cleanup denied by screening gate"
                        " | management_pending=%s | missing=%s",
                        self._management_pending,
                        self._format_screening_stages(missing),
                    )
                    self._management_pending = True
                    self._cave_cleanup_after_management = True
                    self._stage = "open_nest"
                    self._child = object()
                    self._collect_only_after_empty = False
                    return self._choose_current(frame, detections)
            target = self._cave_child.choose(frame, detections)
            if target is None and self._cave_child.is_complete():
                if self.standalone_stage == "cave":
                    self._begin_home_recovery("standalone cave completed")
                    return self._choose_current(frame, detections)
                self.completed_management_cycles += 1
                self._capacity_checked = True
                self._management_pending = False
                self._screening_completed.clear()
                self._egg_pile_failures = 0
                self._egg_pile_capacity_check_pending = False
                self._egg_pile_capacity_rechecked = False
                self._egg_pile_retry_pending = False
                expected = self._cave_child.expected_population
                if expected is not None:
                    self._cave_population = expected
                    self._hatched_since_cave_read = 0
                self._screening_baseline_population = (
                    expected
                    if expected is not None
                    else self._pending_screening_population
                )
                self._pending_screening_population = None
                self._cave_cleanup_after_management = False
                self.logger.info(
                    "Hatch full | completed management cycle %d | restarting Phase A"
                    " | cave≈%s",
                    self.completed_management_cycles,
                    "?" if expected is None else f"{expected}/{self.capacity_limit}",
                )
                self._stage = "hatch"
                self._child = self._new_hatch()
                self._hatch_baseline = 0
                self._start_hatch_cycle()
                return self._choose_current(frame, detections)
            return target
        return None

    def _capture_recovery_evidence(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> None:
        """Keep the frame a failed return-to-home could not describe."""

        if self.home_recovery_snapshots is None:
            return
        self.home_recovery_snapshots.capture(
            frame,
            detections,
            reason=self._recovery_reason or "unknown",
            stage=self._recovery_child.last_stage(),
            rounds=self._recovery_rounds,
            forest_trips=self._recovery_child.forest_trips(),
            measured_base=_egg_pile_base_center(frame),
            expected_base=HOME_PILE_BASE,
        )

    def _begin_home_recovery(self, reason: str) -> None:
        if self.standalone_stage is not None and self._standalone_started:
            self._standalone_returning = True
        if "parent_stats_unreadable" in reason:
            failure_key = f"{self._stage}:parent_stats_unreadable"
            failures = self._screening_recovery_failures.get(failure_key, 0) + 1
            self._screening_recovery_failures[failure_key] = failures
            self.logger.warning(
                "Hatch screening | repeated calibration failure"
                " | key=%s | failure=%d/%d",
                failure_key,
                failures,
                MAX_SCREENING_RECOVERY_FAILURES,
            )
            if failures >= MAX_SCREENING_RECOVERY_FAILURES:
                self._screening_blocked = True
                self._stage = "screening_blocked"
                self._no_target_since = None
                self._recovery_reason = reason
                self.logger.error(
                    "Hatch screening | calibration recovery exhausted"
                    " | key=%s | blocking hatch instead of looping forever",
                    failure_key,
                )
                return
        self.logger.warning("Hatch full | recovering to centered home | %s", reason)
        if self._stage != "recover_home":
            # A fresh episode, not one of its own retries: start the wall-clock
            # budget and forget what the previous episode learned about the
            # screen it was looking at.
            self._recovery_started_at = self.clock()
            self._recovery_forest_exhausted = False
        elif self._recovery_child.forest_trips():
            # The retry gets a new child with fresh counters.  Carry this one
            # fact across, or each retry buys another round trip that has
            # already been shown not to move the camera.
            self._recovery_forest_exhausted = True
        self._stage = "recover_home"
        self._child = HatchHomeRecoveryPlanner(
            reference_width=self.reference_width,
            logger=self.logger,
            max_forest_trips=0 if self._recovery_forest_exhausted else 1,
        )
        self._no_target_since = None
        self._recovery_reason = reason

    def _begin_capacity_preflight(self, reason: str) -> None:
        """Check/cull dinosaur capacity before the first hatch interaction."""

        self.logger.info(
            "Hatch capacity | starting preflight | reason=%s",
            reason,
        )
        self._stage = "capacity_preflight"
        self._child = CaveCullPlanner(
            self.reader,
            threshold=self.cull_threshold,
            reference_width=self.reference_width,
            safe_margin=self.cave_safe_margin,
            bottom_exclusion_px=self.cave_bottom_exclusion_px,
            capacity_limit=self.capacity_limit,
            capacity_read_retries=self.capacity_read_retries,
            cave_recenter_checks=self.cave_recenter_checks,
            capacity_snapshots=self.capacity_snapshots,
            # Preflight may read capacity, but it must never delete dinosaurs
            # before the required parent-screening stages have completed.
            allow_cull=False,
            logger=self.logger,
        )
        self._no_target_since = None

    def _start_cave_cleanup(self) -> None:
        """Enter the guarded cave cleanup stage after collection is complete."""

        self.logger.info(
            "Hatch full | screening gate passed; cave cleanup permitted"
            " | completed=%s",
            self._format_screening_stages(self._screening_completed),
        )
        self._stage = "cave"
        self._child = CaveCullPlanner(
            self.reader,
            threshold=self.cull_threshold,
            reference_width=self.reference_width,
            safe_margin=self.cave_safe_margin,
            bottom_exclusion_px=self.cave_bottom_exclusion_px,
            capacity_limit=self.capacity_limit,
            capacity_read_retries=self.capacity_read_retries,
            cave_recenter_checks=self.cave_recenter_checks,
            capacity_snapshots=self.capacity_snapshots,
            logger=self.logger,
        )

    def _queue_management(
        self,
        *,
        population: int | None,
        cave_cleanup_after: bool,
    ) -> None:
        """Require a full screening pass before optionally entering the cave."""

        self._management_pending = True
        self._pending_screening_population = population
        self._cave_cleanup_after_management = cave_cleanup_after
        self._screening_completed.clear()

    def _finish_management_after_collection(self) -> None:
        """Continue after a completed screening and its egg collection."""

        if self._cave_cleanup_after_management:
            self._start_cave_cleanup()
            return
        self.completed_management_cycles += 1
        self._screening_baseline_population = self._pending_screening_population
        self._pending_screening_population = None
        self._management_pending = False
        self._screening_completed.clear()
        self.logger.info(
            "Hatch full | completed growth screening %d | baseline=%s/%d"
            " | restarting Phase A",
            self.completed_management_cycles,
            "?"
            if self._screening_baseline_population is None
            else self._screening_baseline_population,
            self.capacity_limit,
        )
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._start_hatch_cycle()

    def _remember_capacity_preflight_result(self) -> None:
        """Persist a valid capacity read before cave recentering can recover."""

        child = self._capacity_child
        reading = child.last_capacity
        if not child.capacity_readable or reading is None:
            return
        self._cave_population = reading
        self._hatched_since_cave_read = 0
        if child.cull_required:
            if not self._management_pending:
                self.logger.warning(
                    "Hatch capacity | cleanup blocked until screening completes"
                    " | required=%s",
                    self._format_screening_stages(SCREENING_STAGES),
                )
                self._queue_management(
                    population=reading,
                    cave_cleanup_after=True,
                )
            return
        self._capacity_checked = True

    def _new_hatch(self) -> hatch_feature.HatchPlanner:
        return hatch_feature.HatchPlanner(**self._hatch_kwargs)

    def _start_hatch_cycle(self) -> None:
        """Snapshot the dashboard boost permission for this hatch pass."""

        self._boost_attempted = False
        self._boost_confirmation_pending = False
        self._boost_enabled_for_cycle = False
        self._boost_revisit_pending = False
        self._collect_done_for_cycle = False
        if self.boost_inventory is None:
            return
        inventory = self.boost_inventory.snapshot()
        self._boost_enabled_for_cycle = inventory.enabled and inventory.remaining > 0
        self.logger.info(
            "Hatch boost | next cycle permission=%s | budget=%d",
            self._boost_enabled_for_cycle,
            inventory.remaining,
        )

    def _start_standalone(self) -> None:
        assert self.standalone_stage is not None
        self._standalone_started = True
        self._standalone_returning = False
        self._recovery_reason = None
        self.logger.info("Hatch stage | starting | stage=%s", self.standalone_stage)
        if self.standalone_stage == "hatch":
            self._stage = "hatch"
            self._child = self._new_hatch()
            self._hatch_baseline = 0
            self._start_hatch_cycle()
            return
        if self.standalone_stage == "cave":
            self._stage = "cave"
            self._child = CaveCullPlanner(
                self.reader,
                threshold=self.cull_threshold,
                reference_width=self.reference_width,
                safe_margin=self.cave_safe_margin,
                bottom_exclusion_px=self.cave_bottom_exclusion_px,
                capacity_limit=self.capacity_limit,
                capacity_read_retries=self.capacity_read_retries,
                cave_recenter_checks=self.cave_recenter_checks,
                capacity_snapshots=self.capacity_snapshots,
                logger=self.logger,
            )
            return
        self._stage = "open_nest"
        self._child = object()

    def _start_standalone_nest_phase(self) -> None:
        assert self.standalone_stage is not None
        if self.standalone_stage == "attack":
            self._start_replacement("attack")
        elif self.standalone_stage == "hp":
            self._start_replacement("hp")
        elif self.standalone_stage == "top":
            self._stage = "top"
            self._child = AutoPlaceRoundPlanner(
                TOP_RULE,
                reference_width=self.reference_width,
                logger=self.logger,
            )
        elif self.standalone_stage == "mass":
            self._stage = "mass"
            self._child = AutoPlaceRoundPlanner(
                MASS_RULE,
                reference_width=self.reference_width,
                logger=self.logger,
            )
        elif self.standalone_stage == "collect":
            self._start_collect()
        else:
            raise RuntimeError(
                f"standalone stage does not use My Nest: {self.standalone_stage}"
            )

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
        wait_seconds = float(self._hatch_kwargs["rescan_interval_seconds"])
        wait_source = "config"
        if self._observed_cooldown_until is not None:
            wait_seconds = max(0.0, self._observed_cooldown_until - self.clock())
            wait_source = "screen-batch"
        self.logger.info(
            "Hatch full | nest eggs collected after empty incubator | "
            "cooldown wait=%.0fs | source=%s",
            wait_seconds,
            wait_source,
        )
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._start_hatch_cycle()
        self._collect_only_after_empty = False
        self._empty_rescan_wait = True
        self._hatch_child.begin_rescan_wait(
            "collected all nest eggs after no ready incubator eggs",
            seconds=wait_seconds,
        )
        self._observed_cooldown_until = None

    def _observe_hatch_cooldown(
        self,
        frame: Frame,
        by_type: dict[str, list[Detection]],
    ) -> None:
        if hatch_feature.INCUBATOR_TITLE not in by_type:
            return
        if by_type.get(hatch_feature.HATCH_LABEL):
            return
        seconds = hatch_feature.read_hatch_cooldown_seconds(
            frame.image,
            self.reader,
            reference_width=self.reference_width,
        )
        if seconds is not None:
            self._observed_cooldown_until = self.clock() + seconds

    def _boost_allowed(self, by_type: dict[str, list[Detection]]) -> bool:
        return (
            self._boost_enabled_for_cycle
            and not self._boost_attempted
            and hatch_feature.INCUBATOR_TITLE in by_type
            and not by_type.get(hatch_feature.HATCH_LABEL)
            and bool(by_type.get(hatch_feature.CLOSE_BUTTON))
            and (
                self.boost_inventory is None
                or self.boost_inventory.snapshot().remaining > 0
            )
        )

    def _eggs_cooling(self) -> bool:
        """Whether the incubator currently holds eggs mid-cooldown."""

        return (
            self._observed_cooldown_until is not None
            and self._observed_cooldown_until > self.clock()
        )

    def _should_use_hatch_boost(
        self,
        by_type: dict[str, list[Detection]],
    ) -> bool:
        # 空孵化器不按加速:加速期從按下就開始倒數,蛋要等收蛋後才
        # 入孵化器,先按等於白燒加速時間。收蛋後回訪時再按。
        return self._boost_allowed(by_type) and self._eggs_cooling()

    def _start_replacement(self, kind: str) -> None:
        if kind == "attack":
            self._stage = "attack"
            self._child = AttackReplacementTestPlanner(
                self.reader,
                reference_width=self.reference_width,
                rule=ATTACK_RULE,
                stat_guards=self.stat_upgrade_guards,
                minimum_consistent_stat_reads=self.minimum_consistent_stat_reads,
                stat_read_retries=self.stat_read_retries,
                parent_stats_snapshots=self.parent_stats_snapshots,
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
            stat_guards=self.stat_upgrade_guards,
            minimum_consistent_stat_reads=self.minimum_consistent_stat_reads,
            stat_read_retries=self.stat_read_retries,
            parent_stats_snapshots=self.parent_stats_snapshots,
            logger=self.logger,
        )

    def _start_next_screening_stage(self) -> None:
        """Resume the first unproven parent stage, then collect eggs."""

        missing = self._missing_screening_stages()
        if not missing:
            self.logger.info(
                "Hatch full | screening complete | completed=%s",
                self._format_screening_stages(self._screening_completed),
            )
            self._start_collect()
            return
        next_stage = missing[0]
        self.logger.info(
            "Hatch full | screening stage starting | stage=%s | completed=%s",
            next_stage,
            self._format_screening_stages(self._screening_completed),
        )
        if next_stage in {"attack", "hp"}:
            self._start_replacement(next_stage)
            return
        self._stage = next_stage
        rule = TOP_RULE if next_stage == "top" else MASS_RULE
        self._child = AutoPlaceRoundPlanner(
            rule,
            reference_width=self.reference_width,
            logger=self.logger,
        )

    def _missing_screening_stages(self) -> tuple[str, ...]:
        return tuple(
            stage for stage in SCREENING_STAGES
            if stage not in self._screening_completed
        )

    @staticmethod
    def _format_screening_stages(stages: Sequence[str] | set[str]) -> str:
        selected = set(stages)
        ordered = [stage for stage in SCREENING_STAGES if stage in selected]
        return ",".join(ordered) if ordered else "none"

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
        if self.standalone_stage == self._stage:
            self._begin_home_recovery(
                f"standalone {self.standalone_stage} completed"
            )
            return
        completed_stage = self._stage
        self._screening_recovery_failures.pop(
            f"{completed_stage}:parent_stats_unreadable",
            None,
        )
        self._screening_completed.add(completed_stage)
        self.logger.info(
            "Hatch full | screening stage completed | stage=%s",
            completed_stage,
        )
        self._start_next_screening_stage()

    def _advance_autoplace_if_done(self) -> None:
        child = self._autoplace_child
        if not child.is_complete():
            return
        if self.standalone_stage == self._stage:
            self._begin_home_recovery(
                f"standalone {self.standalone_stage} completed"
            )
            return
        completed_stage = self._stage
        self._screening_completed.add(completed_stage)
        self.logger.info(
            "Hatch full | screening stage completed | stage=%s",
            completed_stage,
        )
        self._start_next_screening_stage()

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
    def _capacity_child(self) -> CaveCullPlanner:
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


def _hypot(offset: tuple[float, float]) -> float:
    return math.hypot(offset[0], offset[1])


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


def _retarget(detection: Detection, target_type: str) -> Target:
    """Keep a matched button's box while assigning a workflow-specific action."""

    return _target(
        Detection(
            type=target_type,
            x=detection.x,
            y=detection.y,
            confidence=detection.confidence,
            bbox=detection.bbox,
            metadata={**detection.metadata, "source_type": detection.type},
        )
    )


def nest_mask_close_target(frame: Frame, reference_width: float) -> Target:
    """The mask tap that dismisses the My Nest panel back to the home screen.

    Shared with the hunting side: a hunt that finds this panel open has no
    vocabulary for it and would otherwise tap map controls it cannot reach.
    """

    return _synthetic(
        NEST_MASK_CLOSE,
        *_scaled(frame, NEST_MASK_POINT, reference_width),
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
