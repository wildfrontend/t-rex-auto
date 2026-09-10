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
import pathlib
from functools import lru_cache
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
from .cooldown_boost import (
    BOOST_ACTIONS,
    BOOST_CANCEL,
    BOOST_CLOSE,
    BOOST_OPEN,
    CooldownBoostVisit,
)
from .cull import (
    EXPECTED_CAPACITY,
    PANEL_CAPACITY_REGION,
    PANEL_CLOSE_POINT,
    PANEL_OPEN_POINT,
    PANEL_TITLE,
    CapacityRead,
    locate_panel_capacity,
    probe_dino_count,
    should_cull,
)
from .digits import DigitReader
from .hatch_inventory import HatchBoostInventoryStore
from .models import Detection, Frame, Target, VerificationResult
from .nests import (
    ATTACK_AUTOPLACE_RULE,
    ATTACK_RULE,
    DEFAULT_STAT_UPGRADE_GUARDS,
    HP_AUTOPLACE_RULE,
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
from .targeting import best_detection, detection_target, swipe_target, synthetic_target

# Full-workflow synthetic actions and newly cropped screen anchors.
OPEN_NEST = "hatch_full_open_nest"
NEST_GEAR = "hatch_nest_gear"
NEST_BUBBLE = "hatch_nest_bubble"
RECOVERY_BUBBLE_DISMISS = "hatch_recovery_bubble_dismiss"
PANEL_OPEN = "hatch_my_dino_open"
PANEL_CLOSE = "hatch_my_dino_close"
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
# Original incubator: one ticket boost bar at the bottom.
HATCH_BOOST_POINT = (450.0, 1380.0)
_BOOST_BAR_SAMPLE = (380, 1355, 520, 1405)
# Incubator v2: a new permanent egg-speed bar pushes the ticket boost down.
HATCH_BOOST_POINT_V2 = (450.0, 1420.0)
_BOOST_BAR_SAMPLE_V2 = (380, 1405, 520, 1455)
# 按鈕帶中段(避開左側 50% 圖示與右側票券圖示)的取樣框,900 寬座標。
_BOOST_BAR_MIN_SATURATION = 80.0

CAVE_SWIPE = "hatch_cave_swipe"
CAVE_RECENTER = "hatch_cave_recenter"
CAVE_SELECT_BUTTON = "hatch_cave_select_button"
CAVE_CONTINUOUS_BUTTON = "hatch_cave_continuous_button"
CAVE_CLOSE_BUTTON = "hatch_cave_close_button"
SELECT_TAG_HEADER = "hatch_cull_tag_header"
SELECT_WEAKEST_BUTTON = "hatch_select_weakest_button"
SELECT_CHOOSE_BUTTON = "hatch_select_choose_button"
# The select-dino list carries four look-alike buttons in one row: green
# "strongest", cyan "choose", yellow "weakest" and yellow "top". A template
# cut from the wrong one still matches at ~1.0, and confidence alone cannot
# tell them apart - that mistake culled the strongest dinosaurs instead of
# the weakest. Anchor the tap to the calibrated position as well.
SELECT_WEAKEST_BUTTON_POINT = (640.0, 1297.0)
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
# Empty map well clear of the bubble, the home anchor and the egg pile.
BUBBLE_DISMISS_POINT = (700.0, 1180.0)
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
    # The server-error dialog is modal and can appear at any point, not only
    # at launch; it blocks every other action until RESTART is pressed.
    "server_error_restart_button",
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
FULL_HATCH_STAGES: frozenset[str] = frozenset(
    {*SCREENING_STAGES, "collect", "cave", "hatch"}
)
# What the default hatch+hunt cycle runs when no explicit stage list is given.
# Mass placement is deliberately absent: it is the least valuable screening
# pass per minute spent, and every stage delays the first hunt of a session.
# It stays in FULL_HATCH_STAGES so the custom workflow can still select it.
DEFAULT_FULL_HATCH_STAGES: frozenset[str] = FULL_HATCH_STAGES - {"mass"}

PLACE_SORT_BEST = "hatch_place_sort_best"
PLACE_SORT_LEVEL = "hatch_place_sort_level"
PLACE_SORT_ATTACK = "hatch_place_sort_attack"
PLACE_SORT_HP = "hatch_place_sort_hp"
PLACE_HDR_BEST = "hatch_place_hdr_best"
PLACE_HDR_LEVEL = "hatch_place_hdr_level"
PLACE_HDR_ATTACK = "hatch_place_hdr_attack"
PLACE_HDR_HP = "hatch_place_hdr_hp"
# The redesigned auto-place dialog moved upward, but the dropdown order is
# unchanged. Coordinates are in the 900-wide reference layout.
AUTOPLACE_SORT_HEADER_POINT = (450.0, 576.0)
AUTOPLACE_SORT_BEST_POINT = (450.0, 630.0)
AUTOPLACE_SORT_LEVEL_POINT = (450.0, 678.0)
AUTOPLACE_SORT_ATTACK_POINT = (450.0, 726.0)
AUTOPLACE_SORT_HP_POINT = (450.0, 774.0)

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    **hatch_feature.DEFAULT_TARGET_ACTIONS,
    **replacement_feature.DEFAULT_TARGET_ACTIONS,
    OPEN_NEST: "tap",
    NEST_GEAR: "tap",
    AUTOPLACE_SORT_HEADER: "tap",
    PLACE_SORT_BEST: "tap",
    PLACE_SORT_LEVEL: "tap",
    PLACE_SORT_ATTACK: "tap",
    PLACE_SORT_HP: "tap",
    AUTOPLACE_MASK_CLOSE: "tap",
    AUTOPLACE_BUTTON: "tap",
    AUTOPLACE_YES: "tap",
    COLLECT_EGGS_BUTTON: "tap",
    NEST_MASK_CLOSE: "tap",
    HATCH_DETAIL_CLOSE: "tap",
    HATCH_BOOST_BUTTON: "tap",
    HATCH_BOOST_CONFIRM: "tap",
    BOOST_OPEN: "tap",
    BOOST_CLOSE: "tap",
    BOOST_CANCEL: "tap",
    CAVE_SWIPE: "swipe",
    CAVE_RECENTER: "swipe",
    CAVE: "tap",
    RECOVERY_BUBBLE_DISMISS: "tap",
    PANEL_OPEN: "tap",
    PANEL_CLOSE: "tap",
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
    PLACE_SORT_ATTACK: 2500,
    PLACE_SORT_HP: 2500,
    AUTOPLACE_MASK_CLOSE: 2500,
    AUTOPLACE_BUTTON: 4000,
    AUTOPLACE_YES: 5000,
    COLLECT_EGGS_BUTTON: 5000,
    NEST_MASK_CLOSE: 3000,
    HATCH_DETAIL_CLOSE: 3000,
    HATCH_BOOST_BUTTON: 3000,
    HATCH_BOOST_CONFIRM: 3000,
    BOOST_OPEN: 3500,
    BOOST_CLOSE: 3000,
    BOOST_CANCEL: 3000,
    CAVE_SWIPE: 3000,
    CAVE_RECENTER: 3500,
    CAVE: 4000,
    RECOVERY_BUBBLE_DISMISS: 2500,
    PANEL_OPEN: 3000,
    PANEL_CLOSE: 2500,
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
    AUTOPLACE_SORT_HEADER: (
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        PLACE_SORT_ATTACK,
        PLACE_SORT_HP,
    ),
    PLACE_SORT_BEST: (PLACE_HDR_BEST,),
    PLACE_SORT_LEVEL: (PLACE_HDR_LEVEL,),
    PLACE_SORT_ATTACK: (PLACE_HDR_ATTACK,),
    PLACE_SORT_HP: (PLACE_HDR_HP,),
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
    BOOST_OPEN: (hatch_feature.INCUBATOR_TITLE,),
    BOOST_CLOSE: (hatch_feature.HOME_ANCHOR,),
    BOOST_CANCEL: (hatch_feature.INCUBATOR_TITLE,),
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
DEFAULT_SUCCESS_DISAPPEARANCES: dict[str, tuple[str, ...]] = {
    **replacement_feature.DEFAULT_SUCCESS_DISAPPEARANCES,
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
        PLACE_SORT_ATTACK,
        PLACE_SORT_HP,
        AUTOPLACE_MASK_CLOSE,
    }
)
MAX_NAVIGATION_RETRIES = 2
MAX_SCREENING_RECOVERY_FAILURES = 3

# The centred home map is proven by the egg-pile base rather than by the pile
# itself, because the pile artwork changes with its contents while the base
# stays put.  Reference and tolerance live together so the "is it centred" test
# and the correction that follows it cannot drift apart.
HOME_PILE_BASE: tuple[float, float] = (450.0, 1455.0)
HOME_PILE_TOLERANCE = 100.0
# Frames to let the map glide after a measured drag before treating a missing
# pile as proof the drag went wrong.
PILE_SETTLE_FRAMES = 3

# `HOME_PILE_BASE` describes where the pile sits when the camera still has room
# to travel, but the home map has a hard bottom edge.  An S9 trace measured the
# basin base parked at y~=1290 with the map already scrolled to that edge: the
# remaining 165px error was unreachable, so four "corrections" each swiped, let
# the map spring straight back, and re-measured the same offset before recovery
# gave up and the run died.  A drag that returns the camera to within this many
# pixels of where it started moved nothing, and repeating it cannot help.
CAMERA_LIMIT_PROGRESS_PX = 12.0
# One stalled drag can also be a dropped gesture, so require two in a row
# before concluding the camera is against its edge.
MAX_CAMERA_LIMIT_HITS = 2

# The cyan basin has to be told apart from the small fixed incubator nests,
# which share its exact hue.  Width alone cannot do it: the basin's cyan water
# narrows as eggs are consumed, and an S9 v0.0.73 frame measured it at 140px
# against a 121px nest - close enough that the original 170px gate discarded
# the real pile, left recovery with no measurement at all, and fused hatching
# off while the map was already centred.  Height separates them cleanly, since
# the basin is a deep bowl (118px there) while every nest is a shallow dish
# (43-67px across that same frame).
CYAN_STRIP_MIN_WIDTH = 170.0
CYAN_BASIN_MIN_WIDTH = 120.0
CYAN_BASIN_MIN_HEIGHT = 90.0

# The cyan strip above only exists on the upgraded stone basin.  The starter
# nest is straw on brick with no cyan anywhere, so that account measured
# nothing at all and every "am I home yet" test answered no forever.  Its base
# is matched as a template instead, over a scale sweep: unlike the basin, the
# whole straw nest grows with its contents (226x112 straw at six eggs, 264x131
# a little later - a uniform 1.17x), while its centre stays on the same map
# point.  Sweep bounds sit a little outside both observed sizes.
HOME_BASE_REFERENCE: tuple[int, int] = (900, 1600)
HOME_BASE_MIN_CONFIDENCE = 0.65
HOME_BASE_SCALES: tuple[float, ...] = tuple(
    round(0.75 + 0.05 * step, 2) for step in range(15)
)
# The higher-level lava nest has no cyan strip and is too different from the
# starter straw nest for that template sweep.  Its broad red/orange/yellow base
# is stable, however.  These bounds describe the one connected warm-colour
# component observed on the 900x1600 home map while rejecting the much smaller
# orange incubator nests around it.
LAVA_BASE_HSV_LOWER = (0, 100, 70)
LAVA_BASE_HSV_UPPER = (40, 255, 255)
LAVA_BASE_MIN_AREA = 8_000.0
LAVA_BASE_WIDTH_RANGE = (190.0, 280.0)
LAVA_BASE_HEIGHT_RANGE = (110.0, 220.0)
LAVA_BASE_BOTTOM_INSET = 18.0
# The S13 growth-stage pile uses desaturated blue stone.  Its coloured strip
# is narrower and bluer than the upgraded cyan basin, but unlike the generic
# dark-structure fallback it cannot merge with roaming dinosaurs below the
# pile.  Production pixels sit tightly around OpenCV hue 108.
BLUE_STONE_HSV_LOWER = (100, 70, 70)
BLUE_STONE_HSV_UPPER = (120, 255, 255)
BLUE_STONE_MIN_AREA = 1_000.0
BLUE_STONE_WIDTH_RANGE = (120.0, 280.0)
BLUE_STONE_HEIGHT_RANGE = (25.0, 120.0)
# The blue pixels describe the cloth/stone face above the structural base,
# while the cyan locator describes the lower common map anchor.  Normalize
# the skin-specific colour centroid before any shared centering or tap logic
# consumes it.  Live centred S13 evidence measures the blue centroid at
# y~=1415, 40px above HOME_PILE_BASE.
BLUE_STONE_ANCHOR_Y_OFFSET = 40.0
_HOME_BASE_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "hatch"
    / "templates"
    / "hatch-home-straw-base.png"
)
_home_base_template: np.ndarray | None = None
_home_base_template_loaded = False


def set_home_base_template(path: str | Path | None) -> None:
    """Point the straw-nest base matcher at an instance's own copy."""

    global _HOME_BASE_TEMPLATE_PATH, _home_base_template, _home_base_template_loaded
    if path is not None:
        _HOME_BASE_TEMPLATE_PATH = Path(path)
    _home_base_template = None
    _home_base_template_loaded = False

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
        PLACE_HDR_ATTACK,
        PLACE_HDR_HP,
        # 展開的排序選單選項也要掃:規劃看不見它們時會重按表頭,
        # 把剛打開的選單又關上,top/mass 階段就此死循環。
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        PLACE_SORT_ATTACK,
        PLACE_SORT_HP,
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
        PLACE_HDR_ATTACK,
        PLACE_HDR_HP,
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
        PLACE_SORT_ATTACK,
        PLACE_SORT_HP,
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
        # Preflight reads capacity off the My Dinosaurs panel, so its title
        # has to be visible to the scan that drives that stage.
        PANEL_TITLE,
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
        # Parks over the home anchor, so recovery has to see it to clear it.
        NEST_BUBBLE,
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


def is_centered_home_frame(frame: Frame) -> bool:
    """Prove centred, unobscured home from frame structure alone.

    Post-action verification deliberately scans only the expected successor
    templates.  S9's animated My Nest icon scored 0.833 and then 0.809 on two
    otherwise identical home frames, so requiring that single template made a
    successful incubator close look like a failure and reopened it.  The
    bright side strips plus a known, centred large pile are independent of the
    changing HUD artwork and reject the dimmed item-information overlay.
    """

    if not _is_bright_outdoor_map(frame):
        return False
    if not hatch_feature.has_home_pile_structure(
        frame,
        egg_pile_point=(450.0, 1330.0),
        reference_width=900.0,
    ):
        return False
    offset = home_pile_offset(frame)
    if offset is None:
        return False
    scale = frame.width / 900.0
    return max(abs(offset[0]), abs(offset[1])) <= HOME_PILE_TOLERANCE * scale


def is_centered_home_screen(frame: Frame, detections: Sequence[Detection]) -> bool:
    """Return whether the normal home map (not the shifted cave view) is ready."""

    types = {item.type for item in detections}
    # The map exit control is the one foreground type the handoff itself taps
    # to get home, and the game keeps painting it on the frame that follows.
    # Letting it veto the centred-home proof makes the target unreachable by
    # definition: an S9 v0.0.70 trace matched it at 0.957 on all eleven frames
    # of a handoff, so the test could never pass, the handoff toggled
    # map_exit -> recenter seven times, and the timeout fused the hatch side
    # off. The geometric proof below is the reliable signal here, so only
    # judge this frame by the foreground types that cannot share it.
    blocking = HOME_FOREGROUND_TYPES - {HUNT_MAP_EXIT}
    if any(item.type == CAVE for item in detections) or types & blocking:
        return False
    # The home anchor can be clipped after a successful map return even when
    # the stable egg-pile base is exactly centred. The Forest control is a
    # second named outdoor-map landmark; require it for this narrow fallback
    # instead of accepting an arbitrary bright screen with cyan pixels.
    if hatch_feature.HOME_ANCHOR not in types and FOREST_RECENTER not in types:
        return False
    # The recovery planner and HatchPlanner must share the same proof of an
    # actionable home pile. Without this gate recovery can complete while the
    # hatch child still sees an unknown screen and starts a Back loop.
    return is_centered_home_frame(frame)


def _home_pile_click_blocked(
    frame: Frame,
    detections: Sequence[Detection],
) -> bool:
    """Whether home geometry is centred but the pile click falls in chat."""

    return is_centered_home_screen(frame, detections) and (
        hatch_feature.home_pile_tap_point(
            frame,
            egg_pile_point=(450.0, 1330.0),
            reference_width=900.0,
        )
        is None
    )


def _hatch_boost_geometry(
    frame: Frame,
    *,
    reference_width: float = 900.0,
) -> tuple[tuple[float, float], tuple[int, int, int, int]]:
    """Return the ticket boost point and sample for the visible UI version."""

    if hatch_feature.uses_permanent_boost_layout(
        frame.image,
        reference_width=reference_width,
    ):
        return HATCH_BOOST_POINT_V2, _BOOST_BAR_SAMPLE_V2
    return HATCH_BOOST_POINT, _BOOST_BAR_SAMPLE


def _hatch_boost_point(
    frame: Frame,
    *,
    reference_width: float = 900.0,
) -> tuple[int, int]:
    point, _ = _hatch_boost_geometry(frame, reference_width=reference_width)
    return _scaled(frame, point, reference_width)


def _hatch_boost_ready(
    frame: Frame,
    *,
    reference_width: float = 900.0,
) -> bool:
    """Return whether the incubator cooldown-boost bar is pressable.

    The bar keeps its template shape while a boost is running, but the game
    desaturates it to gray for the countdown; normalized template matching is
    brightness-invariant, so color saturation is the only reliable signal.
    """

    if frame.image.size == 0:
        return False
    _, sample = _hatch_boost_geometry(frame, reference_width=reference_width)
    scale = frame.width / reference_width
    x0, y0, x1, y1 = (round(value * scale) for value in sample)
    roi = frame.image[y0:y1, x0:x1]
    if not roi.size:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= _BOOST_BAR_MIN_SATURATION


def _load_home_base_template() -> np.ndarray | None:
    global _home_base_template, _home_base_template_loaded
    if not _home_base_template_loaded:
        _home_base_template_loaded = True
        _home_base_template = cv2.imread(
            str(_HOME_BASE_TEMPLATE_PATH),
            cv2.IMREAD_COLOR,
        )
    return _home_base_template


def _straw_base_center(frame: Frame) -> tuple[float, float] | None:
    """Locate the starter nest's brick base, which carries no cyan at all."""

    template = _load_home_base_template()
    if template is None:
        return None
    image = frame.image
    if (frame.width, frame.height) != HOME_BASE_REFERENCE:
        image = cv2.resize(image, HOME_BASE_REFERENCE, interpolation=cv2.INTER_AREA)
    # Same floor the cyan search uses: the base never rides in the top half,
    # and a smaller haystack pays for the scale sweep.  Both sides are then
    # halved again, because matchTemplate costs scale with the searched pixel
    # count and a sweep pays that cost once per scale.  The 2px of reference
    # precision this trades away is nothing against a 100px tolerance.
    top = 850
    region = cv2.resize(image[top:, :], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    best: tuple[float, float, float] | None = None
    for scale in HOME_BASE_SCALES:
        sized = cv2.resize(
            template,
            None,
            fx=scale * 0.5,
            fy=scale * 0.5,
            interpolation=cv2.INTER_AREA,
        )
        if sized.shape[0] > region.shape[0] or sized.shape[1] > region.shape[1]:
            continue
        _, confidence, _, location = cv2.minMaxLoc(
            cv2.matchTemplate(region, sized, cv2.TM_CCOEFF_NORMED)
        )
        if best is None or confidence > best[0]:
            best = (
                confidence,
                (location[0] + sized.shape[1] / 2) * 2.0,
                top + (location[1] + sized.shape[0] / 2) * 2.0,
            )
    if best is None or best[0] < HOME_BASE_MIN_CONFIDENCE:
        return None
    frame_scale = frame.width / 900.0
    return (best[1] * frame_scale, best[2] * frame_scale)


def _lava_base_center(frame: Frame) -> tuple[float, float] | None:
    """Locate the red/orange upgraded nest and return its map anchor point."""

    image = frame.image
    if image.size == 0:
        return None
    scale = frame.width / 900.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, LAVA_BASE_HSV_LOWER, LAVA_BASE_HSV_UPPER)
    count, _, stats, _ = cv2.connectedComponentsWithStats(warm)
    candidates: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        center_x = x + width / 2.0
        if (
            area >= LAVA_BASE_MIN_AREA * scale * scale
            and LAVA_BASE_WIDTH_RANGE[0] * scale
            <= width
            <= LAVA_BASE_WIDTH_RANGE[1] * scale
            and LAVA_BASE_HEIGHT_RANGE[0] * scale
            <= height
            <= LAVA_BASE_HEIGHT_RANGE[1] * scale
            and y >= 850 * scale
            and 150 * scale <= center_x <= 750 * scale
        ):
            candidates.append((area, x, y, width, height))
    if not candidates:
        return None
    _, x, y, width, height = max(candidates)
    return (
        x + width / 2.0,
        y + height - LAVA_BASE_BOTTOM_INSET * scale,
    )


def _blue_stone_base_center(frame: Frame) -> tuple[float, float] | None:
    """Locate blue stone and return the common structural base anchor."""

    image = frame.image
    if image.size == 0:
        return None
    scale = frame.width / 900.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_STONE_HSV_LOWER, BLUE_STONE_HSV_UPPER)
    count, _, stats, centers = cv2.connectedComponentsWithStats(blue)
    candidates: list[tuple[int, float, float]] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        center_x, center_y = centers[index]
        if (
            area >= BLUE_STONE_MIN_AREA * scale * scale
            and BLUE_STONE_WIDTH_RANGE[0] * scale
            <= width
            <= BLUE_STONE_WIDTH_RANGE[1] * scale
            and BLUE_STONE_HEIGHT_RANGE[0] * scale
            <= height
            <= BLUE_STONE_HEIGHT_RANGE[1] * scale
            and y >= 650 * scale
            and 150 * scale <= center_x <= 750 * scale
        ):
            candidates.append((area, float(center_x), float(center_y)))
    if not candidates:
        return None
    # The main pile is the lowest qualifying wide blue base.  Small fixed
    # nests and blue map decorations fail the width/area gates above.
    _, center_x, center_y = max(candidates, key=lambda item: (item[2], item[0]))
    return center_x, center_y + BLUE_STONE_ANCHOR_Y_OFFSET * scale


def _egg_pile_base_center(frame: Frame) -> tuple[float, float] | None:
    """Locate the stable base of the variable-looking home egg pile."""

    if frame.image.size == 0:
        return None
    scale = frame.width / 900.0
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, (75, 70, 70), (105, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(cyan)
    candidates: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        center_x, center_y = centers[index]
        if (
            area < 350 * scale * scale
            or y < 850 * scale
            or not 150 * scale <= center_x <= 750 * scale
        ):
            continue
        if width >= CYAN_STRIP_MIN_WIDTH * scale:
            # A wide, flat strip: its centroid already sits on the base line.
            candidates.append((float(center_x), float(center_y)))
        elif (
            width >= CYAN_BASIN_MIN_WIDTH * scale
            and height >= CYAN_BASIN_MIN_HEIGHT * scale
        ):
            # A deep bowl of water. Its centroid floats in the middle of the
            # basin rather than on the map anchor, so report the bottom edge
            # the flat-strip case measures directly.
            candidates.append((float(center_x), float(y + height)))
    if candidates:
        return max(candidates, key=lambda center: center[1])
    blue_stone = _blue_stone_base_center(frame)
    if blue_stone is not None:
        return blue_stone
    lava = _lava_base_center(frame)
    if lava is not None:
        return lava
    straw = _straw_base_center(frame)
    if straw is not None:
        return straw
    # Never use the style-neutral dark structure as a camera measurement.  On
    # crowded S13 screens it joins the pile to moving dinosaurs, shifting the
    # reported base by hundreds of pixels between frames.  An unknown future
    # style must fail closed until it gets its own stable visual anchor.
    return None


def _egg_pile_safe_tap(frame: Frame) -> tuple[int, int] | None:
    """Choose a point on the eggs from a known stable pile base.

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
    x = round(pile[0])
    y = round(pile[1] - hatch_feature.HOME_PILE_TAP_OFFSET_PX * scale)
    if y >= frame.height - round(
        hatch_feature.HOME_PILE_TAP_BOTTOM_EXCLUSION_PX * scale
    ):
        return None
    return x, y


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
        max_forest_trips: int = 0,
        # ADB map drags can be damped by the game's camera inertia.  The S13
        # recovery trace reduced a 474px vertical error to 161px in two
        # strictly improving moves, but two was not enough to cross the 100px
        # acceptance tolerance.  Four remains bounded; a non-improving move
        # still stops immediately below rather than spending the extra budget.
        max_measured_corrections: int = 4,
        max_hunt_dialog_dismissals: int = 3,
        max_bubble_dismissals: int = 3,
        applied_swipes: Sequence[tuple[int, int, int, int]] = (),
    ) -> None:
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self.max_back_attempts = max(0, max_back_attempts)
        self.required_home_frames = max(1, required_home_frames)
        # Kept as a compatibility argument for callers built against older
        # releases.  Entering Forest never recentres the home camera; recovery
        # must stay on one measurable map instead.
        self.max_forest_trips = 0
        self.max_hunt_dialog_dismissals = max(0, max_hunt_dialog_dismissals)
        self.max_bubble_dismissals = max(0, max_bubble_dismissals)
        self.max_measured_corrections = max(0, max_measured_corrections)
        self._stage = "inspect"
        self._back_attempts = 0
        self._home_frames = 0
        self._recenter_swipes = 0
        self._cave_recovery_required = False
        self._recenter_end = 0
        self._forest_trips = 0
        self._hunt_dialog_dismissals = 0
        self._bubble_dismissals = 0
        self._measured_corrections = 0
        self._last_offset: tuple[float, float] | None = None
        self._camera_limit_hits = 0
        # Set once the camera proves it cannot close the remaining offset, so
        # the completion gate stops demanding a position the map cannot reach.
        self._camera_at_limit = False
        self._autoplace_notice_without_no = False
        self._applied_swipes = list(applied_swipes)
        # Swipes inherited from the cave planner or an earlier episode were
        # aimed at a screen this planner has not measured. Undoing one is right
        # when the home map is proven but the pile is missing - that is the
        # cave return having pushed it out of frame. It is wrong on a frame
        # that shows neither: on s9 the very first recovery frame carried only
        # the Forest control, the inherited cave leg was replayed backwards
        # anyway, and the 400px gap it opened survived all four measured
        # corrections. Require the home anchor before reaching inherited legs.
        self._inherited_swipes = len(applied_swipes)
        self._pile_settle_frames = 0
        self._applied_offsets: list[tuple[float, float] | None] = [None] * len(applied_swipes)
        self._pending_swipe: tuple[int, int, int, int] | None = None
        self._pending_measured_offset: tuple[float, float] | None = None
        self._pending_undo_swipe: tuple[int, int, int, int] | None = None
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

    def camera_history(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(self._applied_swipes)

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
                self._applied_offsets.append(self._pending_measured_offset)
        elif target_type == RECOVERY_UNDO:
            # Treat successful corrections as a real LIFO stack.  A numeric
            # "undo count" is not sufficient: after undoing B and applying C,
            # the next rollback must remove C, not select B a second time.
            if (
                self._pending_undo_swipe is not None
                and self._applied_swipes
                and self._applied_swipes[-1] == self._pending_undo_swipe
            ):
                self._applied_swipes.pop()
                self._applied_offsets.pop()
                self._last_offset = next(
                    (
                        offset
                        for offset in reversed(self._applied_offsets)
                        if offset is not None
                    ),
                    None,
                )
        elif target_type == RECOVERY_FOREST:
            self._forest_trips += 1
        self._pending_swipe = None
        self._pending_measured_offset = None
        self._pending_undo_swipe = None
        self._pending_cave_leg = False
        self._stage = f"verify_{target_type}"

    def on_action_failure(self, target_type: str) -> None:
        self.logger.error("Hatch recovery | action failed | target=%s", target_type)
        self._pending_swipe = None
        self._pending_measured_offset = None
        self._pending_undo_swipe = None
        self._pending_cave_leg = False
        self._stage = f"failed_{target_type}"
        self._failed = True

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete or self._failed:
            return None
        by_type = _group(detections)
        exact_prompt = any(
            target_type in by_type
            for target_type in (
                AUTOPLACE_PROMPT,
                SELECT_CONFIRM_PROMPT,
                NESTED_PARENT_WARNING,
            )
        )
        notice_only = AUTOPLACE_NOTICE in by_type and not exact_prompt
        notice_misread = notice_only and best_detection(by_type.get(CONFIRM_NO)) is None
        # A broad notice without its defining No control is not foreground.
        # Exclude only that evidence from the home geometry checks, so a
        # task-complete toast cannot make a visible, measurable home map look
        # like an opaque dialog.  Keep the original detections for all named
        # prompt, close and escape rules below.
        home_detections = (
            [item for item in detections if item.type != AUTOPLACE_NOTICE]
            if notice_misread
            else detections
        )
        home_by_type = _group(home_detections)
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
        # A camera parked against the map edge cannot satisfy the centred
        # offset, so fall back to the weaker "this really is the home map"
        # test for it.  The click-blocked guard below still has to pass, which
        # is what the hatch flow actually depends on.
        centered = is_centered_home_screen(frame, home_detections)
        if not centered and self._camera_at_limit:
            centered = is_home_screen(frame, home_detections)
        if centered and not _home_pile_click_blocked(
            frame,
            home_detections,
        ):
            self._home_frames += 1
            self._stage = f"confirm_home_{self._home_frames}/{self.required_home_frames}"
            if self._home_frames >= self.required_home_frames:
                self._complete = True
                self._stage = "done"
            return None
        self._home_frames = 0

        known_prompt = exact_prompt or notice_only
        if known_prompt:
            no = best_detection(by_type.get(CONFIRM_NO))
            if no is None:
                if notice_only:
                    # AUTOPLACE_NOTICE is a deliberately broad layout
                    # detector.  The task-complete toast seen on S13 has the
                    # same yellow/white composition but no confirmation
                    # controls.  It is evidence against the reading, not a
                    # reason to abort recovery; fall through to the map and
                    # home-pile rules below.  Exact prompt templates retain
                    # the conservative failure path because a partially drawn
                    # real confirmation must never be dismissed blindly.
                    if not self._autoplace_notice_without_no:
                        self.logger.warning(
                            "Hatch recovery | auto-place layout without a No "
                            "button; treating it as a misread and continuing"
                        )
                        self._autoplace_notice_without_no = True
                else:
                    self.logger.error("Hatch recovery | known prompt has no No button")
                    self._stage = "prompt_without_no"
                    self._failed = True
                    return None
            else:
                self._autoplace_notice_without_no = False
                self._stage = "cancel_prompt"
                return synthetic_target(RECOVERY_NO, no.x, no.y)
        else:
            self._autoplace_notice_without_no = False

        if AUTOPLACE_TITLE in by_type or SELECT_TITLE in by_type or NEST_TITLE in by_type:
            self._stage = "close_mask_layer"
            return synthetic_target(
                RECOVERY_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )

        # The idle "auto growth result" card appears on its own after the game
        # has run unattended, and it dims the whole map behind it: the egg
        # pile cannot be measured, so the pile tap fails and recovery cannot
        # prove a centred home.  One S9 run lost the hatch workflow that way
        # at 22:50 and spent the next hour hunting only.  The card carries no
        # close button - tapping the dimmed margin beside it dismisses it -
        # so reuse the mask point the nest and auto-place layers already use.
        # The growth-result layout detector is intentionally colour-based so
        # it can support the game's changing artwork.  A snowy S9 home map
        # with a large egg pile can resemble that layout, though.  A real
        # growth card dims the outdoor map; a bright map with its home anchor
        # is stronger evidence that this is a false layout match and must be
        # allowed to recenter rather than spending the Back ladder.
        if STARTUP_GROWTH_RESULT in by_type and not is_home_screen(
            frame, home_detections
        ):
            self._stage = "close_growth_result_mask"
            return synthetic_target(
                RECOVERY_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )

        claim = best_detection(by_type.get(hatch_feature.CLAIM_BUTTON))
        if claim is not None:
            # Collecting a completed hatch/battle result is safer than Back:
            # it preserves the result and leads to another named screen.
            self._stage = "collect_result"
            return synthetic_target(RECOVERY_CLAIM, claim.x, claim.y)

        # Android Back dismisses the maximum-population dialog but does not
        # close the egg detail beneath it. Once the dialog is gone, use the
        # structurally proven red-X/yellow-button pair to leave the detail and
        # continue toward a real home screen before the capacity preflight.
        detail_close = _unready_egg_detail_close(frame)
        if detail_close is not None:
            self._stage = "close_hatch_detail"
            return synthetic_target(HATCH_DETAIL_CLOSE, *detail_close)

        close = best_detection(
            [
                *by_type.get(hatch_feature.CLOSE_BUTTON, ()),
                *by_type.get(CAVE_CLOSE_BUTTON, ()),
            ]
        )
        if close is not None:
            self._stage = "close_named_screen"
            return synthetic_target(RECOVERY_CLOSE, close.x, close.y)

        map_exit = best_detection(by_type.get(HUNT_MAP_EXIT))
        if map_exit is not None:
            # A combined hatch+hunt process can be restarted during the hunt
            # cooldown. Android Back does not leave this map, but its explicit
            # exit control is stable and already template-gated.
            self._stage = "leave_hunt_map"
            return synthetic_target(RECOVERY_MAP_EXIT, map_exit.x, map_exit.y)

        # A hunt prompt covers the map controls the rung above needs, and Back
        # does not close it: six presses against one left the button's box and
        # confidence identical every frame at pixel_change=0.000, and the
        # escape ladder spent its whole budget on them before failing. Clear
        # the prompt first so the named exit can be seen at all.
        hunt_prompt = best_detection(
            [
                *by_type.get("hunt_button", ()),
                *by_type.get("hunt_max_group_button", ()),
            ]
        )
        if hunt_prompt is not None:
            dialog_close = best_detection(by_type.get(HUNT_DIALOG_CLOSE))
            if dialog_close is not None:
                self._stage = "close_hunt_dialog"
                return synthetic_target(
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
                return synthetic_target(
                    RECOVERY_HUNT_DIALOG_DISMISS,
                    *_scaled(frame, HUNT_DIALOG_DISMISS_POINT, self.reference_width),
                )

        # Correcting the camera against a landmark that is still on screen
        # outranks any blind escape below: those only open and close screens,
        # and none of them can put the map back where it belongs.
        recenter = self._recenter_target(
            frame,
            home_detections,
            home_by_type,
            vectors,
        )
        if recenter is not None:
            return recenter

        # A nest bubble (管理 / 升級) parks over the top-left corner, which is
        # exactly where the home anchor sits: s9 read the anchor at 0.675
        # instead of 0.990 and recovery could not prove centered home on a
        # frame that was otherwise perfect - offset 95px, well inside
        # tolerance. Dismiss it before spending the escape ladder on Back.
        bubble = best_detection(by_type.get(NEST_BUBBLE))
        if bubble is not None and self._bubble_dismissals < self.max_bubble_dismissals:
            self._bubble_dismissals += 1
            self._stage = (
                f"dismiss_nest_bubble_{self._bubble_dismissals}"
                f"/{self.max_bubble_dismissals}"
            )
            return synthetic_target(
                RECOVERY_BUBBLE_DISMISS,
                *_scaled(frame, BUBBLE_DISMISS_POINT, self.reference_width),
            )

        forest = best_detection(by_type.get(FOREST_RECENTER))
        own_swipe_pending = len(self._applied_swipes) > self._inherited_swipes
        home_anchor_seen = bool(by_type.get(hatch_feature.HOME_ANCHOR))
        # A drag leaves the pile out of frame while the map is still gliding,
        # so the first frames after one are not evidence the correction went
        # wrong. Undoing them immediately cancels the very move that was
        # working: s9 measured 146px, corrected four times, undid all four, and
        # ended at 160px - then fused hatching off. Let the map settle first.
        if (
            forest is not None
            and self._applied_swipes
            and self._pile_settle_frames < PILE_SETTLE_FRAMES
        ):
            self._pile_settle_frames += 1
            return None
        if (
            forest is not None
            and self._applied_swipes
            and (own_swipe_pending or home_anchor_seen)
        ):
            # This control proves we are already on the home map; it is not a
            # recenter command.  If our own last measured drag made the pile
            # disappear, reverse that exact drag without leaving the map.
            undo = self._undo_target(
                reason="measured landmark disappeared on the home map"
            )
            if undo is not None:
                return undo

        if self._back_attempts < self.max_back_attempts:
            self._back_attempts += 1
            self._stage = f"back_{self._back_attempts}/{self.max_back_attempts}"
            return synthetic_target(RECOVERY_BACK, frame.width // 2, frame.height // 2)

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
        measured_offset: tuple[float, float] | None = None,
    ) -> Target:
        """Issue a camera gesture, holding it until the action is confirmed."""

        if target_type == RECOVERY_RECENTER:
            self._pending_swipe = (x1, y1, x2, y2)
            self._pending_measured_offset = measured_offset
            self._pending_cave_leg = cave_leg
        return swipe_target(target_type, x1, y1, x2, y2)

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
        pile_structure = hatch_feature.home_pile_structure_base(
            frame,
            egg_pile_point=(450.0, 1330.0),
            reference_width=900.0,
        )
        click_blocked = pile_structure is not None and (
            hatch_feature.home_pile_tap_point(
                frame,
                egg_pile_point=(450.0, 1330.0),
                reference_width=900.0,
            )
            is None
        )
        if (
            max(abs(offset[0]), abs(offset[1])) <= HOME_PILE_TOLERANCE * scale
            and not click_blocked
        ):
            return None
        if self._last_offset is not None and (
            abs(_hypot(offset) - _hypot(self._last_offset))
            <= CAMERA_LIMIT_PROGRESS_PX * scale
        ):
            # The drag neither improved nor worsened the offset: the map did
            # not move at all, which on this map means the camera is against
            # an edge.  Accept the position instead of burning the remaining
            # attempts on gestures the game will keep springing back, since
            # everything downstream only needs a stable, unobscured home map.
            self._camera_limit_hits += 1
            if self._camera_limit_hits >= MAX_CAMERA_LIMIT_HITS and not click_blocked:
                self.logger.warning(
                    "Hatch recovery | camera at map edge; accepting home"
                    " offset (%.0f,%.0f)px after %d stalled corrections",
                    offset[0],
                    offset[1],
                    self._camera_limit_hits,
                )
                self._last_offset = None
                self._camera_limit_hits = 0
                self._camera_at_limit = True
                return None
        elif (
            self._last_offset is not None
            and _hypot(offset) >= _hypot(self._last_offset)
        ):
            # The latest correction did not converge.  Reverse that exact
            # gesture before trying another measurement; continuing from the
            # regressed camera position walks the map further from home.
            self.logger.error(
                "Hatch recovery | measured correction did not reduce the offset"
                " | before=(%.0f,%.0f) | after=(%.0f,%.0f)",
                self._last_offset[0],
                self._last_offset[1],
                offset[0],
                offset[1],
            )
            return self._undo_target(reason="measured correction regressed")
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
        # A fresh drag earns a fresh settle window; without this the wait is
        # spent once and every later correction is undone on its first frame.
        self._pile_settle_frames = 0
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
        return self._swipe(
            RECOVERY_RECENTER,
            x1,
            y1,
            x2,
            y2,
            measured_offset=offset,
        )

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

    def _undo_target(self, *, reason: str = "no home landmark") -> Target | None:
        """Reverse this planner's own camera move once the landmarks are gone.

        Every branch that can prove or measure the home position needs either
        the home anchor or the egg pile.  A gesture that pushes both out of
        the viewport therefore blinds the planner to its own mistake, and no
        later branch can repair it: the escape ladder below only opens and
        closes screens.  Replaying the gesture backwards is the one move that
        restores something to measure against.
        """

        if not self._applied_swipes:
            return None
        x1, y1, x2, y2 = self._applied_swipes[-1]
        self._pending_undo_swipe = self._applied_swipes[-1]
        self._stage = f"undo_recenter_latest/{len(self._applied_swipes)}"
        self.logger.warning(
            "Hatch recovery | %s after own camera move"
            " | replaying (%d,%d)->(%d,%d) backwards",
            reason,
            x1,
            y1,
            x2,
            y2,
        )
        return self._swipe(RECOVERY_UNDO, x2, y2, x1, y1)


class AutoPlaceRoundPlanner:
    """Converge one nest tag, set auto-place sorting, and apply it."""

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
            ATTACK_AUTOPLACE_RULE.tag: (
                nest_filter_feature.TAG_ATTACK,
                nest_filter_feature.TAG_HDR_ATTACK,
                PLACE_SORT_ATTACK,
                PLACE_HDR_ATTACK,
                AUTOPLACE_SORT_ATTACK_POINT,
            ),
            HP_AUTOPLACE_RULE.tag: (
                nest_filter_feature.TAG_HP,
                nest_filter_feature.TAG_HDR_HP,
                PLACE_SORT_HP,
                PLACE_HDR_HP,
                AUTOPLACE_SORT_HP_POINT,
            ),
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
        elif target_type in (
            PLACE_SORT_BEST,
            PLACE_SORT_LEVEL,
            PLACE_SORT_ATTACK,
            PLACE_SORT_HP,
        ):
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
            gear = best_detection(by_type.get(NEST_GEAR))
            if gear is None:
                return None
            return detection_target(gear)
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
                return detection_target(desired_option)
            return synthetic_target(
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
            return synthetic_target(
                AUTOPLACE_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )
        if self._stage == "verify_settings_closed":
            if AUTOPLACE_TITLE in by_type:
                self._stage = "close_settings"
                return self.choose(frame, detections)
            if NEST_TITLE not in by_type:
                return None
            button = best_detection(by_type.get(AUTOPLACE_BUTTON))
            if button is None:
                return None
            return detection_target(button)
        if self._stage == "after_autoplace":
            if AUTOPLACE_PROMPT in by_type or AUTOPLACE_NOTICE in by_type:
                yes = best_detection(by_type.get(CONFIRM_YES))
                if yes is None:
                    self._stage = "blocked_notice_without_yes"
                    return None
                return synthetic_target(AUTOPLACE_YES, yes.x, yes.y)
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


@lru_cache(maxsize=1)
def _load_panel_template() -> Image | None:
    """Load the My Dinosaurs title art that anchors the capacity crop."""

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "assets"
        / "hatch"
        / "templates"
        / "hatch-my-dino-title.png"
    )
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


class PanelCapacityPlanner:
    """Read the population off the My Dinosaurs panel and close it again.

    Interface-compatible with the slice of CaveCullPlanner that preflight
    uses, so it can stand in wherever capacity only needs reading. It never
    culls: the panel is a measurement, and deleting dinosaurs stays with the
    cave planner that was built for it.

    Preflight used to reach this figure through the cave HUD, which meant
    driving the camera to the hatch home first. That is the position s9's
    146px pile offset keeps missing, so a reading that needs no camera at all
    removes the dependency rather than fighting it.
    """

    def __init__(
        self,
        reader: DigitReader,
        *,
        threshold: int,
        capacity_limit: int = EXPECTED_CAPACITY,
        reference_width: float = 900.0,
        capacity_read_retries: int = 2,
        capacity_snapshots: CapacitySnapshot | None = None,
        panel_template: Image | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.reader = reader
        self.threshold = threshold
        self.capacity_limit = capacity_limit
        self.reference_width = reference_width
        self.capacity_read_retries = max(1, capacity_read_retries)
        self.capacity_snapshots = capacity_snapshots
        self.logger = logger or logging.getLogger("dino_bot")
        # Located by ink rather than by the detector: the title renders a
        # little differently each opening, so a grayscale template match tops
        # out near 0.6 and cannot be separated from a miss.
        self.panel_template = (
            panel_template if panel_template is not None else _load_panel_template()
        )
        self._stage = "open_panel"
        self._complete = False
        self._capacity_readable: bool | None = None
        self._cull_required = False
        self._last_capacity: int | None = None
        self._failures = 0

    def last_stage(self) -> str:
        return f"panel_capacity_{self._stage}"

    def is_complete(self) -> bool:
        return self._complete

    def camera_history(self) -> tuple[tuple[int, int, int, int], ...]:
        # This planner never moves the camera, so it has nothing to hand on.
        return ()

    @property
    def capacity_readable(self) -> bool:
        return self._capacity_readable is True

    @property
    def cull_required(self) -> bool:
        return self._cull_required

    @property
    def last_capacity(self) -> int | None:
        return self._last_capacity

    def planning_detection_types(self) -> frozenset[str] | None:
        return frozenset({PANEL_TITLE})

    def on_action_success(self, target_type: str) -> None:
        return None

    def on_action_failure(self, target_type: str) -> None:
        return None

    def choose(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> Target | None:
        if self._complete:
            return None
        region = locate_panel_capacity(frame.image, self.panel_template)
        panel_open = region is not None
        if not panel_open:
            if self._stage == "close_panel":
                # The close tap landed; the reading is already banked.
                self._complete = True
                return None
            self._stage = "open_panel"
            return synthetic_target(
                PANEL_OPEN,
                *_scaled(frame, PANEL_OPEN_POINT, self.reference_width),
            )
        assert region is not None
        read = probe_dino_count(
            frame.image,
            self.reader,
            reference_width=self.reference_width,
            expected_capacity=self.capacity_limit,
            capacity_region=region,
        )
        if read.reason != "ok" or read.count is None:
            self._failures += 1
            if self.capacity_snapshots is not None:
                self.capacity_snapshots.capture(
                    frame,
                    read,
                    stage=self.last_stage(),
                    attempts=self._failures,
                )
            if self._failures < self.capacity_read_retries:
                self._stage = "reading"
                self.logger.warning(
                    "Hatch capacity | panel unreadable | retry=%d/%d | reason=%s",
                    self._failures,
                    self.capacity_read_retries,
                    read.reason,
                )
                return None
            self._capacity_readable = False
            self._stage = "close_panel"
            self.logger.error(
                "Hatch capacity | panel unreadable after %d reads | reason=%s",
                self._failures,
                read.reason,
            )
            return synthetic_target(
                PANEL_CLOSE,
                *_scaled(frame, PANEL_CLOSE_POINT, self.reference_width),
            )
        self._capacity_readable = True
        self._last_capacity = read.count
        self._cull_required = should_cull(read.count, self.threshold)
        self.logger.info(
            "Hatch capacity | panel=%d/%d | threshold=%d | cull=%s",
            read.count,
            self.capacity_limit,
            self.threshold,
            self._cull_required,
        )
        self._stage = "close_panel"
        return synthetic_target(
            PANEL_CLOSE,
            *_scaled(frame, PANEL_CLOSE_POINT, self.reference_width),
        )


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
        self.reference_width = reference_width
        self.safe_margin = max(0, safe_margin)
        self.bottom_exclusion_px = max(0, bottom_exclusion_px)
        self.selection_size = max(1, selection_size)
        self.allow_cull = bool(allow_cull)
        if capacity_limit <= 0:
            raise ValueError("capacity_limit must be greater than zero")
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")
        if threshold > capacity_limit:
            raise ValueError("threshold cannot exceed capacity_limit")
        self.capacity_limit = capacity_limit
        self.threshold = threshold
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
        self._applied_return_swipes: list[tuple[int, int, int, int]] = []
        self._pending_return_swipe: tuple[int, int, int, int] | None = None
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

    def camera_history(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(self._applied_return_swipes)

    def is_complete(self) -> bool:
        return self._complete

    def _read_capacity(self, frame: Frame) -> CapacityRead:
        # The My Dinosaurs panel carries the same figure on a white card that
        # nothing overlaps, so when it happens to be open it is strictly the
        # better source: the cave HUD prints its digits over the map, where s9
        # lost the slash to the terrain ("32110" for 321/370) and read nothing
        # at all over dark forest. Fall through to the HUD when the panel is
        # not up, which is the ordinary case.
        panel = probe_dino_count(
            frame.image,
            self.reader,
            reference_width=self.reference_width,
            expected_capacity=self.capacity_limit,
            capacity_region=PANEL_CAPACITY_REGION,
        )
        if panel.reason == "ok":
            return panel
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
            # A changed camera view needs fresh, consecutive HUD samples.
            self._capacity_failures = 0
            self._reset_capacity_confirmation()
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
            if self._pending_return_swipe is not None:
                self._applied_return_swipes.append(self._pending_return_swipe)
                self._pending_return_swipe = None
            self._return_swipes += 1
            self._stage = "recenter"

    def on_action_failure(self, target_type: str) -> None:
        if target_type == CAVE_SWIPE:
            self.navigator.on_swipe_result(moved=False)
            return
        self._pending_return_swipe = None
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
                return swipe_target(CAVE_SWIPE, x1, y1, x2, y2)
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
                if 0 < self._navigation_swipes < len(self.navigator.swipe_vectors):
                    # S9 can reveal the cave after only the vertical move,
                    # while a building still overlaps the fixed capacity HUD.
                    # The overlap can make the glyph reader return either
                    # unparsed text or a plausible but wrong denominator
                    # (312/370 became 312/8 in a live S9 trace). Complete
                    # only the remaining calibrated outbound gesture; never
                    # start a fresh path from an unknown cave view or repeat
                    # a successful swipe to try to manufacture a read.
                    step = self.navigator.next_step(
                        cave_visible=False, frame_width=frame.width
                    )
                    if step.kind == SWIPE:
                        assert step.vector is not None
                        self.logger.warning(
                            "Hatch cave | capacity unreadable after retries"
                            " | reason=%s | glyphs=%r"
                            " | completing remaining calibrated move"
                            " | swipe=%s",
                            read.reason,
                            read.text,
                            step.vector,
                        )
                        return swipe_target(CAVE_SWIPE, *step.vector)
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
            return detection_target(cave)
        if self._stage == "open_cave":
            cave = self._safe_cave(frame, by_type.get(CAVE))
            return detection_target(cave) if cave is not None else None
        if self._stage == "cave_screen":
            button = best_detection(by_type.get(CAVE_SELECT_BUTTON))
            return detection_target(button) if button is not None else None
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
                return detection_target(option)
            return synthetic_target(
                SELECT_TAG_HEADER,
                *_scaled(frame, (228.0, 204.0), self.reference_width),
            )
        if self._stage == "select_weakest":
            if SELECT_TITLE not in by_type:
                return None
            button = _near(
                by_type.get(SELECT_WEAKEST_BUTTON),
                _scaled(frame, SELECT_WEAKEST_BUTTON_POINT, self.reference_width),
                60,
            )
            return detection_target(button) if button is not None else None
        if self._stage == "confirm_selection":
            if SELECT_TITLE not in by_type:
                return None
            button = best_detection(by_type.get(SELECT_CHOOSE_BUTTON))
            return detection_target(button) if button is not None else None
        if self._stage == "start_battle":
            button = best_detection(by_type.get(CAVE_CONTINUOUS_BUTTON))
            return detection_target(button) if button is not None else None
        if self._stage == "battle_result":
            claim = best_detection(by_type.get(hatch_feature.CLAIM_BUTTON))
            return detection_target(claim) if claim is not None else None
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
                self._pending_return_swipe = (x1, y1, x2, y2)
                return swipe_target(CAVE_RECENTER, x1, y1, x2, y2)
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
        return best_detection(safe)


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
        allow_extreme_specialization_parent: bool = False,
        expel_below_hp: int = 0,
        expel_below_attack: int = 0,
        expel_dry_run: bool = True,
        auto_place_specializations: bool = False,
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
        enabled_stages: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.reader = reader
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        if capacity_limit <= 0:
            raise ValueError("capacity_limit must be greater than zero")
        if cull_threshold <= 0:
            raise ValueError("cull_threshold must be greater than zero")
        if cull_threshold > capacity_limit:
            raise ValueError("cull_threshold cannot exceed capacity_limit")
        self.capacity_limit = capacity_limit
        self.cull_threshold = cull_threshold
        self.stat_upgrade_guards = dict(stat_upgrade_guards)
        self.allow_extreme_specialization_parent = bool(
            allow_extreme_specialization_parent
        )
        self.auto_place_specializations = bool(auto_place_specializations)
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
        selected_stages = (
            DEFAULT_FULL_HATCH_STAGES
            if enabled_stages is None
            else frozenset(enabled_stages)
        )
        unknown_stages = selected_stages - FULL_HATCH_STAGES
        if unknown_stages:
            raise ValueError(
                "unsupported full hatch stages: " + ", ".join(sorted(unknown_stages))
            )
        if standalone_stage is None and "hatch" not in selected_stages:
            raise ValueError("full hatch stages must include hatch")
        self.enabled_stages = selected_stages
        self._screening_stages = tuple(
            stage for stage in SCREENING_STAGES if stage in selected_stages
        )
        self._collect_enabled = "collect" in selected_stages
        self._cave_enabled = "cave" in selected_stages
        self._custom_cycle = enabled_stages is not None and standalone_stage is None
        self._collect_before_hatch = False
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
            reader=reader,
            expel_below_hp=expel_below_hp,
            expel_below_attack=expel_below_attack,
            expel_dry_run=expel_dry_run,
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
        self._boost_visit: CooldownBoostVisit | None = None
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
        # A hatch tap that never reaches the claim screen can be the game's
        # maximum-population dialog.  Keep that signal separate from egg-pile
        # calibration so recovery always proves the configured cave capacity
        # before another hatch tap is allowed.
        self._hatch_capacity_check_pending = False
        self._egg_pile_capacity_rechecked = False
        self._egg_pile_retry_pending = False
        self._autoplace_without_no_button = False
        self._screening_blocked = False
        self._capacity_blocked = False
        # A combined Hatch+Hunt run may make one bounded map round-trip after
        # a failed capacity preflight. The hunt map recentres a HUD that can be
        # obscured by home-map scenery. Custom runs without culling can then
        # spend a retry interval hunting; other runs retain the capacity fuse.
        self._capacity_camera_refresh_used = False
        self.population_limit_reached = False
        self.capacity_retry_pending = False
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
        if self._boost_visit is not None:
            return f"cooldown_boost:{self._boost_visit.phase}"
        child_stage = getattr(self._child, "last_stage", None)
        detail = child_stage() if callable(child_stage) else child_stage
        return f"full_{self._stage}:{detail or '-'}"

    def is_complete(self) -> bool:
        # A standalone/full-only run has no hunt owner to fall back to. Stop
        # safely after calibration is blocked instead of spinning forever.
        return (
            self._complete
            or self._egg_pile_blocked
            or self._screening_blocked
            or self._capacity_blocked
        )

    def is_hatch_blocked(self) -> bool:
        """Whether calibration or a disabled safety stage blocks hatching."""

        return (
            self._egg_pile_blocked
            or self._screening_blocked
            or self._capacity_blocked
        )

    def begin_hunt_map_capacity_refresh(self) -> bool:
        """Let a combined workflow make one no-hunt camera refresh attempt.

        This is deliberately offered only after the ordinary cave preflight
        has exhausted its own calibrated moves and refused the HUD. A
        standalone hatch run has no verified hunt handoff owner, so it remains
        safely blocked instead of trying to leave the map itself.
        """

        if (
            self._capacity_camera_refresh_used
            or self.population_limit_reached
            or not self._capacity_blocked
            or self._stage != "capacity_blocked"
        ):
            return False
        self._capacity_camera_refresh_used = True
        self._capacity_blocked = False
        self._stage = "capacity_camera_refresh"
        self._no_target_since = None
        self.logger.warning(
            "Hatch capacity | unreadable after calibrated cave route; "
            "requesting one hunt-map camera refresh"
        )
        return True

    def complete_hunt_map_capacity_refresh(self) -> bool:
        """Restart the preflight only after the outer handoff proved home."""

        if self._stage != "capacity_camera_refresh":
            return False
        self.logger.info(
            "Hatch capacity | hunt-map camera refresh returned home; retrying preflight"
        )
        self._begin_capacity_preflight("after hunt-map camera refresh")
        return True

    def fail_hunt_map_capacity_refresh(self, reason: str) -> bool:
        """Consume the one refresh attempt and restore the conservative fuse."""

        if self._stage != "capacity_camera_refresh":
            return False
        self._capacity_blocked = True
        self._stage = "capacity_blocked"
        self._no_target_since = None
        self.logger.error(
            "Hatch capacity | hunt-map camera refresh failed; hatching remains blocked"
            " | reason=%s",
            reason,
        )
        return True

    def _population_limit_pending(self) -> bool:
        if (
            self._cave_enabled or not self._capacity_checked
            or self._cave_population is None or self._stage != "hatch"
        ):
            return False
        estimate = (
            self._cave_population + self._hatched_since_cave_read
            + self._hatch_child.hatched - self._hatch_baseline
        )
        return estimate >= self.cull_threshold

    def next_ready_delay_ms(self) -> int:
        if self._boost_visit is not None:
            return 0
        method = getattr(self._child, "next_ready_delay_ms", None)
        delay = int(method()) if callable(method) else 0
        return delay

    def boost_ready_delay_ms(self) -> int | None:
        if self.capacity_retry_pending:
            return None
        if self.boost_inventory is None or self.standalone_stage is not None:
            return None
        delay = self.boost_inventory.ready_delay_seconds()
        return None if delay is None else math.ceil(delay * 1000)

    def hunt_cooldown_delay_ms(self) -> int:
        # Same stale-flag hazard as `is_hunt_cooldown_active`: report "no
        # cooldown" rather than assert when a non-hatch child is installed.
        if not self._empty_rescan_wait:
            return 0
        if not isinstance(self._child, hatch_feature.HatchPlanner):
            return 0
        return self._child.next_ready_delay_ms()

    def defer_boost_visit(self, reason: str) -> None:
        if self.boost_inventory is not None:
            self.boost_inventory.defer(reason)

    def begin_boost_visit(self, *, keep_incubator_open: bool = False) -> bool:
        """Check boost eligibility on a proven home or incubator screen.

        Capacity/calibration fuses belong to hatching. This visit never clears
        them and never runs a hatch, collection, screening or cull action.
        """
        if self._boost_visit is not None:
            return True
        if self.boost_ready_delay_ms() != 0:
            return False
        assert self.boost_inventory is not None
        self._observed_cooldown_until = None
        self._boost_visit = CooldownBoostVisit(
            self.boost_inventory, clock=self.clock, logger=self.logger,
            keep_incubator_open=keep_incubator_open,
        )
        return True

    def boost_visit_active(self) -> bool:
        return self._boost_visit is not None

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
        if self._boost_visit is not None:
            return HATCH_DETECTION_TYPES | RECOVERY_DETECTION_TYPES
        if self._stage == "hatch":
            return HATCH_DETECTION_TYPES
        if self._stage == "open_nest":
            return NEST_BASE_DETECTION_TYPES
        if self._stage in {"attack", "hp"}:
            return (
                NEST_AUTOPLACE_DETECTION_TYPES
                if isinstance(self._child, AutoPlaceRoundPlanner)
                else NEST_REPLACEMENT_DETECTION_TYPES
            )
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

        # `_empty_rescan_wait` only describes a countdown a HatchPlanner owns.
        # Several stages park a non-hatch child (recovery, My Nest, cave) while
        # that flag is still set, so prove the child before reading it rather
        # than asserting on it: a stale flag must answer "no cooldown to hunt",
        # not crash the run.
        return (
            self._empty_rescan_wait
            and self._boost_visit is None
            and isinstance(self._child, hatch_feature.HatchPlanner)
            and self._child.next_ready_delay_ms() > 0
        )

    def begin_home_collection(self) -> bool:
        """Collect before hatching on a custom cycle's normal return home."""
        if (
            not self._custom_cycle or not self._collect_enabled
            or not self._capacity_checked or self.is_complete()
            or self._stage != "hatch"
        ):
            return False
        self._collect_before_hatch = True
        self._observed_cooldown_until = None
        self._enter_open_nest(collect_only=True)
        return True

    def begin_interim_collection(self) -> bool:
        """Spend a hunt idle window on one collect-only nest round.

        把剩餘冷卻釘進 observed deadline:收蛋結束後的等待會接續原本的
        倒數,而不是從設定值重新起算(`_start_empty_rescan_wait` 會在
        沒有 observed 值時退回整段設定時間)。
        """

        if (
            self.capacity_retry_pending or not self._collect_enabled
            or not self.is_hunt_cooldown_active()
        ):
            return False
        if self._observed_cooldown_until is None:
            self._observed_cooldown_until = (
                self.clock() + self.hunt_cooldown_delay_ms() / 1000
            )
        self.logger.info(
            "Hatch full | interim nest collection during hunt idle | resume=%.0fs",
            max(0.0, self._observed_cooldown_until - self.clock()),
        )
        self._enter_open_nest(collect_only=True)
        return True

    def abort_interim_collection(self, reason: str) -> bool:
        """Undo an errand that never reached the nest and resume the cooldown.

        `begin_interim_collection` clears the rescan wait so the nest round can
        run, which also makes `is_hunt_cooldown_active` false. A caller that
        gives up therefore cannot simply hand control back: the hunt side would
        read "cooldown over" and bounce straight into another handoff. Restart
        the wait from the pinned deadline instead, so the abandoned errand costs
        the remaining cooldown and nothing more.
        """

        if self._stage != "open_nest":
            return False
        self.logger.warning(
            "Hatch full | interim collection abandoned | %s",
            reason,
        )
        self._start_empty_rescan_wait()
        return True

    def begin_home_recovery(self, reason: str) -> bool:
        """Let a combined workflow hand a mis-centred home back to recovery.

        Recovery owns the measured drag that pulls the egg pile back to its
        reference position. A caller that can see the pile but cannot accept it
        as centred has found exactly that job, so route it here rather than
        leaving the home screen to look for a fix elsewhere.
        """

        self._begin_home_recovery(reason)
        return True

    def reset_workflow(self) -> None:
        self.population_limit_reached = False
        self.capacity_retry_pending = False
        self._collect_before_hatch = False
        if self._custom_cycle and not self._cave_enabled:
            self._capacity_checked = False
        if self._boost_visit is not None:
            self.defer_boost_visit("workflow interrupted; recheck game state")
            self._boost_visit = None
        self._egg_pile_failures = 0
        self._egg_pile_blocked = False
        self._egg_pile_capacity_check_pending = False
        self._hatch_capacity_check_pending = False
        self._egg_pile_capacity_rechecked = False
        self._egg_pile_retry_pending = False
        self._screening_blocked = False
        self._capacity_blocked = False
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
        if target_type in BOOST_ACTIONS:
            if self._boost_visit is not None:
                self._boost_visit.on_action_success(target_type)
            return
        if target_type == RECOVERY_NO:
            self.logger.warning(
                "Hatch full | cancelled unexpected auto-place confirmation"
                " | no auto-place round in progress"
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
            if target_type == hatch_feature.CLAIM_BUTTON:
                self._pending_claim_verification = False
            if target_type == hatch_feature.CLOSE_BUTTON:
                if self.standalone_stage == "hatch":
                    self._begin_home_recovery("standalone hatch completed")
                    return
                hatched = hatch.hatched - self._hatch_baseline
                if hatched > 0:
                    self._hatched_since_cave_read += hatched
                    self._hatch_baseline = hatch.hatched
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
                    management_trigger = capacity_trigger or (
                        growth_trigger and bool(self._screening_stages)
                    )
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
                    if capacity_trigger and not self._cave_enabled:
                        self._hatch_capacity_check_pending = True
                        self._begin_home_recovery("verify safe population before hunting only")
                        return
                    if management_trigger and not self._management_pending:
                        self._queue_management(
                            population=estimate,
                            cave_cleanup_after=capacity_trigger,
                        )
                    if management_trigger:
                        self._begin_queued_management()
                    elif self._custom_cycle and self._collect_done_for_cycle:
                        self._start_empty_rescan_wait()
                    elif self._collect_enabled:
                        self._collect_before_hatch = self._custom_cycle
                        self._enter_open_nest(collect_only=True)
                    else:
                        self._start_empty_rescan_wait()
                else:
                    if self._collect_done_for_cycle:
                        # 加速回訪的收尾:本週期已收過蛋,直接進入等待。
                        self._collect_done_for_cycle = False
                        self._start_empty_rescan_wait()
                        return
                    if self._collect_enabled:
                        self.logger.info(
                            "Hatch full | no ready incubator eggs | "
                            "collect all nest eggs before cooldown"
                        )
                        self._collect_before_hatch = self._custom_cycle
                        self._enter_open_nest(collect_only=True)
                    else:
                        self.logger.info(
                            "Hatch custom | collect stage disabled | entering cooldown"
                        )
                        self._start_empty_rescan_wait()
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
            if isinstance(self._child, AutoPlaceRoundPlanner):
                child = self._autoplace_child
                child.on_action_success(target_type)
                self._advance_autoplace_if_done()
            else:
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

    def on_action_success_context(
        self,
        target: Target,
        frame: Frame,
        detections: Sequence[Detection],
        result: VerificationResult,
    ) -> None:
        if target.type in BOOST_ACTIONS:
            self.on_action_success(target.type)
            return
        if self._stage in ("attack", "hp"):
            if isinstance(self._child, AutoPlaceRoundPlanner):
                self.on_action_success(target.type)
            else:
                child = self._replacement_child
                child.on_action_success_context(target, frame, detections, result)
                self._advance_replacement_if_done()
            return
        self.on_action_success(target.type)

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
        if target.type in BOOST_ACTIONS:
            if self._boost_visit is not None:
                self._boost_visit.on_action_failure(target.type)
            return True
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
        if target_type in BOOST_ACTIONS:
            if self._boost_visit is not None:
                self._boost_visit.on_action_failure(target_type)
            return
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
            if target_type == hatch_feature.HATCH_BUTTON:
                self._capacity_checked = False
                self._hatch_capacity_check_pending = True
                self.logger.warning(
                    "Hatch capacity | hatch tap had no verified claim"
                    " | checking configured capacity before retry"
                )
                self._begin_home_recovery(
                    "hatch tap failed; capacity recheck required"
                )
                return
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
        if self.is_complete() and not self.boost_visit_active():
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
            interruption = best_detection(by_type.get(target_type))
            if interruption is not None:
                self._no_target_since = None
                return detection_target(interruption)
        # Stop before another egg or boost when verified claims reach the
        # configured stop line. Re-read the HUD and recenter before hunting.
        if self._population_limit_pending():
            self._hatch_capacity_check_pending = True
            self._begin_home_recovery("verify safe population before further hatching")
            return self.choose(frame, detections)
        # Check boosts during a normal incubator visit, after ready eggs.
        # Keep the panel open so the child can finish its scan and read timers.
        # Only this check owns a confirmation raised by its boost-button action.
        if self._boost_visit is None and self.boost_ready_delay_ms() == 0:
            safe_panel = (
                self._stage == "hatch"
                and hatch_feature.INCUBATOR_TITLE in by_type
                and hatch_feature.CLOSE_BUTTON in by_type
                and not any(key in by_type for key in (
                    CONFIRM_YES, CONFIRM_NO, hatch_feature.CLAIM_BUTTON,
                    hatch_feature.EXPEL_BUTTON, hatch_feature.HATCH_LABEL,
                    hatch_feature.HATCH_BUTTON,
                ))
                and _unready_egg_detail_close(frame) is None
            )
            if safe_panel:
                self.begin_boost_visit(keep_incubator_open=True)
        if self._boost_visit is not None:
            visit = self._boost_visit
            target = visit.choose(
                frame, detections,
                home_centered=is_centered_home_screen(frame, detections),
                home_point=_egg_pile_safe_tap(frame),
                button_ready=_hatch_boost_ready(frame, reference_width=self.reference_width),
                button_point=_hatch_boost_point(frame, reference_width=self.reference_width),
            )
            if visit.used and self._stage == "hatch":
                self._observe_hatch_cooldown(frame, by_type)
            if not visit.complete:
                return target
            self._boost_visit = None
            self._no_target_since = None
            self.logger.info("Cooldown boost | visit complete | resume=%s", self._stage)
            if self.is_complete():
                # Returning from an independent visit must not unlock or
                # restart an unsafe hatch workflow, even when the visit failed.
                return None
            if visit.failed:
                self._begin_home_recovery("cooldown boost visit could not return home")
            elif (
                visit.used
                and self._empty_rescan_wait
                and isinstance(self._child, hatch_feature.HatchPlanner)
            ):
                # The boost changed the egg timer; discard the old wait if no
                # new timer could be read, so the next hatch pass rechecks it.
                # A boost can also start from a non-hatch panel, so the child
                # only owns a rescan wait when it is really a HatchPlanner.
                remaining = max(
                    0.0, (self._observed_cooldown_until or self.clock()) - self.clock()
                )
                self._child.begin_rescan_wait(
                    "cooldown boost updated egg timer", seconds=remaining
                )
                self._observed_cooldown_until = None
        # Same rule one step further: the parent-replacement confirmation also
        # carries cyan Yes and red No buttons, so the broad auto-place layout
        # matches it too. Its own exact templates prove which dialog is really
        # in front, and cancelling one aborts the swap the screening stage just
        # asked for - then the stage restarts and asks again, forever.
        replacing_parent = (
            SELECT_CONFIRM_PROMPT in by_type or NESTED_PARENT_WARNING in by_type
        )
        # Who is allowed to raise this confirmation is a property of the running
        # child planner, not of the stage name. Top/mass always auto-place, but
        # with auto_place_specializations the attack/hp stages run the very same
        # AutoPlaceRoundPlanner, and it is sitting in after_autoplace waiting to
        # tap Yes. Keying this guard on the stage list cancelled that dialog,
        # which aborted the round, recovered home, and restarted the identical
        # stage - a ~25s loop that never completed a single screening pass.
        own_autoplace = isinstance(self._child, AutoPlaceRoundPlanner)
        if (
            not own_autoplace
            and not replacing_parent
            and (AUTOPLACE_PROMPT in by_type or AUTOPLACE_NOTICE in by_type)
        ):
            no = best_detection(by_type.get(CONFIRM_NO))
            if no is None:
                # No No button is evidence against the reading, not a dead end.
                # The layout detector is deliberately broad and a dinosaur
                # detail card satisfies it - white panel, red action button -
                # so returning None here parked the run on one frame it could
                # not act on: 109 identical errors and zero actions in six
                # minutes, invisible to the retry and stall counters because
                # no action was ever attempted. Fall through and let the rules
                # that know the other screens have their turn.
                if not self._autoplace_without_no_button:
                    self._autoplace_without_no_button = True
                    self.logger.warning(
                        "Hatch full | auto-place layout without a No button;"
                        " treating it as a misread and continuing"
                    )
            else:
                self._autoplace_without_no_button = False
                return synthetic_target(RECOVERY_NO, no.x, no.y)
        else:
            self._autoplace_without_no_button = False
        hatch_result_visible = bool(
            by_type.get(hatch_feature.CLAIM_BUTTON)
            or by_type.get(hatch_feature.EXPEL_BUTTON)
        )
        auto_battle = (
            None
            if hatch_result_visible
            else best_detection(by_type.get(STARTUP_AUTO_BATTLE_CLOSE))
        )
        if auto_battle is not None:
            self._no_target_since = None
            return detection_target(auto_battle)
        if (
            not hatch_result_visible
            # The My Nest card is also a tall white panel with a centred
            # green button.  Its auto-place control therefore satisfies the
            # launch growth-result colour heuristic.  Never use the startup
            # shortcut while the nest title proves that this is already the
            # nest screen, or the synthetic tap lands on auto-place.
            and NEST_TITLE not in by_type
            # Recovery owns the screen once it starts.  This branch taps the
            # shortcut *inside* the card to reach the nest; recovery wants the
            # card gone so it can measure the egg pile.  Letting this run
            # first made the recovery rung for the same card unreachable, and
            # on 2026-08-31 S9 re-tapped the shortcut three times, exhausted
            # the Back ladder and dropped the hatch workflow for the session.
            and self._stage != "recover_home"
            # A real startup result card darkens the map.  Conversely, the
            # bright home map is independently proven by its anchor and side
            # strips, and its egg pile can satisfy this broad colour layout
            # detector.  Never open My Nest from that false match.
            and not is_home_screen(frame, detections)
            and (growth_result := best_detection(by_type.get(STARTUP_GROWTH_RESULT))) is not None
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
            return synthetic_target(
                STARTUP_NEST_SHORTCUT,
                *_scaled(frame, shortcut_point, self.reference_width),
            )

        # No egg-pile tap is due while waiting. A shifted home frame must not
        # discard the countdown and start recovery before Hunt can take over.
        # Foreground panels/startup overlays still follow their own handling.
        if self.is_hunt_cooldown_active() and is_home_screen(frame, detections):
            return None

        # The redesigned incubator's real bottom-right close button also
        # matches the hunt-dialog close template (S9: 0.998 hatch vs 0.910
        # hunt). The exact incubator title is stronger screen identity, so it
        # must outrank that generic hunt-map interruption. Without this guard
        # the planner closes the incubator, returns home, and reopens it every
        # ten seconds without ever hatching an egg.
        if (
            self._stage == "hatch"
            and hatch_feature.INCUBATOR_TITLE not in by_type
            and any(item.type in HUNT_ACTIVE_TYPES for item in detections)
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
                # A calibration failure blocks unsafe hatch taps, but the
                # combined hatch+hunt owner can continue useful hunting.  The
                # old completion flag stopped the entire Bot after one bad
                # recovery even though HatchHuntPlanner already has a safe
                # blocked-hatch fallback.
                self._egg_pile_blocked = True
                self._stage = "hatch_blocked"
                self._no_target_since = None
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
                self.logger.info(
                    "Hatch full | centered home confirmed | recovered=%s",
                    self._recovery_reason or "unknown",
                )
                if (
                    self._egg_pile_capacity_check_pending
                    or self._hatch_capacity_check_pending
                ):
                    reason = (
                        "hatch tap had no verified claim"
                        if self._hatch_capacity_check_pending
                        else "egg pile tap had no response"
                    )
                    self._begin_capacity_preflight(
                        reason
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
        if self.is_complete():
            return None
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
            if self.capacity_retry_pending:
                if self.is_hunt_cooldown_active():
                    return None
                if is_centered_home_screen(frame, detections):
                    self._begin_capacity_preflight("scheduled capacity recheck")
                    return self._choose_current(frame, detections)
                self._hatch_capacity_check_pending = True
                self._begin_home_recovery("return home before scheduled capacity recheck")
                return None
            if _home_pile_click_blocked(frame, detections):
                if not self._capacity_checked:
                    # The pile is temporarily in the chat band. Capacity
                    # preflight only swipes the map, so it is safe to proceed
                    # without issuing a pile tap first.
                    self._begin_capacity_preflight(
                        "home pile is in bottom chat band; swipe-only preflight"
                    )
                    return self._choose_current(frame, detections)
                self._begin_home_recovery(
                    "home pile click would land in bottom chat band"
                )
                return self._choose_current(frame, detections)
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
                return synthetic_target(HATCH_DETAIL_CLOSE, *detail_close)
            target = self._hatch_child.choose(frame, detections)
            if target is not None and target.type == hatch_feature.EGG_PILE:
                if not self._capacity_checked:
                    self._begin_capacity_preflight(
                        "before first incubator access"
                    )
                    return self._choose_current(frame, detections)
                safe_point = _egg_pile_safe_tap(frame)
                if safe_point is not None:
                    return synthetic_target(hatch_feature.EGG_PILE, *safe_point)
                # The child retains its fixed point for the lightweight hatch
                # mode, but the full workflow must never tap it blindly: a
                # missed pile can open a roaming dinosaur and corrupt the
                # management state. Let the outer timeout enter bounded home
                # recovery instead.
                return None
            return target
        if self._stage == "capacity_preflight":
            target = self._capacity_child.choose(frame, detections)
            self._remember_capacity_preflight_result()
            if target is None and self._capacity_child.is_complete():
                if not self._capacity_child.capacity_readable:
                    self._egg_pile_capacity_check_pending = False
                    self._hatch_capacity_check_pending = False
                    self._capacity_checked = False
                    if (
                        self._custom_cycle and not self._cave_enabled
                        and self._capacity_camera_refresh_used
                    ):
                        self._start_capacity_retry_wait()
                        return None
                    self.logger.error(
                        "Hatch capacity | preflight failed; configured capacity unreadable; "
                        "hatching stopped safely"
                    )
                    self._capacity_blocked = True
                    self._stage = "capacity_blocked"
                    return None
                self.capacity_retry_pending = False
                if self._capacity_child.cull_required:
                    self._egg_pile_capacity_check_pending = False
                    self._hatch_capacity_check_pending = False
                    self._egg_pile_capacity_rechecked = False
                    self._egg_pile_retry_pending = False
                    if not self._cave_enabled:
                        self._capacity_blocked = True
                        self.population_limit_reached = True
                        self._stage = "capacity_blocked"
                        self.logger.info(
                            "Hatch custom | safe population reached | capacity=%d/%d"
                            " | hatching stopped; hunt remains active",
                            self._cave_population, self.capacity_limit,
                        )
                        return None
                    if self._capacity_blocked:
                        return None
                    self._begin_queued_management()
                    return self._choose_current(frame, detections)
                self._capacity_checked = True
                if self._screening_baseline_population is None:
                    self.logger.info(
                        "Hatch capacity | starting initial selected management"
                        " | capacity=%d/%d | stages=%s",
                        self._cave_population,
                        self.capacity_limit,
                        ",".join(
                            stage
                            for stage in (*self._screening_stages, "collect")
                            if stage in self.enabled_stages
                        )
                        or "none",
                    )
                    self._queue_management(
                        population=self._cave_population,
                        cave_cleanup_after=False,
                    )
                    self._begin_queued_management()
                    return self._choose_current(frame, detections)
                if self._egg_pile_capacity_check_pending:
                    self._egg_pile_capacity_check_pending = False
                    self._egg_pile_capacity_rechecked = True
                self._hatch_capacity_check_pending = False
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
            anchor = best_detection(by_type.get(hatch_feature.HOME_ANCHOR))
            if anchor is None:
                return None
            # The S9 entrance artwork changes slightly with the nest state.
            # Its live 2026-08-16 variant scores 0.833 against the canonical
            # image, so the manifest accepts it at 0.82.  A lower visual
            # threshold must not turn into permission to tap a matching egg
            # behind a dimmed item/detail overlay: require the independently
            # measured, bright and centred home map before using the match.
            if not is_centered_home_screen(frame, detections):
                return None
            return synthetic_target(OPEN_NEST, anchor.x, anchor.y)
        if self._stage in ("attack", "hp"):
            if isinstance(self._child, AutoPlaceRoundPlanner):
                target = self._autoplace_child.choose(frame, detections)
                if target is None:
                    self._advance_autoplace_if_done()
                    if not isinstance(self._child, AutoPlaceRoundPlanner):
                        return self._choose_current(frame, detections)
            else:
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
            button = best_detection(by_type.get(COLLECT_EGGS_BUTTON))
            return detection_target(button) if button is not None else None
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
                    if self._collect_before_hatch:
                        self._collect_before_hatch = False
                        self._collect_only_after_empty = False
                        self._observed_cooldown_until = None
                        self._stage = "hatch"
                        self._child = self._new_hatch()
                        self._hatch_baseline = 0
                        self._start_hatch_cycle()
                        self._collect_done_for_cycle = True
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
                if not self._cave_child.capacity_readable:
                    self._capacity_checked = False
                    self._capacity_blocked = True
                    self._stage = "capacity_blocked"
                    self._no_target_since = None
                    self.logger.error(
                        "Hatch full | cleanup capacity unreadable"
                        " | blocking hatch instead of restarting Phase A"
                    )
                    return None
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
        self._collect_before_hatch = False
        if self._custom_cycle and not self._cave_enabled:
            # The replacement child loses any pending hatch count, so prove
            # population again before allowing more hatches after recovery.
            self._capacity_checked = False
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
            # budget for the bounded same-map recovery episode.
            self._recovery_started_at = self.clock()
        history = (
            self._child.camera_history()
            if isinstance(self._child, (CaveCullPlanner, HatchHomeRecoveryPlanner))
            else ()
        )
        self._stage = "recover_home"
        self._child = HatchHomeRecoveryPlanner(
            reference_width=self.reference_width,
            logger=self.logger,
            applied_swipes=history,
        )
        # Recovery replaces the HatchPlanner that owns the rescan countdown, so
        # the cooldown-hunting window has to close with it.  Leaving the flag
        # set let `is_hunt_cooldown_active` reach for `_hatch_child` while a
        # HatchHomeRecoveryPlanner was installed, which asserted and killed the
        # whole run mid-recovery.
        self._empty_rescan_wait = False
        self._no_target_since = None
        self._recovery_reason = reason

    def _begin_capacity_preflight(self, reason: str) -> None:
        """Check/cull dinosaur capacity before the first hatch interaction."""

        self.logger.info(
            "Hatch capacity | starting preflight | reason=%s",
            reason,
        )
        self._stage = "capacity_preflight"
        # Preflight only reads; it must never delete dinosaurs before the
        # required screening stages have run. The panel is a pure measurement
        # and needs no camera position, which is what the cave route kept
        # failing to reach.
        self._child = PanelCapacityPlanner(
            self.reader,
            threshold=self.cull_threshold,
            capacity_limit=self.capacity_limit,
            reference_width=self.reference_width,
            capacity_read_retries=self.capacity_read_retries,
            capacity_snapshots=self.capacity_snapshots,
            logger=self.logger,
        )
        self._no_target_since = None

    def _start_cave_cleanup(self) -> None:
        """Enter the guarded cave cleanup stage after collection is complete."""

        if not self._cave_enabled:
            self._capacity_blocked = True
            self.logger.error(
                "Hatch custom | cave cleanup required but cave stage is disabled"
                " | switching to hunt"
            )
            return
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

    def _begin_queued_management(self) -> None:
        """Start only the selected safe management stages for this cycle."""

        if self._screening_stages or self._collect_enabled:
            self._enter_open_nest(collect_only=False)
            return
        if self._cave_cleanup_after_management:
            self._start_cave_cleanup()
            return
        self._finish_management_after_collection()

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

        self._collect_done_for_cycle = self._custom_cycle and self._collect_enabled

    def _remember_capacity_preflight_result(self) -> None:
        """Persist a valid capacity read before cave recentering can recover."""

        child = self._capacity_child
        reading = child.last_capacity
        if not child.capacity_readable or reading is None:
            return
        self._cave_population = reading
        self._hatched_since_cave_read = 0
        if child.cull_required:
            if not self._cave_enabled:
                # The capacity child must finish returning home before the
                # blocked flag permits the combined planner to start hunting.
                return
            if not self._management_pending:
                self.logger.warning(
                    "Hatch capacity | cleanup blocked until screening completes"
                    " | required=%s",
                    self._format_screening_stages(self._screening_stages),
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
        """Reset hatch-only bookkeeping; the boost schedule is independent."""
        self._collect_done_for_cycle = False

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

    def _start_capacity_retry_wait(self) -> None:
        """Pause hatching after a failed read and proven return home.

        Reuse the normal hunting/handoff window, but keep collection and boost
        visits disabled until a fresh capacity preflight succeeds. Never use
        the old estimate to permit another hatch.
        """
        seconds = max(60.0, float(self._hatch_kwargs["rescan_interval_seconds"]))
        self.capacity_retry_pending = True
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._hatch_baseline = 0
        self._start_hatch_cycle()
        self._collect_only_after_empty = False
        self._collect_before_hatch = False
        self._empty_rescan_wait = True
        self._observed_cooldown_until = None
        self._hatch_child.begin_rescan_wait(
            "capacity unreadable; recheck before hatching", seconds=seconds,
        )
        self.logger.warning(
            "Hatch capacity | unreadable after repositioning; hunt before recheck"
            " | retry_in=%.0fs", seconds,
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
        if any(key in by_type for key in (
            CONFIRM_YES, CONFIRM_NO, hatch_feature.CLAIM_BUTTON,
            hatch_feature.EXPEL_BUTTON, hatch_feature.HATCH_BUTTON,
        )):
            return
        seconds = hatch_feature.read_hatch_cooldown_seconds(
            frame.image,
            self.reader,
            reference_width=self.reference_width,
        )
        if seconds is not None:
            self._observed_cooldown_until = self.clock() + seconds

    def _start_replacement(self, kind: str) -> None:
        if self.auto_place_specializations:
            self._stage = kind
            rule = ATTACK_AUTOPLACE_RULE if kind == "attack" else HP_AUTOPLACE_RULE
            self._child = AutoPlaceRoundPlanner(
                rule,
                reference_width=self.reference_width,
                logger=self.logger,
            )
            return
        self._start_manual_replacement(kind)

    def _start_manual_replacement(self, kind: str) -> None:
        if kind == "attack":
            self._stage = "attack"
            self._child = AttackReplacementTestPlanner(
                self.reader,
                reference_width=self.reference_width,
                rule=ATTACK_RULE,
                stat_guards=self.stat_upgrade_guards,
                allow_extreme_specialization_parent=self.allow_extreme_specialization_parent,
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
            allow_extreme_specialization_parent=self.allow_extreme_specialization_parent,
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
            if self._collect_enabled:
                self._start_collect()
            else:
                self.logger.info("Hatch custom | collect stage disabled | skipping")
                self._finish_management_after_collection()
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
            stage for stage in self._screening_stages
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
    def _capacity_child(self) -> CaveCullPlanner | PanelCapacityPlanner:
        # Preflight reads through the panel; the cave planner still owns the
        # reads that accompany an actual cull.
        assert isinstance(self._child, (CaveCullPlanner, PanelCapacityPlanner))
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


def nest_mask_close_target(frame: Frame, reference_width: float) -> Target:
    """The mask tap that dismisses the My Nest panel back to the home screen.

    Shared with the hunting side: a hunt that finds this panel open has no
    vocabulary for it and would otherwise tap map controls it cannot reach.
    """

    return synthetic_target(
        NEST_MASK_CLOSE,
        *_scaled(frame, NEST_MASK_POINT, reference_width),
    )
