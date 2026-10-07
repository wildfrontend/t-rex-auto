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
from .hatch_result import HatchVerdict, judge_newborn, read_hatch_result_stats
from .models import Detection, Frame, Image, Target
from .nest_readout import Stats
from .targeting import best_detection, detection_target, swipe_target, synthetic_target

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
# Incubator v3 hatches every ready egg at once: the header's 全部孵化 button
# opens one 孵化結果 panel whose 領取全部 claims the whole batch. The header
# button keeps its shape when nothing is ready and only turns gray, so it is
# located by colour (synthetic target) rather than by template.
HATCH_ALL_BUTTON = "hatch_all_button"
RESULT_TITLE = "hatch_result_title"
CLAIM_ALL_BUTTON = "hatch_claim_all_button"
# The fixed bottom-centre home HUD button (全部孵化 + ready-egg count). It
# opens the incubator straight into the same 孵化結果 panel, so hatching no
# longer depends on centring the map and finding the pile. Matched by its
# 全部孵化 text, so the green 自動放置 control can never be mistaken for it.
HOME_HATCH_ALL_BUTTON = "hatch_home_hatch_all_button"

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    EGG_PILE: "tap",
    HATCH_LABEL: "tap",
    HATCH_BUTTON: "tap",
    CLAIM_BUTTON: "tap",
    EXPEL_BUTTON: "tap",
    CLOSE_BUTTON: "tap",
    SCROLL: "swipe",
    HATCH_ALL_BUTTON: "tap",
    CLAIM_ALL_BUTTON: "tap",
    HOME_HATCH_ALL_BUTTON: "tap",
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
    # The cracking-egg animation takes about 1.5s before the result panel.
    HATCH_ALL_BUTTON: 8000,
    CLAIM_ALL_BUTTON: 4000,
    HOME_HATCH_ALL_BUTTON: 8000,
}

# What must be visible after each tap for it to count as done. Claiming may
# land on the next egg's detail page or back on the grid, so it accepts both.
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    EGG_PILE: (INCUBATOR_TITLE,),
    HATCH_LABEL: (HATCH_BUTTON,),
    HATCH_BUTTON: (CLAIM_BUTTON,),
    CLAIM_BUTTON: (HATCH_BUTTON, INCUBATOR_TITLE, HATCH_LABEL, CLAIM_BUTTON),
    CLOSE_BUTTON: (HOME_ANCHOR,),
    HATCH_ALL_BUTTON: (CLAIM_ALL_BUTTON, RESULT_TITLE),
    HOME_HATCH_ALL_BUTTON: (CLAIM_ALL_BUTTON, RESULT_TITLE),
    CLAIM_ALL_BUTTON: (INCUBATOR_TITLE,),
}

# One successfully claimed dinosaur is one hatch workflow cycle. This stays
# feature-local because the shared config's cycle target normally belongs to
# hunt (mail_reward_collect_button).
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (CLAIM_BUTTON, CLAIM_ALL_BUTTON)

# Timer text positions in the original 900x1600 incubator grid. The visible
# grid has three columns and three rows; each timer is read without the clock
# icon or progress bar. The reader accepts both ``HHMMSS`` (the colon dots are
# too small to survive segmentation) and ``HH?MM?SS``.
HATCH_TIMER_REGIONS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x0, y0, x1, y1)
    for y0, y1 in ((608.0, 638.0), (873.0, 903.0), (1138.0, 1168.0))
    for x0, x1 in ((200.0, 350.0), (380.0, 530.0), (560.0, 710.0))
)

# Incubator v2 adds a permanent egg-speed bar above the ticket boost. That
# leaves the same three-column egg grid in place but compacts it upward by
# roughly forty pixels. The orange permanent-speed bar is a stable layout
# marker and prevents us from trying both coordinate sets over arbitrary egg
# artwork (DigitReader deliberately cannot reject every non-digit glyph).
HATCH_TIMER_REGIONS_V2: tuple[tuple[float, float, float, float], ...] = tuple(
    (x0, y0, x1, y1)
    for y0, y1 in ((565.0, 595.0), (833.0, 863.0), (1100.0, 1130.0))
    for x0, x1 in ((200.0, 350.0), (380.0, 530.0), (560.0, 710.0))
)
HATCH_V2_PERMANENT_BOOST_SAMPLE = (380.0, 1280.0, 520.0, 1340.0)
HATCH_V2_PERMANENT_BOOST_MIN_SATURATION = 80.0

# Incubator v3 drops the permanent egg-speed bar for a row of small icons and
# adds an 升級 / count / 全部孵化 header under the title. The egg grid keeps
# the v2 position (verified: the v2 timer regions read all nine timers), while
# the ticket boost bar moves up into the space the orange bar used. The orange
# 升級 button is the layout marker; it must be checked before the v2 marker,
# because a ready v3 ticket bar is saturated inside the v2 sample.
HATCH_V3_UPGRADE_SAMPLE = (195.0, 232.0, 315.0, 278.0)
HATCH_V3_HATCH_ALL_SAMPLE = (585.0, 232.0, 705.0, 278.0)
HATCH_V3_HATCH_ALL_POINT = (645.0, 255.0)
# Live 900x1600 samples: 升級 H20/S213/V242; active 全部孵化 H85/S106/V202;
# the gray (nothing ready) button S0; both dim to V<30 under the cutscene.
HATCH_V3_MIN_SATURATION = 60.0
HATCH_V3_MIN_VALUE = 150.0
# The 孵化結果 panel prints the batch size as 孵化的恐龍 N.
HATCH_RESULT_COUNT_REGION = (400.0, 322.0, 520.0, 360.0)
# The v3 header prints eggs-in-incubator / slots (e.g. 9/24); the egg count
# bounds how many dinosaurs one 全部孵化 can add.
HATCH_V3_EGG_COUNT_REGION = (375.0, 225.0, 525.0, 285.0)


def _sample_hsv(
    image: Image,
    sample: tuple[float, float, float, float],
    reference_width: float,
) -> tuple[float, float, float] | None:
    if image.size == 0 or image.ndim < 3 or image.shape[1] <= 0 or reference_width <= 0:
        return None
    scale = image.shape[1] / reference_width
    x0, y0, x1, y1 = (round(value * scale) for value in sample)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return (
        float(hsv[:, :, 0].mean()),
        float(hsv[:, :, 1].mean()),
        float(hsv[:, :, 2].mean()),
    )


def uses_header_layout(image: Image, *, reference_width: float = 900.0) -> bool:
    """Return whether the v3 incubator header (升級 / 全部孵化) is visible."""

    upgrade = _sample_hsv(image, HATCH_V3_UPGRADE_SAMPLE, reference_width)
    if upgrade is None:
        return False
    hue, saturation, value = upgrade
    return (
        8.0 <= hue <= 30.0
        and saturation >= HATCH_V3_MIN_SATURATION
        and value >= HATCH_V3_MIN_VALUE
    ) or hatch_all_ready(image, reference_width=reference_width)


def hatch_all_ready(image: Image, *, reference_width: float = 900.0) -> bool:
    """Return whether the v3 全部孵化 button is lit (ready eggs exist)."""

    sample = _sample_hsv(image, HATCH_V3_HATCH_ALL_SAMPLE, reference_width)
    if sample is None:
        return False
    hue, saturation, value = sample
    return (
        70.0 <= hue <= 100.0
        and saturation >= HATCH_V3_MIN_SATURATION
        and value >= HATCH_V3_MIN_VALUE
    )


def hatch_all_point(frame: Frame, *, reference_width: float = 900.0) -> tuple[int, int]:
    scale = frame.width / reference_width
    return (
        round(HATCH_V3_HATCH_ALL_POINT[0] * scale),
        round(HATCH_V3_HATCH_ALL_POINT[1] * scale),
    )


def read_incubator_egg_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
) -> int | None:
    """Read the eggs-in-incubator count from the v3 header, or ``None``."""

    if image.ndim < 2 or image.shape[1] <= 0 or reference_width <= 0:
        return None
    scale = image.shape[1] / reference_width
    x0, y0, x1, y1 = (round(value * scale) for value in HATCH_V3_EGG_COUNT_REGION)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    eggs, sep, slots = reader.read(crop).partition("/")
    if not sep or not eggs.isdigit() or not slots.isdigit():
        return None
    count, total = int(eggs), int(slots)
    return count if 0 <= count <= total else None


# White ready-egg count on the HUD button's dark-blue badge (S16: "22"),
# relative to the matched 全部孵化 text: the button moves right when the green
# 自動放置 button is shown beside it.
HOME_HATCH_ALL_COUNT_OFFSET = (-10.0, -50.0, 50.0, -18.0)
HOME_HATCH_ALL_TEXT_CENTER = (450.0, 1418.0)


def read_home_hatch_all_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
    text_center: tuple[float, float] | None = None,
) -> int | None:
    """Read the ready-egg count on the home 全部孵化 button, or ``None``.

    ``text_center`` is the matched 全部孵化 text in frame pixels; it defaults
    to the button's centred position.
    """

    if image.ndim < 3 or image.shape[1] <= 0 or reference_width <= 0:
        return None
    scale = image.shape[1] / reference_width
    cx, cy = (
        text_center
        if text_center is not None
        else (
            HOME_HATCH_ALL_TEXT_CENTER[0] * scale,
            HOME_HATCH_ALL_TEXT_CENTER[1] * scale,
        )
    )
    dx0, dy0, dx1, dy1 = HOME_HATCH_ALL_COUNT_OFFSET
    x0, y0, x1, y1 = (
        round(cx + dx0 * scale),
        round(cy + dy0 * scale),
        round(cx + dx1 * scale),
        round(cy + dy1 * scale),
    )
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # The reader expects dark glyphs on a light ground.
    glyphs = np.where(gray > 200, 0, 255).astype(np.uint8)
    text = reader.read(cv2.cvtColor(glyphs, cv2.COLOR_GRAY2BGR)).replace("?", "")
    if not text.isdigit() or len(text) > 3:
        return None
    count = int(text)
    return count if count > 0 else None


def read_hatch_result_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
) -> int | None:
    """Read the batch size on the 孵化結果 panel, or ``None``."""

    if image.ndim < 2 or image.shape[1] <= 0 or reference_width <= 0:
        return None
    scale = image.shape[1] / reference_width
    x0, y0, x1, y1 = (round(value * scale) for value in HATCH_RESULT_COUNT_REGION)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    text = reader.read(crop).replace("?", "")
    if not text.isdigit() or len(text) > 3:
        return None
    count = int(text)
    return count if count > 0 else None


def uses_permanent_boost_layout(
    image: Image,
    *,
    reference_width: float = 900.0,
) -> bool:
    """Return whether the v2 permanent egg-speed bar is visible."""

    if image.size == 0 or image.ndim < 2 or image.shape[1] <= 0 or reference_width <= 0:
        return False
    scale = image.shape[1] / reference_width
    x0, y0, x1, y1 = (
        round(value * scale) for value in HATCH_V2_PERMANENT_BOOST_SAMPLE
    )
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= HATCH_V2_PERMANENT_BOOST_MIN_SATURATION

# The home egg pile is placed in the lower half of the map, but cave/recovery
# gestures can leave it hundreds of pixels above its calibrated position. Its
# artwork also changes between accounts, so this locator measures the broad
# dark/coloured structure instead of depending on a particular image or hue.
HOME_PILE_STRUCTURE_REGION: tuple[float, float, float, float] = (
    250.0,
    800.0,
    650.0,
    1600.0,
)
HOME_PILE_MIN_STRUCTURE_AREA = 1_000.0
HOME_PILE_MIN_STRUCTURE_WIDTH = 150.0
# A roaming dinosaur can touch one side of the pile in a captured frame and
# widen the connected component by roughly 100px.  Keep enough room for that
# observed 352px union while still rejecting a component spanning almost the
# entire 400px search strip.
HOME_PILE_MAX_STRUCTURE_WIDTH = 370.0
HOME_PILE_MIN_STRUCTURE_ROWS = 4
HOME_PILE_MIN_STRUCTURE_ROW_WIDTH = 110
HOME_PILE_TAP_OFFSET_PX = 100
# On the 900x1600 game viewport the chat bar starts around y=1540. Keep a
# small 20px margin above it: the lower map can still be used for swipes, but
# no hatch tap may land in the chat bar.
HOME_PILE_TAP_BOTTOM_EXCLUSION_PX = 80


def home_pile_structure_base(
    frame: Frame,
    *,
    egg_pile_point: tuple[float, float],
    reference_width: float = 900.0,
) -> tuple[float, float] | None:
    """Locate a style-neutral home pile and return its bottom-centre anchor.

    The returned Y coordinate is the structure's bottom edge.  On the shifted
    blue-stone S13 pile this remains the same map anchor as the cyan centroid,
    lava inset, and straw-template centre used by the full hatch workflow.
    Requiring a broad component with several wide rows rejects roaming
    dinosaurs and the small fixed incubator nests around it.
    """

    if frame.image.size == 0 or frame.width <= 0 or reference_width <= 0:
        return None
    scale = frame.width / reference_width
    rx0, ry0, rx1, ry1 = (
        round(value * scale) for value in HOME_PILE_STRUCTURE_REGION
    )
    x0 = max(rx0, round((egg_pile_point[0] - 220.0) * scale))
    y0 = max(0, ry0)
    x1 = min(rx1, round((egg_pile_point[0] + 220.0) * scale))
    y1 = min(frame.height, ry1)
    roi = frame.image[y0:y1, x0:x1]
    if roi.size == 0:
        return None

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
    max_width = HOME_PILE_MAX_STRUCTURE_WIDTH * scale
    max_area = roi.shape[0] * roi.shape[1] * 0.70
    expected_x = egg_pile_point[0] * scale - x0
    candidates: list[tuple[int, float, float]] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        center_x = x + width / 2.0
        if (
            area >= min_area
            and area <= max_area
            and width >= min_width
            and width <= max_width
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
                candidates.append(
                    (
                        area,
                        x0 + center_x,
                        y0 + y + height,
                    )
                )
    if not candidates:
        return None
    _, center_x, bottom_y = max(candidates)
    return center_x, float(bottom_y)


def has_home_pile_structure(
    frame: Frame,
    *,
    egg_pile_point: tuple[float, float],
    reference_width: float = 900.0,
) -> bool:
    """Return whether the lower map contains a style-neutral home pile."""

    return (
        home_pile_structure_base(
            frame,
            egg_pile_point=egg_pile_point,
            reference_width=reference_width,
        )
        is not None
    )


# The home HUD now has a fixed bottom-centre button: teal 全部孵化 while eggs are
# ready, green 自動放置 otherwise. It is not attached to the pile - it only
# covers the pile when the map is centred, exactly where the base-minus-offset
# tap lands (2026-10-06: S9 opened the auto-place confirmation three times and
# fused hatching off). Its dark-blue egg-count badge also passed the blue-stone
# pile locator on S16, which then reported a centred pile that was really at
# x~240. Live 900x1600 frames: a 164x77 component at (368,1360), 62% filled.
PILE_BUTTON_HSV_LOWER = (35, 60, 120)
PILE_BUTTON_HSV_UPPER = (105, 255, 255)
PILE_BUTTON_WIDTH_RANGE = (130.0, 200.0)
PILE_BUTTON_HEIGHT_RANGE = (55.0, 100.0)
PILE_BUTTON_MIN_FILL = 0.45
PILE_BUTTON_MIN_Y = 1000.0
PILE_BUTTON_MAX_X_OFFSET = 180.0
# Clearance kept above the button: lands on the front eggs of the pile.
PILE_BUTTON_TAP_ABOVE_PX = 70.0


PILE_BUTTON_MASK_MARGIN_PX = 10.0


def pile_action_button(
    frame: Frame,
    *,
    reference_width: float = 900.0,
) -> tuple[int, int, int, int] | None:
    """Return the bottom HUD button row's ``(x0, y0, x1, y1)``, or ``None``.

    The row can hold teal 全部孵化 alone (centred) or green 自動放置 beside it
    (S16 2026-10-07 16:44: 258-434 and 470-637). A centred pile hides behind
    the whole row, so every caller works with the union of the buttons.
    """

    if frame.image.size == 0 or frame.width <= 0 or reference_width <= 0:
        return None
    scale = frame.width / reference_width
    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PILE_BUTTON_HSV_LOWER, PILE_BUTTON_HSV_UPPER)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, float, float]] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        center_x = x + width / 2.0
        if (
            PILE_BUTTON_WIDTH_RANGE[0] * scale <= width <= PILE_BUTTON_WIDTH_RANGE[1] * scale
            and PILE_BUTTON_HEIGHT_RANGE[0] * scale
            <= height
            <= PILE_BUTTON_HEIGHT_RANGE[1] * scale
            and area >= PILE_BUTTON_MIN_FILL * width * height
            and y >= PILE_BUTTON_MIN_Y * scale
            and abs(center_x - frame.width / 2.0) <= PILE_BUTTON_MAX_X_OFFSET * scale
        ):
            candidates.append((area, x, y, x + width, y + height))
    if not candidates:
        return None
    return (
        min(item[1] for item in candidates),
        min(item[2] for item in candidates),
        max(item[3] for item in candidates),
        max(item[4] for item in candidates),
    )


def without_pile_action_button(
    frame: Frame,
    *,
    reference_width: float = 900.0,
) -> Frame:
    """Return the frame with the HUD button painted over as plain snow.

    Every pile locator measures colour or dark structure; the button carries
    both, so it is removed before any of them looks at the map.
    """

    button = pile_action_button(frame, reference_width=reference_width)
    if button is None:
        return frame
    margin = round(PILE_BUTTON_MASK_MARGIN_PX * frame.width / reference_width)
    x0, y0, x1, y1 = button
    image = frame.image.copy()
    image[
        max(0, y0 - margin) : y1 + margin,
        max(0, x0 - margin) : x1 + margin,
    ] = 255
    return Frame(image)


def clear_of_pile_button(
    frame: Frame,
    point: tuple[int, int],
    *,
    reference_width: float = 900.0,
) -> tuple[int, int]:
    """Move a pile tap that would hit the HUD button up onto the eggs."""

    button = pile_action_button(frame, reference_width=reference_width)
    if button is None:
        return point
    scale = frame.width / reference_width
    x0, y0, x1, _ = button
    margin = PILE_BUTTON_MASK_MARGIN_PX * scale
    safe_y = y0 - PILE_BUTTON_TAP_ABOVE_PX * scale
    if point[1] <= safe_y or not x0 - margin <= point[0] <= x1 + margin:
        return point
    return point[0], round(safe_y)


def home_pile_tap_point(
    frame: Frame,
    *,
    egg_pile_point: tuple[float, float],
    reference_width: float = 900.0,
) -> tuple[int, int] | None:
    """Return a safe egg-pile click point, excluding the bottom chat band."""

    base = home_pile_structure_base(
        frame,
        egg_pile_point=egg_pile_point,
        reference_width=reference_width,
    )
    if base is None:
        return None
    scale = frame.width / reference_width
    x = round(base[0])
    y = round(base[1] - HOME_PILE_TAP_OFFSET_PX * scale)
    if y >= frame.height - round(HOME_PILE_TAP_BOTTOM_EXCLUSION_PX * scale):
        return None
    return clear_of_pile_button(frame, (x, y), reference_width=reference_width)


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
    regions = (
        HATCH_TIMER_REGIONS_V2
        if uses_header_layout(image, reference_width=reference_width)
        or uses_permanent_boost_layout(image, reference_width=reference_width)
        else HATCH_TIMER_REGIONS
    )
    values: list[int] = []
    for x0, y0, x1, y1 in regions:
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
        reader: DigitReader | None = None,
        expel_below_hp: int = 0,
        expel_below_attack: int = 0,
        expel_dry_run: bool = True,
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
        self.reader = reader
        # Both floors must be set for the judgement to run at all: a zero floor
        # would expel everything that fails the other one.
        self.expel_below_hp = max(0, expel_below_hp)
        self.expel_below_attack = max(0, expel_below_attack)
        self.expel_dry_run = bool(expel_dry_run)
        self._best_hp = 0
        self._best_attack = 0
        self.logger = logger or logging.getLogger("dino_bot")
        self.clock = clock
        self._wait_until: float | None = None
        self._scrolls_done = 0
        self._home_failures = 0
        self._stage = "start"
        self.hatched = 0
        # Batch size of the open 孵化結果 panel, credited when 領取全部 is
        # verified; the ready labels last seen on the grid are the fallback
        # when the panel's count cannot be read.
        self._pending_batch: int | None = None
        self._ready_labels_seen = 0
        # Dinosaurs the caller still allows before its population stop line;
        # ``None`` means unlimited. A batch that might exceed it falls back to
        # the single-egg path, which re-checks the line after every claim.
        self.hatch_all_headroom: int | None = None
        self._headroom_fallback_logged = False

    def _record_best_stats(self, verdict: HatchVerdict | None) -> None:
        """Log the breeding line's best HP and attack when either improves.

        Only on a new record, so the line's progress is visible without a
        line per hatch. Manual screening used to print every nest parent; that
        pass is gone, and the hatch-result screen is now the only place the
        bot sees a full stat block.
        """

        if verdict is None or verdict.stats is None:
            return
        stats = verdict.stats
        improved = []
        if stats.hp > self._best_hp:
            self._best_hp = stats.hp
            improved.append("hp")
        if stats.attack > self._best_attack:
            self._best_attack = stats.attack
            improved.append("attack")
        if not improved:
            return
        self.logger.info(
            "Hatch best | new %s record | hp=%d | attack=%d | this hatch=%s",
            "+".join(improved),
            self._best_hp,
            self._best_attack,
            self._format_stats(stats),
        )

    def _judge_newborn(self, frame: Frame) -> HatchVerdict | None:
        """Judge the newborn on screen, or None when judging is switched off.

        Both floors and a reader are required. Without them the screen is
        claimed exactly as before, so an unconfigured instance cannot start
        expelling by accident.
        """

        if self.reader is None:
            return None
        if not self.expel_below_hp or not self.expel_below_attack:
            return None
        stats = read_hatch_result_stats(
            frame,
            self.reader,
            reference_width=self.reference_width,
        )
        return judge_newborn(
            stats,
            hp_floor=self.expel_below_hp,
            attack_floor=self.expel_below_attack,
        )

    @staticmethod
    def _format_stats(stats: Stats | None) -> str:
        if stats is None:
            return "unreadable"
        return f"{stats.hp}/{stats.attack}/{stats.speed}"

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
        elif target_type == CLAIM_ALL_BUTTON:
            batch = self._pending_batch or max(1, self._ready_labels_seen)
            self.hatched += batch
            self.logger.info(
                "Hatch | claimed batch of %d%s | hatched=%d",
                batch,
                "" if self._pending_batch else " (count unreadable; estimated)",
                self.hatched,
            )
            self._pending_batch = None
            self._ready_labels_seen = 0
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

        claim_all = best_detection(by_type.get(CLAIM_ALL_BUTTON))
        if claim_all is not None:
            # 選擇 and 驅逐 share this panel and are never targets: the whole
            # batch is always kept, exactly like the single-egg claim.
            if self.reader is not None:
                count = read_hatch_result_count(
                    frame.image,
                    self.reader,
                    reference_width=self.reference_width,
                )
                if count is not None:
                    self._pending_batch = count
            self._stage = "claim_all"
            return detection_target(claim_all)
        if RESULT_TITLE in by_type:
            # The panel is up but its claim button is not matched yet (still
            # animating in). Never fall through to the dimmed close button
            # behind it.
            self._stage = "result_settling"
            return None
        claim = best_detection(by_type.get(CLAIM_BUTTON))
        if claim is not None:
            expel = best_detection(by_type.get(EXPEL_BUTTON))
            verdict = self._judge_newborn(frame)
            if verdict is not None and verdict.expel and expel is not None:
                if self.expel_dry_run:
                    self.logger.info(
                        "Hatch expel | %s | WOULD EXPEL (dry run) | %s"
                        " | claiming instead",
                        self._format_stats(verdict.stats),
                        verdict.reason,
                    )
                else:
                    self.logger.info(
                        "Hatch expel | %s | expelling | %s",
                        self._format_stats(verdict.stats),
                        verdict.reason,
                    )
                    self._stage = "expel"
                    return detection_target(expel)
            elif verdict is not None:
                self.logger.info(
                    "Hatch expel | %s | keeping | %s",
                    self._format_stats(verdict.stats),
                    verdict.reason,
                )
            self._record_best_stats(verdict)
            self._stage = "claim"
            return detection_target(claim)
        hatch_button = best_detection(by_type.get(HATCH_BUTTON))
        if hatch_button is not None:
            self._stage = "detail"
            return detection_target(hatch_button)
        if INCUBATOR_TITLE in by_type:
            labels = by_type.get(HATCH_LABEL)
            if hatch_all_ready(
                frame.image, reference_width=self.reference_width
            ) and self._batch_fits(frame):
                self._stage = "hatch_all"
                self._ready_labels_seen = len(labels or ())
                return synthetic_target(
                    HATCH_ALL_BUTTON,
                    *hatch_all_point(frame, reference_width=self.reference_width),
                )
            if labels:
                self._stage = "grid"
                # Template hits in one visual row can differ by a few pixels
                # vertically. Lock onto the top row first, then choose its
                # leftmost egg so processing is deterministic row-major.
                return detection_target(self._top_left(labels, frame))
            # On v3 a gray 全部孵化 already proves no egg anywhere is ready.
            if self._scrolls_done < self.max_scrolls and not uses_header_layout(
                frame.image, reference_width=self.reference_width
            ):
                self._stage = "scroll"
                return self._scroll_target(frame)
            close = best_detection(by_type.get(CLOSE_BUTTON))
            if close is not None:
                self._stage = "close"
                return detection_target(close)
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
            home_hatch_all = best_detection(by_type.get(HOME_HATCH_ALL_BUTTON))
            if home_hatch_all is not None:
                ready = (
                    None
                    if self.reader is None
                    else read_home_hatch_all_count(
                        frame.image,
                        self.reader,
                        reference_width=self.reference_width,
                        text_center=(home_hatch_all.x, home_hatch_all.y),
                    )
                )
                if self._ready_fits(ready):
                    self._stage = "home_hatch_all"
                    self._ready_labels_seen = ready or 0
                    return detection_target(home_hatch_all)
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
    def _top_left(items: list[Detection], frame: Frame) -> Detection:
        top_y = min(item.y for item in items)
        row_tolerance = max(12, int(frame.width * 0.06))
        top_row = [item for item in items if item.y <= top_y + row_tolerance]
        return min(top_row, key=lambda item: item.x)

    def _ready_fits(self, ready: int | None) -> bool:
        headroom = self.hatch_all_headroom
        if headroom is None:
            return True
        if ready is not None and ready <= headroom:
            self._headroom_fallback_logged = False
            return True
        if not self._headroom_fallback_logged:
            self._headroom_fallback_logged = True
            self.logger.info(
                "Hatch | home 全部孵化 skipped | ready=%s exceeds headroom=%d"
                " | opening the incubator instead",
                "?" if ready is None else ready,
                headroom,
            )
        return False

    def _batch_fits(self, frame: Frame) -> bool:
        headroom = self.hatch_all_headroom
        if headroom is None:
            return True
        eggs = (
            None
            if self.reader is None
            else read_incubator_egg_count(
                frame.image,
                self.reader,
                reference_width=self.reference_width,
            )
        )
        if eggs is not None and eggs <= headroom:
            self._headroom_fallback_logged = False
            return True
        if not self._headroom_fallback_logged:
            self._headroom_fallback_logged = True
            self.logger.info(
                "Hatch | 全部孵化 skipped | eggs=%s exceeds headroom=%d"
                " | hatching one egg at a time",
                "?" if eggs is None else eggs,
                headroom,
            )
        return False

    def _scale(self, frame: Frame) -> float:
        return frame.width / self.reference_width

    def _egg_pile_target(self, frame: Frame) -> Target | None:
        scale = self._scale(frame)
        pile = home_pile_structure_base(
            frame,
            egg_pile_point=self.egg_pile_point,
            reference_width=self.reference_width,
        )
        if pile is None:
            x, y = clear_of_pile_button(
                frame,
                (
                    int(self.egg_pile_point[0] * scale),
                    int(self.egg_pile_point[1] * scale),
                ),
                reference_width=self.reference_width,
            )
        else:
            point = home_pile_tap_point(
                frame,
                egg_pile_point=self.egg_pile_point,
                reference_width=self.reference_width,
            )
            if point is None:
                self._stage = "home_tap_blocked"
                return None
            x, y = point
        return synthetic_target(EGG_PILE, x, y)

    def _scroll_target(self, frame: Frame) -> Target:
        scale = self._scale(frame)
        x0, y0, x1, y1 = (value * scale for value in self.scroll_vector)
        return swipe_target(
            SCROLL,
            int(x0),
            int(y0),
            int(x1),
            int(y1),
            duration_ms=self.scroll_duration_ms,
        )
