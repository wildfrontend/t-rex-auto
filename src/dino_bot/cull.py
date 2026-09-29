"""Dino culling (Phase C) offline pieces: capacity readout and threshold.

docs/auto-hatch-plan.md §4.5: after collecting eggs the bot slides the map to
the cave view, reads the N/M dino counter from the top-left HUD, and only runs
the cull flow when the count reaches the configured threshold. The
screen-driving part (cave template, swipe calibration, battle loop) needs
on-device work; the judgement implemented here is verifiable offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from .digits import DigitReader, binarize, parse_fraction
from .models import Image

# HUD capacity readout in 900-wide reference space, valid on the cave view
# where nothing overlaps it (the home view can be obscured by badges). The
# whole UI anchors top-left and scales with frame width, so both axes rescale
# by width, same as ExclusionZone.
# Start below the top edge of the glyphs' row.  Live frames can contain tiny
# confetti particles at y=236-238; including one after the fraction turns an
# otherwise valid read into ``280/350?``. The digits themselves remain fully
# connected from y=239 onward at 900-wide reference scale.
CAPACITY_REGION = (10.0, 239.0, 110.0, 258.0)
# The My Dinosaurs panel prints the same N/M as plain black digits on the
# panel's white card. Nothing overlaps it, it does not animate, and the panel
# is a fixed overlay rather than part of the map - so unlike the cave HUD it
# needs no camera position at all. The cave HUD misread 321/370 as "32110" by
# losing the slash to the scenery behind it, and returned nothing at all over
# dark forest; this region reads first time.
PANEL_CAPACITY_REGION = (370.0, 355.0, 530.0, 405.0)
# The card's height changes with its contents, so the fraction does not sit at
# a fixed y: two live captures put the title at 292 and at 325. The gap from
# the title's top edge down to the digits is stable (63 and 60), so anchor the
# crop on the detected title instead of the screen.
PANEL_CAPACITY_FROM_TITLE = (
    25.0,  # left, relative to the title's left edge
    63.0,  # top, relative to the title's top edge
    185.0,  # right
    113.0,  # bottom
)
PANEL_TITLE = "hatch_my_dino_title"
# The title renders slightly differently between openings - live captures
# measured the same characters at 188x43 and at 183x40 - so a grayscale
# template match tops out around 0.6 and cannot be told from a miss. Matching
# the ink alone, across a small scale sweep, separates cleanly: 0.89 with the
# panel open against 0.20 without it.
PANEL_TITLE_INK_LEVEL = 120
PANEL_TITLE_SCALES = tuple(round(1.0 + i * 0.02, 2) for i in range(-4, 5))
PANEL_TITLE_MIN_SCORE = 0.8
# Opens the panel from the home map: the second control down the left edge.
PANEL_OPEN_POINT = (60.0, 288.0)
# Any point outside the card dismisses it.
PANEL_CLOSE_POINT = (450.0, 1500.0)
EXPECTED_CAPACITY = 350
# The capacity HUD is drawn over the map.  Coloured particles and dinosaurs
# can touch the final digit and make the grayscale connected-component reader
# merge them into ``4``/``9`` or erase the digit at the crop edge.  The HUD
# digits themselves are grayscale, so saturated pixels are safe to discard.
CAPACITY_MAX_SATURATION = 80


@dataclass(frozen=True, slots=True)
class CapacityRead:
    """One attempt at the configured N/M HUD, including why it was rejected.

    ``read_dino_count`` collapses every failure to None, which is the right
    contract for the planner but leaves a diagnostic bundle unable to tell a
    HUD that is absent from one that is present but unreadable. The two want
    opposite fixes - a navigation change versus a crop or glyph change - so
    the evidence writer needs the distinction that the planner discards.
    """

    count: int | None
    text: str
    fraction: tuple[int, int] | None
    # Absolute crop in frame pixels, so a saved frame can be re-cropped by hand.
    region: tuple[int, int, int, int]
    reason: str

    @property
    def ok(self) -> bool:
        return self.count is not None


def _clean_capacity_crop(crop: Image) -> tuple[Image, bool]:
    """Blank out ink that the crop's own edge cuts through.

    Unlike the stat panels, this HUD sits over the map, so dark scenery can
    reach into the calibrated region from the side. Every glyph of the
    readout falls entirely inside that region; anything the border truncates
    is not one of them. Left in place it is read as an extra digit and
    poisons the value - a stone outline once turned 114/200 into 114/2002 -
    and it cannot be told apart by size, having been as slim as a real digit.

    A blob that merges with the last digit is erased along with it, which
    reads short and fails the denominator check: the run stops instead of
    acting on a number the scenery had a hand in.
    """

    ink = binarize(crop)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    width = ink.shape[1]
    cleaned = crop.copy()
    edge_ink_removed = False
    for index in range(1, count):
        x = stats[index, cv2.CC_STAT_LEFT]
        if x == 0 or x + stats[index, cv2.CC_STAT_WIDTH] >= width:
            cleaned[labels == index] = 255
            edge_ink_removed = True
    return cleaned, edge_ink_removed


def _without_truncated_ink(crop: Image) -> Image:
    """Compatibility wrapper returning the cleaned capacity crop."""

    return _clean_capacity_crop(crop)[0]


def _without_colored_ink(crop: Image) -> Image:
    """Remove saturated map particles before matching grayscale HUD digits."""

    if crop.ndim != 3:
        return crop.copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    cleaned = crop.copy()
    cleaned[hsv[:, :, 1] > CAPACITY_MAX_SATURATION] = 255
    return cleaned


def _repair_truncated_denominator(
    text: str,
    expected_capacity: int,
    *,
    edge_ink_removed: bool,
) -> str | None:
    """Restore only a known denominator suffix lost at the crop edge.

    This is deliberately narrower than accepting any OCR value near the
    configured capacity.  A repair is allowed only when the visible
    denominator is an exact prefix of the configured value and the image
    contained ink touching the crop edge, which is the signature of the
    observed ``380`` -> ``38`` failure.
    """

    if not edge_ink_removed or text.count("/") != 1:
        return None
    numerator, denominator = text.split("/")
    expected = str(expected_capacity)
    if (
        not numerator.isdigit()
        or not denominator.isdigit()
        or not denominator
        or len(denominator) >= len(expected)
        or not expected.startswith(denominator)
    ):
        return None
    return f"{numerator}/{expected}"


def _capacity_text_candidates(
    crop: Image,
    reader: DigitReader,
) -> list[tuple[str, bool]]:
    """Read raw and colour-cleaned variants, preserving edge evidence."""

    candidates: list[tuple[str, bool]] = []
    for variant in (crop, _without_colored_ink(crop)):
        cleaned, edge_ink_removed = _clean_capacity_crop(variant)
        candidates.append((reader.read(cleaned), edge_ink_removed))
    return candidates


def locate_panel_capacity(
    image: Image,
    template: Image,
) -> tuple[float, float, float, float] | None:
    """Find the My Dinosaurs panel and return its capacity crop.

    Returns None when the panel is not on screen. The card's height varies
    with its contents, so the fraction is anchored to the title rather than to
    the frame: two live captures put the title 33px apart while the gap down
    to the digits stayed at 63 and 60.
    """

    if image.size == 0 or template.size == 0:
        return None
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl = template if template.ndim == 2 else cv2.cvtColor(
        template, cv2.COLOR_BGR2GRAY
    )
    haystack = ((grey < PANEL_TITLE_INK_LEVEL) * 255).astype("uint8")
    needle = ((tpl < PANEL_TITLE_INK_LEVEL) * 255).astype("uint8")
    best_score = 0.0
    best_point: tuple[int, int] | None = None
    for scale in PANEL_TITLE_SCALES:
        scaled = (
            needle
            if scale == 1.0
            else cv2.resize(
                needle, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        )
        if scaled.shape[0] > haystack.shape[0] or scaled.shape[1] > haystack.shape[1]:
            continue
        _, score, _, point = cv2.minMaxLoc(
            cv2.matchTemplate(haystack, scaled, cv2.TM_CCOEFF_NORMED)
        )
        if score > best_score:
            best_score, best_point = score, point
    if best_point is None or best_score < PANEL_TITLE_MIN_SCORE:
        return None
    left, top, right, bottom = PANEL_CAPACITY_FROM_TITLE
    x, y = best_point
    return (x + left, y + top, x + right, y + bottom)


def probe_dino_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
    expected_capacity: int = EXPECTED_CAPACITY,
    capacity_region: tuple[float, float, float, float] = CAPACITY_REGION,
) -> CapacityRead:
    """Read a dino count from the given region and report what happened.

    The default region is the cave-view HUD. Pass ``PANEL_CAPACITY_REGION`` to
    read the same figure off the My Dinosaurs panel instead, where the digits
    sit on a white card that nothing overlaps.
    """

    if expected_capacity <= 0:
        raise ValueError("expected_capacity must be greater than zero")

    height, width = image.shape[0], image.shape[1]
    scale = width / reference_width
    x0, y0, x1, y1 = (int(value * scale) for value in capacity_region)
    region = (x0, y0, x1, y1)
    # A frame smaller than the calibrated HUD (a portrait/landscape flip, or a
    # capture that lost the emulator chrome insets) would otherwise crop to an
    # empty array and read as an ordinary unparsed miss.
    if x0 >= x1 or y0 >= y1 or x1 > width or y1 > height:
        return CapacityRead(None, "", None, region, "region_outside_frame")
    crop = image[y0:y1, x0:x1]
    candidates = _capacity_text_candidates(crop, reader)
    text = candidates[0][0]

    # Prefer an exact read from any preprocessing variant.  This handles the
    # observed coloured obstruction turning the final 0 into a 4 or 9.
    for candidate_text, _ in candidates:
        fraction = parse_fraction(candidate_text)
        if fraction is None:
            continue
        count, capacity = fraction
        if capacity == expected_capacity and 0 <= count <= capacity:
            return CapacityRead(count, candidate_text, fraction, region, "ok")

    # If the final glyph was erased together with an edge-touching obstruction,
    # restore it only when the visible denominator is an exact prefix of the
    # configured capacity.  This keeps foreign or malformed values fail-safe.
    for candidate_text, edge_ink_removed in candidates:
        repaired = _repair_truncated_denominator(
            candidate_text,
            expected_capacity,
            edge_ink_removed=edge_ink_removed,
        )
        if repaired is None:
            continue
        fraction = parse_fraction(repaired)
        assert fraction is not None
        count, capacity = fraction
        if 0 <= count <= capacity:
            return CapacityRead(count, repaired, fraction, region, "ok")

    fraction = parse_fraction(text)
    if fraction is None:
        return CapacityRead(None, text, None, region, "unparsed")
    count, capacity = fraction
    if capacity != expected_capacity:
        return CapacityRead(None, text, fraction, region, "unexpected_capacity")
    if not 0 <= count <= capacity:
        return CapacityRead(None, text, fraction, region, "count_out_of_range")
    return CapacityRead(count, text, fraction, region, "ok")


def read_dino_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
    expected_capacity: int = EXPECTED_CAPACITY,
) -> int | None:
    """Read the cave-view dino count, or None when the readout is not trusted.

    Digit reading cannot reject foreign glyphs, so a mispositioned crop could
    parse as a plausible number; requiring the configured denominator is what
    keeps a bad read from triggering (or suppressing) a cull. Frames narrower
    than ~600px lose the slash to downscaling and fail safe to None.
    """

    return probe_dino_count(
        image,
        reader,
        reference_width=reference_width,
        expected_capacity=expected_capacity,
    ).count


def should_cull(count: int, threshold: int) -> bool:
    """Plan rule: below the threshold skip cleaning, at/above it cull."""

    return count >= threshold
