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


def probe_dino_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
    expected_capacity: int = EXPECTED_CAPACITY,
) -> CapacityRead:
    """Read the cave-view dino count and report what happened either way."""

    if expected_capacity <= 0:
        raise ValueError("expected_capacity must be greater than zero")

    height, width = image.shape[0], image.shape[1]
    scale = width / reference_width
    x0, y0, x1, y1 = (int(value * scale) for value in CAPACITY_REGION)
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
