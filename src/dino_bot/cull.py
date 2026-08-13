"""Dino culling (Phase C) offline pieces: capacity readout and threshold.

docs/auto-hatch-plan.md §4.5: after collecting eggs the bot slides the map to
the cave view, reads the N/M dino counter from the top-left HUD, and only runs
the cull flow when the count reaches the configured threshold. The
screen-driving part (cave template, swipe calibration, battle loop) needs
on-device work; the judgement implemented here is verifiable offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .digits import DigitReader, parse_fraction
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
# What the map's outline reads as when it reaches into the crop: either no
# glyph at all, or - being a thin diagonal stroke - a slash. Both are equally
# untrustworthy, and neither can be told from a genuine digit by shape alone.
SCENERY_GLYPHS = "?/"


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


def _denominator_clipped_by_scenery(
    text: str, expected_capacity: int
) -> tuple[int, int] | None:
    """Recover a readout the map's outline reached into.

    In the cave view the HUD sits over the map, and a dark scenery outline
    lands right after the capacity. Depending on how wide the numbers are it
    either stays separate - adding one glyph nobody can name, ``149/200?`` -
    or touches the final digit and swallows it, ``188/20?``. As strings the
    two are indistinguishable, so the configured capacity is what tells them
    apart. The count is what the plan acts on and is read in full either way.

    The denominator only proves the crop landed on the right HUD, so it is
    waived narrowly: trailing noise may be dropped only if what remains is
    exactly the configured capacity, and a swallowed digit only if the
    lengths agree and every legible digit matches. A misplaced crop satisfies
    neither. A digit that reads as the wrong number - rather than as
    unreadable - still goes down the ordinary ``unexpected_capacity`` path.
    """

    left, separator, right = text.partition("/")
    if not separator or not left.isdigit():
        return None
    expected_text = str(expected_capacity)
    trimmed = right.rstrip(SCENERY_GLYPHS)
    if trimmed != right and trimmed == expected_text:
        return int(left), expected_capacity
    if len(right) != len(expected_text):
        return None
    if sum(1 for glyph in right if glyph in SCENERY_GLYPHS) != 1:
        return None
    if any(
        glyph not in SCENERY_GLYPHS and glyph != want
        for glyph, want in zip(right, expected_text, strict=True)
    ):
        return None
    return int(left), expected_capacity


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
    text = reader.read(image[y0:y1, x0:x1])
    fraction = parse_fraction(text) or _denominator_clipped_by_scenery(
        text, expected_capacity
    )
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
