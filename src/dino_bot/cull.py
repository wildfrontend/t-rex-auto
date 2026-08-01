"""Dino culling (Phase C) offline pieces: capacity readout and threshold.

docs/auto-hatch-plan.md §4.5: after collecting eggs the bot slides the map to
the cave view, reads the N/350 dino counter from the top-left HUD, and only
runs the cull flow when the count exceeds the configured threshold. The
screen-driving part (cave template, swipe calibration, battle loop) needs
on-device work; the judgement implemented here is verifiable offline.
"""

from __future__ import annotations

from .digits import DigitReader
from .models import Image

# HUD capacity readout in 900-wide reference space, valid on the cave view
# where nothing overlaps it (the home view can be obscured by badges). The
# whole UI anchors top-left and scales with frame width, so both axes rescale
# by width, same as ExclusionZone.
# Start below the top edge of the glyphs' row.  Live frames can contain tiny
# confetti particles at y=236-238; including one after ``/350`` turns an
# otherwise valid read into ``280/350?``.  The digits themselves remain fully
# connected from y=239 onward at 900-wide reference scale.
CAPACITY_REGION = (10.0, 239.0, 110.0, 258.0)
EXPECTED_CAPACITY = 350


def read_dino_count(
    image: Image,
    reader: DigitReader,
    *,
    reference_width: float = 900.0,
) -> int | None:
    """Read the cave-view dino count, or None when the readout is not trusted.

    Digit reading cannot reject foreign glyphs, so a mispositioned crop could
    parse as a plausible number; requiring the exact /350 denominator is what
    keeps a bad read from triggering (or suppressing) a cull. Frames narrower
    than ~600px lose the slash to downscaling and fail safe to None.
    """

    scale = image.shape[1] / reference_width
    x0, y0, x1, y1 = (int(value * scale) for value in CAPACITY_REGION)
    fraction = reader.read_fraction(image[y0:y1, x0:x1])
    if fraction is None:
        return None
    count, capacity = fraction
    if capacity != EXPECTED_CAPACITY or not 0 <= count <= capacity:
        return None
    return count


def should_cull(count: int, threshold: int) -> bool:
    """Plan rule: at or below the threshold skip cleaning, above it cull."""

    return count > threshold
