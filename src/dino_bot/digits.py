"""Template-based digit reading for hatch phases B/C.

The game renders every number the bot cares about as dark hand-drawn glyphs
on a light background (stat bars, counters, the N/M capacity readout), so a
full OCR stack is unnecessary: binarize, split into glyphs by connected
components, and match each glyph against a small labelled set cropped from
reference screenshots (``assets/hatch/digits``).

Callers must crop regions that contain digits (and ``/``) only. There is no
reliable rejection of other glyphs: hand-drawn letters score inside the digit
range (``o`` genuinely is ``0``), so a mispositioned crop over text yields a
plausible-looking number rather than ``None``. Guard call sites structurally
instead, e.g. ``read_fraction`` demanding the configured denominator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import Image

INK_THRESHOLD = 110
# Glyphs are resized to one canonical raster before comparison so the same
# set serves both the large counters and the small stat rows.
GLYPH_SIZE = (24, 32)
MIN_GLYPH_AREA = 12
MIN_MATCH_SCORE = 0.60
NARROW_ONE_MAX_ASPECT = 0.45
NARROW_ONE_MAX_SCORE_GAP = 0.05
# Anti-aliasing varies slightly between Select Dino rows.  A live ``6`` can
# consequently score a few points closer to the single shipped ``5`` or ``8``
# template.  Their enclosed-hole counts are stable, so prefer the matching
# topology only while the pixel scores are still close enough to be ambiguous.
TOPOLOGY_MAX_SCORE_GAP = 0.08
# Small HP zeroes can gain a short anti-aliased tail and score closer to the
# shipped ``6`` template.  Both glyphs have one enclosed hole, but a zero's
# hole spans most of the glyph height while a six's hole is confined to the
# lower bowl.  Only override a close 6/0 score when that structural proof is
# present.
ZERO_MAX_SCORE_GAP = 0.05
ZERO_MIN_HOLE_HEIGHT_RATIO = 0.55


class DigitReadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Glyph:
    char: str
    raster: np.ndarray
    enclosed_holes: int


def _enclosed_hole_boxes(
    raster: np.ndarray,
) -> list[tuple[int, int, int, int, int]]:
    """Return connected background boxes fully enclosed by a white glyph."""

    background = (raster == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background,
        connectivity=4,
    )
    holes: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        if (
            np.any(labels[0, :] == label)
            or np.any(labels[-1, :] == label)
            or np.any(labels[:, 0] == label)
            or np.any(labels[:, -1] == label)
        ):
            continue
        x, y, width, height, area = stats[label]
        holes.append((int(x), int(y), int(width), int(height), int(area)))
    return holes


def _count_enclosed_holes(raster: np.ndarray) -> int:
    """Count background regions fully enclosed by a white glyph raster."""

    return len(_enclosed_hole_boxes(raster))


def _prefer_matching_topology(
    best_char: str,
    best_score: float,
    scores: dict[str, float],
    glyph_holes: dict[str, int],
    source_holes: int,
) -> tuple[str, float]:
    """Resolve a close pixel match using the glyph's enclosed-hole count."""

    matching = [
        (score, char)
        for char, score in scores.items()
        if glyph_holes.get(char) == source_holes
    ]
    if not matching:
        return best_char, best_score
    topology_score, topology_char = max(matching)
    if (
        topology_char != best_char
        and topology_score >= MIN_MATCH_SCORE
        and best_score - topology_score <= TOPOLOGY_MAX_SCORE_GAP
    ):
        return topology_char, topology_score
    return best_char, best_score


def _prefer_narrow_one(
    best_char: str,
    best_score: float,
    scores: dict[str, float],
    bbox: tuple[int, int, int, int],
) -> tuple[str, float]:
    """Correct close 1/7 matches using the source glyph's aspect ratio.

    Canonical resizing intentionally removes size differences, but in the
    game's small stat font ``1`` is roughly half as wide as ``7``.  Lower
    Select Dino rows can otherwise turn 2310 into 2370 by a tiny score margin.
    """

    _, _, width, height = bbox
    one_score = scores.get("1", 0.0)
    if (
        best_char == "7"
        and height > 0
        and width / height <= NARROW_ONE_MAX_ASPECT
        and one_score >= MIN_MATCH_SCORE
        and best_score - one_score <= NARROW_ONE_MAX_SCORE_GAP
    ):
        return "1", one_score
    return best_char, best_score


def _prefer_tall_hole_zero(
    best_char: str,
    best_score: float,
    scores: dict[str, float],
    raster: np.ndarray,
) -> tuple[str, float]:
    """Correct a close 0/6 match using the enclosed hole's vertical span."""

    zero_score = scores.get("0", 0.0)
    holes = _enclosed_hole_boxes(raster)
    if (
        best_char == "6"
        and len(holes) == 1
        and raster.shape[0] > 0
        and holes[0][3] / raster.shape[0] >= ZERO_MIN_HOLE_HEIGHT_RATIO
        and zero_score >= MIN_MATCH_SCORE
        and best_score - zero_score <= ZERO_MAX_SCORE_GAP
    ):
        return "0", zero_score
    return best_char, best_score


def binarize(image: Image) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, INK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return ink


def _merge_by_column(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge boxes that overlap horizontally (a colon is two stacked dots)."""

    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[0]):
        x, y, w, h = box
        if merged:
            mx, my, mw, mh = merged[-1]
            overlap = min(mx + mw, x + w) - max(mx, x)
            if overlap > 0.4 * min(w, mw):
                x0 = min(mx, x)
                y0 = min(my, y)
                x1 = max(mx + mw, x + w)
                y1 = max(my + mh, y + h)
                merged[-1] = (x0, y0, x1 - x0, y1 - y0)
                continue
        merged.append(box)
    return merged


def segment_glyphs(image: Image) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Return (bbox, canonical raster) per glyph, left to right."""

    ink = binarize(image)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < MIN_GLYPH_AREA:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    results = []
    for x, y, w, h in _merge_by_column(boxes):
        raster = cv2.resize(ink[y : y + h, x : x + w], GLYPH_SIZE, interpolation=cv2.INTER_AREA)
        # Re-binarize: resizing leaves gray edge pixels that would never equal
        # the (also re-binarized) templates under exact comparison.
        _, raster = cv2.threshold(raster, 127, 255, cv2.THRESH_BINARY)
        results.append(((x, y, w, h), raster))
    return results


class DigitReader:
    def __init__(self, glyph_dir: Path, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("dino_bot")
        self.glyphs: list[Glyph] = []
        for path in sorted(Path(glyph_dir).glob("*.png")):
            raster = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raster is None:
                raise DigitReadError(f"Cannot read glyph template: {path}")
            char = {"slash": "/", "colon": ":"}.get(path.stem, path.stem)
            if len(char) != 1:
                raise DigitReadError(f"Glyph file name must map to one character: {path}")
            raster = cv2.resize(raster, GLYPH_SIZE, interpolation=cv2.INTER_AREA)
            _, raster = cv2.threshold(raster, 127, 255, cv2.THRESH_BINARY)
            self.glyphs.append(Glyph(char, raster, _count_enclosed_holes(raster)))
        if not self.glyphs:
            raise DigitReadError(f"No glyph templates found in {glyph_dir}")

    def read(self, image: Image) -> str:
        """Read every recognizable glyph left to right; '?' for misses."""

        chars = []
        for bbox, raster in segment_glyphs(image):
            best_char, best_score = "?", 0.0
            scores: dict[str, float] = {}
            glyph_holes: dict[str, int] = {}
            for glyph in self.glyphs:
                score = float(np.mean(raster == glyph.raster))
                scores[glyph.char] = max(scores.get(glyph.char, 0.0), score)
                glyph_holes[glyph.char] = glyph.enclosed_holes
                if score > best_score:
                    best_char, best_score = glyph.char, score
            best_char, best_score = _prefer_matching_topology(
                best_char,
                best_score,
                scores,
                glyph_holes,
                _count_enclosed_holes(raster),
            )
            best_char, best_score = _prefer_narrow_one(
                best_char,
                best_score,
                scores,
                bbox,
            )
            best_char, best_score = _prefer_tall_hole_zero(
                best_char,
                best_score,
                scores,
                raster,
            )
            chars.append(best_char if best_score >= MIN_MATCH_SCORE else "?")
        return "".join(chars)

    def read_int(self, image: Image) -> int | None:
        """Read a pure-number region; any stray or unknown glyph rejects it."""

        text = self.read(image)
        if not text or not text.isdigit():
            return None
        return int(text)

    def read_fraction(self, image: Image) -> tuple[int, int] | None:
        """Read an ``N/M`` capacity counter."""

        return parse_fraction(self.read(image))


def parse_fraction(text: str) -> tuple[int, int] | None:
    """Parse an ``N/M`` readout, or None when the glyphs do not spell one.

    Kept separate from ``DigitReader.read_fraction`` so a caller that needs to
    report *why* a read failed can hold on to the raw glyph text and still
    apply the identical accept/reject rule.
    """

    if "?" in text or text.count("/") != 1:
        return None
    left, right = text.split("/")
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)
