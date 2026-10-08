"""Read the My Nest tag dropdown by text, for tags the player renamed.

The tag header and the dropdown options are otherwise recognised by
screenshots of their default labels (攻擊特化, HP特化, ...). A player can rename
those tags in the game, and a renamed tag no longer looks like its screenshot.
For every role whose configured name differs from the default, this detector
reads the dropdown area with OCR and reports the same detection types the
templates would have, so the planners never learn the tag was renamed.

Roles still on their default names keep using the templates: they are proven
on every account, cost nothing, and need no OCR runtime at all.
"""

from __future__ import annotations

import difflib
import logging
import unicodedata
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from .models import BoundingBox, Detection, Frame
from .nest_filter import (
    TAG_ATTACK,
    TAG_HDR_ATTACK,
    TAG_HDR_HP,
    TAG_HDR_MASS,
    TAG_HDR_TOP,
    TAG_HP,
    TAG_MASS,
    TAG_TOP,
)

# The game's own names for the tags a player can rename. 所有 is the built-in
# "show everything" entry and cannot be renamed, so it is not listed.
DEFAULT_TAG_NAMES: dict[str, str] = {
    "attack": "攻擊特化",
    "hp": "HP特化",
    "top": "頂尖",
    "mass": "量產",
}
ROLE_TYPES: dict[str, tuple[str, str]] = {
    # role: (dropdown option type, collapsed header type)
    "attack": (TAG_ATTACK, TAG_HDR_ATTACK),
    "hp": (TAG_HP, TAG_HDR_HP),
    "top": (TAG_TOP, TAG_HDR_TOP),
    "mass": (TAG_MASS, TAG_HDR_MASS),
}

# Reference-space (900x1600) area holding the collapsed header (y~166) and the
# open dropdown below it (one option every ~43px, 所有 at y~211 down to
# 攻擊特化 at y~383, with room for a couple more rows).
OCR_REGION = (60, 130, 520, 520)
# Header text sits on the y~166 row; anything lower is a dropdown option.
HEADER_MAX_Y = 188
# Names this short are matched exactly; one wrong character is another word.
FUZZY_MIN_LENGTH = 3
FUZZY_MIN_RATIO = 0.8
MIN_OCR_SCORE = 0.5


class TextReader(Protocol):
    def read(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Return (text, score, (x0, y0, x1, y1)) for each line in ``image``."""


def normalize_tag_name(name: str) -> str:
    """Fold width, case and spacing, so "ＨＰ 特化" and "hp特化" compare equal."""

    folded = unicodedata.normalize("NFKC", name).casefold()
    return "".join(folded.split())


def custom_tag_names(names: Mapping[str, str] | None) -> dict[str, str]:
    """Roles whose configured name differs from the game's default."""

    custom: dict[str, str] = {}
    for role, default in DEFAULT_TAG_NAMES.items():
        name = (names or {}).get(role)
        if not name or not name.strip():
            continue
        if normalize_tag_name(name) != normalize_tag_name(default):
            custom[role] = name.strip()
    return custom


def match_role(text: str, names: Mapping[str, str]) -> str | None:
    """Return the role whose configured name ``text`` reads as, if any."""

    read = normalize_tag_name(text)
    if not read:
        return None
    best: tuple[float, str] | None = None
    for role, name in names.items():
        wanted = normalize_tag_name(name)
        if read == wanted:
            return role
        if min(len(read), len(wanted)) < FUZZY_MIN_LENGTH:
            continue
        ratio = difflib.SequenceMatcher(None, read, wanted).ratio()
        if ratio >= FUZZY_MIN_RATIO and (best is None or ratio > best[0]):
            best = (ratio, role)
    return best[1] if best is not None else None


class RapidOcrReader:
    """Offline Chinese/English line reader, loaded on first use."""

    def __init__(self) -> None:
        self._engine: Any = None

    def read(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
        result = self._engine(image)
        lines: list[tuple[str, float, tuple[int, int, int, int]]] = []
        if result.boxes is None or result.txts is None:
            return lines
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            bounds = (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
            lines.append((str(text), float(score), bounds))
        return lines


def ocr_available() -> bool:
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


class TagLabelOcrDetector:
    """Report renamed tag labels under the template detection types."""

    def __init__(
        self,
        names: Mapping[str, str],
        *,
        reader: TextReader | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.names = custom_tag_names(names)
        self.reader = reader or RapidOcrReader()
        self.logger = logger or logging.getLogger("dino_bot")
        self.types = frozenset(
            target_type
            for role in self.names
            for target_type in ROLE_TYPES[role]
        )

    def detect(self, frame: Frame) -> list[Detection]:
        return self._detect(frame)

    def detect_types(
        self,
        frame: Frame,
        target_types: set[str] | frozenset[str],
    ) -> list[Detection]:
        if not self.types & frozenset(target_types):
            return []
        return [item for item in self._detect(frame) if item.type in target_types]

    def _detect(self, frame: Frame) -> list[Detection]:
        if not self.names or frame.image.size == 0:
            return []
        x0, y0, x1, y1 = OCR_REGION
        crop = frame.image[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        detections: list[Detection] = []
        for text, score, (bx0, by0, bx1, by1) in self.reader.read(crop):
            if score < MIN_OCR_SCORE:
                continue
            role = match_role(text, self.names)
            if role is None:
                continue
            option_type, header_type = ROLE_TYPES[role]
            center_x = x0 + (bx0 + bx1) // 2
            center_y = y0 + (by0 + by1) // 2
            target_type = header_type if center_y <= HEADER_MAX_Y else option_type
            detections.append(
                Detection(
                    target_type,
                    center_x,
                    center_y,
                    score,
                    BoundingBox(x0 + bx0, y0 + by0, bx1 - bx0, by1 - by0),
                    {"detector": "ocr", "text": text, "role": role},
                )
            )
        return detections
