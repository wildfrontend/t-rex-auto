"""Read the newborn's stats off the hatch-result screen and judge it.

The screen that follows a hatch shows one dinosaur with HP, attack and speed,
above a 獲取 (claim) and a 驅逐 (expel) button. Until now the bot always
claimed. Purity breeding produces a steady trickle of failures - a line aimed
at HP that lands on 1000/20, or an attack line that lands on 20/444 - and
every one of those consumes a cave slot until a later cull evicts it.

Two things make this screen safe to act on where the nest screens were not:
the digits sit on flat colour with no animation, and the decision is taken
once, on a screen that only ever shows a newborn. Nest parents never pass
through here, so a seeded 5880/1 breeder cannot be expelled by this code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .digits import DigitReader
from .models import Frame
from .nest_readout import Stats

# Measured from the 900x1600 reference frame. The three stat rows sit on pale
# cyan bands 89px apart; the digits start right of the icon that opens each
# band. Speed is read for the log only - it takes no part in the decision.
REFERENCE_WIDTH = 900.0
STAT_ROW_HEIGHT = 60
STAT_ROWS: dict[str, int] = {"hp": 846, "attack": 935, "speed": 1024}
STAT_X1, STAT_X2 = 390, 574

# HP and attack are drawn in blue, speed in black. A plain grey threshold sees
# only the black row, so the mask has to accept blue ink as well.
_BLUE_MARGIN = 60
_DARK_LEVEL = 100


@dataclass(frozen=True, slots=True)
class HatchVerdict:
    """What to do with the newborn, and why."""

    stats: Stats | None
    expel: bool
    reason: str

    @property
    def readable(self) -> bool:
        return self.stats is not None


def _ink_mask(patch: np.ndarray) -> np.ndarray:
    """Blue and black glyphs become ink; the pale band becomes paper."""

    blue, _green, red = cv2.split(patch.astype(np.int16))
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = ((blue - red) > _BLUE_MARGIN) | (grey < _DARK_LEVEL)
    out = np.full(mask.shape, 255, np.uint8)
    out[mask] = 0
    return out


def read_hatch_result_stats(
    frame: Frame,
    reader: DigitReader,
    *,
    reference_width: float = REFERENCE_WIDTH,
) -> Stats | None:
    """Read the newborn's three stats, or None if any row is unreadable."""

    scale = frame.width / reference_width
    values: dict[str, int] = {}
    for name, top in STAT_ROWS.items():
        y1 = round(top * scale)
        y2 = round((top + STAT_ROW_HEIGHT) * scale)
        x1 = round(STAT_X1 * scale)
        x2 = round(STAT_X2 * scale)
        patch = frame.image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        value = reader.read_int(_ink_mask(patch))
        if value is None:
            return None
        values[name] = value
    return Stats(hp=values["hp"], attack=values["attack"], speed=values["speed"])


def judge_newborn(
    stats: Stats | None,
    *,
    hp_floor: int,
    attack_floor: int,
) -> HatchVerdict:
    """Expel only a newborn that reaches neither breeding target.

    A specialized line deliberately floors the opposing stat, so 5880/1 and
    10/854 are successes, not failures - clearing either floor is enough to be
    kept. Only a dinosaur that misses both, like 1000/20 or 20/444, is a purity
    line that drifted and is worth expelling.

    An unreadable screen is always kept. Claiming a weak newborn costs one cave
    slot that the existing cull already handles; expelling a strong one cannot
    be undone.
    """

    if stats is None:
        return HatchVerdict(None, False, "stats unreadable")
    if stats.hp >= hp_floor:
        return HatchVerdict(stats, False, f"hp {stats.hp} >= {hp_floor}")
    if stats.attack >= attack_floor:
        return HatchVerdict(stats, False, f"attack {stats.attack} >= {attack_floor}")
    return HatchVerdict(
        stats,
        True,
        f"hp {stats.hp} < {hp_floor} and attack {stats.attack} < {attack_floor}",
    )
