"""Bounded on-device test for opening the left nest parent.

The planner is deliberately narrower than the full replacement workflow.  It
requires the My Nest attack-specialization screen, reads all six parent stats,
and permits one tap on the left dinosaur body.  It stops as soon as Select
Dino is verified and never exposes a candidate-row action.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .digits import DigitReader
from .models import Detection, Frame, Target
from .nest_readout import read_attack_parents

NEST_TITLE = "hatch_nest_title"
SELECT_TITLE = "hatch_select_title"
TAG_HDR_ATTACK = "hatch_tag_hdr_attack"
PARENT_LEFT = "hatch_parent_left"

OPEN_TAG_OPTIONS = frozenset(
    {
        "hatch_tag_all",
        "hatch_tag_mass",
        "hatch_tag_top",
        "hatch_tag_hp",
        "hatch_tag_attack",
    }
)

DEFAULT_TARGET_ACTIONS: dict[str, str] = {PARENT_LEFT: "tap"}
DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {PARENT_LEFT: 3000}
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    PARENT_LEFT: (SELECT_TITLE,),
}
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (PARENT_LEFT,)


class ParentOpenTestPlanner:
    """Read both parents and tap the left parent body at most once."""

    def __init__(
        self,
        reader: DigitReader,
        *,
        reference_width: float = 900.0,
        left_parent_point: tuple[float, float] = (264.0, 407.0),
        attack_header_point: tuple[float, float] = (217.0, 166.0),
        logger: logging.Logger | None = None,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.reader = reader
        self.reference_width = reference_width
        self.left_parent_point = left_parent_point
        self.attack_header_point = attack_header_point
        self.logger = logger or logging.getLogger("dino_bot")
        self._stage = "start"
        self._issued = False
        self._complete = False

    def last_stage(self) -> str:
        return self._stage

    def is_complete(self) -> bool:
        return self._complete

    def on_action_success(self, target_type: str) -> None:
        if target_type == PARENT_LEFT:
            self._stage = "select_dino_opened"
            self._complete = True

    def on_action_failure(self, target_type: str) -> None:
        if target_type == PARENT_LEFT:
            # This rehearsal is intentionally one-tap-only.  A failed verify
            # must not lead to another parent tap through planner recovery.
            self._stage = "parent_open_failed"
            self._complete = True

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)

        if SELECT_TITLE in by_type:
            self._stage = "select_dino_already_open"
            self._complete = True
            return None
        if self._complete or self._issued:
            return None
        if NEST_TITLE not in by_type:
            self._stage = "waiting_for_nest"
            return None
        if any(target_type in by_type for target_type in OPEN_TAG_OPTIONS):
            self._stage = "tag_menu_open"
            return None
        if not self._attack_header_is_foreground(frame, by_type.get(TAG_HDR_ATTACK)):
            self._stage = "attack_filter_required"
            return None

        parents = read_attack_parents(frame.image, self.reader)
        if parents is None:
            self._stage = "parent_stats_unreadable"
            self.logger.warning("Hatch parent test | parent stats unreadable; refusing tap")
            return None

        left, right = parents
        self.logger.info(
            "Hatch parent test | left=%d/%d/%d | right=%d/%d/%d",
            left.hp,
            left.attack,
            left.speed,
            right.hp,
            right.attack,
            right.speed,
        )
        self._issued = True
        self._stage = "open_left_parent"
        return self._synthetic_target(
            PARENT_LEFT,
            *self._scaled(frame, self.left_parent_point),
        )

    def _attack_header_is_foreground(
        self,
        frame: Frame,
        items: list[Detection] | None,
    ) -> bool:
        if not items:
            return False
        expected = self._scaled(frame, self.attack_header_point)
        max_distance = 45.0 * frame.width / self.reference_width
        return any(
            (item.x - expected[0]) ** 2 + (item.y - expected[1]) ** 2
            <= max_distance**2
            for item in items
        )

    def _scaled(self, frame: Frame, point: tuple[float, float]) -> tuple[int, int]:
        scale = frame.width / self.reference_width
        return (round(point[0] * scale), round(point[1] * scale))

    @staticmethod
    def _synthetic_target(target_type: str, x: int, y: int) -> Target:
        detection = Detection(
            type=target_type,
            x=x,
            y=y,
            confidence=1.0,
            metadata={"synthetic": True},
        )
        return Target(
            type=target_type,
            x=x,
            y=y,
            confidence=1.0,
            detection=detection,
        )
