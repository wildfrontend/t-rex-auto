"""Safe T7 test planner for Select Dino filtering and attack sorting.

The user opens a parent and leaves the game on the "選擇恐龍" screen. This
planner converges the foreground tag dropdown to 所有, the sort dropdown to
攻擊力, and verifies the visible attack values are descending. It may toggle
the sort-direction arrow once, but it never returns a dinosaur-row target.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .digits import DigitReader
from .models import Detection, Frame, Target
from .nest_readout import read_candidate_rows

SELECT_TITLE = "hatch_select_title"
TAG_HEADER = "hatch_select_tag_header"
SORT_HEADER = "hatch_select_sort_header"
SORT_DIRECTION = "hatch_select_sort_direction"

TAG_ALL = "hatch_tag_all"
TAG_HDR_ALL = "hatch_tag_hdr_all"
SORT_HP = "hatch_sort_hp"
SORT_ATTACK = "hatch_sort_attack"
SORT_HDR_ATTACK = "hatch_sort_hdr_attack"

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    TAG_HEADER: "tap",
    TAG_ALL: "tap",
    SORT_HEADER: "tap",
    SORT_ATTACK: "tap",
    SORT_DIRECTION: "tap",
}
DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    TAG_HEADER: 2500,
    TAG_ALL: 2500,
    SORT_HEADER: 2500,
    SORT_ATTACK: 2500,
    SORT_DIRECTION: 2500,
}
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    TAG_HEADER: (TAG_ALL,),
    TAG_ALL: (TAG_HDR_ALL,),
    SORT_HEADER: (SORT_HP, SORT_ATTACK),
    SORT_ATTACK: (SORT_HDR_ATTACK,),
}
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = ()


class SelectSortTestPlanner:
    """Converge Select Dino to all-tags, attack-descending, then complete."""

    def __init__(
        self,
        reader: DigitReader,
        *,
        reference_width: float = 900.0,
        tag_header_point: tuple[float, float] = (228.0, 204.0),
        sort_header_point: tuple[float, float] = (649.0, 355.0),
        direction_point: tuple[float, float] = (552.0, 355.0),
        logger: logging.Logger | None = None,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.reader = reader
        self.reference_width = reference_width
        self.tag_header_point = tag_header_point
        self.sort_header_point = sort_header_point
        self.direction_point = direction_point
        self.logger = logger or logging.getLogger("dino_bot")
        self._stage = "start"
        self._tag_ready = False
        self._sort_ready = False
        self._direction_taps = 0
        self._complete = False

    def last_stage(self) -> str:
        return self._stage

    def is_complete(self) -> bool:
        return self._complete

    def on_action_success(self, target_type: str) -> None:
        if target_type == TAG_ALL:
            self._tag_ready = True
        elif target_type == SORT_ATTACK:
            self._sort_ready = True
        elif target_type == SORT_DIRECTION:
            self._direction_taps += 1

    def on_action_failure(self, target_type: str) -> None:
        return None

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            self._stage = "sort_done"
            return None

        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)
        if SELECT_TITLE not in by_type:
            self._stage = "waiting_for_select_dino"
            return None

        if not self._tag_ready:
            menu_all = self._closest(
                by_type.get(TAG_ALL),
                self._scaled(frame, (228.0, 249.0)),
            )
            if menu_all is not None:
                self._stage = "select_all"
                return self._target(menu_all)
            header_all = self._closest(
                by_type.get(TAG_HDR_ALL),
                self._scaled(frame, self.tag_header_point),
                max_distance=self._scaled_distance(frame, 45.0),
            )
            if header_all is not None:
                self._tag_ready = True
            else:
                self._stage = "open_tag_filter"
                return self._synthetic_target(
                    TAG_HEADER,
                    *self._scaled(frame, self.tag_header_point),
                )

        if not self._sort_ready:
            menu_attack = self._closest(
                by_type.get(SORT_ATTACK),
                self._scaled(frame, (649.0, 550.0)),
            )
            if menu_attack is not None:
                self._stage = "select_attack_sort"
                return self._target(menu_attack)
            header_attack = self._closest(
                by_type.get(SORT_HDR_ATTACK),
                self._scaled(frame, self.sort_header_point),
                max_distance=self._scaled_distance(frame, 50.0),
            )
            if header_attack is not None:
                self._sort_ready = True
            else:
                self._stage = "open_sort"
                return self._synthetic_target(
                    SORT_HEADER,
                    *self._scaled(frame, self.sort_header_point),
                )

        # Read the whole visible panel. Attack-heavy accounts can have a long
        # equal-value plateau at the top, so five rows may contain no usable
        # direction signal even though a lower value is visible farther down.
        rows = read_candidate_rows(frame.image, self.reader, max_rows=9)
        direction = self._descending_direction([row.attack for row in rows])
        if direction is None:
            self._stage = "direction_unreadable"
            self.logger.warning(
                "Hatch filter | cannot verify attack direction | rows=%s",
                [row.attack for row in rows],
            )
            return None
        self.logger.info(
            "Hatch filter | attack order=%s | descending=%s",
            [row.attack for row in rows],
            direction,
        )
        if direction:
            self._complete = True
            self._stage = "sort_done"
            return None
        if self._direction_taps >= 1:
            self._stage = "direction_failed"
            self.logger.error("Hatch filter | direction still ascending after one toggle")
            return None
        self._stage = "toggle_direction"
        return self._synthetic_target(
            SORT_DIRECTION,
            *self._scaled(frame, self.direction_point),
        )

    @staticmethod
    def _descending_direction(values: list[int]) -> bool | None:
        for first, second in zip(values, values[1:], strict=False):
            if first != second:
                return first > second
        return None

    def _scaled(self, frame: Frame, point: tuple[float, float]) -> tuple[int, int]:
        scale = frame.width / self.reference_width
        return (round(point[0] * scale), round(point[1] * scale))

    def _scaled_distance(self, frame: Frame, distance: float) -> float:
        return distance * frame.width / self.reference_width

    @staticmethod
    def _closest(
        items: list[Detection] | None,
        point: tuple[int, int],
        *,
        max_distance: float | None = None,
    ) -> Detection | None:
        if not items:
            return None
        item = min(items, key=lambda hit: (hit.x - point[0]) ** 2 + (hit.y - point[1]) ** 2)
        if max_distance is not None:
            distance_sq = (item.x - point[0]) ** 2 + (item.y - point[1]) ** 2
            if distance_sq > max_distance**2:
                return None
        return item

    @staticmethod
    def _target(detection: Detection) -> Target:
        return Target(
            type=detection.type,
            x=detection.x,
            y=detection.y,
            confidence=detection.confidence,
            detection=detection,
        )

    @staticmethod
    def _synthetic_target(target_type: str, x: int, y: int) -> Target:
        detection = Detection(
            type=target_type,
            x=x,
            y=y,
            confidence=1.0,
            metadata={"synthetic": True},
        )
        return SelectSortTestPlanner._target(detection)
