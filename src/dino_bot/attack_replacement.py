"""Bounded end-to-end attack-parent replacement workflow.

For each parent, the planner opens Select Dino, converges to all-tags and
attack-descending, compares visible candidates, and selects only a strictly
stronger attack candidate.  A known confirmation prompt is required before
the affirmative button is allowed.  When no upgrade exists, the list closes
through the outside mask and the current parent is preserved.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from . import nest_filter as nest_filter_feature
from . import select_sort as select_sort_feature
from .digits import DigitReader
from .models import Detection, Frame, Target
from .nest_filter import NestTagFilterTestPlanner
from .nest_readout import SELECT_ROW_PITCH, read_attack_parents, read_candidate_rows
from .nests import (
    ATTACK_RULE,
    ReplacementRule,
    Stats,
    descending_prefix,
    pick_replacement,
    primary_of,
)
from .overlays import CONFIRM_YES, NESTED_PARENT_WARNING, SELECT_CONFIRM_PROMPT
from .parent_open import NEST_TITLE, OPEN_TAG_OPTIONS, SELECT_TITLE
from .select_sort import SelectSortTestPlanner

PARENT_LEFT = "hatch_parent_left"
PARENT_RIGHT = "hatch_parent_right"
CANDIDATE_ROW = "hatch_candidate_row"
NESTED_PARENT_YES = "hatch_nested_parent_yes"
SELECT_MASK_CLOSE = "hatch_select_mask_close"

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    **nest_filter_feature.DEFAULT_TARGET_ACTIONS,
    **select_sort_feature.DEFAULT_TARGET_ACTIONS,
    PARENT_LEFT: "tap",
    PARENT_RIGHT: "tap",
    CANDIDATE_ROW: "tap",
    NESTED_PARENT_YES: "tap",
    CONFIRM_YES: "tap",
    SELECT_MASK_CLOSE: "tap",
}
DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    **nest_filter_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    **select_sort_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    PARENT_LEFT: 3000,
    PARENT_RIGHT: 3000,
    CANDIDATE_ROW: 2500,
    NESTED_PARENT_YES: 3000,
    CONFIRM_YES: 3000,
    SELECT_MASK_CLOSE: 3000,
}
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    **nest_filter_feature.DEFAULT_SUCCESS_TRANSITIONS,
    **select_sort_feature.DEFAULT_SUCCESS_TRANSITIONS,
    PARENT_LEFT: (SELECT_TITLE,),
    PARENT_RIGHT: (SELECT_TITLE,),
    CANDIDATE_ROW: (SELECT_CONFIRM_PROMPT, NESTED_PARENT_WARNING),
    NESTED_PARENT_YES: (SELECT_CONFIRM_PROMPT, NEST_TITLE),
    CONFIRM_YES: (NEST_TITLE,),
    SELECT_MASK_CLOSE: (NEST_TITLE,),
}
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (CONFIRM_YES, SELECT_MASK_CLOSE)


class AttackReplacementTestPlanner:
    """Run the attack replacement rule for left and right parents once."""

    def __init__(
        self,
        reader: DigitReader,
        *,
        reference_width: float = 900.0,
        left_parent_point: tuple[float, float] = (264.0, 407.0),
        right_parent_point: tuple[float, float] = (523.0, 407.0),
        attack_header_point: tuple[float, float] = (217.0, 166.0),
        candidate_point: tuple[float, float] = (350.0, 435.0),
        mask_close_point: tuple[float, float] = (50.0, 800.0),
        rule: ReplacementRule = ATTACK_RULE,
        nest_filter_option: str = nest_filter_feature.TAG_ATTACK,
        nest_filter_header: str = nest_filter_feature.TAG_HDR_ATTACK,
        select_sort_option: str = select_sort_feature.SORT_ATTACK,
        select_sort_header: str = select_sort_feature.SORT_HDR_ATTACK,
        select_sort_menu_point: tuple[float, float] = (649.0, 550.0),
        logger: logging.Logger | None = None,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.reader = reader
        self.reference_width = reference_width
        self.parent_points = (left_parent_point, right_parent_point)
        self.attack_header_point = attack_header_point
        self.candidate_point = candidate_point
        self.mask_close_point = mask_close_point
        self.rule = rule
        self.nest_filter_option = nest_filter_option
        self.nest_filter_header = nest_filter_header
        self.select_sort_option = select_sort_option
        self.select_sort_header = select_sort_header
        self.select_sort_menu_point = select_sort_menu_point
        self.logger = logger or logging.getLogger("dino_bot")
        self._stage = "filter_attack"
        self._side = 0
        self._current_parent: Stats | None = None
        self._filter_planner = NestTagFilterTestPlanner(
            reference_width=reference_width,
            target_label=rule.tag,
            target_option_type=nest_filter_option,
            target_header_type=nest_filter_header,
        )
        self._select_planner: SelectSortTestPlanner | None = None
        self._complete = False

    def last_stage(self) -> str:
        return self._stage

    def is_complete(self) -> bool:
        return self._complete

    def on_action_success(self, target_type: str) -> None:
        if self._stage.startswith("filter_"):
            self._filter_planner.on_action_success(target_type)
            if target_type == self.nest_filter_option:
                self._stage = "nest_left"
            return
        if target_type in (PARENT_LEFT, PARENT_RIGHT):
            self._stage = self._side_stage("select")
            self._select_planner = SelectSortTestPlanner(
                self.reader,
                reference_width=self.reference_width,
                sort_option_type=self.select_sort_option,
                sort_header_type=self.select_sort_header,
                sort_menu_point=self.select_sort_menu_point,
                primary_attr=self.rule.primary,
                sort_label=self.rule.sort_option,
                logger=self.logger,
            )
            return
        if target_type == CANDIDATE_ROW:
            self._stage = self._side_stage("confirm")
            return
        if target_type == NESTED_PARENT_YES:
            self._stage = self._side_stage("after_nested")
            return
        if target_type in (CONFIRM_YES, SELECT_MASK_CLOSE):
            self._advance_parent()
            return
        if self._select_planner is not None:
            self._select_planner.on_action_success(target_type)

    def on_action_failure(self, target_type: str) -> None:
        self._stage = f"failed_{target_type}"
        self._complete = True

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            return None
        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)

        if self._stage.startswith("filter_"):
            return self._filter_planner.choose(frame, detections)
        if self._stage.startswith("nest_"):
            return self._choose_parent(frame, by_type)
        if self._stage.startswith("select_"):
            return self._choose_select(frame, detections, by_type)
        if self._stage.startswith("confirm_"):
            return self._choose_confirmation(by_type)
        if self._stage.startswith("after_nested_"):
            return self._choose_after_nested_warning(by_type)
        return None

    def _choose_parent(
        self,
        frame: Frame,
        by_type: dict[str, list[Detection]],
    ) -> Target | None:
        if SELECT_TITLE in by_type:
            self._stage = "unexpected_select_dino"
            return None
        if NEST_TITLE not in by_type:
            return None
        if any(target_type in by_type for target_type in OPEN_TAG_OPTIONS):
            self._stage = "tag_menu_open"
            return None
        if not self._nest_filter_header_is_foreground(
            frame,
            by_type.get(self.nest_filter_header),
        ):
            self._stage = "target_filter_required"
            return None

        parents = read_attack_parents(frame.image, self.reader)
        if parents is None:
            self._stage = "parent_stats_unreadable"
            self.logger.warning(
                "Hatch %s | parent stats unreadable; refusing tap",
                self.rule.tag,
            )
            return None
        self._current_parent = parents[self._side]
        self.logger.info(
            "Hatch %s | side=%s | parent=%s | pair=%s,%s",
            self.rule.tag,
            self._side_name,
            self._format_stats(self._current_parent),
            self._format_stats(parents[0]),
            self._format_stats(parents[1]),
        )
        target_type = PARENT_LEFT if self._side == 0 else PARENT_RIGHT
        self._stage = self._side_stage("open")
        return self._synthetic_target(
            target_type,
            *self._scaled(frame, self.parent_points[self._side]),
        )

    def _choose_select(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        by_type: dict[str, list[Detection]],
    ) -> Target | None:
        if SELECT_TITLE not in by_type:
            return None
        if self._current_parent is None or self._select_planner is None:
            self._stage = "missing_parent_context"
            self._complete = True
            return None

        target = self._select_planner.choose(frame, detections)
        if target is not None:
            return target
        if not self._select_planner.is_complete():
            if self._select_planner.last_stage() == "direction_unreadable":
                plateau_rows = read_candidate_rows(frame.image, self.reader)
                if self._equal_parent_plateau(plateau_rows):
                    self.logger.info(
                        "Hatch %s | side=%s | equal %s plateau=%s"
                        " | decision=keep parent without further search",
                        self.rule.tag,
                        self._side_name,
                        self.rule.sort_option,
                        [primary_of(row, self.rule) for row in plateau_rows],
                    )
                    return self._close_list(frame)
            return None

        raw_rows = read_candidate_rows(frame.image, self.reader)
        if not raw_rows:
            self._stage = "candidate_stats_unreadable"
            self.logger.warning(
                "Hatch %s | candidate stats unreadable; stopping",
                self.rule.tag,
            )
            self._complete = True
            return None
        rows = descending_prefix(raw_rows, self.rule)
        if len(rows) < len(raw_rows):
            self.logger.warning(
                "Hatch %s | side=%s | ignored non-descending OCR tail"
                " | raw=%s | trusted=%s",
                self.rule.tag,
                self._side_name,
                [primary_of(row, self.rule) for row in raw_rows],
                [primary_of(row, self.rule) for row in rows],
            )
        replacement_index = pick_replacement(self._current_parent, rows, self.rule)
        if replacement_index is None:
            self.logger.info(
                "Hatch %s | side=%s | candidates=%s | decision=keep parent",
                self.rule.tag,
                self._side_name,
                [self._format_stats(row) for row in rows],
            )
            return self._close_list(frame)

        replacement = rows[replacement_index]
        self.logger.info(
            "Hatch %s | side=%s | candidates=%s | decision=select row %d (%s)",
            self.rule.tag,
            self._side_name,
            [self._format_stats(row) for row in rows],
            replacement_index + 1,
            self._format_stats(replacement),
        )

        x, y = self._scaled(frame, self.candidate_point)
        y += round(
            replacement_index * SELECT_ROW_PITCH * frame.width / self.reference_width
        )
        self._stage = self._side_stage("choose")
        return self._synthetic_target(CANDIDATE_ROW, x, y)

    def _choose_confirmation(
        self,
        by_type: dict[str, list[Detection]],
    ) -> Target | None:
        if NESTED_PARENT_WARNING in by_type:
            yes = self._best(by_type.get(CONFIRM_YES))
            if yes is None:
                self._stage = "nested_confirmation_yes_missing"
                self._complete = True
                return None
            self._stage = self._side_stage("remove_nested")
            return self._synthetic_target(NESTED_PARENT_YES, yes.x, yes.y)
        if SELECT_CONFIRM_PROMPT not in by_type:
            self._stage = "confirmation_prompt_missing"
            self._complete = True
            return None
        button = self._best(by_type.get(CONFIRM_YES))
        if button is None:
            self._stage = "confirmation_yes_missing"
            self._complete = True
            return None
        self._stage = self._side_stage("replace")
        return self._target(button)

    def _choose_after_nested_warning(
        self,
        by_type: dict[str, list[Detection]],
    ) -> Target | None:
        if NEST_TITLE in by_type:
            self._advance_parent()
            return None
        if SELECT_CONFIRM_PROMPT not in by_type:
            return None
        yes = self._best(by_type.get(CONFIRM_YES))
        if yes is None:
            self._stage = "confirmation_yes_missing_after_nested"
            self._complete = True
            return None
        self._stage = self._side_stage("replace")
        return self._target(yes)

    def _close_list(self, frame: Frame) -> Target:
        self._stage = self._side_stage("close")
        return self._synthetic_target(
            SELECT_MASK_CLOSE,
            *self._scaled(frame, self.mask_close_point),
        )

    def _advance_parent(self) -> None:
        if self._side == 0:
            self._side = 1
            self._current_parent = None
            self._select_planner = None
            self._stage = "nest_right"
        else:
            self._stage = "replacement_done"
            self._complete = True

    def _equal_parent_plateau(self, rows: list[Stats]) -> bool:
        return (
            self._current_parent is not None
            and len(rows) >= 5
            and all(
                primary_of(row, self.rule) == primary_of(self._current_parent, self.rule)
                for row in rows
            )
        )

    def _nest_filter_header_is_foreground(
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

    def _side_stage(self, stage: str) -> str:
        return f"{stage}_{self._side_name}"

    @property
    def _side_name(self) -> str:
        return "left" if self._side == 0 else "right"

    @staticmethod
    def _format_stats(stats: Stats) -> str:
        return f"{stats.hp}/{stats.attack}/{stats.speed}"

    @staticmethod
    def _best(items: list[Detection] | None) -> Detection | None:
        if not items:
            return None
        return max(items, key=lambda item: item.confidence)

    @staticmethod
    def _target(detection: Detection) -> Target:
        return Target(
            detection.type,
            detection.x,
            detection.y,
            detection.confidence,
            detection,
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
        return Target(target_type, x, y, 1.0, detection)
