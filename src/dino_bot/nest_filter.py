"""Safe on-device test planner for the Phase B nest tag dropdown.

This planner intentionally covers only T7's first reversible step: while the
user is already on the "我的巢" screen, converge the remembered tag filter to
"攻擊特化" and stop after the collapsed header confirms it. It cannot enter
the nest screen, touch a parent, auto-place dinosaurs, or collect eggs.
"""

from __future__ import annotations

from collections.abc import Sequence

from .dropdowns import SELECT, Sighting, next_step
from .models import Detection, Frame, Target

NEST_TITLE = "hatch_nest_title"
FILTER_HEADER = "hatch_tag_filter_header"

TAG_ALL = "hatch_tag_all"
TAG_MASS = "hatch_tag_mass"
TAG_TOP = "hatch_tag_top"
TAG_HP = "hatch_tag_hp"
TAG_ATTACK = "hatch_tag_attack"

TAG_HDR_ALL = "hatch_tag_hdr_all"
TAG_HDR_MASS = "hatch_tag_hdr_mass"
TAG_HDR_TOP = "hatch_tag_hdr_top"
TAG_HDR_HP = "hatch_tag_hdr_hp"
TAG_HDR_ATTACK = "hatch_tag_hdr_attack"

TARGET_LABEL = "攻擊特化"

OPTION_LABELS: dict[str, str] = {
    TAG_ALL: "所有",
    TAG_MASS: "量產",
    TAG_TOP: "頂尖",
    TAG_HP: "HP特化",
    TAG_ATTACK: TARGET_LABEL,
}
HEADER_LABELS: dict[str, str] = {
    TAG_HDR_ALL: "所有",
    TAG_HDR_MASS: "量產",
    TAG_HDR_TOP: "頂尖",
    TAG_HDR_HP: "HP特化",
    TAG_HDR_ATTACK: TARGET_LABEL,
}

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    FILTER_HEADER: "tap",
    TAG_ATTACK: "tap",
    TAG_HP: "tap",
}
DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    FILTER_HEADER: 2500,
    TAG_ATTACK: 2500,
    TAG_HP: 2500,
}
DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    FILTER_HEADER: tuple(OPTION_LABELS),
    TAG_ATTACK: (TAG_HDR_ATTACK,),
    TAG_HP: (TAG_HDR_HP,),
}
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (TAG_ATTACK,)


class NestTagFilterTestPlanner:
    """Converge the nest tag dropdown to attack specialization, then idle."""

    def __init__(
        self,
        *,
        reference_width: float = 900.0,
        header_point: tuple[float, float] = (228.0, 168.0),
        target_label: str = TARGET_LABEL,
        target_option_type: str = TAG_ATTACK,
        target_header_type: str = TAG_HDR_ATTACK,
    ) -> None:
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.reference_width = reference_width
        self.header_point = header_point
        self.target_label = target_label
        self.target_option_type = target_option_type
        self.target_header_type = target_header_type
        self._stage = "start"
        self._complete = False

    def last_stage(self) -> str:
        return self._stage

    def on_action_success(self, target_type: str) -> None:
        if target_type == self.target_option_type:
            self._complete = True

    def on_action_failure(self, target_type: str) -> None:
        return None

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._complete:
            self._stage = "filter_done"
            return None

        by_type: dict[str, list[Detection]] = {}
        for item in detections:
            by_type.setdefault(item.type, []).append(item)

        if NEST_TITLE not in by_type:
            self._stage = "waiting_for_nest"
            return None

        sightings: list[Sighting] = []
        for target_type, label in HEADER_LABELS.items():
            sightings.extend(
                Sighting(label, item.x, item.y, in_header=True)
                for item in by_type.get(target_type, ())
            )
        for target_type, label in OPTION_LABELS.items():
            sightings.extend(
                Sighting(label, item.x, item.y, in_header=False)
                for item in by_type.get(target_type, ())
            )

        step = next_step(
            self.target_label,
            sightings,
            self._scaled_header_point(frame),
        )
        if step.kind == SELECT:
            option = self._best(by_type.get(self.target_option_type))
            if option is None:
                self._stage = "menu_missing_target"
                return None
            self._stage = "select_target"
            return self._target(option)

        # Even when the header already says attack, open and reselect it once.
        # This verifies both menu expansion and option selection on device.
        header_hits = [
            item
            for target_type in HEADER_LABELS
            for item in by_type.get(target_type, ())
        ]
        if header_hits:
            header = self._best(header_hits)
            assert header is not None
            point = (header.x, header.y)
        else:
            # An unsupported remembered label (for example 速度特化) has no
            # template. The nest title makes this fixed header tap safe.
            point = self._scaled_header_point(frame)
        self._stage = "open_filter"
        return self._synthetic_target(FILTER_HEADER, *point)

    def _scaled_header_point(self, frame: Frame) -> tuple[int, int]:
        scale = frame.width / self.reference_width
        return (
            round(self.header_point[0] * scale),
            round(self.header_point[1] * scale),
        )

    @staticmethod
    def _best(items: list[Detection] | None) -> Detection | None:
        if not items:
            return None
        return max(items, key=lambda item: item.confidence)

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
        return NestTagFilterTestPlanner._target(detection)
