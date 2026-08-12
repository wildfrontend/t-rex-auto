"""Beginner auto-hatch workflow.

This mode keeps the normal incubator loop, then performs one simple
nest-management pass: switch to ``所有``, auto-place once, and collect once.
When combined with hunting, every verified return to the home screen also
runs one collect-only nest pass.  It deliberately never reads parent stats,
sorts nests, opens the cave, or removes dinosaurs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from . import hatch as hatch_feature
from . import nest_filter as nest_filter_feature
from .full_hatch import (
    AUTOPLACE_BUTTON,
    AUTOPLACE_NOTICE,
    AUTOPLACE_PROMPT,
    COLLECT_EGGS_BUTTON,
    NEST_MASK_CLOSE,
    OPEN_NEST,
    STARTUP_AUTO_BATTLE_CLOSE,
    STARTUP_DETECTION_TYPES,
    STARTUP_GROWTH_RESULT,
    STARTUP_NEST_SHORTCUT,
    STARTUP_SIMPLE_INTERRUPTS,
    is_home_screen,
)
from .models import Detection, Frame, Target
from .overlays import AUTOPLACE_UNAVAILABLE, CONFIRM_YES, INCUBATOR_FULL_TOAST
from .parent_open import NEST_TITLE

DEFAULT_TARGET_ACTIONS: dict[str, str] = {
    **hatch_feature.DEFAULT_TARGET_ACTIONS,
    **nest_filter_feature.DEFAULT_TARGET_ACTIONS,
    OPEN_NEST: "tap",
    AUTOPLACE_BUTTON: "tap",
    "hatch_beginner_autoplace_yes": "tap",
    COLLECT_EGGS_BUTTON: "tap",
    NEST_MASK_CLOSE: "tap",
    STARTUP_AUTO_BATTLE_CLOSE: "tap",
    STARTUP_NEST_SHORTCUT: "tap",
    **{target_type: "tap" for target_type in STARTUP_SIMPLE_INTERRUPTS},
}

DEFAULT_POST_ACTION_DELAYS_MS: dict[str, int] = {
    **hatch_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    **nest_filter_feature.DEFAULT_POST_ACTION_DELAYS_MS,
    OPEN_NEST: 3500,
    AUTOPLACE_BUTTON: 4000,
    "hatch_beginner_autoplace_yes": 5000,
    COLLECT_EGGS_BUTTON: 5000,
    NEST_MASK_CLOSE: 3000,
    STARTUP_AUTO_BATTLE_CLOSE: 3000,
    STARTUP_NEST_SHORTCUT: 4000,
    **{target_type: 5000 for target_type in STARTUP_SIMPLE_INTERRUPTS},
}

DEFAULT_SUCCESS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    **hatch_feature.DEFAULT_SUCCESS_TRANSITIONS,
    **nest_filter_feature.DEFAULT_SUCCESS_TRANSITIONS,
    OPEN_NEST: (NEST_TITLE,),
    AUTOPLACE_BUTTON: (
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        AUTOPLACE_UNAVAILABLE,
        CONFIRM_YES,
        NEST_TITLE,
    ),
    "hatch_beginner_autoplace_yes": (NEST_TITLE,),
    # ``孵化器已滿`` is a successful, non-destructive collection attempt.
    # It must leave the nest instead of repeatedly tapping collect.
    COLLECT_EGGS_BUTTON: (INCUBATOR_FULL_TOAST, NEST_TITLE),
    NEST_MASK_CLOSE: (hatch_feature.HOME_ANCHOR,),
    STARTUP_AUTO_BATTLE_CLOSE: (hatch_feature.HOME_ANCHOR,),
    STARTUP_NEST_SHORTCUT: (NEST_TITLE,),
}

# A hatched dinosaur is the only productive cycle the shared engine counts.
DEFAULT_CYCLE_COMPLETE_TARGETS: tuple[str, ...] = (hatch_feature.CLAIM_BUTTON,)

AUTOPLACE_YES = "hatch_beginner_autoplace_yes"

HATCH_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        *hatch_feature.DEFAULT_TARGET_ACTIONS,
        # The incubator title is state evidence, not an action.  HatchPlanner
        # requires it before it may tap a ready egg or close the grid.
        hatch_feature.INCUBATOR_TITLE,
        hatch_feature.HOME_ANCHOR,
        hatch_feature.EXPEL_BUTTON,
    }
)
NEST_DETECTION_TYPES: frozenset[str] = frozenset(
    {
        *STARTUP_DETECTION_TYPES,
        hatch_feature.HOME_ANCHOR,
        NEST_TITLE,
        *nest_filter_feature.DEFAULT_TARGET_ACTIONS,
        *nest_filter_feature.HEADER_LABELS,
        AUTOPLACE_BUTTON,
        AUTOPLACE_PROMPT,
        AUTOPLACE_NOTICE,
        AUTOPLACE_UNAVAILABLE,
        CONFIRM_YES,
        COLLECT_EGGS_BUTTON,
        INCUBATOR_FULL_TOAST,
    }
)


class BeginnerHatchPlanner:
    """Hatch eggs and run one guarded beginner nest pass per incubator visit."""

    def __init__(
        self,
        *,
        egg_pile_point: tuple[float, float],
        reference_width: float = 900.0,
        scroll_vector: tuple[float, float, float, float] = (450, 1100, 450, 500),
        scroll_duration_ms: int = 400,
        max_scrolls: int = 0,
        rescan_interval_seconds: float = 600.0,
        require_home_anchor: bool = True,
        home_failure_limit: int = 3,
        home_backoff_seconds: float = 30.0,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.reference_width = reference_width
        self.logger = logger or logging.getLogger("dino_bot")
        self._hatch_kwargs = dict(
            egg_pile_point=egg_pile_point,
            reference_width=reference_width,
            scroll_vector=scroll_vector,
            scroll_duration_ms=scroll_duration_ms,
            max_scrolls=max_scrolls,
            rescan_interval_seconds=rescan_interval_seconds,
            require_home_anchor=require_home_anchor,
            home_failure_limit=home_failure_limit,
            home_backoff_seconds=home_backoff_seconds,
            logger=self.logger,
        )
        if clock is not None:
            self._hatch_kwargs["clock"] = clock
        self._stage = "hatch"
        self._child: object = self._new_hatch()
        self._filter: nest_filter_feature.NestTagFilterTestPlanner | None = None
        self._autoplace_requested = False
        self._collect_requested = False
        self._home_collection = False
        self._blocked_reason: str | None = None
        self.management_rounds = 0
        self.home_collection_rounds = 0

    def last_stage(self) -> str:
        child_stage = getattr(self._child, "last_stage", None)
        detail = child_stage() if callable(child_stage) else child_stage
        return f"beginner_{self._stage}:{detail or '-'}"

    def is_complete(self) -> bool:
        return self._blocked_reason is not None

    def is_hatch_blocked(self) -> bool:
        return self._blocked_reason is not None

    def continue_hunting_when_blocked(self) -> bool:
        """An unverified beginner management action must stop the whole run."""

        return False

    def next_ready_delay_ms(self) -> int:
        method = getattr(self._child, "next_ready_delay_ms", None)
        return int(method()) if callable(method) else 0

    def is_hunt_cooldown_active(self) -> bool:
        """Whether the post-management incubator wait can be spent hunting."""

        return self._stage == "hatch" and self.next_ready_delay_ms() > 0

    def begin_interim_collection(self) -> bool:
        """Keep beginner hunt mode to one management round per hatch cycle.

        Full hatch can use an idle hunting window for an extra collect-only
        errand.  Beginner mode intentionally stays at its documented single
        ``所有 → 自動放置 → 收集`` pass, so it declines that optional errand.
        """

        return False

    def begin_home_collection(self) -> bool:
        """Collect once after hatch-hunt has verified a return home.

        Keep the existing incubator child so completing this short errand
        resumes the original cooldown (or its now-ready incubator) instead of
        starting a fresh wait.  The dedicated flag also prevents a second
        handoff frame from arming the same collection twice.
        """

        if self._stage != "hatch" or self._home_collection:
            return False
        self.logger.info("Beginner hatch | returned home | collecting all eggs")
        self._stage = "open_nest_collect"
        self._home_collection = True
        self._collect_requested = False
        return True

    def planning_detection_types(self) -> frozenset[str] | None:
        if self._stage == "hatch":
            return HATCH_DETECTION_TYPES
        return NEST_DETECTION_TYPES

    def reset_workflow(self) -> None:
        self._stage = "hatch"
        self._child = self._new_hatch()
        self._filter = None
        self._autoplace_requested = False
        self._collect_requested = False
        self._home_collection = False
        self._blocked_reason = None

    def on_action_success(self, target_type: str) -> None:
        if target_type == STARTUP_NEST_SHORTCUT:
            self._stage = "open_nest"
            return
        if target_type in {*STARTUP_SIMPLE_INTERRUPTS, STARTUP_AUTO_BATTLE_CLOSE}:
            return
        if self._stage == "hatch":
            hatch = self._hatch_child
            hatch.on_action_success(target_type)
            if target_type == hatch_feature.CLOSE_BUTTON:
                self._stage = "open_nest"
                self._child = object()
            return
        if self._stage == "select_all" and self._filter is not None:
            self._filter.on_action_success(target_type)
            return
        if self._stage == "autoplace":
            if target_type == AUTOPLACE_BUTTON:
                self._stage = "autoplace_result"
            elif target_type == AUTOPLACE_YES:
                self._stage = "collect"
            return
        if self._stage == "collect" and target_type == COLLECT_EGGS_BUTTON:
            self._stage = "close_nest"
            return
        if self._stage == "close_nest" and target_type == NEST_MASK_CLOSE:
            self._stage = "verify_nest_closed"

    def on_action_failure(self, target_type: str) -> None:
        if self._stage == "hatch":
            self._hatch_child.on_action_failure(target_type)
            return
        # Auto-place and collect can both mutate the game even if the visual
        # transition was missed. Do not guess or send either action twice.
        if target_type in {AUTOPLACE_BUTTON, AUTOPLACE_YES, COLLECT_EGGS_BUTTON}:
            self._blocked_reason = f"unverified {target_type}"
            self._stage = "blocked"
            self.logger.error("Beginner hatch | stopped safely | %s", self._blocked_reason)

    def choose(self, frame: Frame, detections: Sequence[Detection]) -> Target | None:
        if self._blocked_reason is not None:
            return None
        by_type = _group(detections)
        startup = self._choose_startup(frame, by_type)
        if startup is not None:
            return startup

        if self._stage == "hatch":
            return self._hatch_child.choose(frame, detections)
        if self._stage == "open_nest":
            if NEST_TITLE in by_type:
                self._start_all_filter()
                return self.choose(frame, detections)
            anchor = _best(by_type.get(hatch_feature.HOME_ANCHOR))
            return _synthetic(OPEN_NEST, anchor.x, anchor.y) if anchor else None
        if self._stage == "open_nest_collect":
            if NEST_TITLE in by_type:
                self._stage = "collect"
                return self.choose(frame, detections)
            anchor = _best(by_type.get(hatch_feature.HOME_ANCHOR))
            return _synthetic(OPEN_NEST, anchor.x, anchor.y) if anchor else None
        if self._stage == "select_all":
            assert self._filter is not None
            target = self._filter.choose(frame, detections)
            if target is None and self._filter.is_complete():
                self._stage = "autoplace"
                return self.choose(frame, detections)
            return target
        if self._stage == "autoplace":
            if NEST_TITLE not in by_type or self._autoplace_requested:
                return None
            button = _best(by_type.get(AUTOPLACE_BUTTON))
            if button is None:
                return None
            self._autoplace_requested = True
            return _target(button)
        if self._stage == "autoplace_result":
            # This toast proves auto-place has nothing left to do.  The
            # collection button remains visible and usable behind it, so move
            # straight to the one guarded collect action instead of waiting
            # for the toast to disappear or trying auto-place again.
            if AUTOPLACE_UNAVAILABLE in by_type:
                self.logger.info(
                    "Beginner hatch | no nest available for auto-place | collecting all eggs"
                )
                self._stage = "collect"
                return self.choose(frame, detections)
            # The game has multiple confirmation wordings (best attributes,
            # level order, and localized variants).  Once the one allowed
            # auto-place tap has completed, a visible explicit "是" button is
            # sufficient proof of this modal; do not depend on its body text.
            yes = _best(by_type.get(CONFIRM_YES))
            if yes is not None:
                return _synthetic(AUTOPLACE_YES, yes.x, yes.y)
            if AUTOPLACE_PROMPT in by_type or AUTOPLACE_NOTICE in by_type:
                return None
            if NEST_TITLE in by_type:
                self._stage = "collect"
                return self.choose(frame, detections)
            return None
        if self._stage == "collect":
            if NEST_TITLE not in by_type or self._collect_requested:
                return None
            button = _best(by_type.get(COLLECT_EGGS_BUTTON))
            if button is None:
                return None
            self._collect_requested = True
            return _target(button)
        if self._stage == "close_nest":
            if NEST_TITLE not in by_type:
                self._stage = "verify_nest_closed"
                return self.choose(frame, detections)
            return _synthetic(
                NEST_MASK_CLOSE,
                *_scaled(frame, (50.0, 800.0), self.reference_width),
            )
        if self._stage == "verify_nest_closed":
            if NEST_TITLE in by_type:
                self._stage = "close_nest"
                return self.choose(frame, detections)
            if not is_home_screen(frame, detections):
                return None
            if self._home_collection:
                self.home_collection_rounds += 1
                self.logger.info(
                    "Beginner hatch | home collection round %d complete",
                    self.home_collection_rounds,
                )
                self._stage = "hatch"
                self._home_collection = False
                self._collect_requested = False
                return self._hatch_child.choose(frame, detections)
            self.management_rounds += 1
            self.logger.info(
                "Beginner hatch | management round %d complete | waiting for incubator",
                self.management_rounds,
            )
            self._stage = "hatch"
            self._child = self._new_hatch()
            self._hatch_child.begin_rescan_wait("beginner nest round complete")
            self._filter = None
            self._autoplace_requested = False
            self._collect_requested = False
            return None
        return None

    def _choose_startup(
        self,
        frame: Frame,
        by_type: dict[str, list[Detection]],
    ) -> Target | None:
        for target_type in STARTUP_SIMPLE_INTERRUPTS:
            interrupt = _best(by_type.get(target_type))
            if interrupt is not None:
                return _target(interrupt)
        # The lightweight auto-battle modal detector is colour/layout based.
        # A hatch-result card can share those colours, but the explicit claim
        # or expel controls prove this is not a startup overlay.  Do not let
        # the heuristic steal priority from the actual hatch result.
        hatch_result_visible = bool(
            by_type.get(hatch_feature.CLAIM_BUTTON)
            or by_type.get(hatch_feature.EXPEL_BUTTON)
        )
        auto_battle = (
            None
            if hatch_result_visible
            else _best(by_type.get(STARTUP_AUTO_BATTLE_CLOSE))
        )
        if auto_battle is not None:
            return _target(auto_battle)
        growth = (
            None
            # My Nest has the same tall white card and centred green control
            # as the launch growth-result layout.  The green control is
            # auto-place, so never synthesize the startup shortcut once the
            # explicit nest title proves that the nest is already open.
            if hatch_result_visible or NEST_TITLE in by_type
            else _best(by_type.get(STARTUP_GROWTH_RESULT))
        )
        if growth is None:
            return None
        point = (
            (450.0, 1270.0)
            if growth.metadata.get("shortcut_layout") == "centered_nest"
            else (592.0, 1265.0)
        )
        return _synthetic(STARTUP_NEST_SHORTCUT, *_scaled(frame, point, self.reference_width))

    def _start_all_filter(self) -> None:
        self._stage = "select_all"
        self._filter = nest_filter_feature.NestTagFilterTestPlanner(
            reference_width=self.reference_width,
            target_label="所有",
            target_option_type=nest_filter_feature.TAG_ALL,
            target_header_type=nest_filter_feature.TAG_HDR_ALL,
        )

    def _new_hatch(self) -> hatch_feature.HatchPlanner:
        return hatch_feature.HatchPlanner(**self._hatch_kwargs)

    @property
    def _hatch_child(self) -> hatch_feature.HatchPlanner:
        assert isinstance(self._child, hatch_feature.HatchPlanner)
        return self._child


def _group(detections: Sequence[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    for item in detections:
        grouped.setdefault(item.type, []).append(item)
    return grouped


def _best(items: list[Detection] | None) -> Detection | None:
    return max(items, key=lambda item: item.confidence) if items else None


def _scaled(
    frame: Frame,
    point: tuple[float, float],
    reference_width: float,
) -> tuple[int, int]:
    scale = frame.width / reference_width
    return round(point[0] * scale), round(point[1] * scale)


def _target(detection: Detection) -> Target:
    return Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )


def _synthetic(target_type: str, x: int, y: int) -> Target:
    detection = Detection(
        type=target_type,
        x=x,
        y=y,
        confidence=1.0,
        metadata={"synthetic": True},
    )
    return _target(detection)
