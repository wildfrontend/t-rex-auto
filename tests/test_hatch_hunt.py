from __future__ import annotations

import numpy as np

from dino_bot import hatch
from dino_bot.full_hatch import STARTUP_GROWTH_RESULT, STARTUP_NEST_SHORTCUT
from dino_bot.hatch_hunt import HatchHuntPlanner
from dino_bot.models import BoundingBox, Detection, Frame, Target, VerificationResult


def frame() -> Frame:
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1448:1460, 330:573] = (220, 180, 20)
    return Frame(image)


def detection(target_type: str, x: int, y: int) -> Detection:
    return Detection.from_bbox(
        target_type,
        BoundingBox(x - 5, y - 5, 10, 10),
        1.0,
    )


def target(target_type: str, x: int, y: int) -> Target:
    item = detection(target_type, x, y)
    return Target(target_type, x, y, 1.0, item)


class StubHatch:
    def __init__(self, cooldown_ms: int) -> None:
        self.cooldown_ms = cooldown_ms
        self.next_target: Target | None = None
        self.successes: list[str] = []
        self.blocked = False

    def choose(self, frame: Frame, detections: list[Detection]) -> Target | None:
        return self.next_target

    def next_ready_delay_ms(self) -> int:
        return self.cooldown_ms

    def planning_detection_types(self) -> frozenset[str]:
        return frozenset({"hatch_button"})

    def is_hunt_cooldown_active(self) -> bool:
        return self.cooldown_ms > 0

    def is_hatch_blocked(self) -> bool:
        return self.blocked

    def on_action_success(self, target_type: str) -> None:
        self.successes.append(target_type)

    def reset_workflow(self) -> None:
        self.cooldown_ms = 0

    def last_stage(self) -> str:
        return "stub_hatch"


class StubHunt:
    completion_type = "hunt_confirm_button"
    center_anchor_type = "map_center_egg"
    map_exit_type = "map_exit_nest_button"
    forest_recenter_type = "forest_recenter_button"
    mailbox_type = "mailbox_button"
    dinosaur_type = "dinosaur"
    hunt_button_types = frozenset({"hunt_button", "hunt_confirm_button"})
    own_path_types = frozenset({"own_hunt_path"})

    def __init__(self) -> None:
        self.next_target: Target | None = None
        self.delay_ms = 0
        self.planning_types: frozenset[str] | None = frozenset({self.dinosaur_type})
        self.full_types = frozenset(
            {self.dinosaur_type, "duplicate_login_close_button"}
        )
        self.successes: list[str] = []
        self.recenter_requests: list[str] = []
        self.reset_count = 0

    def choose(self, frame: Frame, detections: list[Detection]) -> Target | None:
        return self.next_target

    def next_ready_delay_ms(self) -> int:
        return self.delay_ms

    def request_external_recenter(self, reason: str) -> None:
        self.recenter_requests.append(reason)

    def on_action_success(self, target_type: str) -> None:
        self.successes.append(target_type)

    def reset_workflow(self) -> None:
        self.reset_count += 1

    def planning_detection_types(self) -> frozenset[str] | None:
        return self.planning_types

    def full_detection_types(self) -> frozenset[str]:
        return self.full_types

    def should_finalize_verification_early(
        self,
        target: Target,
        result: VerificationResult,
        checks: int,
    ) -> bool:
        return target.type == "hunt_button" and result.pixel_change == 0 and checks >= 2

    def last_stage(self) -> str:
        return "stub_hunt"

    def take_blind_escape(self):
        return None

    def last_rejections(self) -> dict[str, int]:
        return {}

    def last_idle_seconds(self) -> float:
        return 0.0

    def last_recenter_reason(self):
        return None

    def last_blind_seconds(self) -> float:
        return 0.0

    def last_supply(self) -> int:
        return 0

    def anchor_measured(self) -> bool:
        return False


def planner(cooldown_ms: int = 600_000) -> tuple[HatchHuntPlanner, StubHatch, StubHunt]:
    hatch_planner = StubHatch(cooldown_ms)
    hunt_planner = StubHunt()
    combined = HatchHuntPlanner(  # type: ignore[arg-type]
        hatch_planner,
        hunt_planner,
        handoff_seconds=30,
    )
    return combined, hatch_planner, hunt_planner


def test_long_hatch_cooldown_switches_to_hunt_and_keeps_action_owner() -> None:
    combined, _, hunt_planner = planner()
    hunt_planner.next_target = target("dinosaur", 300, 700)

    chosen = combined.choose(frame(), [])
    assert chosen is not None and chosen.type == "dinosaur"
    assert combined.planning_detection_types() == frozenset({"dinosaur"})

    combined.on_action_success(chosen.type)
    assert hunt_planner.successes == ["dinosaur"]

    inert_button = target("hunt_button", 301, 546)
    assert combined.should_finalize_verification_early(
        inert_button,
        VerificationResult(False, "no change", pixel_change=0.0),
        2,
    )


def test_hatch_mode_uses_hatch_scoped_detection_types() -> None:
    combined, _, _ = planner(cooldown_ms=0)

    assert combined.planning_detection_types() == frozenset({"hatch_button"})


def test_hunt_full_scan_does_not_include_inactive_hatch_templates() -> None:
    combined, _, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hunt_planner.planning_types = None

    requested = combined.planning_detection_types()

    assert requested == hunt_planner.full_types
    assert "hatch_button" not in requested


def test_blocked_hatch_falls_back_to_hunting_in_combined_mode() -> None:
    combined, hatch_planner, hunt_planner = planner(cooldown_ms=0)
    hatch_planner.blocked = True
    hunt_planner.next_target = target("dinosaur", 300, 700)

    chosen = combined.choose(frame(), [])

    assert chosen is not None and chosen.type == "dinosaur"
    assert combined.planning_detection_types() == frozenset({"dinosaur"})
    assert not combined.is_complete()


def test_short_cooldown_is_reserved_for_handoff_without_starting_hunt() -> None:
    combined, _, hunt_planner = planner(cooldown_ms=20_000)
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []) is None
    assert hunt_planner.successes == []


def test_hunt_wait_never_sleeps_past_hatch_handoff_deadline() -> None:
    combined, _, hunt_planner = planner(cooldown_ms=100_000)
    hunt_planner.delay_ms = 120_000
    assert combined.choose(frame(), []) is None
    assert combined.next_ready_delay_ms() == 70_000


def test_handoff_requests_recenter_when_hunt_map_is_visible() -> None:
    combined, hatch_planner, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)

    chosen = combined.choose(
        frame(),
        [detection("map_exit_nest_button", 840, 1295)],
    )
    assert chosen is not None and chosen.type == "map_exit_nest_button"
    assert hunt_planner.recenter_requests == ["hatch cooldown handoff"]
    assert {"hatch_button", "dinosaur"} <= set(
        combined.planning_detection_types() or ()
    )


def test_handoff_requires_two_centered_frames_before_resuming_hatch() -> None:
    combined, hatch_planner, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 0
    hatch_planner.next_target = target(hatch.EGG_PILE, 450, 1330)
    centered = [detection(hatch.HOME_ANCHOR, 59, 561)]

    assert combined.choose(frame(), centered) is None
    chosen = combined.choose(frame(), centered)
    assert chosen is not None and chosen.type == hatch.EGG_PILE
    assert hunt_planner.reset_count == 1


def test_startup_interruption_during_hunt_restarts_hatch_first() -> None:
    combined, hatch_planner, hunt_planner = planner()
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []).type == "dinosaur"  # type: ignore[union-attr]

    hatch_planner.next_target = target(STARTUP_NEST_SHORTCUT, 592, 1265)
    chosen = combined.choose(
        frame(),
        [detection(STARTUP_GROWTH_RESULT, 307, 1265)],
    )

    assert chosen is not None and chosen.type == STARTUP_NEST_SHORTCUT
    assert hatch_planner.cooldown_ms == 0
    assert hunt_planner.reset_count == 1


def test_hunt_idle_window_triggers_interim_collection_errand() -> None:
    combined, hatch_planner, hunt_planner = planner()
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []) is not None  # 進入 hunt 模式

    calls: list[bool] = []
    hatch_planner.begin_interim_collection = lambda: calls.append(True) or True

    hunt_planner.next_target = None
    hunt_planner.delay_ms = 30_000  # 狩獵側全目標冷卻中
    combined.choose(frame(), [])
    assert calls == [True]
    assert combined._mode == "handoff"


def test_hunt_idle_errand_skipped_when_handback_is_near() -> None:
    combined, hatch_planner, hunt_planner = planner(cooldown_ms=100_000)
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []) is not None

    calls: list[bool] = []
    hatch_planner.begin_interim_collection = lambda: calls.append(True) or True

    hunt_planner.next_target = None
    hunt_planner.delay_ms = 30_000
    combined.choose(frame(), [])
    # 距離正式交棒不到 margin(30s handoff + 90s),不值得跑差事。
    assert calls == []
    assert combined._mode == "hunt"
