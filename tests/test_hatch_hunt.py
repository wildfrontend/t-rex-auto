from __future__ import annotations

import contextlib

from pathlib import Path

import numpy as np
import pytest

from dino_bot import hatch
from dino_bot.full_hatch import (
    NEST_MASK_CLOSE,
    STARTUP_GROWTH_RESULT,
    STARTUP_NEST_SHORTCUT,
)
from dino_bot.hatch_hunt import (
    MAX_ANCHOR_ONLY_HANDOFF_FRAMES,
    MAX_HANDOFF_PROGRESS_EXTENSIONS,
    MAX_HOME_ANCHOR_SETTLE_FRAMES,
    HatchHuntPlanner,
)
from dino_bot.models import BoundingBox, Detection, Frame, Target, VerificationResult
from dino_bot.parent_open import NEST_TITLE


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
    reference_width = 900.0

    def __init__(self, cooldown_ms: int) -> None:
        self.cooldown_ms = cooldown_ms
        self.next_target: Target | None = None
        self.successes: list[str] = []
        self.blocked = False
        self.continue_hunting = True
        self.success_contexts: list[tuple[str, str]] = []
        self.camera_refresh_available = False
        self.camera_refresh_active = False
        self.camera_refresh_started = 0
        self.camera_refresh_completed = 0
        self.camera_refresh_failed: list[str] = []

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

    def continue_hunting_when_blocked(self) -> bool:
        return self.continue_hunting

    def begin_hunt_map_capacity_refresh(self) -> bool:
        if not self.camera_refresh_available or not self.blocked:
            return False
        self.camera_refresh_available = False
        self.camera_refresh_started += 1
        self.camera_refresh_active = True
        self.blocked = False
        return True

    def complete_hunt_map_capacity_refresh(self) -> bool:
        if not self.camera_refresh_active:
            return False
        self.camera_refresh_active = False
        self.camera_refresh_completed += 1
        return True

    def fail_hunt_map_capacity_refresh(self, reason: str) -> bool:
        self.camera_refresh_active = False
        self.camera_refresh_failed.append(reason)
        self.blocked = True
        return True

    def on_action_success(self, target_type: str) -> None:
        self.successes.append(target_type)

    def on_action_success_context(
        self,
        target: Target,
        frame: Frame,
        detections: list[Detection],
        result: VerificationResult,
    ) -> None:
        self.success_contexts.append((target.type, result.reason))

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
    # The hunt scan carries NEST_TITLE so a stray My Nest panel is visible.
    assert combined.planning_detection_types() == frozenset(
        {"dinosaur", NEST_TITLE}
    )

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


def test_hatch_mode_forwards_verified_success_context() -> None:
    combined, hatch_planner, _ = planner(cooldown_ms=0)
    selected = target("hatch_candidate_row", 350, 435)
    result = VerificationResult(True, "previous UI disappeared: hatch_select_title")
    hatch_planner.next_target = selected

    assert combined.choose(frame(), []) == selected
    combined.on_action_success_context(selected, frame(), [], result)

    assert hatch_planner.success_contexts == [(selected.type, result.reason)]


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
    # The hunt scan carries NEST_TITLE so a stray My Nest panel is visible.
    assert combined.planning_detection_types() == frozenset(
        {"dinosaur", NEST_TITLE}
    )
    assert not combined.is_complete()


def test_capacity_refresh_round_trip_enters_and_leaves_hunt_without_hunting() -> None:
    combined, hatch_planner, hunt_planner = planner(cooldown_ms=0)
    hatch_planner.blocked = True
    hatch_planner.camera_refresh_available = True
    hunt_planner.next_target = target("forest_recenter_button", 841, 1296)

    # The only outbound action is the map switch, never a dinosaur target.
    outbound = combined.choose(frame(), [])
    assert outbound is not None and outbound.type == "forest_recenter_button"
    assert combined.workflow_status()["stage"] == "capacity_camera_refresh"
    assert {"hatch_button", "dinosaur"} <= set(
        combined.planning_detection_types() or ()
    )
    combined.on_action_success(outbound.type)

    # As soon as hunt-map evidence arrives, hand straight back to home instead
    # of allowing another hunt planner action.
    hunt_planner.next_target = target("map_exit_nest_button", 841, 1295)
    inbound = combined.choose(
        frame(), [detection("map_exit_nest_button", 841, 1295)]
    )
    assert inbound is not None and inbound.type == "map_exit_nest_button"
    assert hunt_planner.recenter_requests == ["capacity camera refresh"]
    combined.on_action_success(inbound.type)

    hatch_planner.next_target = target("hatch_button", 450, 1330)
    centered = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert combined.choose(frame(), centered) is None
    resumed = combined.choose(frame(), centered)
    assert resumed is not None and resumed.type == "hatch_button"
    assert hatch_planner.camera_refresh_started == 1
    assert hatch_planner.camera_refresh_completed == 1
    assert "dinosaur" not in hunt_planner.successes


def test_safety_blocked_hatch_stops_the_combined_workflow() -> None:
    combined, hatch_planner, hunt_planner = planner(cooldown_ms=0)
    hatch_planner.blocked = True
    hatch_planner.continue_hunting = False
    hunt_planner.next_target = target("dinosaur", 300, 700)

    assert combined.is_complete()
    assert combined.choose(frame(), []) is None


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


def test_handoff_arms_optional_home_collection_before_resuming_hatch() -> None:
    combined, hatch_planner, _ = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 0
    hatch_planner.next_target = target("collect_after_home", 640, 1315)
    calls: list[bool] = []
    hatch_planner.begin_home_collection = lambda: calls.append(True) or True
    centered = [detection(hatch.HOME_ANCHOR, 59, 561)]

    assert combined.choose(frame(), centered) is None
    chosen = combined.choose(frame(), centered)

    assert calls == [True]
    assert chosen is not None and chosen.type == "collect_after_home"


@pytest.mark.parametrize("boost_delay", [0, 30_000])
def test_boost_deadline_does_not_interrupt_hunt_or_shorten_wait(boost_delay):
    combined, full, hunt = planner(cooldown_ms=3_600_000)
    full.boost_ready_delay_ms = lambda: boost_delay
    hunt.next_target = target("dinosaur", 300, 700)
    combined.choose(frame(), [])
    assert combined.choose(frame(), []).type == "dinosaur"
    assert combined._mode == "hunt"
    hunt.delay_ms = 60_000
    assert combined.next_ready_delay_ms() == 60_000
    assert not hunt.recenter_requests
    full.cooldown_ms = 0
    combined.choose(frame(), [])
    assert combined._mode == "handoff"
    assert combined._handoff_reason == "cooldown"


@pytest.mark.parametrize("fuse", ["_egg_pile_blocked", "_screening_blocked", "_capacity_blocked"])
def test_blocked_hatch_does_not_leave_hunt_for_boost(tmp_path, fuse):
    from dino_bot.digits import DigitReader
    from dino_bot.full_hatch import FullHatchPlanner
    from dino_bot.hatch_inventory import HatchBoostInventoryStore

    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    inventory.set_enabled(True)
    full = FullHatchPlanner(
        DigitReader(Path(__file__).parents[1] / "assets/hatch/digits"),
        egg_pile_point=(450, 1330), boost_inventory=inventory,
    )
    setattr(full, fuse, True)
    full._stage = "hatch_blocked"
    child = full._child
    hunt = StubHunt()
    hunt.next_target = target("dinosaur", 300, 700)
    hunt.delay_ms = 3_600_000
    combined = HatchHuntPlanner(full, hunt)
    combined._mode = "hunt"
    assert combined.choose(frame(), []).type == "dinosaur"
    assert combined.next_ready_delay_ms() == 3_600_000
    assert combined.workflow_status()["stage"] == "hatch_blocked_hunt"
    assert full._child is child and getattr(full, fuse)
    assert inventory.snapshot().remaining == 100


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


def test_stray_nest_panel_during_hunt_is_closed_then_handed_back() -> None:
    combined, hatch_planner, hunt_planner = planner()
    hunt_planner.next_target = target("dinosaur", 300, 700)

    chosen = combined.choose(frame(), [])
    assert chosen is not None and chosen.type == "dinosaur"

    # 巢面板意外開著:它蓋住地圖上的每個狩獵控制項,所以關閉遮罩要勝過
    # 狩獵側提出的目標。
    panel = [detection(NEST_TITLE, 450, 240)]
    chosen = combined.choose(frame(), panel)
    assert chosen is not None and chosen.type == NEST_MASK_CLOSE
    assert (chosen.x, chosen.y) == (50, 800)

    # 這一下屬於孵蛋側的字彙,回呼要送到孵蛋 planner。
    combined.on_action_success(chosen.type)
    assert hatch_planner.successes == [NEST_MASK_CLOSE]

    # 面板關掉後把控制權交還孵蛋側重新判斷冷卻,而不是原地接著狩獵。
    hatch_planner.next_target = target("hatch_button", 450, 800)
    chosen = combined.choose(frame(), [])
    assert chosen is not None and chosen.type == "hatch_button"
    assert combined._mode == "hatch"


def test_handoff_that_never_centers_gives_up_instead_of_toggling_forever() -> None:
    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]

    assert combined.choose(frame(), map_view) is not None
    assert combined._mode == "handoff"

    # 期限一過就不再敲同一組按鈕:冷卻交棒是非做不可的,所以交給孵蛋側
    # 自己的恢復流程,而不是留在原地空轉。
    now[0] = 95.0
    recoveries: list[str] = []
    hatch_planner.begin_home_recovery = lambda reason: recoveries.append(reason) or True
    hatch_planner.next_target = target("hatch_button", 450, 800)

    chosen = combined.choose(frame(), map_view)

    assert len(recoveries) == 1
    assert chosen is not None and chosen.type == "hatch_button"
    assert combined._mode == "hatch"


def test_verified_step_towards_home_extends_the_handoff_deadline() -> None:
    """S9 trace: a working exit sequence was guillotined 2s in.

    The exit was planned at 10:28:55, verified at 10:28:57 and the deadline
    fired at 10:29:04, throwing away progress that was two taps from home.
    A verified step has to buy the sequence time to finish.
    """

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]

    assert combined.choose(frame(), map_view) is not None
    assert combined._mode == "handoff"

    # 88 秒後才點到離開鍵並驗證成功:原本再 2 秒就會被砍掉。
    now[0] = 88.0
    combined.on_action_success("map_exit_nest_button")

    now[0] = 95.0
    assert not combined._handoff_expired()
    assert combined._mode == "handoff"

    # 期限只是延後,不是取消。
    now[0] = 119.0
    assert combined._handoff_expired()


def test_a_visible_hatch_home_is_given_time_to_settle() -> None:
    """Seeing the home anchor must not trigger another camera move.

    The cloud wipe between the cave and the home map covers the frame for a
    few cycles. Tapping Forest again during it restarts the wipe, so the
    geometric proof never gets a clean frame: on s9 this lost all seven
    handoffs and eventually fused the hatch side off.
    """

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("forest_recenter_button", 841, 1296)
    assert combined.choose(frame(), [detection("map_exit_nest_button", 840, 1295)])

    home_view = [
        detection("hatch_home_anchor", 450, 800),
        detection("forest_recenter_button", 841, 1296),
    ]
    recentres: list[str] = []
    hunt_planner.request_external_recenter = lambda reason: recentres.append(reason)

    # While the anchor is visible the handoff waits instead of tapping.
    for _ in range(MAX_HOME_ANCHOR_SETTLE_FRAMES):
        assert combined.choose(frame(), home_view) is None
    assert recentres == []


@contextlib.contextmanager
def _home_proof(*, bright: bool, pile: bool, offset: tuple[float, float] | None):
    """Drive the three centred-home gates without rendering a real map."""

    import dino_bot.hatch_hunt as module

    saved = (
        module._is_bright_outdoor_map,
        module.hatch_feature.has_home_pile_structure,
        module.home_pile_offset,
        module.is_centered_home_screen,
    )
    module._is_bright_outdoor_map = lambda _frame: bright
    module.hatch_feature.has_home_pile_structure = lambda *a, **k: pile
    module.home_pile_offset = lambda _frame: offset
    # A home that is off-centre by more than the tolerance is, by definition,
    # not yet a centred home; without this the handoff completes on the second
    # frame and the settle path under test is never reached.
    module.is_centered_home_screen = lambda _frame, _detections: False
    try:
        yield
    finally:
        (
            module._is_bright_outdoor_map,
            module.hatch_feature.has_home_pile_structure,
            module.home_pile_offset,
            module.is_centered_home_screen,
        ) = saved


def test_an_off_centre_home_goes_to_recovery_not_another_recentre() -> None:
    """Recentring cannot close a measured pile offset; the hatch side can.

    s9 logged the identical (-4,146) residue on two separate handoffs after
    recentring, while hatch recovery's measured drag closes exactly that gap
    and had already rescued a cooldown handoff the same afternoon.
    """

    combined, hatch_planner, hunt_planner = planner()
    combined._handoff_reason = "errand"
    combined._mode = "handoff"
    recoveries: list[str] = []
    hatch_planner.begin_home_recovery = lambda reason: recoveries.append(reason) or True

    with _home_proof(bright=True, pile=True, offset=(-4.0, 146.0)):
        handed = combined._offer_home_to_recovery(frame(), [])

    assert handed is True
    assert len(recoveries) == 1
    assert combined._mode == "hatch"
    assert combined._handoff_deadline is None


def test_a_settled_off_centre_home_reaches_recovery_through_choose() -> None:
    """The handover must be wired into the handoff, not just callable.

    Covers the path an operator actually hits: the map arrives, the settle
    wait expires, and the next decision has to be recovery rather than one
    more recenter.
    """

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("forest_recenter_button", 841, 1296)
    assert combined.choose(frame(), [detection("map_exit_nest_button", 840, 1295)])

    recoveries: list[str] = []
    hatch_planner.begin_home_recovery = lambda reason: recoveries.append(reason) or True
    hatch_planner.next_target = target("hatch_recovery_recenter", 452, 727)
    recentres: list[str] = []
    hunt_planner.request_external_recenter = lambda reason: recentres.append(reason)
    home_view = [
        detection("hatch_home_anchor", 450, 800),
        detection("forest_recenter_button", 841, 1296),
    ]

    with _home_proof(bright=True, pile=True, offset=(-4.0, 146.0)):
        for _ in range(MAX_HOME_ANCHOR_SETTLE_FRAMES):
            combined.choose(frame(), home_view)
        chosen = combined.choose(frame(), home_view)

    assert len(recoveries) == 1
    assert recentres == []
    assert chosen is not None and chosen.type == "hatch_recovery_recenter"


def test_a_cave_frame_is_not_handed_to_recovery() -> None:
    """No visible pile means the map never left the cave - a different fault.

    Recovery's correction is a drag measured against the pile, so a frame
    without one gives it nothing to aim at; that case keeps the recenter path.
    """

    combined, hatch_planner, hunt_planner = planner()
    combined._handoff_reason = "errand"
    combined._mode = "handoff"
    recoveries: list[str] = []
    hatch_planner.begin_home_recovery = lambda reason: recoveries.append(reason) or True

    with _home_proof(bright=False, pile=False, offset=None):
        assert combined._offer_home_to_recovery(frame(), []) is False
    with _home_proof(bright=True, pile=False, offset=None):
        assert combined._offer_home_to_recovery(frame(), []) is False

    assert recoveries == []
    assert combined._mode == "handoff"


def test_alternating_exit_and_recentre_buys_no_extension() -> None:
    """Two buttons that summon each other are not progress towards home.

    Exit and recentre share the map corner: tapping one reveals the other. On
    s9 that alternation was verified nine times in 90s, spent all three
    extensions, and kept the handoff alive on a map that never moved.
    """

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]
    assert combined.choose(frame(), map_view) is not None

    for step in range(8):
        now[0] = 10.0 + step * 5.0
        combined.on_action_success(
            "map_exit_nest_button" if step % 2 else "forest_recenter_button"
        )

    assert combined._handoff_extensions == 0


def test_repeating_one_step_still_counts_as_progress() -> None:
    """Retrying a single real step is not the ping-pong this guards against."""

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]
    assert combined.choose(frame(), map_view) is not None

    now[0] = 80.0
    combined.on_action_success("map_exit_nest_button")
    now[0] = 100.0
    combined.on_action_success("map_exit_nest_button")

    assert combined._handoff_extensions == 2


def test_handoff_extension_is_capped_so_a_toggling_map_still_gives_up() -> None:
    """A map that keeps tapping without ever centring is still a stall."""

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]
    assert combined.choose(frame(), map_view) is not None

    # 一直在離開/置中之間來回,但永遠到不了首頁。
    for step in range(10):
        now[0] = 80.0 + step * 20.0
        combined.on_action_success("map_exit_nest_button")

    assert combined._handoff_extensions == MAX_HANDOFF_PROGRESS_EXTENSIONS

    recoveries: list[str] = []
    hatch_planner.begin_home_recovery = lambda reason: recoveries.append(reason) or True
    hatch_planner.next_target = target("hatch_button", 450, 800)
    chosen = combined.choose(frame(), map_view)

    assert len(recoveries) == 1
    assert chosen is not None and chosen.type == "hatch_button"
    assert combined._mode == "hatch"


def test_unrelated_taps_do_not_extend_the_handoff_deadline() -> None:
    """Only steps towards home count; a hunt tap must not renew the window."""

    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    map_view = [detection("map_exit_nest_button", 840, 1295)]
    assert combined.choose(frame(), map_view) is not None

    now[0] = 88.0
    combined.on_action_success("dinosaur")

    assert combined._handoff_extensions == 0
    now[0] = 95.0
    assert combined._handoff_expired()


def test_timed_out_errand_is_abandoned_back_to_hunting() -> None:
    now = [0.0]
    combined, hatch_planner, hunt_planner = planner()
    combined.clock = lambda: now[0]
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []) is not None

    aborts: list[str] = []
    hatch_planner.begin_interim_collection = lambda: True
    hatch_planner.abort_interim_collection = (
        lambda reason: aborts.append(reason) or True
    )
    hunt_planner.next_target = None
    hunt_planner.delay_ms = 30_000
    combined.choose(frame(), [])
    assert combined._mode == "handoff"

    # 差事是可選的,期限到了就退回狩獵,並把下一次差事往後推。
    now[0] = 95.0
    hunt_planner.next_target = target("dinosaur", 300, 700)
    chosen = combined.choose(frame(), [])

    assert len(aborts) == 1
    assert chosen is not None and chosen.type == "dinosaur"
    assert combined._mode == "hunt"


def test_nest_panel_that_never_closes_hands_back_to_hatch_recovery() -> None:
    combined, hatch_planner, hunt_planner = planner()
    hunt_planner.next_target = target("dinosaur", 300, 700)
    assert combined.choose(frame(), []) is not None

    panel = [detection(NEST_TITLE, 450, 240)]
    for _ in range(combined.nest_close_attempt_limit):
        chosen = combined.choose(frame(), panel)
        assert chosen is not None and chosen.type == NEST_MASK_CLOSE

    # 敲不掉的面板是這個 planner 讀不懂的畫面:交給孵蛋側的恢復流程,
    # 而不是無限地敲遮罩。
    hatch_planner.next_target = target("hatch_button", 450, 800)
    chosen = combined.choose(frame(), panel)
    assert chosen is not None and chosen.type == "hatch_button"
    assert combined._mode == "hatch"


def test_handoff_leaves_a_map_parked_with_its_centre_egg_centred() -> None:
    """S9 trace: a centred hunt egg stalled every handoff for its full deadline.

    The centre egg proves the hunt map is in a normal state, never that the
    hatch home is reachable, so a map parked with it centred satisfied the
    "wait one more frame" test on every frame.  The observed run sat 13px from
    centre and burned 90s per handoff in a 152s loop (91s stalled, 61s
    hunting) while the nest button it needed was visible the whole time.
    """

    combined, hatch_planner, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    # The measured live frame: egg centre (450, 787) on a 900x1600 screen is
    # 13px from centre, and the nest button is visible alongside it.
    parked = [
        detection("map_center_egg", 450, 787),
        detection("map_exit_nest_button", 840, 1295),
        detection("mailbox_button", 841, 1208),
    ]

    # A brief wait is still allowed: a transient missed HUD anchor is normal.
    for _ in range(MAX_ANCHOR_ONLY_HANDOFF_FRAMES):
        assert combined.choose(frame(), parked) is None

    # But it must then leave the map by its own controls rather than waiting
    # out the deadline.
    chosen = combined.choose(frame(), parked)
    assert chosen is not None and chosen.type == "map_exit_nest_button"
    assert hunt_planner.recenter_requests == ["hatch cooldown handoff"]


def test_handoff_anchor_wait_survives_a_dropped_egg_detection() -> None:
    combined, hatch_planner, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    centred = [detection("map_center_egg", 450, 787)]
    # The map is unchanged; the animated egg merely failed to match this frame.
    dropped = [detection("dinosaur", 300, 700)]

    assert combined.choose(frame(), centred) is None
    assert combined.choose(frame(), dropped) is None
    assert combined.choose(frame(), centred) is None
    assert combined._anchor_only_frames == 3

    # A fourth frame passes the bound, so the handoff acts on the map instead
    # of sitting on it until the deadline fires.
    assert combined.choose(frame(), centred) is not None


def test_handoff_anchor_wait_resets_when_the_egg_leaves_the_centre() -> None:
    """The bounded wait is for consecutive frames, not a lifetime budget."""

    combined, hatch_planner, hunt_planner = planner()
    assert combined.choose(frame(), []) is None
    hatch_planner.cooldown_ms = 20_000
    hunt_planner.next_target = target("map_exit_nest_button", 840, 1295)
    centred = [detection("map_center_egg", 450, 787)]
    off_centre = [
        detection("map_center_egg", 450, 200),
        detection("map_exit_nest_button", 840, 1295),
    ]

    assert combined.choose(frame(), centred) is None
    # An off-centre egg is not the stalling case, so the streak restarts.
    assert combined.choose(frame(), off_centre) is not None
    for _ in range(MAX_ANCHOR_ONLY_HANDOFF_FRAMES):
        assert combined.choose(frame(), centred) is None
    assert combined.choose(frame(), centred) is not None


def test_population_stop_keeps_hunting_without_cooldown_handoff():
    combined, full, hunt = planner(cooldown_ms=0)
    full.blocked = True
    full.population_limit_reached = True
    full.next_target = target(hatch.HATCH_LABEL, 270, 436)
    hunt.next_target = target("dinosaur", 300, 700)
    hunt.delay_ms = 60_000
    for _ in range(3):
        assert combined.choose(frame(), []).type == "dinosaur"
        assert not combined.is_complete()
        assert combined.next_ready_delay_ms() == 60_000
    assert combined.workflow_status()["stage"] == "population_limit_hunt"
    assert combined.workflow_status()["cooldown_remaining_seconds"] is None
    assert not hunt.recenter_requests


def test_failed_capacity_read_hunts_then_hands_back_for_a_fresh_preflight():
    from dino_bot.digits import DigitReader
    from dino_bot.full_hatch import PANEL_OPEN, FullHatchPlanner

    now = [1000.0]
    full = FullHatchPlanner(
        DigitReader(Path(__file__).parents[1] / 'assets/hatch/digits'),
        egg_pile_point=(450, 1330), enabled_stages=('collect', 'hatch'),
        clock=lambda: now[0],
    )
    full._begin_capacity_preflight('test unreadable capacity after returning home')
    full._capacity_camera_refresh_used = True
    full._capacity_child._complete = True
    full._capacity_child._capacity_readable = False
    hunt = StubHunt()
    hunt.next_target = target('dinosaur', 300, 700)
    hunt.delay_ms = 60_000
    combined = HatchHuntPlanner(full, hunt, clock=lambda: now[0])
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert combined.choose(frame(), home).type == 'dinosaur'
    state = combined.workflow_status()
    assert state['stage'] == 'capacity_retry_hunt'
    assert state['hatch_blocked'] is False
    assert state['cooldown_remaining_seconds'] == 600
    assert not combined.is_complete()
    assert combined.next_ready_delay_ms() == 60_000
    assert combined.choose(frame(), home).type == 'dinosaur'
    assert not hunt.recenter_requests
    now[0] += 601
    assert combined.choose(frame(), home) is None
    chosen = combined.choose(frame(), home)
    assert chosen.type == PANEL_OPEN
    assert full._stage == 'capacity_preflight'
    assert not full._capacity_checked


def test_custom_camera_refresh_failure_to_read_flows_into_periodic_hunting_retry():
    from dino_bot.digits import DigitReader
    from dino_bot.full_hatch import PANEL_OPEN, FullHatchPlanner

    full = FullHatchPlanner(
        DigitReader(Path(__file__).parents[1] / 'assets/hatch/digits'),
        egg_pile_point=(450, 1330), enabled_stages=('collect', 'hatch'),
    )
    hunt = StubHunt()
    combined = HatchHuntPlanner(full, hunt)
    full._begin_capacity_preflight('test first failed read')
    full._capacity_child._complete = True
    full._capacity_child._capacity_readable = False
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert combined.choose(frame(), home) is None
    assert full.is_hatch_blocked() and not full.capacity_retry_pending
    hunt.next_target = target('forest_recenter_button', 841, 1296)
    assert combined.choose(frame(), home).type == 'forest_recenter_button'
    assert combined.workflow_status()['stage'] == 'capacity_camera_refresh'
    combined.on_action_success('forest_recenter_button')
    hunt.next_target = target('map_exit_nest_button', 841, 1295)
    assert combined.choose(frame(), [detection('map_exit_nest_button', 841, 1295)]).type == (
        'map_exit_nest_button'
    )
    combined.on_action_success('map_exit_nest_button')
    assert combined.choose(frame(), home) is None
    assert combined.choose(frame(), home).type == PANEL_OPEN
    full._capacity_child._complete = True
    full._capacity_child._capacity_readable = False
    hunt.next_target = target('dinosaur', 300, 700)
    assert combined.choose(frame(), home).type == 'dinosaur'
    assert combined.workflow_status()['stage'] == 'capacity_retry_hunt'
    assert full.capacity_retry_pending and not full.is_hatch_blocked()
    assert full._capacity_checked is False
    assert not full.begin_hunt_map_capacity_refresh()
