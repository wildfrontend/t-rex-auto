from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dino_bot import hatch, nest_filter
from dino_bot.cull import CAPACITY_REGION
from dino_bot.digits import DigitReader
from dino_bot.full_hatch import (
    AUTOPLACE_BUTTON,
    AUTOPLACE_MASK_CLOSE,
    AUTOPLACE_PROMPT,
    AUTOPLACE_SORT_HEADER,
    AUTOPLACE_TITLE,
    AUTOPLACE_YES,
    CAVE_CLOSE_BUTTON,
    CAVE_CONTINUOUS_BUTTON,
    CAVE_RECENTER,
    CAVE_SELECT_BUTTON,
    CAVE_SWIPE,
    COLLECT_EGGS_BUTTON,
    HATCH_BOOST_BUTTON,
    HATCH_BOOST_CONFIRM,
    HATCH_DETAIL_CLOSE,
    NEST_GEAR,
    NEST_MASK_CLOSE,
    OPEN_NEST,
    PLACE_HDR_BEST,
    PLACE_HDR_LEVEL,
    PLACE_SORT_BEST,
    PLACE_SORT_LEVEL,
    RECOVERY_BACK,
    RECOVERY_FOREST,
    RECOVERY_MAP_EXIT,
    RECOVERY_MASK_CLOSE,
    RECOVERY_NO,
    RECOVERY_RECENTER,
    SELECT_CHOOSE_BUTTON,
    SELECT_WEAKEST_BUTTON,
    STARTUP_AUTO_BATTLE_CLOSE,
    STARTUP_GROWTH_RESULT,
    STARTUP_NEST_SHORTCUT,
    AutoPlaceRoundPlanner,
    CaveCullPlanner,
    FullHatchPlanner,
    HatchHomeRecoveryPlanner,
    is_centered_home_screen,
    is_home_screen,
)
from dino_bot.hatch_inventory import HatchBoostInventoryStore
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.nests import MASS_RULE, TOP_RULE
from dino_bot.overlays import CONFIRM_NO, CONFIRM_YES, SELECT_CONFIRM_PROMPT
from dino_bot.parent_open import NEST_TITLE, SELECT_TITLE

REPO = Path(__file__).resolve().parent.parent
GLYPHS = REPO / "assets" / "hatch" / "digits"
FIXTURES = REPO / "tests" / "fixtures" / "hatch"


def frame(image: np.ndarray | None = None) -> Frame:
    if image is None:
        image = np.full((1600, 900, 3), 255, dtype=np.uint8)
        # Stable cyan base line of the centred home egg pile.
        image[1448:1460, 330:573] = (220, 180, 20)
    return Frame(
        image
    )


def detection(target_type: str, x: int = 100, y: int = 100) -> Detection:
    return Detection.from_bbox(target_type, BoundingBox(x - 5, y - 5, 10, 10), 0.99)


def capacity_frame() -> Frame:
    image = frame().image.copy()
    crop = cv2.imread(str(FIXTURES / "hud_282_350.png"))
    assert crop is not None
    x0, y0 = int(CAPACITY_REGION[0]), int(CAPACITY_REGION[1])
    image[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return frame(image)


def test_top_autoplace_round_requires_screen_anchors_and_known_prompt(caplog) -> None:
    planner = AutoPlaceRoundPlanner(TOP_RULE)
    nest = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_TOP, 228, 168),
    ]

    target = planner.choose(frame(), nest)
    assert target is not None and target.type == nest_filter.FILTER_HEADER
    planner.on_action_success(target.type)

    target = planner.choose(frame(), nest + [detection(nest_filter.TAG_TOP, 228, 297)])
    assert target is not None and target.type == nest_filter.TAG_TOP
    planner.on_action_success(target.type)

    target = planner.choose(frame(), nest + [detection(NEST_GEAR, 210, 276)])
    assert target is not None and target.type == NEST_GEAR
    planner.on_action_success(target.type)

    settings = [
        detection(AUTOPLACE_TITLE, 450, 565),
        detection(PLACE_SORT_BEST, 450, 702),
    ]
    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_BEST
    planner.on_action_success(target.type)

    target = planner.choose(frame(), settings + [detection(PLACE_HDR_BEST, 450, 648)])
    assert target is not None and target.type == AUTOPLACE_MASK_CLOSE
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        nest + [detection(AUTOPLACE_BUTTON, 450, 1315)],
    )
    assert target is not None and target.type == AUTOPLACE_BUTTON
    planner.on_action_success(target.type)

    prompt = [
        detection(AUTOPLACE_PROMPT, 450, 723),
        detection(CONFIRM_YES, 365, 987),
    ]
    target = planner.choose(frame(), prompt)
    assert target is not None and target.type == AUTOPLACE_YES
    with caplog.at_level("INFO"):
        planner.on_action_success(target.type)
    assert planner.is_complete()
    assert "tag=頂尖 | sort=最佳屬性組合 | completed with confirmation" in caplog.text


def test_mass_autoplace_round_selects_level_and_confirms_application() -> None:
    planner = AutoPlaceRoundPlanner(MASS_RULE)
    nest = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_MASS, 228, 168),
    ]

    target = planner.choose(frame(), nest)
    assert target is not None and target.type == nest_filter.FILTER_HEADER
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        nest + [detection(nest_filter.TAG_MASS, 228, 254)],
    )
    assert target is not None and target.type == nest_filter.TAG_MASS
    planner.on_action_success(target.type)

    target = planner.choose(frame(), nest + [detection(NEST_GEAR, 210, 276)])
    assert target is not None and target.type == NEST_GEAR
    planner.on_action_success(target.type)

    settings = [
        detection(AUTOPLACE_TITLE, 450, 565),
        detection(PLACE_SORT_LEVEL, 450, 749),
    ]
    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_LEVEL
    planner.on_action_success(target.type)

    target = planner.choose(frame(), settings + [detection(PLACE_HDR_LEVEL, 450, 648)])
    assert target is not None and target.type == AUTOPLACE_MASK_CLOSE
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        nest + [detection(AUTOPLACE_BUTTON, 450, 1315)],
    )
    assert target is not None and target.type == AUTOPLACE_BUTTON
    planner.on_action_success(target.type)

    prompt = [
        detection(AUTOPLACE_PROMPT, 450, 723),
        detection(CONFIRM_YES, 365, 987),
    ]
    target = planner.choose(frame(), prompt)
    assert target is not None and target.type == AUTOPLACE_YES
    planner.on_action_success(target.type)
    assert planner.is_complete()


def test_autoplace_sort_dropdown_is_opened_when_target_is_not_visible() -> None:
    planner = AutoPlaceRoundPlanner(TOP_RULE)
    planner._stage = "settings"
    target = planner.choose(frame(), [detection(AUTOPLACE_TITLE, 450, 565)])
    assert target is not None and target.type == AUTOPLACE_SORT_HEADER
    assert (target.x, target.y) == (450, 648)


def test_cave_below_threshold_recenters_without_entering() -> None:
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    first = planner.choose(capacity_frame(), [])
    assert first is not None and first.type == CAVE_SWIPE
    planner.on_action_success(first.type)
    second = planner.choose(capacity_frame(), [])
    assert second is not None and second.type == CAVE_SWIPE
    planner.on_action_success(second.type)

    target = planner.choose(capacity_frame(), [detection("hatch_cave", 209, 1150)])
    assert target is not None and target.type == CAVE_RECENTER
    assert (target.x, target.y) == (600, 800)
    assert target.detection.metadata["swipe"] == {
        "x2": 350,
        "y2": 800,
        "duration_ms": 400,
    }
    planner.on_action_success(target.type)
    target = planner.choose(capacity_frame(), [detection("hatch_cave", 209, 1150)])
    assert target is not None and target.type == CAVE_RECENTER
    assert (target.x, target.y) == (450, 600)
    assert target.detection.metadata["swipe"]["y2"] == 1050
    planner.on_action_success(target.type)
    assert planner.choose(
        capacity_frame(), [detection(hatch.HOME_ANCHOR, 59, 561)]
    ) is None
    assert planner.choose(
        capacity_frame(), [detection(hatch.HOME_ANCHOR, 59, 561)]
    ) is None
    assert planner.is_complete()


def test_cave_below_threshold_can_use_hud_when_cave_is_clipped() -> None:
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    for _ in range(2):
        swipe = planner.choose(capacity_frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    target = planner.choose(capacity_frame(), [])

    assert target is not None and target.type == CAVE_RECENTER
    assert planner.capacity_readable


def test_cave_inside_hunt_bottom_exclusion_is_never_opened() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        safe_margin=80,
        bottom_exclusion_px=180,
    )
    unsafe = detection("hatch_cave", 450, 1450)

    target = planner.choose(capacity_frame(), [unsafe])
    assert target is not None and target.type == CAVE_SWIPE

    planner._stage = "open_cave"
    assert planner.choose(capacity_frame(), [unsafe]) is None


def test_cave_inside_hunt_side_exclusion_is_never_opened() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        safe_margin=80,
        bottom_exclusion_px=180,
    )
    planner._stage = "open_cave"
    assert planner.choose(
        capacity_frame(),
        [detection("hatch_cave", 40, 1150)],
    ) is None


def test_cave_above_threshold_runs_weakest_continuous_battle(
    monkeypatch, caplog
) -> None:
    caplog.set_level("INFO")
    monkeypatch.setattr("dino_bot.full_hatch.read_dino_count", lambda *args, **kwargs: 301)
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    target = planner.choose(frame(), [detection("hatch_cave", 209, 1150)])
    assert target is not None and target.type == "hatch_cave"
    planner.on_action_success(target.type)

    target = planner.choose(frame(), [detection(CAVE_SELECT_BUTTON, 450, 1190)])
    assert target is not None and target.type == CAVE_SELECT_BUTTON
    planner.on_action_success(target.type)

    select = [
        detection(SELECT_TITLE, 450, 300),
        detection(nest_filter.TAG_HDR_ALL, 228, 204),
        detection(SELECT_WEAKEST_BUTTON, 259, 1304),
        detection(SELECT_CHOOSE_BUTTON, 450, 1304),
    ]
    target = planner.choose(frame(), select)
    assert target is not None and target.type == SELECT_WEAKEST_BUTTON
    planner.on_action_success(target.type)
    target = planner.choose(frame(), select)
    assert target is not None and target.type == SELECT_CHOOSE_BUTTON
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(), [detection(CAVE_CONTINUOUS_BUTTON, 560, 1304)]
    )
    assert target is not None and target.type == CAVE_CONTINUOUS_BUTTON
    planner.on_action_success(target.type)
    target = planner.choose(frame(), [detection(hatch.CLAIM_BUTTON, 450, 1170)])
    assert target is not None and target.type == hatch.CLAIM_BUTTON
    planner.on_action_success(target.type)
    assert (
        "Hatch cave | cull completed | before=301/350 | selected=40"
        " | expected_after=261 | result=claim_verified"
    ) in caplog.text


def test_capacity_probe_records_required_cull_without_opening_cave(monkeypatch) -> None:
    monkeypatch.setattr("dino_bot.full_hatch.read_dino_count", lambda *args, **kwargs: 301)
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        allow_cull=False,
    )
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    target = planner.choose(frame(), [detection("hatch_cave", 209, 1150)])

    assert target is not None and target.type == CAVE_RECENTER
    assert target.type != "hatch_cave"
    assert planner.capacity_readable
    assert planner.cull_required


def make_full_planner() -> FullHatchPlanner:
    return FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
    )


def test_full_hatch_scopes_detection_by_workflow_phase() -> None:
    planner = make_full_planner()

    hatch_types = planner.planning_detection_types()
    assert hatch.HATCH_BUTTON in hatch_types
    assert hatch.HOME_ANCHOR in hatch_types
    assert STARTUP_AUTO_BATTLE_CLOSE in hatch_types
    assert "dinosaur" not in hatch_types
    assert "own_hunt_path" not in hatch_types

    planner.on_action_success(hatch.CLOSE_BUTTON)
    nest_types = planner.planning_detection_types()
    assert NEST_TITLE in nest_types
    assert SELECT_TITLE in nest_types
    assert set(nest_filter.HEADER_LABELS) <= nest_types
    assert "dinosaur" not in nest_types
    assert "own_hunt_path" not in nest_types


def test_full_hatch_can_request_full_scan_for_every_workflow_phase() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        stage_scoped_scan=False,
    )

    assert planner.planning_detection_types() is None
    planner._stage = "attack"
    assert planner.planning_detection_types() is None
    planner._stage = "cave"
    assert planner.planning_detection_types() is None


def test_attack_filter_header_survives_into_next_scoped_planning_frame() -> None:
    planner = make_full_planner()
    planner._start_replacement("attack")
    planner._replacement_child.on_action_success(nest_filter.TAG_ATTACK)

    visible = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_ATTACK, 217, 166),
    ]
    scoped = [
        item
        for item in visible
        if item.type in planner.planning_detection_types()
    ]

    planner.choose(frame(), scoped)

    assert planner._replacement_child.last_stage() != "target_filter_required"


def test_full_hatch_preflights_capacity_before_first_egg_pile_tap() -> None:
    planner = make_full_planner()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    assert planner._stage == "capacity_preflight"
    assert "hatch_cave" in planner.planning_detection_types()
    planner.on_action_success(target.type)

    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    planner.on_action_success(target.type)

    cave = [detection("hatch_cave", 209, 1150)]
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)

    assert planner.choose(capacity_frame(), home) is None
    target = planner.choose(capacity_frame(), home)
    assert target is not None and target.type == hatch.EGG_PILE
    assert planner._capacity_checked


def test_full_capacity_preflight_screens_before_required_cull(monkeypatch) -> None:
    monkeypatch.setattr("dino_bot.full_hatch.read_dino_count", lambda *args, **kwargs: 321)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        cull_threshold=320,
    )
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    for _ in range(2):
        target = planner.choose(frame(), home)
        assert target is not None and target.type == CAVE_SWIPE
        planner.on_action_success(target.type)

    cave = [detection("hatch_cave", 209, 1150)]
    for _ in range(2):
        target = planner.choose(frame(), cave)
        assert target is not None and target.type == CAVE_RECENTER
        assert target.type != "hatch_cave"
        planner.on_action_success(target.type)

    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)

    assert target is not None and target.type == OPEN_NEST
    assert planner._management_pending
    assert not planner._capacity_checked


def test_cleanup_gate_reopens_nest_when_one_screening_stage_is_missing() -> None:
    planner = make_full_planner()
    planner._stage = "verify_nest_closed"
    planner._child = object()
    planner._management_pending = True
    planner._collect_only_after_empty = False
    planner._screening_completed = {"attack", "hp", "top"}
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    target = planner.choose(frame(), home)

    assert target is not None and target.type == OPEN_NEST
    assert target.type != CAVE_SWIPE
    assert planner._missing_screening_stages() == ("mass",)


def test_cleanup_gate_allows_cave_only_after_every_screening_stage() -> None:
    planner = make_full_planner()
    planner._stage = "verify_nest_closed"
    planner._child = object()
    planner._management_pending = True
    planner._collect_only_after_empty = False
    planner._screening_completed = {"attack", "hp", "top", "mass"}
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    target = planner.choose(frame(), home)

    assert target is not None and target.type == CAVE_SWIPE
    assert planner._stage == "cave"


def test_workflow_reset_preserves_and_resumes_incomplete_screening() -> None:
    planner = make_full_planner()
    planner._management_pending = True
    planner._screening_completed = {"attack"}

    planner.reset_workflow()

    assert planner._stage == "recover_home"
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == OPEN_NEST
    planner.on_action_success(target.type)
    assert planner._stage == "hp"


def test_full_flow_enters_nest_only_after_a_verified_hatch_claim() -> None:
    planner = make_full_planner()
    planner.on_action_success(hatch.CLAIM_BUTTON)
    planner.on_action_success(hatch.CLOSE_BUTTON)
    target = planner.choose(frame(), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert target is not None and target.type == OPEN_NEST
    assert (target.x, target.y) == (59, 561)


def test_full_flow_collects_all_nest_eggs_before_empty_rescan_wait() -> None:
    now = [1000.0]
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        clock=lambda: now[0],
    )
    planner.on_action_success(hatch.CLOSE_BUTTON)

    target = planner.choose(frame(), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert target is not None and target.type == OPEN_NEST
    planner.on_action_success(target.type)

    nest = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_ALL, 228, 168),
    ]
    target = planner.choose(frame(), nest)
    assert target is not None and target.type == nest_filter.FILTER_HEADER
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        nest + [detection(nest_filter.TAG_ALL, 228, 297)],
    )
    assert target is not None and target.type == nest_filter.TAG_ALL
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        nest + [detection(COLLECT_EGGS_BUTTON, 650, 1315)],
    )
    assert target is not None and target.type == COLLECT_EGGS_BUTTON
    planner.on_action_success(target.type)

    target = planner.choose(frame(), nest)
    assert target is not None and target.type == NEST_MASK_CLOSE
    planner.on_action_success(target.type)

    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), home) is None
    assert planner.next_ready_delay_ms() > 0
    assert planner.is_hunt_cooldown_active()

    now[0] += 601
    assert not planner.is_hunt_cooldown_active()
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE


def test_full_hatch_accumulates_ten_claims_before_management() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        batch_hatch_count=12,
    )
    planner._hatch_child.hatched = 4
    planner.on_action_success(hatch.CLOSE_BUTTON)
    assert planner._stage == "open_nest"
    assert planner._collect_only_after_empty
    assert planner._batch_hatched == 4

    planner._stage = "hatch"
    planner._child = planner._new_hatch()
    planner._hatch_child.hatched = 8
    planner.on_action_success(hatch.CLOSE_BUTTON)
    assert planner._stage == "open_nest"
    assert not planner._collect_only_after_empty
    assert planner._batch_hatched == 12
    assert planner._batch_hunt_ready


def test_full_hatch_uses_observed_batch_timer_for_rescan_wait() -> None:
    now = [1000.0]
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        clock=lambda: now[0],
    )
    planner._observed_cooldown_until = now[0] + 3725
    planner._start_empty_rescan_wait()
    assert planner.next_ready_delay_ms() == 3_725_000


def test_interim_collection_pins_remaining_cooldown() -> None:
    now = [1000.0]
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        clock=lambda: now[0],
    )
    planner._start_empty_rescan_wait()
    remaining_ms = planner.next_ready_delay_ms()
    assert planner.is_hunt_cooldown_active()

    assert planner.begin_interim_collection() is True
    assert planner._stage == "open_nest"
    assert planner._collect_only_after_empty
    # 差事結束後的等待必須接續原本的倒數,而不是重新起算。
    assert planner._observed_cooldown_until == now[0] + remaining_ms / 1000

    # 差事進行中不再宣告可狩獵,也不可重複觸發。
    assert not planner.is_hunt_cooldown_active()
    assert planner.begin_interim_collection() is False


def boost_ready_frame() -> Frame:
    """Home incubator frame whose boost bar shows the saturated ready state."""

    ready = frame()
    ready.image[1355:1405, 380:520] = (30, 140, 240)
    return ready


def test_boost_permission_applies_only_to_the_next_hatch_cycle(tmp_path) -> None:
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        boost_inventory=inventory,
    )
    grid = [
        detection(hatch.INCUBATOR_TITLE, 450, 40),
        detection(hatch.CLOSE_BUTTON, 800, 1380),
    ]

    # The default-off setting is snapshotted when this hatch cycle begins.
    inventory.set_enabled(True)
    target = planner.choose(boost_ready_frame(), grid)
    assert target is not None and target.type == hatch.CLOSE_BUTTON

    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._observed_cooldown_until = planner.clock() + 300  # 蛋冷卻中
    target = planner.choose(boost_ready_frame(), grid)
    assert target is not None and target.type == HATCH_BOOST_BUTTON
    planner.on_action_success(target.type)

    prompt = grid + [
        detection(CONFIRM_YES, 365, 850),
        detection(CONFIRM_NO, 535, 850),
    ]
    target = planner.choose(boost_ready_frame(), prompt)
    assert target is not None and target.type == HATCH_BOOST_CONFIRM
    planner.on_action_success(target.type)
    assert inventory.snapshot().remaining == 99


def test_boost_skipped_while_countdown_bar_is_gray(tmp_path) -> None:
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        boost_inventory=inventory,
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._observed_cooldown_until = planner.clock() + 300  # 蛋冷卻中
    grid = [
        detection(hatch.INCUBATOR_TITLE, 450, 40),
        detection(hatch.CLOSE_BUTTON, 800, 1380),
    ]

    # The default frame keeps the bar desaturated (active countdown): the
    # planner must fall through instead of pressing the dead button.
    target = planner.choose(frame(), grid)
    assert target is not None and target.type == hatch.CLOSE_BUTTON
    assert planner._boost_attempted is True
    assert inventory.snapshot().remaining == 100


def test_boost_deferred_while_incubator_is_empty(tmp_path) -> None:
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        boost_inventory=inventory,
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._capacity_checked = True
    grid = [
        detection(hatch.INCUBATOR_TITLE, 450, 40),
        detection(hatch.CLOSE_BUTTON, 800, 1380),
    ]

    # 空孵化器(沒有冷卻讀數):即使按鈕帶是橘色也不按,改標記回訪。
    target = planner.choose(boost_ready_frame(), grid)
    assert target is not None and target.type == hatch.CLOSE_BUTTON
    assert planner._boost_revisit_pending is True
    assert inventory.snapshot().remaining == 100

    # 收蛋完成回到主畫面:轉入回訪孵化器的孵化階段。
    planner._collect_only_after_empty = True
    planner._stage = "verify_nest_closed"
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    planner.choose(frame(), home)
    assert planner._stage == "hatch"
    assert planner._boost_revisit_pending is False
    assert planner._collect_done_for_cycle is True

    # 回訪結束關閉孵化器:直接進入等待,不再重複收蛋。
    planner.on_action_success(hatch.CLOSE_BUTTON)
    assert planner._collect_done_for_cycle is False
    assert planner._empty_rescan_wait is True


def test_home_screen_requires_bright_unobscured_map_and_no_foreground() -> None:
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    cave_home = home + [detection("hatch_cave", 209, 1150)]
    assert is_home_screen(frame(), home)
    assert is_centered_home_screen(frame(), home)
    assert is_home_screen(frame(), cave_home)
    assert not is_centered_home_screen(frame(), cave_home)
    assert not is_home_screen(
        frame(),
        home + [detection(NEST_TITLE, 450, 260)],
    )
    dimmed = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    assert not is_home_screen(dimmed, home)


def test_home_recovery_unwinds_prompt_select_nest_then_confirms_two_frames() -> None:
    planner = HatchHomeRecoveryPlanner()
    dimmed = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    target = planner.choose(
        dimmed,
        [
            detection(SELECT_CONFIRM_PROMPT, 450, 700),
            detection(CONFIRM_NO, 535, 880),
        ],
    )
    assert target is not None and target.type == RECOVERY_NO
    planner.on_action_success(target.type)

    target = planner.choose(dimmed, [detection(SELECT_TITLE, 450, 300)])
    assert target is not None and target.type == RECOVERY_MASK_CLOSE
    planner.on_action_success(target.type)
    target = planner.choose(dimmed, [detection(NEST_TITLE, 450, 260)])
    assert target is not None and target.type == RECOVERY_MASK_CLOSE
    planner.on_action_success(target.type)

    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), home) is None
    assert not planner.is_complete()
    assert planner.choose(frame(), home) is None
    assert planner.is_complete()


def test_home_recovery_recenters_shifted_map_before_accepting_home() -> None:
    planner = HatchHomeRecoveryPlanner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection("hatch_cave", 209, 1150),
        ],
    )
    assert target is not None and target.type == RECOVERY_RECENTER
    assert (target.x, target.y) == (600, 800)
    assert target.detection.metadata["swipe"]["x2"] == 350
    planner.on_action_success(target.type)
    # The horizontal return moves the cave off screen before the map is fully
    # restored. Recovery must still perform the remembered vertical return.
    target = planner.choose(
        frame(),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )
    assert target is not None and target.type == RECOVERY_RECENTER
    assert (target.x, target.y) == (450, 600)
    assert target.detection.metadata["swipe"]["y2"] == 1050


def test_full_flow_recovers_shifted_cave_view_before_tapping_egg_pile() -> None:
    planner = make_full_planner()
    shifted_home = [
        detection(hatch.HOME_ANCHOR, 59, 561),
        detection("hatch_cave", 209, 1150),
    ]

    target = planner.choose(frame(), shifted_home)
    assert target is not None and target.type == RECOVERY_RECENTER
    assert target.type != hatch.EGG_PILE
    assert (target.x, target.y) == (600, 800)
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )
    assert target is not None and target.type == RECOVERY_RECENTER
    assert (target.x, target.y) == (450, 600)
    planner.on_action_success(target.type)

    centered_home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), centered_home) is None
    target = planner.choose(frame(), centered_home)
    assert target is not None and target.type == CAVE_SWIPE
    assert target.type != hatch.EGG_PILE


def test_full_flow_resumes_vertical_recovery_after_restart_mid_return() -> None:
    planner = make_full_planner()
    partially_returned = np.full((1600, 900, 3), 255, dtype=np.uint8)
    partially_returned[1085:1097, 342:585] = (220, 180, 20)

    target = planner.choose(
        frame(partially_returned),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )

    assert target is not None and target.type == RECOVERY_RECENTER
    assert (target.x, target.y) == (450, 600)
    assert target.detection.metadata["swipe"]["y2"] == 1050


def test_full_flow_immediately_leaves_active_hunt_map_on_startup() -> None:
    planner = make_full_planner()

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [
            detection("hunt_button", 515, 550),
            detection("map_exit_nest_button", 841, 1295),
        ],
    )

    assert target is not None and target.type == RECOVERY_MAP_EXIT
    assert (target.x, target.y) == (841, 1295)
    assert planner._stage == "recover_home"


def test_full_flow_tracks_shifted_egg_pile_instead_of_tapping_roaming_dinosaur() -> None:
    planner = make_full_planner()
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1532:1544, 324:567] = (220, 180, 20)

    target = planner.choose(
        frame(image),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )

    assert target is not None and target.type == CAVE_SWIPE
    assert target.type != hatch.EGG_PILE


def test_full_flow_uses_nest_shortcut_before_tapping_visible_home() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection(STARTUP_GROWTH_RESULT, 307, 1265),
        ],
    )

    assert target is not None and target.type == STARTUP_NEST_SHORTCUT
    assert (target.x, target.y) == (592, 1265)
    planner.on_action_success(target.type)
    assert planner._stage == "recover_home"

    close = planner.choose(frame(), [detection(NEST_TITLE, 450, 260)])
    assert close is not None and close.type == RECOVERY_MASK_CLOSE


def test_full_flow_prefers_nested_auto_battle_overlay_during_startup() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection(STARTUP_GROWTH_RESULT, 307, 1265),
            detection(STARTUP_AUTO_BATTLE_CLOSE, 50, 800),
        ],
    )

    assert target is not None and target.type == STARTUP_AUTO_BATTLE_CLOSE
    assert (target.x, target.y) == (50, 800)


def test_full_flow_prioritizes_hatch_result_over_startup_false_positive() -> None:
    planner = make_full_planner()

    target = planner.choose(
        frame(),
        [
            detection(hatch.CLAIM_BUTTON, 330, 1242),
            detection(hatch.EXPEL_BUTTON, 570, 1242),
            detection(STARTUP_AUTO_BATTLE_CLOSE, 50, 800),
        ],
    )

    assert target is not None and target.type == hatch.CLAIM_BUTTON


def test_standalone_attack_recovers_opens_nest_and_runs_only_attack() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        standalone_stage="attack",
    )
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    assert planner.choose(frame(), home) is None
    open_nest = planner.choose(frame(), home)
    assert open_nest is not None and open_nest.type == OPEN_NEST
    planner.on_action_success(open_nest.type)

    assert planner._stage == "attack"
    assert planner.standalone_stage == "attack"


def test_standalone_cave_starts_after_shared_home_preflight() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        standalone_stage="cave",
    )
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)

    assert target is not None and target.type == CAVE_SWIPE
    assert planner._stage == "cave"


def test_full_flow_clears_relogin_before_tapping_visible_home() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection("duplicate_login_close_button", 730, 310),
        ],
    )

    assert target is not None and target.type == "duplicate_login_close_button"
    assert (target.x, target.y) == (730, 310)


def test_failed_egg_pile_tap_enters_recovery_before_any_retry() -> None:
    planner = make_full_planner()
    planner.on_action_failure(hatch.EGG_PILE)

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )

    assert target is not None and target.type == RECOVERY_BACK


def test_last_claim_can_finish_on_unready_egg_detail_and_close_safely() -> None:
    planner = make_full_planner()
    planner.on_action_failure(hatch.CLAIM_BUTTON)
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1136:1238, 339:560] = (40, 180, 255)
    image[1146:1213, 652:723] = (115, 125, 255)

    target = planner.choose(frame(image), [])

    assert target is not None and target.type == HATCH_DETAIL_CLOSE
    assert 650 <= target.x <= 723
    assert 1146 <= target.y <= 1213
    assert planner._hatch_child.hatched == 1


def test_mass_filter_dropped_tap_retries_without_abandoning_round() -> None:
    planner = make_full_planner()
    planner._stage = "mass"
    planner._child = AutoPlaceRoundPlanner(MASS_RULE)
    nest = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_TOP, 217, 166),
    ]

    target = planner.choose(frame(), nest)
    assert target is not None and target.type == nest_filter.FILTER_HEADER
    planner.on_action_failure(target.type)

    retry = planner.choose(frame(), nest)
    assert retry is not None and retry.type == nest_filter.FILTER_HEADER
    assert planner._stage == "mass"


def test_home_recovery_uses_named_cave_close_before_falling_back_to_back() -> None:
    planner = HatchHomeRecoveryPlanner()
    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [
            detection(CAVE_SELECT_BUTTON, 450, 1189),
            detection(CAVE_CLOSE_BUTTON, 735, 1302),
        ],
    )
    assert target is not None and target.type == "hatch_recovery_close"
    assert (target.x, target.y) == (735, 1302)


def test_home_recovery_uses_hunt_map_exit_instead_of_android_back() -> None:
    planner = HatchHomeRecoveryPlanner()

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [detection("map_exit_nest_button", 841, 1295)],
    )

    assert target is not None and target.type == RECOVERY_MAP_EXIT
    assert (target.x, target.y) == (841, 1295)


def test_home_recovery_uses_forest_round_trip_when_home_anchor_is_clipped() -> None:
    planner = HatchHomeRecoveryPlanner()
    forest = detection("forest_recenter_button", 841, 1295)

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [forest],
    )
    assert target is not None and target.type == RECOVERY_FOREST
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [detection("map_exit_nest_button", 841, 1295)],
    )
    assert target is not None and target.type == RECOVERY_MAP_EXIT


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_full_flow_blind_screen_uses_bounded_back_then_requires_home_proof() -> None:
    clock = FakeClock()
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        recovery_timeout_seconds=2,
        clock=clock,
    )
    unknown = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    assert planner.choose(unknown, []) is None
    clock.now += 3
    target = planner.choose(unknown, [])
    assert target is not None and target.type == RECOVERY_BACK
    planner.on_action_success(target.type)

    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
