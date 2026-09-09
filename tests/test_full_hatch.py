from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from dino_bot import hatch, nest_filter
from dino_bot.attack_replacement import AttackReplacementTestPlanner
from dino_bot.cull import CAPACITY_REGION, CapacityRead
from dino_bot.detection import OpenCvDetector
from dino_bot.digits import DigitReader
from dino_bot.full_hatch import (
    AUTOPLACE_BUTTON,
    AUTOPLACE_MASK_CLOSE,
    AUTOPLACE_NOTICE,
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
    DEFAULT_FULL_HATCH_STAGES,
    DEFAULT_SUCCESS_TRANSITIONS,
    FULL_HATCH_STAGES,
    HATCH_BOOST_BUTTON,
    HATCH_BOOST_CONFIRM,
    HATCH_DETAIL_CLOSE,
    HOME_PILE_BASE,
    NEST_GEAR,
    NEST_MASK_CLOSE,
    OPEN_NEST,
    PLACE_HDR_ATTACK,
    PLACE_HDR_BEST,
    PLACE_HDR_HP,
    PLACE_HDR_LEVEL,
    PLACE_SORT_ATTACK,
    PLACE_SORT_BEST,
    PLACE_SORT_HP,
    PLACE_SORT_LEVEL,
    RECOVERY_BACK,
    RECOVERY_FOREST,
    RECOVERY_HUNT_DIALOG_CLOSE,
    RECOVERY_HUNT_DIALOG_DISMISS,
    RECOVERY_MAP_EXIT,
    RECOVERY_MASK_CLOSE,
    RECOVERY_NO,
    RECOVERY_RECENTER,
    RECOVERY_UNDO,
    SCREENING_STAGES,
    SELECT_CHOOSE_BUTTON,
    SELECT_WEAKEST_BUTTON,
    STANDALONE_STAGES,
    STARTUP_AUTO_BATTLE_CLOSE,
    STARTUP_GROWTH_RESULT,
    STARTUP_NEST_SHORTCUT,
    AutoPlaceRoundPlanner,
    CaveCullPlanner,
    FullHatchPlanner,
    HatchHomeRecoveryPlanner,
    _egg_pile_base_center,
    _egg_pile_safe_tap,
    _hatch_boost_point,
    _hatch_boost_ready,
    home_pile_offset,
    is_centered_home_frame,
    is_centered_home_screen,
    is_home_screen,
    is_unready_egg_detail,
    set_home_base_template,
)
from dino_bot.hatch_inventory import HatchBoostInventoryStore
from dino_bot.models import BoundingBox, Detection, Frame, Target, VerificationResult
from dino_bot.nests import (
    ATTACK_AUTOPLACE_RULE,
    HP_AUTOPLACE_RULE,
    MASS_RULE,
    TOP_RULE,
)
from dino_bot.overlays import CONFIRM_NO, CONFIRM_YES, SELECT_CONFIRM_PROMPT
from dino_bot.parent_open import NEST_TITLE, SELECT_TITLE
from dino_bot.verification import TargetChangedVerifier

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


def detection(
    target_type: str,
    x: int = 100,
    y: int = 100,
    metadata: dict[str, object] | None = None,
) -> Detection:
    return Detection(
        target_type,
        x,
        y,
        0.99,
        BoundingBox(x - 5, y - 5, 10, 10),
        metadata or {},
    )


def capacity_frame() -> Frame:
    image = frame().image.copy()
    crop = cv2.imread(str(FIXTURES / "hud_282_350.png"))
    assert crop is not None
    x0, y0 = int(CAPACITY_REGION[0]), int(CAPACITY_REGION[1])
    image[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return frame(image)


def _capacity_read(count: int | None) -> CapacityRead:
    return CapacityRead(
        count=count,
        text="" if count is None else f"{count}/350",
        fraction=None if count is None else (count, 350),
        region=tuple(int(value) for value in CAPACITY_REGION),
        reason="unparsed" if count is None else "ok",
    )


def occluded_capacity_frame() -> Frame:
    # Exact 100x19 HUD pixels from S9's auto-saved 2026-09-05 20:03:55
    # failure. The building overlaps the slash/denominator; do not guess 303.
    encoded = (FIXTURES / "s9-capacity-occluded-20260905.png.b64").read_text()
    crop = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert crop is not None and crop.shape == (19, 100, 3)
    image = frame().image.copy()
    image[239:258, 10:110] = crop
    return frame(image)


def _patch_capacity(monkeypatch, count: int | None) -> None:
    """Force the HUD read, bypassing the glyph matcher and its fixture crop."""

    monkeypatch.setattr(
        "dino_bot.full_hatch.probe_dino_count",
        lambda *args, **kwargs: _capacity_read(count),
    )


class RecordingCapacitySnapshots:
    def __init__(self) -> None:
        self.captures: list[tuple[CapacityRead, str, int]] = []

    def capture(self, frame, read, *, stage, attempts):
        self.captures.append((read, stage, attempts))
        return Path("capacity-00000000-000000.png")


class RecordingEggPileSnapshots:
    def __init__(self) -> None:
        self.captures: list[dict[str, object]] = []

    def capture(self, frame, detections, **kwargs):
        self.captures.append(kwargs)
        return Path("egg-pile-00000000-000000.png")


def test_top_autoplace_round_requires_screen_anchors_and_known_prompt(caplog) -> None:
    planner = AutoPlaceRoundPlanner(TOP_RULE)
    nest = [
        detection(NEST_TITLE, 450, 260),
        detection(nest_filter.TAG_HDR_TOP, 228, 168),
    ]

    # 表頭已是「頂尖」:標籤選擇直接跳過,進入設定齒輪。
    target = planner.choose(frame(), nest + [detection(NEST_GEAR, 210, 276)])
    assert target is not None and target.type == NEST_GEAR
    planner.on_action_success(target.type)

    settings = [
        detection(AUTOPLACE_TITLE, 450, 490),
        detection(PLACE_SORT_BEST, 450, 630),
    ]
    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_BEST
    planner.on_action_success(target.type)

    target = planner.choose(frame(), settings + [detection(PLACE_HDR_BEST, 450, 576)])
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

    # 表頭已是「量產」:標籤選擇直接跳過,進入設定齒輪。
    target = planner.choose(frame(), nest + [detection(NEST_GEAR, 210, 276)])
    assert target is not None and target.type == NEST_GEAR
    planner.on_action_success(target.type)

    settings = [
        detection(AUTOPLACE_TITLE, 450, 490),
        detection(PLACE_SORT_LEVEL, 450, 678),
    ]
    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_LEVEL
    planner.on_action_success(target.type)

    target = planner.choose(frame(), settings + [detection(PLACE_HDR_LEVEL, 450, 576)])
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


def test_attack_autoplace_uses_attack_sort_and_header() -> None:
    planner = AutoPlaceRoundPlanner(ATTACK_AUTOPLACE_RULE)
    planner._stage = "settings"
    settings = [
        detection(AUTOPLACE_TITLE, 450, 490),
        detection(PLACE_SORT_ATTACK, 450, 726),
    ]

    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_ATTACK
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        settings + [detection(PLACE_HDR_ATTACK, 435, 576)],
    )
    assert target is not None and target.type == AUTOPLACE_MASK_CLOSE


def test_hp_autoplace_uses_hp_sort_and_header() -> None:
    planner = AutoPlaceRoundPlanner(HP_AUTOPLACE_RULE)
    planner._stage = "settings"
    settings = [
        detection(AUTOPLACE_TITLE, 450, 490),
        detection(PLACE_SORT_HP, 450, 774),
    ]

    target = planner.choose(frame(), settings)
    assert target is not None and target.type == PLACE_SORT_HP
    planner.on_action_success(target.type)

    target = planner.choose(
        frame(),
        settings + [detection(PLACE_HDR_HP, 436, 575)],
    )
    assert target is not None and target.type == AUTOPLACE_MASK_CLOSE


def test_live_stat_sort_menu_fixture_detects_attack_and_hp_options() -> None:
    menu = cv2.imread(str(FIXTURES / "autoplace-stat-sort-menu.png"))
    assert menu is not None
    image = np.zeros((1600, 900, 3), dtype=np.uint8)
    height, width = menu.shape[:2]
    image[542 : 542 + height, 340 : 340 + width] = menu
    detector = OpenCvDetector(REPO / "assets" / "hatch" / "manifest.json")

    detections = detector.detect_types(
        frame(image),
        {PLACE_SORT_ATTACK, PLACE_SORT_HP},
    )
    by_type = {item.type: item for item in detections}

    assert by_type[PLACE_SORT_ATTACK].confidence >= 0.99
    assert by_type[PLACE_SORT_HP].confidence >= 0.99


def test_live_stat_sort_header_fixtures_detect_selected_headers() -> None:
    fixtures = {
        "autoplace-attack-sort-header.png": PLACE_HDR_ATTACK,
        "autoplace-hp-sort-header.png": PLACE_HDR_HP,
    }
    detector = OpenCvDetector(REPO / "assets" / "hatch" / "manifest.json")

    for fixture_name, target_type in fixtures.items():
        crop = cv2.imread(str(FIXTURES / fixture_name))
        assert crop is not None
        image = np.zeros((1600, 900, 3), dtype=np.uint8)
        height, width = crop.shape[:2]
        image[548 : 548 + height, 340 : 340 + width] = crop

        detections = detector.detect_types(frame(image), {target_type})

        assert len(detections) == 1
        assert detections[0].confidence >= 0.99


def test_autoplace_sort_dropdown_is_opened_when_target_is_not_visible() -> None:
    planner = AutoPlaceRoundPlanner(TOP_RULE)
    planner._stage = "settings"
    target = planner.choose(frame(), [detection(AUTOPLACE_TITLE, 450, 490)])
    assert target is not None and target.type == AUTOPLACE_SORT_HEADER
    assert (target.x, target.y) == (450, 576)


def test_full_hatch_schedules_top_and_mass_before_collect() -> None:
    assert SCREENING_STAGES == ("attack", "hp", "top", "mass")
    assert "top" not in STANDALONE_STAGES
    assert "mass" not in STANDALONE_STAGES

    # Mass is off in the default cycle, so ask for it explicitly: this test
    # covers the top -> mass -> collect ordering machinery, which still has to
    # work for anyone who selects mass in the custom workflow.
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        enabled_stages=FULL_HATCH_STAGES,
    )
    planner._management_pending = True
    planner._screening_completed = {"attack", "hp"}
    planner._start_next_screening_stage()
    assert planner._stage == "top"
    assert planner._autoplace_child.rule == TOP_RULE

    planner._autoplace_child._complete = True
    planner._advance_autoplace_if_done()
    assert planner._stage == "mass"
    assert planner._autoplace_child.rule == MASS_RULE

    planner._autoplace_child._complete = True
    planner._advance_autoplace_if_done()
    assert planner._stage == "collect"


def test_full_hatch_top_stage_confirms_its_own_autoplace_prompt() -> None:
    planner = make_full_planner()
    planner._stage = "top"
    planner._child = AutoPlaceRoundPlanner(TOP_RULE)
    planner._autoplace_child._stage = "after_autoplace"

    target = planner.choose(
        frame(),
        [
            detection(AUTOPLACE_NOTICE, 450, 720),
            detection(CONFIRM_YES, 365, 890),
            detection(CONFIRM_NO, 535, 890),
        ],
    )

    assert target is not None and target.type == AUTOPLACE_YES
    assert (target.x, target.y) == (365, 890)


def test_full_hatch_attack_stage_confirms_its_own_autoplace_prompt() -> None:
    # auto_place_specializations 讓 attack/hp 階段改用 AutoPlaceRoundPlanner,
    # 但攔截誤觸的守衛還停在「只有 top/mass 會開自動放置」的舊假設,於是把該按
    # 「是」的框按成「否」:取消這一輪、回首頁復原、再從同一階段重來。S13 因此
    # 卡在約 25 秒的死循環,跑了七十次動作、completed 永遠是 none。
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        auto_place_specializations=True,
    )
    planner._start_replacement("attack")
    planner._autoplace_child._stage = "after_autoplace"

    target = planner.choose(
        frame(),
        [
            detection(AUTOPLACE_NOTICE, 450, 720),
            detection(CONFIRM_YES, 365, 890),
            detection(CONFIRM_NO, 535, 890),
        ],
    )

    assert target is not None and target.type == AUTOPLACE_YES
    assert (target.x, target.y) == (365, 890)


def test_full_hatch_cancels_unexpected_autoplace_confirmation() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(AUTOPLACE_NOTICE, 450, 720),
            detection(CONFIRM_NO, 535, 890),
        ],
    )

    assert target is not None and target.type == RECOVERY_NO
    assert (target.x, target.y) == (535, 890)
    planner.on_action_success(target.type)
    assert planner._stage == "recover_home"


def test_full_hatch_autoplace_layout_without_a_no_button_keeps_planning() -> None:
    # 廣義版面偵測會把恐龍詳情卡(白面板 + 紅色驅逐鈕)報成自動放置提示,實測信心
    # 0.89。舊寫法在找不到「否」鈕時 return None,於是整輪停在同一幀:六分鐘、109 次
    # 同樣的 ERROR、零動作,而且因為沒有嘗試過任何動作,重試與卡住計數器都不會累積。
    # 沒有「否」鈕正好證明這不是自動放置框,應該讓其他規則接手。
    planner = make_full_planner()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    expected = planner.choose(frame(), home)
    assert expected is not None, "前置條件:這個畫面本來就該有事情可做"

    chosen = planner.choose(
        frame(),
        [detection(AUTOPLACE_NOTICE, 450, 720), *home],
    )

    assert chosen is not None and chosen.type == expected.type


def test_full_hatch_parent_swap_confirmation_beats_false_autoplace_notice() -> None:
    # 替換親代的確認框同樣是「青色是 + 紅色否」,廣義版面偵測會以 0.99 的信心
    # 報成自動放置提示。按下「否」會取消篩選階段剛要求的替換,階段重來後又問
    # 一次,永遠出不去。精確模板在場就證明畫面是哪一種對話框。
    planner = make_full_planner()

    target = planner.choose(
        frame(),
        [
            detection(SELECT_CONFIRM_PROMPT, 450, 700),
            detection(AUTOPLACE_NOTICE, 450, 720),
            detection(CONFIRM_YES, 365, 890),
            detection(CONFIRM_NO, 535, 890),
        ],
    )

    assert target is None or target.type != RECOVERY_NO


def test_full_hatch_device_history_prompt_beats_false_autoplace_notice() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection("device_history_confirm_button", 368, 861),
            detection(AUTOPLACE_NOTICE, 450, 720),
        ],
    )

    assert target is not None and target.type == "device_history_confirm_button"
    assert (target.x, target.y) == (368, 861)


def test_cave_below_threshold_recenters_without_entering() -> None:
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    first = planner.choose(capacity_frame(), [])
    assert first is not None and first.type == CAVE_SWIPE
    planner.on_action_success(first.type)
    second = planner.choose(capacity_frame(), [])
    assert second is not None and second.type == CAVE_SWIPE
    planner.on_action_success(second.type)

    assert planner.choose(
        capacity_frame(), [detection("hatch_cave", 209, 1150)]
    ) is None
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

    assert planner.choose(capacity_frame(), []) is None
    target = planner.choose(capacity_frame(), [])

    assert target is not None and target.type == CAVE_RECENTER
    assert planner.capacity_readable
    assert planner.last_capacity == 282


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
    _patch_capacity(monkeypatch, 301)
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    assert planner.choose(frame(), [detection("hatch_cave", 209, 1150)]) is None
    target = planner.choose(frame(), [detection("hatch_cave", 209, 1150)])
    assert target is not None and target.type == "hatch_cave"
    planner.on_action_success(target.type)

    target = planner.choose(frame(), [detection(CAVE_SELECT_BUTTON, 450, 1190)])
    assert target is not None and target.type == CAVE_SELECT_BUTTON
    planner.on_action_success(target.type)

    select = [
        detection(SELECT_TITLE, 450, 300),
        detection(nest_filter.TAG_HDR_ALL, 228, 204),
        detection(SELECT_WEAKEST_BUTTON, 640, 1297),
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


def test_cave_ignores_weakest_match_at_the_strongest_button(monkeypatch) -> None:
    """A weakest-button hit at a sibling's position must not be tapped.

    The select-dino row holds green "strongest", cyan "choose", yellow
    "weakest" and yellow "top". A template cut from the wrong button still
    matches at ~1.0, so confidence alone once sent the cull into "select
    strongest" and sacrificed the best dinosaurs. Position is the guard.
    """

    _patch_capacity(monkeypatch, 301)
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        planner.on_action_success(swipe.type)
    assert planner.choose(frame(), [detection("hatch_cave", 209, 1150)]) is None
    target = planner.choose(frame(), [detection("hatch_cave", 209, 1150)])
    planner.on_action_success(target.type)
    target = planner.choose(frame(), [detection(CAVE_SELECT_BUTTON, 450, 1190)])
    planner.on_action_success(target.type)

    # x=259 is the green "select strongest" button.
    assert planner.choose(
        frame(),
        [
            detection(SELECT_TITLE, 450, 300),
            detection(nest_filter.TAG_HDR_ALL, 228, 204),
            detection(SELECT_WEAKEST_BUTTON, 259, 1304),
        ],
    ) is None


def test_capacity_probe_records_required_cull_without_opening_cave(monkeypatch) -> None:
    _patch_capacity(monkeypatch, 301)
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        allow_cull=False,
    )
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    assert planner.choose(frame(), [detection("hatch_cave", 209, 1150)]) is None
    target = planner.choose(frame(), [detection("hatch_cave", 209, 1150)])

    assert target is not None and target.type == CAVE_RECENTER
    assert target.type != "hatch_cave"
    assert planner.capacity_readable
    assert planner.cull_required


def test_capacity_requires_two_matching_reads_after_a_mismatch(
    monkeypatch, caplog
) -> None:
    caplog.set_level("INFO")
    readings = iter((324, 325, 325))
    monkeypatch.setattr(
        "dino_bot.full_hatch.probe_dino_count",
        lambda *args, **kwargs: _capacity_read(next(readings)),
    )
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=320,
        allow_cull=False,
    )
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)
    cave = [detection("hatch_cave", 209, 1150)]

    assert planner.choose(frame(), cave) is None
    assert planner.choose(frame(), cave) is None
    assert planner.last_capacity is None
    target = planner.choose(frame(), cave)

    assert target is not None and target.type == CAVE_RECENTER
    assert planner.last_capacity == 325
    assert any(
        "capacity confirmation changed" in record.message for record in caplog.records
    )


def test_capacity_probe_can_allow_extra_retries_for_slow_detection(monkeypatch) -> None:
    _patch_capacity(monkeypatch, None)
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        allow_cull=False,
        capacity_read_retries=4,
    )
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)

    cave = [detection("hatch_cave", 209, 1150)]
    for _ in range(4):
        assert planner.choose(frame(), cave) is None
    assert planner._capacity_failures == 4

    target = planner.choose(frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER


def test_s9_occluded_capacity_finishes_remaining_move_before_failing(caplog) -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=340, capacity_limit=370,
        capacity_read_retries=2,
    )
    blocked = occluded_capacity_frame()
    read = planner._read_capacity(blocked)
    assert read.reason == "unparsed"
    assert read.count is None and read.text == "30310"
    first = planner.choose(frame(), [])
    assert first is not None and first.type == CAVE_SWIPE
    planner.on_action_success(first.type)

    cave = [detection("hatch_cave", 100, 1100)]
    for _ in range(2):
        assert planner.choose(blocked, cave) is None
    target = planner.choose(blocked, cave)
    assert target is not None and target.type == CAVE_SWIPE
    assert (target.x, target.y) == (350, 800)
    assert target.detection.metadata["swipe"]["x2"] == 600
    assert target.detection.metadata["swipe"]["y2"] == 800
    assert not planner.capacity_readable and planner.last_capacity is None
    assert "completing remaining calibrated move" in caplog.text
    planner.on_action_success(target.type)
    assert planner._capacity_failures == 0

    # Still obscured after the full path: no third move and no guessed count.
    for _ in range(2):
        assert planner.choose(blocked, cave) is None
    target = planner.choose(blocked, cave)
    assert target is not None and target.type == CAVE_RECENTER
    assert (target.x, target.y) == (600, 800)
    assert target.detection.metadata["swipe"]["x2"] == 350
    assert planner._navigation_swipes == 2
    assert not planner.capacity_readable
    assert planner.last_capacity is None


@pytest.mark.parametrize("allow_cull", [False, True])
@pytest.mark.parametrize("count", [303, 343])
def test_capacity_after_remaining_move_requires_two_new_reads(
    monkeypatch, allow_cull: bool, count: int,
) -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=340, capacity_limit=370,
        capacity_read_retries=1, allow_cull=allow_cull,
    )
    first = planner.choose(frame(), [])
    planner.on_action_success(first.type)
    cave = [detection("hatch_cave", 100, 1100)]
    blocked = occluded_capacity_frame()
    assert planner.choose(blocked, cave) is None
    move = planner.choose(blocked, cave)
    assert move is not None and move.type == CAVE_SWIPE
    planner.on_action_success(move.type)

    # Model a fresh legible frame, independently of the unavailable live
    # post-swipe screenshot. Navigation must not substitute for confirmation.
    _patch_capacity(monkeypatch, count)
    assert planner.choose(frame(), cave) is None
    assert not planner.capacity_readable and planner.last_capacity is None
    target = planner.choose(frame(), cave)
    assert target is not None
    assert target.type == (
        "hatch_cave" if allow_cull and count >= 340 else CAVE_RECENTER
    )
    assert planner.capacity_readable and planner.last_capacity == count
    assert planner.cull_required is (count >= 340)


def test_occluded_capacity_does_not_start_new_path_from_unknown_cave_view() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=340, capacity_limit=370,
        capacity_read_retries=1,
    )
    blocked = occluded_capacity_frame()
    cave = [detection("hatch_cave", 100, 1100)]
    assert planner.choose(blocked, cave) is None
    target = planner.choose(blocked, cave)
    assert target is not None and target.type == CAVE_RECENTER
    assert planner._navigation_swipes == 0
    assert not planner.capacity_readable


def test_occluded_capacity_remaining_move_failures_are_bounded() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=340, capacity_limit=370,
        capacity_read_retries=1,
    )
    first = planner.choose(frame(), [])
    planner.on_action_success(first.type)
    cave = [detection("hatch_cave", 100, 1100)]
    blocked = occluded_capacity_frame()
    assert planner.choose(blocked, cave) is None
    for _ in range(planner.navigator.max_swipe_failures + 1):
        target = planner.choose(blocked, cave)
        assert target is not None and target.type == CAVE_SWIPE
        assert (target.x, target.y) == (350, 800)
        planner.on_action_failure(target.type)
    target = planner.choose(blocked, cave)
    assert target is not None and target.type == CAVE_RECENTER
    # Only the first outbound move succeeded, so do not undo a dropped move.
    assert (target.x, target.y) == (450, 600)
    assert planner._navigation_swipes == 1
    assert not planner.capacity_readable and planner.last_capacity is None


def test_wrong_capacity_limit_finishes_remaining_move_then_fails_safely() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=340, capacity_limit=370,
        capacity_read_retries=1,
    )
    first = planner.choose(frame(), [])
    planner.on_action_success(first.type)
    cave = [detection("hatch_cave", 100, 1100)]
    assert planner._read_capacity(capacity_frame()).reason == "unexpected_capacity"
    assert planner.choose(capacity_frame(), cave) is None
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_SWIPE
    assert (target.x, target.y) == (350, 800)
    assert target.detection.metadata["swipe"] == {
        "x2": 600,
        "y2": 800,
        "duration_ms": 400,
    }
    planner.on_action_success(target.type)

    # A genuinely wrong configured cap still cannot pass the second read, but
    # the full calibrated route has been tried once before it fails safely.
    assert planner.choose(capacity_frame(), cave) is None
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER
    assert planner._navigation_swipes == 2
    assert not planner.capacity_readable


def test_unreadable_capacity_saves_the_frame_only_once_it_gives_up(monkeypatch) -> None:
    _patch_capacity(monkeypatch, None)
    snapshots = RecordingCapacitySnapshots()
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        capacity_read_retries=2,
        capacity_snapshots=snapshots,
    )
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        planner.on_action_success(swipe.type)

    cave = [detection("hatch_cave", 209, 1150)]
    # The retries all see the same still frame; saving each one would evict the
    # retained set with copies of a single episode.
    for _ in range(2):
        assert planner.choose(frame(), cave) is None
    assert snapshots.captures == []

    planner.choose(frame(), cave)
    assert len(snapshots.captures) == 1
    read, stage, attempts = snapshots.captures[0]
    assert read.reason == "unparsed"
    assert stage == "cave_navigate"
    assert attempts == 3


def test_presumed_navigation_failure_saves_the_frame(monkeypatch) -> None:
    _patch_capacity(monkeypatch, None)
    snapshots = RecordingCapacitySnapshots()
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        capacity_snapshots=snapshots,
    )
    # Exhaust the calibrated swipes without the cave ever matching: the cave
    # template and the HUD have now both missed, and only pixels can say why.
    for _ in range(len(planner.navigator.swipe_vectors)):
        swipe = planner.choose(frame(), [])
        assert swipe is not None and swipe.type == CAVE_SWIPE
        planner.on_action_success(swipe.type)
    # The rescans still hope the HUD stands in for the missing template, so
    # they are not yet a failure worth a frame.
    for _ in range(planner.navigator.max_rescans):
        assert planner.choose(frame(), []) is None
    assert snapshots.captures == []

    planner.choose(frame(), [])
    assert len(snapshots.captures) == 1
    assert snapshots.captures[0][0].reason == "unparsed"


def test_capacity_snapshots_are_optional() -> None:
    planner = CaveCullPlanner(DigitReader(GLYPHS), threshold=300)
    for _ in range(2):
        swipe = planner.choose(frame(), [])
        planner.on_action_success(swipe.type)
    # A blank frame has no HUD; the read fails and must not raise without a
    # writer attached.
    assert planner.choose(frame(), [detection("hatch_cave", 209, 1150)]) is None


def test_cave_recenter_can_allow_extra_checks_for_slow_detection() -> None:
    planner = CaveCullPlanner(
        DigitReader(GLYPHS),
        threshold=300,
        cave_recenter_checks=5,
    )
    planner._stage = "verify_recenter"

    for _ in range(4):
        assert planner.choose(frame(), []) is None
        assert planner._stage == "verify_recenter"
    assert planner.choose(frame(), []) is None
    assert planner._stage == "recenter_failed"


def make_full_planner() -> FullHatchPlanner:
    return FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
    )


def test_unreadable_cleanup_capacity_blocks_hatch_instead_of_restarting() -> None:
    planner = make_full_planner()
    cave = CaveCullPlanner(DigitReader(GLYPHS), threshold=330)
    cave._complete = True
    cave._capacity_readable = False
    planner._stage = "cave"
    planner._child = cave
    planner._management_pending = True
    planner._screening_completed = set(SCREENING_STAGES)
    planner._capacity_checked = True

    assert planner.choose(frame(), []) is None

    assert planner.is_hatch_blocked()
    assert planner._capacity_blocked
    assert not planner._capacity_checked
    assert planner._stage == "capacity_blocked"
    assert planner.completed_management_cycles == 0


@pytest.mark.parametrize(
    ("capacity_limit", "cull_threshold"),
    [(0, 1), (350, 0), (350, 351)],
)
def test_full_hatch_rejects_unsafe_capacity_parameters(
    capacity_limit: int,
    cull_threshold: int,
) -> None:
    with pytest.raises(ValueError):
        FullHatchPlanner(
            DigitReader(GLYPHS),
            egg_pile_point=(450, 1330),
            capacity_limit=capacity_limit,
            cull_threshold=cull_threshold,
        )


def test_attack_autoplace_completes_the_stage_without_manual_screening() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        auto_place_specializations=True,
    )

    planner._start_replacement("attack")
    assert isinstance(planner._child, AutoPlaceRoundPlanner)
    assert planner._autoplace_child.rule == ATTACK_AUTOPLACE_RULE
    assert PLACE_SORT_ATTACK in planner.planning_detection_types()

    planner._autoplace_child._complete = True
    planner._advance_autoplace_if_done()

    # Auto-place already ordered the tag pool by attack, so no per-side
    # re-check follows: the stage is done and screening moves on.
    assert "attack" in planner._screening_completed
    assert not isinstance(planner._child, AttackReplacementTestPlanner)


def test_hp_autoplace_completes_the_stage_without_manual_screening() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        auto_place_specializations=True,
    )

    planner._start_replacement("hp")
    assert planner._autoplace_child.rule == HP_AUTOPLACE_RULE
    assert PLACE_SORT_HP in planner.planning_detection_types()
    planner._autoplace_child._complete = True
    planner._advance_autoplace_if_done()

    assert "hp" in planner._screening_completed
    assert not isinstance(planner._child, AttackReplacementTestPlanner)


def test_custom_full_hatch_runs_only_selected_management_stages() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        enabled_stages=("collect", "hatch"),
    )

    assert planner.enabled_stages == {"collect", "hatch"}
    assert planner._missing_screening_stages() == ()
    assert planner._collect_enabled is True
    assert planner._cave_enabled is False


def test_custom_full_hatch_skips_idle_collection_when_collect_is_disabled() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        enabled_stages=("hatch",),
    )
    planner._start_empty_rescan_wait()

    assert planner.is_hunt_cooldown_active()
    assert planner.begin_interim_collection() is False
    assert planner._stage == "hatch"


def test_custom_full_hatch_stops_hatching_at_safe_population_without_cave() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        enabled_stages=("collect", "hatch"),
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._cave_population = planner.cull_threshold - 1
    planner._screening_baseline_population = planner.cull_threshold - 1
    planner._hatch_child.hatched = 1

    planner.on_action_success(hatch.CLOSE_BUTTON)

    assert not planner.is_hatch_blocked()
    assert planner._hatch_capacity_check_pending
    assert planner._stage == "recover_home"


def test_full_hatch_repeated_action_failure_enters_safe_home_recovery() -> None:
    planner = make_full_planner()
    failed = detection(COLLECT_EGGS_BUTTON, 450, 1300)
    target = Target(failed.type, failed.x, failed.y, failed.confidence, failed)

    assert planner.recover_from_action_failures(
        target,
        "full_collect_button",
        2,
        frame(),
        [],
    )
    assert planner.last_stage().startswith("full_recover_home:")
    assert planner.is_recovery_progress(hatch.CLAIM_BUTTON)
    assert not planner.is_recovery_progress(COLLECT_EGGS_BUTTON)


def test_full_hatch_scopes_detection_by_workflow_phase() -> None:
    planner = make_full_planner()

    planner._stage = "recover_home"
    recovery_types = planner.planning_detection_types()
    assert recovery_types is not None
    assert {
        hatch.HOME_ANCHOR,
        hatch.CLAIM_BUTTON,
        hatch.CLOSE_BUTTON,
        CAVE_CLOSE_BUTTON,
        "map_exit_nest_button",
        "forest_recenter_button",
        CONFIRM_NO,
    } <= recovery_types
    assert "dinosaur" not in recovery_types
    assert "own_hunt_path" not in recovery_types
    assert not recovery_types & {
        nest_filter.TAG_ATTACK,
        nest_filter.TAG_HP,
        nest_filter.TAG_TOP,
        nest_filter.TAG_MASS,
        nest_filter.FILTER_HEADER,
        hatch.EGG_PILE,
    }

    planner._stage = "hatch"
    hatch_types = planner.planning_detection_types()
    assert hatch.HATCH_BUTTON in hatch_types
    assert hatch.INCUBATOR_TITLE in hatch_types
    assert hatch.HOME_ANCHOR in hatch_types
    assert STARTUP_AUTO_BATTLE_CLOSE in hatch_types
    assert "dinosaur" not in hatch_types
    assert "own_hunt_path" not in hatch_types

    planner.on_action_success(hatch.CLOSE_BUTTON)
    nest_types = planner.planning_detection_types()
    assert NEST_TITLE in nest_types
    assert SELECT_TITLE not in nest_types
    assert not set(nest_filter.HEADER_LABELS) & nest_types
    assert len(nest_types) < 15

    planner._start_replacement("attack")
    replacement_types = planner.planning_detection_types()
    assert SELECT_TITLE in replacement_types
    assert set(nest_filter.HEADER_LABELS) <= replacement_types
    assert len(replacement_types) < 55

    planner._stage = "top"
    autoplace_types = planner.planning_detection_types()
    assert {
        NEST_GEAR,
        AUTOPLACE_TITLE,
        AUTOPLACE_BUTTON,
        PLACE_SORT_BEST,
        PLACE_SORT_LEVEL,
    } <= autoplace_types
    assert "hatch_parent_left" not in autoplace_types
    assert len(autoplace_types) < 55

    planner._stage = "collect"
    collect_types = planner.planning_detection_types()
    assert COLLECT_EGGS_BUTTON in collect_types
    assert set(nest_filter.HEADER_LABELS) <= collect_types
    assert NEST_GEAR not in collect_types
    assert len(collect_types) < 55
    assert "dinosaur" not in nest_types
    assert "own_hunt_path" not in nest_types


def test_s9_live_home_anchor_variants_are_recognized() -> None:
    """Real S9 entrance frames varied from 0.833 down to 0.809."""

    fixtures = {
        "s9-home-anchor-20260816.png.b64": (0.82, 0.85),
        "s9-home-anchor-20260816-late.png.b64": (0.79, 0.82),
    }
    detector = OpenCvDetector(REPO / "assets" / "hatch" / "manifest.json")
    for fixture, expected_range in fixtures.items():
        encoded = (FIXTURES / fixture).read_text(encoding="ascii")
        crop = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert crop is not None
        image = np.full((1600, 900, 3), 255, dtype=np.uint8)
        image[540:585, 24:74] = crop

        found = detector.detect_types(Frame(image), {hatch.HOME_ANCHOR})

        assert len(found) == 1, fixture
        assert (found[0].x, found[0].y) == (49, 562)
        assert expected_range[0] <= found[0].confidence < expected_range[1]


def test_open_nest_visual_match_requires_unobscured_centered_home() -> None:
    planner = make_full_planner()
    planner._stage = "open_nest"
    home_anchor = [detection(hatch.HOME_ANCHOR, 49, 562)]

    target = planner.choose(frame(), home_anchor)
    assert target is not None and target.type == OPEN_NEST

    planner = make_full_planner()
    planner._stage = "open_nest"
    dimmed = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    assert planner.choose(dimmed, home_anchor) is None


def test_incubator_close_accepts_centered_home_structure_without_anchor() -> None:
    close = detection(hatch.CLOSE_BUTTON, 798, 1384)
    target = Target(close.type, close.x, close.y, close.confidence, close)
    verifier = TargetChangedVerifier(
        success_transitions={hatch.CLOSE_BUTTON: (hatch.HOME_ANCHOR,)},
        success_frame_predicates={
            hatch.CLOSE_BUTTON: is_centered_home_frame,
        },
    )

    result = verifier.verify(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        frame(),
        target,
        [close],
        [],
    )

    assert result.success
    assert "frame structure" in result.reason


def test_incubator_close_rejects_dimmed_or_shifted_home_structure() -> None:
    assert not is_centered_home_frame(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    )
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1248:1260, 330:573] = (220, 180, 20)
    assert not is_centered_home_frame(frame(shifted))


def test_full_hatch_forwards_direct_candidate_success_to_replacement_child() -> None:
    planner = make_full_planner()
    planner._start_replacement("attack")
    child = planner._replacement_child
    child._stage = "confirm_left"
    child._side = 0
    selected = detection("hatch_candidate_row", 350, 435)
    target = Target(
        selected.type,
        selected.x,
        selected.y,
        selected.confidence,
        selected,
    )

    planner.on_action_success_context(
        target,
        frame(),
        [],
        VerificationResult(True, "previous UI disappeared: hatch_select_title"),
    )

    assert planner._stage == "attack"
    assert child.last_stage() == "nest_right"


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


def test_full_hatch_preflight_starts_initial_screening_before_first_egg_pile_tap() -> None:
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
    assert planner.choose(capacity_frame(), cave) is None
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)
    target = planner.choose(capacity_frame(), cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)

    assert planner.choose(capacity_frame(), home) is None
    target = planner.choose(capacity_frame(), home)
    assert target is not None and target.type == OPEN_NEST
    assert planner._capacity_checked
    assert planner._management_pending
    assert planner._screening_baseline_population is None


@pytest.mark.parametrize("readable_after_move", [False, True])
def test_custom_preflight_recovers_occluded_hud_or_keeps_capacity_fuse(
    readable_after_move: bool,
) -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        enabled_stages=("attack", "top", "collect", "cave", "hatch"),
        capacity_read_retries=1,
    )
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    planner.on_action_success(target.type)
    cave = [detection("hatch_cave", 100, 1100)]
    blocked = occluded_capacity_frame()
    assert planner.choose(blocked, cave) is None
    target = planner.choose(blocked, cave)
    assert target is not None and target.type == CAVE_SWIPE
    assert (target.x, target.y) == (350, 800)
    planner.on_action_success(target.type)

    after = capacity_frame() if readable_after_move else blocked
    assert planner.choose(after, cave) is None
    target = planner.choose(after, cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)
    target = planner.choose(after, cave)
    assert target is not None and target.type == CAVE_RECENTER
    planner.on_action_success(target.type)
    assert planner.choose(after, home) is None
    target = planner.choose(after, home)

    if readable_after_move:
        assert target is not None and target.type == OPEN_NEST
        planner.on_action_success(target.type)
        assert planner._stage == "attack"
        assert planner._capacity_checked and not planner.is_hatch_blocked()
        assert planner._screening_stages == ("attack", "top")
    else:
        assert target is None
        assert planner._capacity_blocked and not planner._capacity_checked
        assert planner.is_hatch_blocked()


def test_full_capacity_preflight_screens_before_required_cull(monkeypatch) -> None:
    _patch_capacity(monkeypatch, 321)
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
    assert planner.choose(frame(), cave) is None
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


def test_capacity_preflight_keeps_cull_reading_when_recenter_needs_recovery(
    monkeypatch,
) -> None:
    _patch_capacity(monkeypatch, 324)
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

    # Capacity remains readable even when the cave template is clipped. The
    # result must be committed before the return-to-home proof can fail.
    assert planner.choose(frame(), []) is None
    target = planner.choose(frame(), [])
    assert target is not None and target.type == CAVE_RECENTER
    assert planner._capacity_child.last_capacity == 324
    assert planner._cave_population == 324
    assert planner._management_pending
    assert not planner._capacity_checked

    planner._begin_home_recovery("capacity preflight recenter failed")
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)

    assert target is not None and target.type == OPEN_NEST
    assert target.type != CAVE_SWIPE
    assert planner._stage == "open_nest"


def test_cleanup_gate_reopens_nest_when_one_screening_stage_is_missing() -> None:
    planner = make_full_planner()
    planner._stage = "verify_nest_closed"
    planner._child = object()
    planner._management_pending = True
    planner._collect_only_after_empty = False
    planner._screening_completed = {"attack"}
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    target = planner.choose(frame(), home)

    assert target is not None and target.type == OPEN_NEST
    assert target.type != CAVE_SWIPE
    # Mass is not part of the default cycle, so it is not owed here either.
    assert planner._missing_screening_stages() == ("hp", "top")


def test_cleanup_gate_allows_cave_only_after_every_screening_stage() -> None:
    planner = make_full_planner()
    planner._stage = "verify_nest_closed"
    planner._child = object()
    planner._management_pending = True
    planner._cave_cleanup_after_management = True
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
    # 表頭已是「所有」:標籤選擇直接跳過,進入收蛋按鈕。
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


def test_collect_only_opens_my_nest_even_when_home_collect_button_is_visible() -> None:
    planner = make_full_planner()
    planner._stage = "open_nest"
    planner._collect_only_after_empty = True
    home = [
        detection(hatch.HOME_ANCHOR, 59, 561),
        detection(COLLECT_EGGS_BUTTON, 450, 1305),
    ]

    target = planner.choose(frame(), home)

    assert target is not None and target.type == OPEN_NEST


def test_hatch_taps_egg_pile_without_repositioning_for_home_collect_button() -> None:
    planner = make_full_planner()
    planner._capacity_checked = True
    home = [
        detection(hatch.HOME_ANCHOR, 59, 561),
        detection(COLLECT_EGGS_BUTTON, 450, 1305),
    ]

    target = planner.choose(frame(), home)

    assert target is not None and target.type == hatch.EGG_PILE


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


def test_permanent_boost_layout_moves_ticket_boost_safely() -> None:
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    panel = cv2.imread(str(FIXTURES / "incubator-v2-boost-panel.png"))
    assert panel is not None
    image[1250:1470, 240:660] = panel
    v2 = frame(image)

    assert _hatch_boost_point(v2) == (450, 1420)
    assert not _hatch_boost_ready(v2)

    # The captured ticket bar is gray because a boost is active. Simulate its
    # saturated ready state without touching the orange permanent-speed bar.
    image[1405:1455, 380:520] = (30, 140, 240)
    assert _hatch_boost_ready(frame(image))


def test_permanent_boost_layout_planner_taps_ticket_bar_center(tmp_path) -> None:
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        boost_inventory=inventory,
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._observed_cooldown_until = planner.clock() + 300

    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    panel = cv2.imread(str(FIXTURES / "incubator-v2-boost-panel.png"))
    assert panel is not None
    image[1250:1470, 240:660] = panel
    image[1405:1455, 380:520] = (30, 140, 240)
    grid = [
        detection(hatch.INCUBATOR_TITLE, 450, 169),
        detection(hatch.CLOSE_BUTTON, 798, 1421),
    ]

    target = planner.choose(frame(image), grid)

    assert target is not None and target.type == HATCH_BOOST_BUTTON
    assert (target.x, target.y) == (450, 1420)


def test_incubator_title_outranks_false_hunt_dialog_close_detection() -> None:
    # S9 v0.0.58 live evidence: the redesigned incubator's real close button
    # scored 0.998 as hatch_close_button and 0.910 as hunt_dialog_close_button.
    # Treating the weaker hit as active-hunt identity closed and reopened the
    # incubator every ten seconds (99 actions, zero completed cycles).
    planner = make_full_planner()
    planner._stage = "hatch"
    planner._child = planner._new_hatch()
    detections = [
        detection(hatch.INCUBATOR_TITLE, 450, 169),
        detection(hatch.CLOSE_BUTTON, 798, 1421),
        detection(hatch.HATCH_LABEL, 241, 535),
        detection("hunt_dialog_close_button", 798, 1421),
    ]

    target = planner.choose(frame(), detections)

    assert target is not None and target.type == hatch.HATCH_LABEL
    assert planner._stage == "hatch"


def test_boost_permission_is_live_and_stock_requires_active_bar(tmp_path) -> None:
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

    # Enabling takes effect without resetting the hatch cycle.
    target = planner.choose(boost_ready_frame(), grid)
    assert target is not None and target.type == hatch.CLOSE_BUTTON
    inventory.set_enabled(True)
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
    assert inventory.snapshot().remaining == 100
    planner.choose(frame(), grid)  # Fresh incubator with a gray active bar.
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
    assert 0 < inventory.ready_delay_seconds() <= 60
    assert inventory.snapshot().remaining == 100


def test_boost_is_due_even_while_incubator_is_empty(tmp_path) -> None:
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

    # 用券排程與蛋的冷卻/數量無關。
    target = planner.choose(boost_ready_frame(), grid)
    assert target is not None and target.type == HATCH_BOOST_BUTTON
    assert inventory.snapshot().remaining == 100


def test_scheduled_visit_preserves_child_and_refreshes_egg_wait(tmp_path, monkeypatch) -> None:
    now = [10_000.0]
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: now[0])
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        boost_inventory=inventory, clock=lambda: now[0],
    )
    child = planner._child
    planner._empty_rescan_wait = True
    child.begin_rescan_wait("existing egg cooldown", seconds=3600)
    planner._screening_completed = {"attack", "hp"}
    monkeypatch.setattr(
        "dino_bot.full_hatch.is_centered_home_screen",
        lambda frame, items: any(item.type == hatch.HOME_ANCHOR for item in items),
    )
    monkeypatch.setattr("dino_bot.full_hatch._egg_pile_safe_tap", lambda frame: (450, 1330))
    monkeypatch.setattr(hatch, "read_hatch_cooldown_seconds", lambda *a, **kw: 600)
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    panel = [detection(hatch.INCUBATOR_TITLE, 450, 40), detection(hatch.CLOSE_BUTTON, 800, 1380)]
    prompt = panel + [detection(CONFIRM_YES, 365, 850), detection(CONFIRM_NO, 535, 850)]
    assert planner.next_ready_delay_ms() == 3_600_000
    assert planner.hunt_cooldown_delay_ms() == 3_600_000
    assert planner.choose(frame(), home) is None
    assert planner.begin_boost_visit()
    assert planner.choose(frame(), home).type == "hatch_cooldown_boost_open"
    assert planner.choose(boost_ready_frame(), panel).type == HATCH_BOOST_BUTTON
    planner.on_action_success(HATCH_BOOST_BUTTON)
    assert planner.choose(boost_ready_frame(), prompt).type == HATCH_BOOST_CONFIRM
    planner.on_action_success(HATCH_BOOST_CONFIRM)
    assert planner.choose(frame(), panel).type == "hatch_cooldown_boost_close"
    planner.on_action_success("hatch_cooldown_boost_close")
    assert planner.choose(frame(), home) is None
    assert planner._boost_visit is None
    assert planner._child is child
    assert planner._screening_completed == {"attack", "hp"}
    assert planner.is_hunt_cooldown_active()
    assert planner.hunt_cooldown_delay_ms() == 600_000
    assert inventory.ready_delay_seconds() == 1800
    planner._start_hatch_cycle()
    assert inventory.ready_delay_seconds() == 1800
    planner.reset_workflow()
    assert inventory.ready_delay_seconds() == 1800


def test_interim_collection_preserves_egg_deadline_when_boost_is_due_sooner(tmp_path) -> None:
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: 10_000.0)
    inventory.set_enabled(True)
    inventory.consume_one(only_if_due=True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        boost_inventory=inventory, clock=lambda: 10_000.0,
    )
    planner._empty_rescan_wait = True
    planner._child.begin_rescan_wait("existing egg cooldown", seconds=3600)
    assert planner.next_ready_delay_ms() == 3_600_000
    assert planner.begin_interim_collection()
    assert planner._observed_cooldown_until == 13_600
    assert planner.abort_interim_collection("test returning from errand")
    assert planner.hunt_cooldown_delay_ms() == 3_600_000


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


def test_centered_home_accepts_clipped_anchor_with_forest_landmark() -> None:
    # A centered pile can remain measurable after the left home anchor is
    # clipped. The second outdoor-map landmark keeps this fallback screen-gated.
    assert is_centered_home_screen(
        frame(),
        [detection("forest_recenter_button", 841, 1296)],
    )
    assert not is_centered_home_screen(frame(), [])


def test_centered_home_survives_a_visible_map_exit_button() -> None:
    landmarks = [
        detection("forest_recenter_button", 841, 1296),
        detection(hatch.HOME_ANCHOR, 59, 561),
    ]
    exit_button = detection("map_exit_nest_button", 841, 1295)
    for landmark in landmarks:
        assert is_centered_home_screen(frame(), [landmark, exit_button])

    # The exit button alone still proves nothing, and a real hunt sheet must
    # still be rejected; only the exit control is allowed to share home.
    assert not is_centered_home_screen(frame(), [exit_button])
    assert not is_centered_home_screen(
        frame(),
        [*landmarks, detection("hunt_confirm_button", 451, 1411)],
    )


def test_centered_home_requires_the_same_pile_proof_as_hatch_child() -> None:
    # A very thin coloured strip is enough for the skin-specific locator to
    # estimate a centred offset, but it is not enough to safely identify the
    # tappable pile. Recovery must not declare home complete in this state,
    # otherwise the hatch child becomes unknown_screen and the outer timeout
    # starts the Back loop seen in the diagnostic.
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1453:1455, 350:550] = (220, 180, 20)
    thin_pile = frame(image)
    home = [detection("forest_recenter_button", 841, 1296)]

    assert home_pile_offset(thin_pile) is not None
    assert not hatch.has_home_pile_structure(
        thin_pile,
        egg_pile_point=(450.0, 1330.0),
        reference_width=900.0,
    )
    assert not is_centered_home_screen(thin_pile, home)


def straw_home_frame(scale: float = 1.0, dy: int = 0) -> Frame:
    """A home frame carrying the starter nest's base and no cyan at all.

    ``scale`` reproduces the nest growing with its contents, which is the whole
    reason the base is matched over a sweep rather than at one size.
    """

    template = cv2.imread(
        str(REPO / "assets" / "hatch" / "templates" / "hatch-home-straw-base.png"),
        cv2.IMREAD_COLOR,
    )
    assert template is not None
    if scale != 1.0:
        template = cv2.resize(template, None, fx=scale, fy=scale)
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    height, width = template.shape[:2]
    top = int(HOME_PILE_BASE[1] + dy - height / 2)
    left = int(HOME_PILE_BASE[0] - width / 2)
    image[top : top + height, left : left + width] = template
    return Frame(image)


def lava_home_frame(dx: int = 0, dy: int = 0) -> Frame:
    """A synthetic warm-colour component with the measured live base geometry."""

    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(
        image,
        (335 + dx, 1318 + dy),
        (563 + dx, 1472 + dy),
        (20, 90, 220),
        thickness=-1,
    )
    return Frame(image)


def blue_stone_home_frame(dx: int = 0, dy: int = 0) -> Frame:
    """The desaturated blue-stone growth stage measured on centred S13."""

    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(
        image,
        # Its colour centroid is 40px above the shared structural anchor.
        (375 + dx, 1387 + dy),
        (525 + dx, 1443 + dy),
        (212, 149, 108),
        thickness=-1,
    )
    return Frame(image)


def test_starter_nest_base_is_measured_by_template_when_no_cyan_exists() -> None:
    # The cyan strip belongs to the upgraded stone basin. The starter nest is
    # straw on brick, so an account that still has one measured nothing at all
    # and never once agreed it had arrived home.
    offset = home_pile_offset(straw_home_frame())
    assert offset is not None
    assert max(abs(offset[0]), abs(offset[1])) <= 2
    assert is_centered_home_screen(
        straw_home_frame(),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )


def test_starter_nest_base_is_found_at_the_sizes_its_contents_produce() -> None:
    # Observed 1.00x and 1.17x on the same nest twenty minutes apart; the base
    # centre stayed on the same map point through it.
    for scale in (0.8, 1.0, 1.17, 1.4):
        offset = home_pile_offset(straw_home_frame(scale=scale))
        assert offset is not None, scale
        assert max(abs(offset[0]), abs(offset[1])) <= 4, scale


def test_starter_nest_base_still_measures_an_off_centre_home() -> None:
    offset = home_pile_offset(straw_home_frame(dy=-120))
    assert offset is not None
    assert 116 <= offset[1] <= 124


def test_upgraded_basin_keeps_its_cyan_measurement_and_skips_the_sweep() -> None:
    # The cyan path must stay first: it is both the calibrated one and the
    # cheap one, and the template sweep costs ~30x more per frame.
    assert home_pile_offset(frame()) == home_pile_offset(frame())
    offset = home_pile_offset(frame())
    assert offset is not None
    assert max(abs(offset[0]), abs(offset[1])) <= 2


def deep_basin_home_frame(dx: int = 0, dy: int = 0) -> Frame:
    """The live upgraded basin: a deep bowl of water, not a painted strip.

    Measured on the S9 v0.0.73 frame that fused hatching off: the basin's cyan
    spans 140x118 with its base at y=1250, while the fixed incubator nests
    around it are the same hue at 105-121 wide but only 43-67 tall.
    """

    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    # Shallow incubator nests: wide enough to pass a width-only gate.  They sit
    # on the map, so a camera move carries them along with the basin.
    for nest_x, nest_y in ((180, 900), (620, 940), (300, 1000)):
        cv2.rectangle(
            image,
            (nest_x + dx, nest_y + dy),
            (nest_x + dx + 121, nest_y + dy + 60),
            (220, 180, 20),
            thickness=-1,
        )
    cv2.rectangle(
        image,
        (381 + dx, 1132 + dy),
        (521 + dx, 1250 + dy),
        (220, 180, 20),
        thickness=-1,
    )
    return Frame(image)


def test_deep_basin_is_measured_by_its_base_not_its_floating_centroid() -> None:
    # A 118px-deep bowl puts its colour centroid ~60px above the map anchor.
    # Reporting that centroid would hand recovery a phantom offset and send it
    # dragging the map away from a home it had already reached.
    base = _egg_pile_base_center(deep_basin_home_frame())
    assert base is not None
    assert abs(base[0] - 450) <= 3
    assert abs(base[1] - 1250) <= 2


def test_shallow_incubator_nests_are_never_mistaken_for_the_basin() -> None:
    # The nests share the basin's exact hue and come within 19px of its width,
    # so only their depth separates them.
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    for nest_x, nest_y in ((180, 900), (620, 940), (300, 1000)):
        cv2.rectangle(
            image,
            (nest_x, nest_y),
            (nest_x + 121, nest_y + 60),
            (220, 180, 20),
            thickness=-1,
        )
    assert _egg_pile_base_center(Frame(image)) is None


def test_herd_merged_home_still_measures_its_offset() -> None:
    # The regression this guards: dinosaurs bridging the nests to the pile made
    # the structure locator return nothing, so recovery had no measurement,
    # swiped blind, undid itself, and fused hatching off on an already-centred
    # home. The cyan basin cannot merge with a herd - dinosaurs are not cyan.
    settled = home_pile_offset(deep_basin_home_frame())
    shifted = deep_basin_home_frame(dy=-205)
    offset = home_pile_offset(shifted)
    assert settled is not None and offset is not None
    # The camera move is measured, not merely detected: recovery drags by this
    # vector, so an offset that ignored the shift would swipe the wrong way.
    assert abs((offset[1] - settled[1]) - 205) <= 2

    planner = HatchHomeRecoveryPlanner()
    target = planner.choose(
        shifted,
        [
            detection(hatch.HOME_ANCHOR, 49, 562),
            detection("forest_recenter_button", 841, 1296),
        ],
    )
    assert target is not None
    assert target.type == RECOVERY_RECENTER


def test_lava_nest_base_proves_centered_home_without_cyan() -> None:
    offset = home_pile_offset(lava_home_frame())
    assert offset is not None
    assert max(abs(offset[0]), abs(offset[1])) <= 2
    assert is_centered_home_screen(
        lava_home_frame(),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )


def test_lava_nest_base_measures_a_shifted_home() -> None:
    offset = home_pile_offset(lava_home_frame(dx=70, dy=-120))
    assert offset is not None
    assert -72 <= offset[0] <= -68
    assert 118 <= offset[1] <= 122


def test_blue_stone_base_is_measured_by_stable_colour() -> None:
    centered = blue_stone_home_frame()
    offset = home_pile_offset(centered)
    assert offset is not None
    assert max(abs(offset[0]), abs(offset[1])) <= 1
    assert _egg_pile_safe_tap(centered) == (450, 1355)
    assert is_centered_home_screen(
        centered,
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )


def test_blue_stone_and_upgraded_basin_share_the_same_pile_hitbox() -> None:
    # Nest skins and egg artwork differ by growth stage, but the large home
    # pile keeps one map position and one interactive footprint.
    blue_point = _egg_pile_safe_tap(blue_stone_home_frame())
    upgraded_point = _egg_pile_safe_tap(frame())
    assert blue_point is not None and upgraded_point is not None
    assert abs(blue_point[0] - upgraded_point[0]) <= 2
    assert abs(blue_point[1] - upgraded_point[1]) <= 2


def test_shifted_blue_stone_base_is_not_accepted_as_centered_home() -> None:
    shifted = blue_stone_home_frame(dy=-315)
    offset = home_pile_offset(shifted)
    assert offset is not None
    assert 314 <= offset[1] <= 316
    assert _egg_pile_safe_tap(shifted) == (450, 1040)
    assert not is_centered_home_screen(
        shifted,
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection("forest_recenter_button", 841, 1296),
        ],
    )


def test_blue_stone_measurement_ignores_dark_roaming_dinosaur_group() -> None:
    shifted = blue_stone_home_frame(dy=-500)
    # A large dark connected group below the pile satisfies the old generic
    # structure fallback and changes shape as dinosaurs roam.  The blue-stone
    # colour anchor must remain tied to the pile itself.
    cv2.rectangle(shifted.image, (300, 1050), (650, 1260), (40, 40, 40), -1)

    offset = home_pile_offset(shifted)

    assert offset is not None
    assert abs(offset[0]) <= 1
    assert 499 <= offset[1] <= 501


def test_regular_small_orange_nest_is_not_a_home_base() -> None:
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (709, 869), (861, 966), (20, 90, 220), thickness=-1)
    assert home_pile_offset(Frame(image)) is None


def test_home_base_template_absence_fails_closed_for_unknown_style(tmp_path) -> None:
    set_home_base_template(tmp_path / "missing.png")
    try:
        assert home_pile_offset(straw_home_frame()) is None
        assert home_pile_offset(frame()) is not None
    finally:
        set_home_base_template(None)


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
    swipe = target.detection.metadata["swipe"]
    # The base sits 364px above centre, so the drag carries it exactly that
    # far.  Replaying the 450px calibrated leg instead would push it 86px
    # past centre, and a larger shortfall would push it off screen entirely -
    # taking with it the only landmark the planner can measure against.
    assert swipe["y2"] - target.y == 364
    assert abs(swipe["x2"] - target.x) <= 15


def test_full_flow_recenters_shifted_unknown_skin_before_tapping_pile() -> None:
    planner = make_full_planner()

    target = planner.choose(
        blue_stone_home_frame(dy=-315),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
    )

    assert target is not None and target.type == RECOVERY_RECENTER
    assert target.type != hatch.EGG_PILE
    swipe = target.detection.metadata["swipe"]
    assert swipe["y2"] - target.y == 315
    assert abs(swipe["x2"] - target.x) <= 1


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


def test_recovery_dismisses_a_hunt_prompt_that_hides_the_map_exit() -> None:
    # 實測的死結:氣泡框蓋住地圖右側的離開鈕,掃描只認得出那顆狩獵按鈕,
    # 而 Back 對氣泡完全無效(連按六次,bbox 與信心度逐格相同)。
    planner = HatchHomeRecoveryPlanner()

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [detection("hunt_button", 515, 550)],
    )

    assert target is not None and target.type == RECOVERY_HUNT_DIALOG_DISMISS
    assert (target.x, target.y) == (110, 1200)
    # 收掉氣泡的證據是被它蓋住的控制項重新露出來,而不是合成目標消失。
    assert DEFAULT_SUCCESS_TRANSITIONS[RECOVERY_HUNT_DIALOG_DISMISS] == (
        "map_exit_nest_button",
        "forest_recenter_button",
    )


def test_recovery_dismisses_the_idle_growth_result_card() -> None:
    """The auto-growth card dims the map and blocks the egg pile.

    It appears on its own after the game idles and carries no close button,
    so recovery used to exhaust its Back ladder against it: one S9 run gave
    up at 22:50, dropped the hatch workflow, and hunted for the rest of the
    hour. Tapping the dimmed margin beside the card dismisses it.
    """

    planner = HatchHomeRecoveryPlanner()

    target = planner.choose(frame(), [detection(STARTUP_GROWTH_RESULT, 450, 1270)])

    assert target is not None
    assert target.type == RECOVERY_MASK_CLOSE
    assert (target.x, target.y) == (50, 800)


def test_recovery_prefers_a_real_close_button_over_tapping_empty_map() -> None:
    planner = HatchHomeRecoveryPlanner()

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [
            detection("hunt_button", 515, 550),
            detection("hunt_dialog_close_button", 700, 400),
        ],
    )

    assert target is not None and target.type == RECOVERY_HUNT_DIALOG_CLOSE
    assert (target.x, target.y) == (700, 400)


def test_recovery_stops_dismissing_a_hunt_prompt_that_never_clears() -> None:
    planner = HatchHomeRecoveryPlanner()
    detections = [detection("hunt_button", 515, 550)]

    for _ in range(planner.max_hunt_dialog_dismissals):
        target = planner.choose(
            frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
            detections,
        )
        assert target is not None and target.type == RECOVERY_HUNT_DIALOG_DISMISS

    # 收不掉就讓位給階梯剩下的逃生手段,而不是永遠點空地圖。
    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        detections,
    )
    assert target is not None and target.type == RECOVERY_BACK


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


def test_full_flow_uses_nest_shortcut_on_dimmed_card_despite_visible_home_anchor() -> None:
    # The card can leave the HUD anchor template-visible behind it. Its dimmed
    # outdoor map, not the anchor alone, distinguishes it from the bright-home
    # false positive covered below.
    image = frame().image.copy()
    image[:] = 70
    planner = make_full_planner()
    target = planner.choose(
        frame(image),
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


def test_full_flow_lets_recovery_dismiss_the_card_instead_of_retapping() -> None:
    """Recovery owns the screen once it starts.

    The shortcut branch taps *inside* the card to open the nest; recovery
    needs the card gone to measure the egg pile.  While the shortcut ran
    first the recovery rung for the same card was unreachable: on 2026-08-31
    S9 re-tapped the shortcut three times, exhausted the Back ladder at
    10:54 and hunted without hatching for the rest of the session.
    """

    planner = make_full_planner()
    card = [
        detection(
            STARTUP_GROWTH_RESULT,
            450,
            1270,
            metadata={"shortcut_layout": "centered_nest"},
        )
    ]

    first = planner.choose(frame(), card)
    assert first is not None and first.type == STARTUP_NEST_SHORTCUT
    planner.on_action_success(first.type)
    assert planner._stage == "recover_home"

    # With the card still up, recovery must dismiss it rather than tap the
    # shortcut again for as long as the card survives.
    for _ in range(3):
        target = planner.choose(frame(), card)
        assert target is not None
        assert target.type == RECOVERY_MASK_CLOSE
        assert (target.x, target.y) == (50, 800)


def test_full_flow_uses_centred_nest_shortcut_on_new_growth_result_layout() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(
                STARTUP_GROWTH_RESULT,
                307,
                1265,
                metadata={"shortcut_layout": "centered_nest"},
            ),
        ],
    )

    assert target is not None and target.type == STARTUP_NEST_SHORTCUT
    assert (target.x, target.y) == (450, 1270)


def test_full_flow_does_not_tap_false_startup_shortcut_on_nest_screen() -> None:
    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 49, 562),
            detection(NEST_TITLE, 450, 261),
            detection(
                STARTUP_GROWTH_RESULT,
                450,
                1270,
                metadata={"shortcut_layout": "centered_nest"},
            ),
        ],
    )

    assert target is None or target.type != STARTUP_NEST_SHORTCUT


def test_full_flow_ignores_growth_layout_match_on_bright_home_map() -> None:
    """S9's snowy home and egg pile can satisfy the broad launch layout."""

    planner = make_full_planner()
    target = planner.choose(
        frame(),
        [
            detection(hatch.HOME_ANCHOR, 49, 562),
            detection(
                STARTUP_GROWTH_RESULT,
                450,
                1270,
                metadata={"shortcut_layout": "centered_nest"},
            ),
        ],
    )

    assert target is None or target.type != STARTUP_NEST_SHORTCUT


def test_home_recovery_ignores_growth_layout_match_on_bright_shifted_home() -> None:
    """A false launch-card match must not consume recovery's escape ladder."""

    image = frame().image.copy()
    image[1448:1460, 330:573] = 255
    image[1328:1340, 330:573] = (220, 180, 20)
    planner = HatchHomeRecoveryPlanner()

    target = planner.choose(
        frame(image),
        [
            detection(hatch.HOME_ANCHOR, 59, 561),
            detection(STARTUP_GROWTH_RESULT, 450, 1270),
        ],
    )

    assert target is not None and target.type == RECOVERY_RECENTER


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


def test_failed_hatch_tap_checks_capacity_before_any_retry() -> None:
    planner = make_full_planner()
    planner._capacity_checked = True

    planner.on_action_failure(hatch.HATCH_BUTTON)

    assert planner._hatch_capacity_check_pending
    assert not planner._capacity_checked
    assert planner._stage == "recover_home"

    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)

    assert target is not None and target.type == CAVE_SWIPE
    assert planner._stage == "capacity_preflight"


def test_failed_egg_pile_tap_checks_full_capacity_before_retry(monkeypatch) -> None:
    _patch_capacity(monkeypatch, 350)
    planner = make_full_planner()
    planner._capacity_checked = True
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    failed_target = Target(
        hatch.EGG_PILE,
        450,
        1330,
        1.0,
        detection(hatch.EGG_PILE, 450, 1330),
    )

    planner.on_action_failure_context(failed_target, frame(), home, 1)
    planner.on_action_failure(hatch.EGG_PILE)
    planner.on_retry_exhausted(failed_target)

    assert planner._egg_pile_capacity_check_pending
    assert not planner.is_hatch_blocked()
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE

    planner.on_action_success(target.type)
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    planner.on_action_success(target.type)

    cave = [detection("hatch_cave", 209, 1150)]
    assert planner.choose(frame(), cave) is None
    for _ in range(2):
        target = planner.choose(frame(), cave)
        assert target is not None and target.type == CAVE_RECENTER
        planner.on_action_success(target.type)

    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == OPEN_NEST
    assert planner._management_pending
    assert not planner._egg_pile_capacity_check_pending


def test_failed_egg_pile_tap_retries_when_capacity_is_below_limit(monkeypatch) -> None:
    _patch_capacity(monkeypatch, 329)
    planner = make_full_planner()
    planner._capacity_checked = True
    planner._screening_baseline_population = 329
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    failed_target = Target(
        hatch.EGG_PILE,
        450,
        1330,
        1.0,
        detection(hatch.EGG_PILE, 450, 1330),
    )

    planner.on_action_failure_context(failed_target, frame(), home, 1)
    planner.on_action_failure(hatch.EGG_PILE)
    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    planner.on_action_success(target.type)
    target = planner.choose(frame(), home)
    assert target is not None and target.type == CAVE_SWIPE
    planner.on_action_success(target.type)

    cave = [detection("hatch_cave", 209, 1150)]
    assert planner.choose(frame(), cave) is None
    for _ in range(2):
        target = planner.choose(frame(), cave)
        assert target is not None and target.type == CAVE_RECENTER
        planner.on_action_success(target.type)

    assert planner.choose(frame(), home) is None
    target = planner.choose(frame(), home)
    assert target is not None and target.type == hatch.EGG_PILE
    assert planner._capacity_checked
    assert planner._egg_pile_capacity_rechecked
    assert not planner._management_pending


def test_egg_pile_still_failing_after_safe_capacity_uses_bounded_fuse() -> None:
    planner = make_full_planner()
    planner._capacity_checked = True
    planner._egg_pile_capacity_rechecked = True
    planner._egg_pile_failures = 2
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    failed_target = Target(
        hatch.EGG_PILE,
        450,
        1330,
        1.0,
        detection(hatch.EGG_PILE, 450, 1330),
    )

    planner.on_action_failure_context(failed_target, frame(), home, 1)
    planner.on_action_failure(hatch.EGG_PILE)

    assert planner._egg_pile_retry_pending
    assert not planner._egg_pile_capacity_check_pending
    assert planner.choose(frame(), home) is None
    assert planner.choose(frame(), home) is None
    assert planner.is_hatch_blocked()


def test_failed_egg_pile_captures_calibration_evidence_and_fuses_hatch() -> None:
    snapshots = RecordingEggPileSnapshots()
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        egg_pile_snapshots=snapshots,
    )
    failed_target = Target(
        hatch.EGG_PILE,
        457,
        1279,
        1.0,
        detection(hatch.EGG_PILE, 457, 1279),
    )

    planner.on_action_failure_context(
        failed_target,
        frame(),
        [detection(hatch.HOME_ANCHOR, 59, 561)],
        1,
    )
    assert len(snapshots.captures) == 1
    assert snapshots.captures[0]["target_x"] == 457
    assert snapshots.captures[0]["proposed_point"] is not None

    planner.on_retry_exhausted(failed_target)
    assert planner.is_hatch_blocked()
    assert planner.is_complete()
    assert planner.choose(frame(), [detection(hatch.HOME_ANCHOR, 59, 561)]) is None


def test_repeated_parent_calibration_recovery_blocks_hatch() -> None:
    planner = make_full_planner()

    for attempt in range(1, 4):
        planner._stage = "hp"
        planner._begin_home_recovery(
            "no actionable target at full_hp:parent_stats_unreadable"
        )
        assert planner.is_hatch_blocked() is (attempt == 3)

    assert planner.last_stage().startswith("full_screening_blocked:")
    assert planner.is_complete()


def test_open_nest_retry_exhaustion_blocks_hatch() -> None:
    planner = make_full_planner()
    failed_target = Target(
        OPEN_NEST,
        49,
        562,
        1.0,
        detection(OPEN_NEST, 49, 562),
    )

    planner.on_retry_exhausted(failed_target)

    assert planner.is_hatch_blocked()
    assert planner.is_complete()


def test_failed_home_recovery_action_retries_without_completing_workflow() -> None:
    planner = make_full_planner()
    planner._management_pending = True
    planner._screening_completed = {"attack", "hp"}
    planner._begin_home_recovery("cave recenter failed")
    shifted_home = [
        detection(hatch.HOME_ANCHOR, 59, 561),
        detection("forest_recenter_button", 841, 1296),
    ]
    dimmed = frame(np.zeros((1600, 900, 3), dtype=np.uint8))

    target = planner.choose(dimmed, shifted_home)
    assert target is not None and target.type == RECOVERY_BACK
    planner.on_action_failure(target.type)

    assert not planner.is_complete()
    assert planner.choose(dimmed, shifted_home) is None
    assert planner._recovery_rounds == 1
    assert planner._management_pending
    assert planner._screening_completed == {"attack", "hp"}

    retry = planner.choose(dimmed, shifted_home)
    assert retry is not None and retry.type == RECOVERY_BACK
    assert not planner.is_complete()


def test_last_claim_can_finish_on_unready_egg_detail_and_close_safely() -> None:
    planner = make_full_planner()
    planner.on_action_failure(hatch.CLAIM_BUTTON)
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1136:1238, 339:560] = (40, 180, 255)
    image[1146:1213, 652:723] = (115, 125, 255)
    assert is_unready_egg_detail(frame(image))

    target = planner.choose(frame(image), [])

    assert target is not None and target.type == HATCH_DETAIL_CLOSE
    assert 650 <= target.x <= 723
    assert 1146 <= target.y <= 1213
    assert planner._hatch_child.hatched == 1


def test_home_recovery_closes_hatch_detail_after_capacity_dialog_is_dismissed() -> None:
    planner = HatchHomeRecoveryPlanner()
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[1136:1238, 339:560] = (40, 180, 255)
    image[1146:1213, 652:723] = (115, 125, 255)

    target = planner.choose(frame(image), [])

    assert target is not None and target.type == HATCH_DETAIL_CLOSE
    assert 650 <= target.x <= 723
    assert 1146 <= target.y <= 1213


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


def test_home_recovery_never_enters_forest_to_recenter_home() -> None:
    planner = HatchHomeRecoveryPlanner()
    forest = detection("forest_recenter_button", 841, 1295)

    target = planner.choose(
        frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [forest],
    )
    assert target is not None and target.type == RECOVERY_BACK
    assert target.type != RECOVERY_FOREST
    assert planner.forest_trips() == 0


def test_home_recovery_ignores_a_notice_without_no_button_and_recenters() -> None:
    """The yellow task-complete toast is not an auto-place confirmation."""

    planner = HatchHomeRecoveryPlanner()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1085:1097, 342:585] = (220, 180, 20)

    target = planner.choose(
        frame(shifted),
        [
            detection(AUTOPLACE_NOTICE, 450, 720),
            detection(hatch.HOME_ANCHOR, 59, 561),
        ],
    )

    assert target is not None and target.type == RECOVERY_RECENTER
    assert not planner.is_failed()


def test_home_recovery_undoes_when_task_toast_outlives_a_measured_pile() -> None:
    """A persistent task banner must not wait or switch to Forest."""

    planner = HatchHomeRecoveryPlanner()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[974:986, 330:573] = (220, 180, 20)
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    nudge = planner.choose(frame(shifted), home)
    assert nudge is not None and nudge.type == RECOVERY_RECENTER
    planner.on_action_success(nudge.type)

    hidden = np.full((1600, 900, 3), 255, dtype=np.uint8)
    hidden[1085:1097, 342:585] = (220, 180, 20)
    obscured = [
        detection(AUTOPLACE_NOTICE, 450, 720),
        detection("forest_recenter_button", 841, 1296),
    ]

    undo = planner.choose(frame(hidden), obscured)
    assert undo is not None and undo.type == RECOVERY_UNDO
    assert undo.type != RECOVERY_FOREST
    assert planner.forest_trips() == 0
    assert (undo.x, undo.y) == (
        nudge.detection.metadata["swipe"]["x2"],
        nudge.detection.metadata["swipe"]["y2"],
    )


def test_home_recovery_keeps_exact_prompt_without_no_button_as_a_safe_failure() -> None:
    planner = HatchHomeRecoveryPlanner()

    assert planner.choose(frame(), [detection(AUTOPLACE_PROMPT, 450, 720)]) is None
    assert planner.is_failed()


def test_home_recovery_keeps_correcting_a_damped_but_converging_camera_move() -> None:
    """S13 needs more than two measured drags when the camera damps a swipe.

    The production trace reduced the vertical offset from 474px to 227px,
    then 161px.  Each move was safe and measurably better, but the old two
    move budget diverted into the Forest round trip before the next measured
    correction could finish centring the visible home pile.
    """

    def shifted_home(offset_y: int) -> Frame:
        image = np.full((1600, 900, 3), 255, dtype=np.uint8)
        image[1448 - offset_y : 1460 - offset_y, 330:573] = (220, 180, 20)
        return frame(image)

    planner = HatchHomeRecoveryPlanner()
    home = [
        detection(hatch.HOME_ANCHOR, 59, 561),
        detection("forest_recenter_button", 841, 1296),
    ]
    for offset_y in (474, 227, 161):
        target = planner.choose(shifted_home(offset_y), home)
        assert target is not None and target.type == RECOVERY_RECENTER
        planner.on_action_success(target.type)


def test_cave_return_history_survives_recovery_and_undo_retries() -> None:
    planner = FullHatchPlanner(DigitReader(GLYPHS), egg_pile_point=(450, 1330))
    cave = CaveCullPlanner(DigitReader(GLYPHS), threshold=340, capacity_limit=370)
    cave._stage = "recenter"
    missing_pile = frame(np.full((1600, 900, 3), 255, dtype=np.uint8))
    home = [detection(hatch.HOME_ANCHOR, 59, 561), detection("forest_recenter_button", 841, 1296)]
    for _ in range(2):
        action = cave.choose(missing_pile, home)
        assert action.type == CAVE_RECENTER
        cave.on_action_success(action.type)
    history = cave.camera_history()
    assert len(history) == 2
    planner._stage = "cave"
    planner._child = cave
    planner._begin_home_recovery("cave return moved pile below frame")
    recovery = planner._child
    undo = recovery.choose(missing_pile, home)
    assert undo.type == RECOVERY_UNDO
    x1, y1, x2, y2 = history[-1]
    assert (undo.x, undo.y) == (x2, y2)
    assert undo.detection.metadata["swipe"]["x2"] == x1
    assert undo.detection.metadata["swipe"]["y2"] == y1
    recovery.on_action_success(undo.type)
    assert recovery.camera_history() == history[:-1]
    planner._begin_home_recovery("retry after interrupted recovery")
    assert planner._child.camera_history() == history[:-1]
    # Once the known base is visible again, geometry must be proved; an undo
    # alone is never treated as a successful return.
    planner._child.choose(frame(), home)
    planner._child.choose(frame(), home)
    assert planner._child.is_complete()


def test_failed_cave_return_swipe_is_not_available_for_undo() -> None:
    cave = CaveCullPlanner(DigitReader(GLYPHS), threshold=340, capacity_limit=370)
    cave._stage = "recenter"
    missing_pile = frame(np.full((1600, 900, 3), 255, dtype=np.uint8))
    action = cave.choose(missing_pile, [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert action.type == CAVE_RECENTER
    cave.on_action_failure(action.type)
    assert cave.camera_history() == ()


def test_home_recovery_undoes_its_own_swipe_when_the_landmark_disappears() -> None:
    planner = HatchHomeRecoveryPlanner()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1085:1097, 342:585] = (220, 180, 20)

    nudge = planner.choose(frame(shifted), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert nudge is not None and nudge.type == RECOVERY_RECENTER
    swipe = nudge.detection.metadata["swipe"]
    planner.on_action_success(nudge.type)

    # Forest proves this is still the same home map, not a recenter command.
    # If the measured pile disappeared after our own swipe, undo immediately
    # on that map instead of leaving it.
    blind = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    home = [detection("forest_recenter_button", 841, 1296)]
    undo = planner.choose(blind, home)
    assert undo is not None and undo.type == RECOVERY_UNDO
    assert (undo.x, undo.y) == (swipe["x2"], swipe["y2"])
    assert undo.detection.metadata["swipe"]["x2"] == nudge.x
    assert undo.detection.metadata["swipe"]["y2"] == nudge.y

    # One undo per applied gesture: the reversal must not become its own loop.
    planner.on_action_success(undo.type)
    next_target = planner.choose(blind, home)
    assert next_target is not None and next_target.type == RECOVERY_BACK
    assert next_target.type != RECOVERY_FOREST


def test_home_recovery_undoes_latest_swipe_after_undo_then_new_nudge() -> None:
    """Regression trace: apply A/B, undo B, apply C, then undo C."""

    def shifted_home(offset_y: int) -> Frame:
        image = np.full((1600, 900, 3), 255, dtype=np.uint8)
        image[1448 - offset_y : 1460 - offset_y, 330:573] = (220, 180, 20)
        return frame(image)

    planner = HatchHomeRecoveryPlanner()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    first = planner.choose(shifted_home(474), home)
    assert first is not None and first.type == RECOVERY_RECENTER
    planner.on_action_success(first.type)

    second = planner.choose(shifted_home(216), home)
    assert second is not None and second.type == RECOVERY_RECENTER
    planner.on_action_success(second.type)

    undo_second = planner._undo_target(reason="test rollback")
    assert undo_second is not None and undo_second.type == RECOVERY_UNDO
    planner.on_action_success(undo_second.type)

    third = planner.choose(shifted_home(149), home)
    assert third is not None and third.type == RECOVERY_RECENTER
    third_swipe = third.detection.metadata["swipe"]
    planner.on_action_success(third.type)

    undo_regression = planner.choose(shifted_home(520), home)
    assert undo_regression is not None and undo_regression.type == RECOVERY_UNDO
    assert (undo_regression.x, undo_regression.y) == (
        third_swipe["x2"],
        third_swipe["y2"],
    )
    assert undo_regression.detection.metadata["swipe"]["x2"] == third.x
    assert undo_regression.detection.metadata["swipe"]["y2"] == third.y

    planner.on_action_success(undo_regression.type)
    assert planner._applied_swipes == [
        (
            first.x,
            first.y,
            first.detection.metadata["swipe"]["x2"],
            first.detection.metadata["swipe"]["y2"],
        )
    ]


def test_home_recovery_cave_route_still_starts_at_its_first_leg_after_a_nudge() -> None:
    planner = HatchHomeRecoveryPlanner()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1085:1097, 342:585] = (220, 180, 20)

    nudge = planner.choose(frame(shifted), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert nudge is not None and nudge.type == RECOVERY_RECENTER
    planner.on_action_success(nudge.type)

    # The cave view appearing afterwards is a calibrated displacement with its
    # own two-leg return.  A measured nudge is not one of those legs, so it
    # must not advance the route's index and leave the map halfway home.
    cave = planner.choose(
        frame(shifted),
        [detection(hatch.HOME_ANCHOR, 59, 561), detection("hatch_cave", 209, 1150)],
    )
    assert cave is not None and cave.type == RECOVERY_RECENTER
    assert (cave.x, cave.y) == (600, 800)


def test_home_recovery_keeps_a_correction_that_a_transition_frame_hid() -> None:
    planner = HatchHomeRecoveryPlanner()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1085:1097, 342:585] = (220, 180, 20)

    nudge = planner.choose(frame(shifted), [detection(hatch.HOME_ANCHOR, 59, 561)])
    assert nudge is not None and nudge.type == RECOVERY_RECENTER
    planner.on_action_success(nudge.type)

    # The frame captured straight after the drag is still animating and
    # matches nothing.  Undoing here would throw away a correction that
    # worked, so the very next move must not be the reversal.
    blank = planner.choose(frame(np.zeros((1600, 900, 3), dtype=np.uint8)), [])
    assert blank is None or blank.type != RECOVERY_UNDO

    # Once the map settles at centre, recovery confirms home instead.
    centered = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert planner.choose(frame(), centered) is None
    assert planner.choose(frame(), centered) is None
    assert planner.is_complete()


def test_full_flow_bounds_total_map_switches_across_recovery_retries() -> None:
    planner = make_full_planner()
    planner._begin_home_recovery("shifted cave view")
    dark = frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    home = [detection("forest_recenter_button", 841, 1296)]
    on_map = [detection("map_exit_nest_button", 841, 1295)]

    switches = 0
    for _ in range(40):
        if planner.is_complete():
            break
        detections = on_map if switches % 2 else home
        target = planner.choose(dark, detections)
        if target is None:
            continue
        if target.type in (RECOVERY_FOREST, RECOVERY_MAP_EXIT):
            switches += 1
        planner.on_action_success(target.type)

    # Home recovery never uses a hunt-map transition as a camera correction.
    assert switches == 0
    assert planner.is_complete()
    assert planner.is_hatch_blocked()
    assert not planner._complete


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


def test_growth_interval_triggers_screening_before_cull_threshold() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._cave_population = 282
    planner._screening_baseline_population = 282
    planner._hatched_since_cave_read = 8
    planner._hatch_child.hatched = 12
    planner.on_action_success(hatch.CLOSE_BUTTON)
    # 估算 302，比最近一次篩選多 20，先篩選但不進洞穴。
    assert planner._management_pending is True
    assert planner._collect_only_after_empty is False
    assert planner._cave_cleanup_after_management is False


def test_cave_estimate_triggers_screening_at_cull_threshold() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    # The screening path must trigger at the configured total-count boundary.
    planner._cave_population = 329
    planner._screening_baseline_population = 329
    planner._hatch_child.hatched = 1
    planner.on_action_success(hatch.CLOSE_BUTTON)
    assert planner._management_pending is True
    assert planner._collect_only_after_empty is False
    assert planner._cave_cleanup_after_management is True


def test_cave_estimate_below_trigger_keeps_collect_only_cycle() -> None:
    planner = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
    )
    planner._child = planner._new_hatch()
    planner._start_hatch_cycle()
    planner._cave_population = 200
    planner._screening_baseline_population = 200
    planner._hatch_child.hatched = 12
    planner.on_action_success(hatch.CLOSE_BUTTON)
    # 估算 212 < 330，維持一般收蛋循環。
    assert planner._management_pending is False
    assert planner._collect_only_after_empty is True


def _edge_home_frame(bar_top: int) -> Frame:
    """Home map whose pile base sits at ``bar_top``, measured by the cyan bar."""

    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    image[bar_top : bar_top + 12, 330:573] = (220, 180, 20)
    return frame(image)


def test_home_recovery_accepts_pile_the_camera_cannot_move_further() -> None:
    """S9 trace: the map bottomed out 165px short and recovery never converged.

    Every nudge swiped, the map sprang back, and the next measurement returned
    the same offset until the attempt budget ran out and the run died.  Two
    stalled corrections now prove the camera is against the edge, so recovery
    accepts the position instead of failing.
    """

    planner = HatchHomeRecoveryPlanner()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    # 165px short of HOME_PILE_BASE, exactly as the live trace measured.
    stuck = _edge_home_frame(1285)

    # The map does not move, so every measurement returns the same offset.
    for _ in range(2):
        nudge = planner.choose(stuck, home)
        assert nudge is not None and nudge.type == RECOVERY_RECENTER
        planner.on_action_success(nudge.type)

    # Two stalled corrections are enough to conclude the map is at its edge.
    planner.choose(stuck, home)
    assert planner._camera_at_limit is True
    # Crucially it did not burn the whole budget nor mark itself failed.
    assert planner.is_failed() is False
    assert planner._measured_corrections < planner.max_measured_corrections

    # It must go on to *prove* home rather than merely stop nudging: the run
    # that crashed did so because recovery never reached a completed state.
    for _ in range(5):
        target = planner.choose(stuck, home)
        if target is not None:
            planner.on_action_success(target.type)
    assert planner.is_complete() is True
    assert planner.is_failed() is False


def test_home_recovery_still_corrects_a_camera_that_is_actually_moving() -> None:
    """A converging camera must not be mistaken for one stuck at the edge."""

    planner = HatchHomeRecoveryPlanner()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]

    first = planner.choose(_edge_home_frame(1100), home)
    assert first is not None and first.type == RECOVERY_RECENTER
    planner.on_action_success(first.type)

    # The offset shrank by well over the stall threshold: keep correcting.
    second = planner.choose(_edge_home_frame(1285), home)
    assert second is not None and second.type == RECOVERY_RECENTER
    assert planner._camera_at_limit is False


def test_cooldown_hunt_queries_survive_a_recovery_child() -> None:
    """Regression: 0.0.65 died mid-recovery with an AssertionError.

    ``_begin_home_recovery`` swaps in a HatchHomeRecoveryPlanner but left the
    rescan flag set, so the next cooldown query reached for ``_hatch_child``
    and asserted.  The flag must clear, and the queries must stay answerable
    whatever child is installed.
    """

    planner = FullHatchPlanner(DigitReader(GLYPHS), egg_pile_point=(450, 1330))
    planner._child = planner._new_hatch()
    planner._empty_rescan_wait = True

    planner._begin_home_recovery("handoff could not confirm centered home")

    assert planner._empty_rescan_wait is False
    assert planner.is_hunt_cooldown_active() is False
    assert planner.hunt_cooldown_delay_ms() == 0


def test_cooldown_hunt_queries_tolerate_a_stale_rescan_flag() -> None:
    """Belt and braces: several stages park a non-hatch child on that flag."""

    planner = FullHatchPlanner(DigitReader(GLYPHS), egg_pile_point=(450, 1330))
    planner._child = object()
    planner._empty_rescan_wait = True

    assert planner.is_hunt_cooldown_active() is False
    assert planner.hunt_cooldown_delay_ms() == 0


def test_default_hatch_cycle_skips_mass_placement() -> None:
    """Mass placement is off by default but still selectable on request.

    It is the least valuable screening pass per minute spent, and every stage
    delays the first hunt of a session.  Removing it from the default must not
    remove the capability, nor leave the cleanup gate waiting on a stage that
    can never run.
    """

    assert "mass" in FULL_HATCH_STAGES
    assert "mass" not in DEFAULT_FULL_HATCH_STAGES
    # Nothing else was dropped along with it.
    assert {"mass"} == FULL_HATCH_STAGES - DEFAULT_FULL_HATCH_STAGES

    planner = make_full_planner()
    assert planner._screening_stages == ("attack", "hp", "top")
    # The cleanup gate must stay satisfiable: it can only ever owe these three.
    assert planner._missing_screening_stages() == ("attack", "hp", "top")
    planner._screening_completed = {"attack", "hp", "top"}
    assert planner._missing_screening_stages() == ()

    # The custom workflow can still ask for it by name.
    explicit = FullHatchPlanner(
        DigitReader(GLYPHS),
        egg_pile_point=(450, 1330),
        max_scrolls=0,
        enabled_stages=("hatch", "collect", "mass"),
    )
    assert explicit._screening_stages == ("mass",)


@pytest.mark.parametrize("ready", [False, True])
def test_incubator_hatches_ready_eggs_and_reads_timer_before_inline_boost(
    tmp_path, monkeypatch, ready,
):
    now = [10_000.0]
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: now[0])
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        boost_inventory=inventory, clock=lambda: now[0],
    )
    planner._capacity_checked = True
    planner._screening_completed = {"attack", "hp", "top"}
    child = planner._child
    monkeypatch.setattr(hatch, "read_hatch_cooldown_seconds", lambda *a, **kw: 600)
    panel = [detection(hatch.INCUBATOR_TITLE, 450, 40), detection(hatch.CLOSE_BUTTON, 800, 1380)]
    ready_egg = panel + [detection(hatch.HATCH_LABEL, 270, 436)]
    screen = boost_ready_frame() if ready else frame()
    chosen = planner.choose(screen, ready_egg)
    assert chosen.type == hatch.HATCH_LABEL
    assert planner._observed_cooldown_until == 10_600
    assert not planner.boost_visit_active()
    planner.on_action_success(chosen.type)
    chosen = planner.choose(screen, [detection(hatch.HATCH_BUTTON, 450, 1185)])
    assert chosen.type == hatch.HATCH_BUTTON
    planner.on_action_success(hatch.HATCH_BUTTON)
    chosen = planner.choose(screen, [detection(hatch.CLAIM_BUTTON, 330, 1242)])
    assert chosen.type == hatch.CLAIM_BUTTON
    planner.on_action_success(hatch.CLAIM_BUTTON)
    assert child.hatched == 1
    if ready:
        assert planner.choose(screen, panel).type == HATCH_BOOST_BUTTON
        planner.on_action_success(HATCH_BOOST_BUTTON)
        prompt = panel + [detection(CONFIRM_YES, 365, 850), detection(CONFIRM_NO, 535, 850)]
        assert planner.choose(screen, prompt).type == HATCH_BOOST_CONFIRM
        planner.on_action_success(HATCH_BOOST_CONFIRM)
        monkeypatch.setattr(hatch, "read_hatch_cooldown_seconds", lambda *a, **kw: 300)
    chosen = planner.choose(frame(), panel)
    assert chosen.type == hatch.CLOSE_BUTTON
    assert not planner.boost_visit_active()
    assert planner._child is child
    assert planner._screening_completed == {"attack", "hp", "top"}
    assert planner._observed_cooldown_until == (10_300 if ready else 10_600)
    assert inventory.snapshot().remaining == (99 if ready else 100)


@pytest.mark.parametrize("stop_line", [330, 350])
def test_custom_preflight_at_stop_line_returns_home_before_hunting(monkeypatch, stop_line):
    _patch_capacity(monkeypatch, stop_line)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        enabled_stages=("collect", "hatch"),
        capacity_limit=350, cull_threshold=stop_line,
    )
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    for _ in range(2):
        chosen = planner.choose(frame(), home)
        assert chosen.type == CAVE_SWIPE
        planner.on_action_success(chosen.type)
    cave = [detection("hatch_cave", 209, 1150)]
    assert planner.choose(frame(), cave) is None
    for _ in range(2):
        chosen = planner.choose(frame(), cave)
        assert chosen.type == CAVE_RECENTER
        assert not planner.is_hatch_blocked()
        planner.on_action_success(chosen.type)
    assert planner.choose(frame(), home) is None
    assert not planner.is_hatch_blocked()
    assert planner.choose(frame(), home) is None
    assert planner.is_hatch_blocked()
    assert planner.population_limit_reached
    assert not planner.begin_hunt_map_capacity_refresh()
    assert not planner.begin_home_collection()
    assert not planner.begin_interim_collection()
    assert planner.choose(frame(), [detection(hatch.HATCH_LABEL)]) is None


@pytest.mark.parametrize("stop_line", [330, 350])
@pytest.mark.parametrize("ready_egg", [False, True])
def test_custom_last_claim_prevents_next_egg_and_boost(tmp_path, stop_line, ready_egg):
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        enabled_stages=("collect", "hatch"), boost_inventory=inventory,
        capacity_limit=350, cull_threshold=stop_line,
    )
    planner._capacity_checked = True
    planner._cave_population = stop_line - 1
    claim = planner.choose(frame(), [detection(hatch.CLAIM_BUTTON, 330, 1242)])
    assert claim.type == hatch.CLAIM_BUTTON
    planner.on_action_success(claim.type)
    panel = [detection(hatch.INCUBATOR_TITLE), detection(hatch.CLOSE_BUTTON, 800, 1380)]
    if ready_egg:
        panel.append(detection(hatch.HATCH_LABEL, 270, 436))
    chosen = planner.choose(boost_ready_frame(), panel)
    assert chosen is None or chosen.type not in {hatch.HATCH_LABEL, HATCH_BOOST_BUTTON}
    assert planner._stage == "recover_home"
    assert planner._hatch_capacity_check_pending
    assert not planner._capacity_checked
    assert not planner.boost_visit_active()
    assert inventory.snapshot().remaining == 100


def test_custom_return_collects_then_hatches_boosts_and_waits_once(tmp_path, monkeypatch):
    now = [10_000.0]
    inventory = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: now[0])
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        enabled_stages=("collect", "hatch"), boost_inventory=inventory,
        clock=lambda: now[0],
    )
    planner._capacity_checked = True
    planner._cave_population = 300
    planner._start_empty_rescan_wait()
    now[0] += 601
    assert planner.begin_home_collection()
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    chosen = planner.choose(frame(), home)
    assert chosen.type == OPEN_NEST
    planner.on_action_success(chosen.type)
    nest = [detection(NEST_TITLE, 450, 260), detection(nest_filter.TAG_HDR_ALL, 228, 168)]
    chosen = planner.choose(frame(), nest + [detection(COLLECT_EGGS_BUTTON, 650, 1315)])
    assert chosen.type == COLLECT_EGGS_BUTTON
    planner.on_action_success(chosen.type)
    chosen = planner.choose(frame(), nest)
    assert chosen.type == NEST_MASK_CLOSE
    planner.on_action_success(chosen.type)
    chosen = planner.choose(frame(), home)
    assert chosen.type == hatch.EGG_PILE
    planner.on_action_success(chosen.type)
    panel = [detection(hatch.INCUBATOR_TITLE), detection(hatch.CLOSE_BUTTON, 800, 1380)]
    chosen = planner.choose(boost_ready_frame(), panel + [detection(hatch.HATCH_LABEL, 270, 436)])
    assert chosen.type == hatch.HATCH_LABEL
    assert not planner.boost_visit_active()
    planner.on_action_success(chosen.type)
    assert planner.choose(frame(), [detection(hatch.HATCH_BUTTON)]).type == hatch.HATCH_BUTTON
    planner.on_action_success(hatch.HATCH_BUTTON)
    assert planner.choose(frame(), [detection(hatch.CLAIM_BUTTON)]).type == hatch.CLAIM_BUTTON
    planner.on_action_success(hatch.CLAIM_BUTTON)
    chosen = planner.choose(boost_ready_frame(), panel)
    assert chosen.type == HATCH_BOOST_BUTTON
    planner.on_action_success(chosen.type)
    prompt = panel + [detection(CONFIRM_YES, 365, 850), detection(CONFIRM_NO, 535, 850)]
    chosen = planner.choose(boost_ready_frame(), prompt)
    assert chosen.type == HATCH_BOOST_CONFIRM
    planner.on_action_success(chosen.type)
    monkeypatch.setattr(hatch, "read_hatch_cooldown_seconds", lambda *a, **kw: 300)
    chosen = planner.choose(frame(), panel)
    assert chosen.type == hatch.CLOSE_BUTTON
    planner.on_action_success(chosen.type)
    assert inventory.snapshot().remaining == 99
    assert planner.is_hunt_cooldown_active()
    assert planner.hunt_cooldown_delay_ms() == 300_000
    assert planner.choose(frame(), home) is None
    assert planner._stage == "hatch"  # No second collection before hunting.


def obscured_s9_capacity_frame() -> Frame:
    # Exact HUD pixels saved at 12:29:44 on 2026-09-08, when 308/370
    # overlapped a building and OCR read 308/1.
    encoded = (FIXTURES / 's9-capacity-obscured-20260908.png.b64').read_text()
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape == (19, 100, 3)
    image = frame().image.copy()
    image[239:258, 10:110] = crop
    return frame(image)


def test_real_s9_wrong_denominator_finishes_known_path_before_retrying():
    planner = CaveCullPlanner(
        DigitReader(GLYPHS), threshold=370, capacity_limit=370,
        capacity_read_retries=1, allow_cull=False,
    )
    blocked = obscured_s9_capacity_frame()
    read = planner._read_capacity(blocked)
    assert read.text == '308/1'
    assert read.reason == 'unexpected_capacity' and read.count is None
    first = planner.choose(frame(), [])
    planner.on_action_success(first.type)
    cave = [detection('hatch_cave', 100, 1100)]
    assert planner.choose(blocked, cave) is None
    second = planner.choose(blocked, cave)
    assert second.type == CAVE_SWIPE
    assert (second.x, second.y) == (350, 800)
    assert second.detection.metadata['swipe']['x2'] == 600
    planner.on_action_success(second.type)
    assert planner.choose(blocked, cave) is None
    back = planner.choose(blocked, cave)
    assert back.type == CAVE_RECENTER
    assert not planner.capacity_readable
    assert planner.last_capacity is None
    assert planner._navigation_swipes == 2


@pytest.mark.parametrize('next_count', [308, 370, None])
def test_custom_unreadable_population_retries_after_wait_and_rechecks_limit(
    tmp_path, monkeypatch, next_count,
):
    now = [1000.0]
    inventory = HatchBoostInventoryStore(tmp_path / 'stats.sqlite3', clock=lambda: now[0])
    inventory.set_enabled(True)
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330),
        enabled_stages=('collect', 'hatch'), capacity_limit=370, cull_threshold=370,
        capacity_read_retries=1, clock=lambda: now[0], boost_inventory=inventory,
    )
    planner._cave_population = 308  # Cached count must not allow a hatch.
    # The immediate map refresh is exhausted; subsequent failures must use
    # the periodic retry window instead of permanently fusing hatching.
    planner._capacity_camera_refresh_used = True
    planner._screening_baseline_population = 298
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    cave = [detection('hatch_cave', 100, 1100)]
    for _ in range(2):
        move = planner.choose(frame(), home)
        assert move.type == CAVE_SWIPE
        planner.on_action_success(move.type)
    blocked = obscured_s9_capacity_frame()
    assert planner.choose(blocked, cave) is None
    for _ in range(2):
        back = planner.choose(blocked, cave)
        assert back.type == CAVE_RECENTER
        planner.on_action_success(back.type)
    assert planner.choose(frame(), home) is None
    assert planner.choose(frame(), home) is None
    assert planner.capacity_retry_pending and not planner.is_hatch_blocked()
    assert not planner._capacity_checked
    assert planner.is_hunt_cooldown_active()
    assert planner.hunt_cooldown_delay_ms() == 600_000
    assert not planner.begin_home_collection()
    assert not planner.begin_interim_collection()
    assert not planner.begin_boost_visit()
    assert planner.choose(boost_ready_frame(), [detection(hatch.HATCH_LABEL)]) is None
    assert inventory.snapshot().remaining == 100
    now[0] += 601
    # Fresh navigation comes before any egg/claim/boost.
    for _ in range(2):
        move = planner.choose(frame(), home)
        assert move.type == CAVE_SWIPE
        planner.on_action_success(move.type)
    monkeypatch.setattr(
        'dino_bot.full_hatch.probe_dino_count',
        lambda *a, **kw: CapacityRead(
            next_count, '' if next_count is None else f'{next_count}/370',
            None if next_count is None else (next_count, 370),
            (10, 239, 110, 258), 'unparsed' if next_count is None else 'ok',
        ),
    )
    assert planner.choose(frame(), cave) is None
    for _ in range(2):
        back = planner.choose(frame(), cave)
        assert back.type == CAVE_RECENTER
        planner.on_action_success(back.type)
    assert planner.choose(frame(), home) is None
    chosen = planner.choose(frame(), home)
    if next_count == 308:
        assert chosen.type == hatch.EGG_PILE
        assert planner._capacity_checked
        assert not planner.capacity_retry_pending
        assert not planner.is_hatch_blocked()
    elif next_count == 370:
        assert chosen is None
        assert planner.population_limit_reached and planner.is_hatch_blocked()
        assert not planner.capacity_retry_pending
    else:
        assert chosen is None
        assert planner.capacity_retry_pending
        assert not planner._capacity_checked
        assert not planner.is_hatch_blocked()
        assert planner.hunt_cooldown_delay_ms() == 600_000


def test_cooldown_preserves_wait_on_shifted_home_and_recovers_only_when_due():
    now = [1000.0]
    planner = FullHatchPlanner(
        DigitReader(GLYPHS), egg_pile_point=(450, 1330), clock=lambda: now[0],
    )
    planner._capacity_checked = True
    planner._start_empty_rescan_wait()
    shifted = np.full((1600, 900, 3), 255, dtype=np.uint8)
    shifted[1085:1097, 342:585] = (220, 180, 20)
    home = [detection(hatch.HOME_ANCHOR, 59, 561)]
    assert is_home_screen(frame(shifted), home)
    assert not is_centered_home_screen(frame(shifted), home)
    assert planner.choose(frame(shifted), home) is None
    assert planner.is_hunt_cooldown_active()
    assert planner._stage == 'hatch'
    now[0] += 601
    planner.choose(frame(shifted), home)
    assert planner._stage == 'recover_home'
    assert not planner.is_hunt_cooldown_active()
