from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dino_bot import hatch
from dino_bot.config import ConfigError, load_config
from dino_bot.hatch import (
    HatchPlanner,
    parse_hatch_timer_text,
    read_hatch_cooldown_seconds,
)
from dino_bot.models import BoundingBox, Detection, Frame


def make_frame(width: int = 900, height: int = 1600) -> Frame:
    return Frame(np.zeros((height, width, 3), dtype=np.uint8))


def detection(type: str, x: int = 100, y: int = 100, confidence: float = 0.95) -> Detection:
    return Detection.from_bbox(type, BoundingBox(x - 5, y - 5, 10, 10), confidence)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_planner(**overrides) -> tuple[HatchPlanner, FakeClock]:
    clock = FakeClock()
    defaults = dict(
        egg_pile_point=(450.0, 1330.0),
        reference_width=900.0,
        scroll_vector=(450.0, 1100.0, 450.0, 500.0),
        scroll_duration_ms=400,
        max_scrolls=2,
        rescan_interval_seconds=600.0,
        require_home_anchor=True,
        home_failure_limit=2,
        home_backoff_seconds=30.0,
        clock=clock,
    )
    defaults.update(overrides)
    return HatchPlanner(**defaults), clock


def test_unknown_screen_yields_no_target() -> None:
    planner, _ = make_planner()
    assert planner.choose(make_frame(), []) is None
    assert planner.last_stage == "unknown_screen"


def test_home_anchor_triggers_scaled_egg_pile_tap() -> None:
    planner, _ = make_planner()
    target = planner.choose(make_frame(width=450, height=800), [detection(hatch.HOME_ANCHOR)])
    assert target is not None
    assert target.type == hatch.EGG_PILE
    assert (target.x, target.y) == (225, 665)


def test_incubator_prefers_leftmost_label_in_top_row() -> None:
    planner, _ = make_planner()
    detections = [
        detection(hatch.INCUBATOR_TITLE, y=40),
        detection(hatch.HATCH_LABEL, x=270, y=404),
        detection(hatch.HATCH_LABEL, x=448, y=400),
        detection(hatch.HATCH_LABEL, x=627, y=396),
        detection(hatch.HATCH_LABEL, x=270, y=668),
    ]
    target = planner.choose(make_frame(), detections)
    assert target is not None
    assert target.type == hatch.HATCH_LABEL
    assert (target.x, target.y) == (270, 404)


def test_incubator_moves_to_next_row_after_top_row_is_gone() -> None:
    planner, _ = make_planner()
    detections = [
        detection(hatch.INCUBATOR_TITLE, y=40),
        detection(hatch.HATCH_LABEL, x=627, y=672),
        detection(hatch.HATCH_LABEL, x=270, y=680),
        detection(hatch.HATCH_LABEL, x=448, y=676),
    ]
    target = planner.choose(make_frame(), detections)
    assert target is not None
    assert (target.x, target.y) == (270, 680)


def test_incubator_without_labels_closes_immediately_by_default() -> None:
    planner, _ = make_planner(max_scrolls=0)
    grid = [detection(hatch.INCUBATOR_TITLE, y=40), detection(hatch.CLOSE_BUTTON, x=800, y=1380)]
    target = planner.choose(make_frame(), grid)
    assert target is not None and target.type == hatch.CLOSE_BUTTON


def test_incubator_scrolling_remains_available_when_explicitly_configured() -> None:
    planner, _ = make_planner(max_scrolls=1)
    grid = [detection(hatch.INCUBATOR_TITLE, y=40), detection(hatch.CLOSE_BUTTON, x=800, y=1380)]
    first = planner.choose(make_frame(), grid)
    assert first is not None and first.type == hatch.SCROLL
    assert first.detection.metadata["swipe"] == {"x2": 450, "y2": 500, "duration_ms": 400}
    planner.on_action_success(hatch.SCROLL)
    second = planner.choose(make_frame(), grid)
    assert second is not None and second.type == hatch.CLOSE_BUTTON


def test_close_starts_rescan_wait_and_resumes() -> None:
    planner, clock = make_planner()
    planner.on_action_success(hatch.CLOSE_BUTTON)
    assert planner.choose(make_frame(), [detection(hatch.HOME_ANCHOR)]) is None
    assert planner.last_stage == "waiting"
    assert planner.next_ready_delay_ms() > 0
    clock.now += 601
    target = planner.choose(make_frame(), [detection(hatch.HOME_ANCHOR)])
    assert target is not None and target.type == hatch.EGG_PILE


def test_beginner_settings_are_per_config_and_reserved_for_later_management(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "hatch": {
                    "beginner_population_limit": 180,
                    "beginner_stat_upgrade_guards": {
                        "attack": {"min_value": 10, "max_value": 120}
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.hatch.beginner_population_limit == 180
    guard = config.hatch.beginner_stat_upgrade_guards["attack"]
    assert (guard.min_value, guard.max_value) == (10, 120)


def test_hatch_timer_parser_accepts_compact_digits_and_rejects_bad_time() -> None:
    assert parse_hatch_timer_text("000537") == 337
    assert parse_hatch_timer_text("00?20?05") == 1205
    assert parse_hatch_timer_text("016099") is None


def test_hatch_cooldown_reader_uses_longest_visible_timer_for_batch() -> None:
    class Reader:
        def __init__(self) -> None:
            self.values = iter(
                [
                    "000537",
                    "000647",
                    "001958",
                    "002005",
                    "005431",
                    "010205",
                    "000000",
                    "??????",
                    "",
                ]
            )

        def read(self, _crop: np.ndarray) -> str:
            return next(self.values)

    assert read_hatch_cooldown_seconds(make_frame().image, Reader()) == 3725


def test_claim_button_takes_priority_and_counts() -> None:
    planner, _ = make_planner()
    detections = [
        detection(hatch.CLAIM_BUTTON, x=330, y=1180),
        detection(hatch.HATCH_BUTTON, x=450, y=1180),
        detection(hatch.INCUBATOR_TITLE, y=40),
        detection(hatch.HATCH_LABEL, x=300, y=500),
    ]
    target = planner.choose(make_frame(), detections)
    assert target is not None and target.type == hatch.CLAIM_BUTTON
    planner.on_action_success(hatch.CLAIM_BUTTON)
    assert planner.hatched == 1


def test_hatch_cycle_completes_only_after_claim() -> None:
    assert hatch.DEFAULT_CYCLE_COMPLETE_TARGETS == (hatch.CLAIM_BUTTON,)


def test_expel_button_is_never_a_target() -> None:
    planner, _ = make_planner()
    assert planner.choose(make_frame(), [detection(hatch.EXPEL_BUTTON)]) is None


def test_repeated_egg_pile_failures_back_off() -> None:
    planner, clock = make_planner()
    home = [detection(hatch.HOME_ANCHOR)]
    assert planner.choose(make_frame(), home) is not None
    planner.on_action_failure(hatch.EGG_PILE)
    planner.on_action_failure(hatch.EGG_PILE)
    assert planner.choose(make_frame(), home) is None
    clock.now += 31
    assert planner.choose(make_frame(), home) is not None


def test_hatch_label_click_offset_targets_egg_body() -> None:
    # 2026-08-01 實機 T5:點「孵化」標籤文字本身不會開詳細頁,必須點標籤
    # 上方的蛋本體(標籤 bbox 左上 + (29,-99) ≈ 蛋中心)。
    manifest = json.loads(
        (Path(__file__).parent.parent / "assets/hatch/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    label = next(t for t in manifest["templates"] if t["type"] == "hatch_label")
    assert label["click_offset"] == [29, -99]


def test_hatch_config_defaults_load(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = load_config(config_path)
    assert config.hatch.rescan_interval_seconds == 600
    assert config.hatch.screening_growth_interval == 20
    assert config.hatch.egg_pile == (450.0, 1330.0)
    assert config.hatch.max_scrolls == 0
    assert config.hatch.manifest == tmp_path / "assets/hatch/manifest.json"
    assert config.hatch.stat_upgrade_guards["hp"].min_delta is None
    assert config.hatch.stat_upgrade_guards["hp"].max_delta is None
    assert config.hatch.stat_upgrade_guards["hp"].multiple_of == 10
    assert config.hatch.stat_upgrade_guards["attack"].min_delta is None
    assert config.hatch.stat_upgrade_guards["attack"].max_delta is None
    assert config.hatch.stat_upgrade_guards["speed"].min_value == 1
    assert config.hatch.stat_upgrade_guards["speed"].max_value == 150
    assert config.hatch.stat_consistent_reads == 2
    assert config.hatch.stat_read_retries == 3


def test_hatch_config_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"hatch": {"egg_pile": [400, 1200], "rescan_interval_seconds": 300,'
        ' "screening_growth_interval": 15,'
        ' "max_scrolls": 6}}',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.hatch.egg_pile == (400.0, 1200.0)
    assert config.hatch.rescan_interval_seconds == 300
    assert config.hatch.screening_growth_interval == 15
    assert config.hatch.max_scrolls == 6


def test_hatch_config_allows_future_stat_ranges(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"hatch": {"stat_upgrade_guards": {"hp": {"max_delta": 40},'
        '"attack": {"max_delta": 4}, "speed": {"max_value": 180}}}}',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.hatch.stat_upgrade_guards["hp"].max_delta == 40
    assert config.hatch.stat_upgrade_guards["attack"].max_delta == 4
    assert config.hatch.stat_upgrade_guards["speed"].max_value == 180


def test_hatch_config_rejects_invalid_stat_calibration_settings(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"hatch": {"stat_consistent_reads": 3, "stat_read_retries": 2}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="stat_read_retries"):
        load_config(config_path)


def test_hatch_config_rejects_zero_stat_multiple(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"hatch": {"stat_upgrade_guards": {"hp": {"multiple_of": 0}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="multiple_of"):
        load_config(config_path)
