from __future__ import annotations

import numpy as np

from dino_bot import hatch
from dino_bot.hatch import HatchPlanner
from dino_bot.models import BoundingBox, Detection, Frame
from dino_bot.config import load_config


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


def test_incubator_prefers_topmost_label() -> None:
    planner, _ = make_planner()
    detections = [
        detection(hatch.INCUBATOR_TITLE, y=40),
        detection(hatch.HATCH_LABEL, x=300, y=900),
        detection(hatch.HATCH_LABEL, x=600, y=400),
    ]
    target = planner.choose(make_frame(), detections)
    assert target is not None
    assert target.type == hatch.HATCH_LABEL
    assert target.y == 400


def test_incubator_without_labels_scrolls_then_closes() -> None:
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


def test_hatch_config_defaults_load(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = load_config(config_path)
    assert config.hatch.rescan_interval_seconds == 600
    assert config.hatch.egg_pile == (450.0, 1330.0)
    assert config.hatch.manifest == tmp_path / "assets/hatch/manifest.json"


def test_hatch_config_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"hatch": {"egg_pile": [400, 1200], "rescan_interval_seconds": 300,'
        ' "max_scrolls": 6}}',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.hatch.egg_pile == (400.0, 1200.0)
    assert config.hatch.rescan_interval_seconds == 300
    assert config.hatch.max_scrolls == 6
