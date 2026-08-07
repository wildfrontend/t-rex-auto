from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import action as action_facade
import capture as capture_facade
import detector as detector_facade
import planner as planner_facade
from dino_bot.actions import AdbActionDriver, AdbClient, RecordingActionDriver
from dino_bot.assets import create_template
from dino_bot.cli import apply_run_timing, build_parser
from dino_bot.config import AppConfig, ConfigError, load_config
from dino_bot.cull import CapacityRead
from dino_bot.detection import (
    DetectorAssetError,
    HuntCapacityDetector,
    HuntTeamAvailabilityDetector,
    OpenCvDetector,
    TargetTooStrongDetector,
)
from dino_bot.engine import BotContext, BotEngine, BotState
from dino_bot.models import (
    ActionCommand,
    BoundingBox,
    Detection,
    Frame,
    Target,
    VerificationResult,
)
from dino_bot.modes import DebugMode, RuntimeMode, TrainingMode
from dino_bot.planning import HuntPlanner, TargetPlanner
from dino_bot.recovery import AdbAppRestarter, BlackScreenRecovery
from dino_bot.stalls import (
    HUD_ZOOM,
    CapacitySnapshotWriter,
    StallSnapshotWriter,
)
from dino_bot.verification import TargetChangedVerifier


def make_frame(value: int = 0, sequence: int = 1) -> Frame:
    return Frame(np.full((100, 160, 3), value, dtype=np.uint8), sequence=sequence)


def make_detection(x: int = 80, y: int = 50, type: str = "resource") -> Detection:
    return Detection.from_bbox(type, BoundingBox(x - 5, y - 5, 10, 10), 0.95)


@pytest.mark.parametrize(
    ("training", "message"),
    [
        ({"fps": 0, "max_images": 500}, "training.fps"),
        ({"fps": 6, "max_images": 500}, "training.fps"),
        ({"fps": 2, "max_images": 501}, "training.max_images"),
    ],
)
def test_config_enforces_training_collection_limits(
    tmp_path: Path,
    training: dict[str, int],
    message: str,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"training": training}), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_config_requires_positive_verification_minimum_checks(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"verify": {"minimum_checks": 0}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="verify.minimum_checks"):
        load_config(config_file)


def test_config_rejects_a_negative_center_distance_limit(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"planner": {"max_center_distance_px": -1}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="planner.max_center_distance_px"):
        load_config(config_file)


def test_cli_fast_speed_profile_reduces_hunt_delays() -> None:
    config = AppConfig(
        root=Path("."),
        click_delay=1500,
        idle_delay=500,
        post_action_delays={
            "hunt_button": 5000,
            "hunt_confirm_button": 3000,
        },
    )

    result = apply_run_timing(config, speed="fast")

    assert result.click_delay == 300
    assert result.idle_delay == 250
    assert result.post_action_delays["dinosaur"] == 300
    assert result.post_action_delays["hunt_button"] == 900
    assert result.post_action_delays["hunt_confirm_button"] == 1200


def test_cli_safe_speed_profile_uses_conservative_delays() -> None:
    config = AppConfig(
        root=Path("."),
        click_delay=300,
        idle_delay=250,
        transition_poll_interval=100,
        post_action_delays={
            "hunt_button": 900,
            "hunt_confirm_button": 1200,
        },
    )

    result = apply_run_timing(config, speed="safe")

    assert result.click_delay == 1500
    assert result.idle_delay == 500
    assert result.transition_poll_interval == 250
    assert result.post_action_delays["dinosaur"] == 1500
    assert result.post_action_delays["hunt_button"] == 5000
    assert result.post_action_delays["hunt_confirm_button"] == 3000
    assert result.timing_profile == "safe"
    assert result.post_action_delays["hatch_cave_swipe"] == 6000
    assert result.post_action_delays["hatch_cave_recenter"] == 7000
    assert result.hatch.capacity_read_retries == 4
    assert result.hatch.cave_recenter_checks == 5
    assert result.hatch.recovery_timeout_seconds == 30.0


def test_cli_leaves_speed_unset_for_config_default() -> None:
    args = build_parser().parse_args(["run"])

    assert args.speed is None


def test_cli_explicit_timing_overrides_profile() -> None:
    result = apply_run_timing(
        AppConfig(root=Path(".")),
        speed="safe",
        dinosaur_delay_ms=700,
        hunt_button_delay_ms=1800,
        hunt_confirm_delay_ms=1200,
        idle_delay_ms=100,
    )

    assert result.post_action_delays["dinosaur"] == 700
    assert result.post_action_delays["hunt_button"] == 1800
    assert result.post_action_delays["hunt_confirm_button"] == 1200
    assert result.idle_delay == 100


def test_cli_parses_terminal_timing_options() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--speed",
            "fast",
            "--hunt-button-delay-ms",
            "1800",
            "--status-port",
            "9876",
        ]
    )

    assert args.speed == "fast"
    assert args.hunt_button_delay_ms == 1800
    assert args.status_port == 9876


def test_cli_parses_json_status_command() -> None:
    args = build_parser().parse_args(["status", "--json", "--actions", "5"])

    assert args.command == "status"
    assert args.json is True
    assert args.actions == 5


def test_cli_parses_diagnostics_bundle_options() -> None:
    args = build_parser().parse_args(
        ["diagnostics", "--include-screenshot", "--log-lines", "750"]
    )

    assert args.command == "diagnostics"
    assert args.include_screenshot is True
    assert args.log_lines == 750


def test_adb_client_discovers_android_sdk_for_current_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adb = tmp_path / "Android" / "Sdk" / "platform-tools" / "adb.exe"
    adb.parent.mkdir(parents=True)
    adb.touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.setattr("dino_bot.actions.shutil.which", lambda _: None)

    assert AdbClient._resolve_executable(None) == str(adb)


def test_compatibility_facades_are_independently_callable() -> None:
    class OneFrameCapture:
        def __init__(self) -> None:
            self.closed = False

        def capture(self) -> Frame:
            return make_frame(7)

        def close(self) -> None:
            self.closed = True

    class OneDetectionDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            return [make_detection(type="resource")]

    provider = OneFrameCapture()
    capture_facade.configure(provider)
    image = capture_facade.capture()
    assert isinstance(image, np.ndarray) and int(image[0, 0, 0]) == 7

    detector_facade.configure(OneDetectionDetector())
    detections = detector_facade.detect(image)
    planner_facade.configure(TargetPlanner(("resource",)))
    target = planner_facade.choose(image, detections)
    assert target is not None and target.type == "resource"

    driver = RecordingActionDriver()
    action_facade.configure(driver, image)
    action_facade.tap(target.x, target.y)
    assert driver.actions == [ActionCommand.tap(target.x, target.y)]
    capture_facade.close()
    assert provider.closed


def test_planner_nearest_center() -> None:
    frame = make_frame()
    detections = [make_detection(10, 10), make_detection(82, 51), make_detection(80, 50, "mail")]
    target = TargetPlanner(("resource",), "nearest_center").choose(frame, detections)
    assert target is not None
    assert (target.x, target.y) == (82, 51)


def test_planner_highest_confidence() -> None:
    frame = make_frame()
    low = Detection("resource", 80, 50, 0.5)
    high = Detection("resource", 10, 10, 0.99)
    target = TargetPlanner(("resource",), "highest_confidence").choose(frame, [low, high])
    assert target is not None
    assert target.confidence == 0.99


def test_planner_prioritizes_target_type_order() -> None:
    frame = make_frame()
    dinosaur = Detection("dinosaur", 80, 50, 0.99)
    hunt_button = Detection("hunt_button", 10, 10, 0.8)
    target = TargetPlanner(("hunt_button", "dinosaur")).choose(
        frame, [dinosaur, hunt_button]
    )
    assert target is not None
    assert target.type == "hunt_button"


def test_planner_blocks_actions_while_failure_alert_is_visible() -> None:
    planner = TargetPlanner(
        ("dinosaur",),
        blocking_types=("duplicate_hunt_alert",),
    )
    detections = [
        make_detection(type="dinosaur"),
        make_detection(type="duplicate_hunt_alert"),
    ]
    assert planner.choose(make_frame(), detections) is None


def test_planner_persists_and_excludes_selected_dinosaurs(tmp_path: Path) -> None:
    history_file = tmp_path / "target-history.json"
    planner = TargetPlanner(
        ("dinosaur",),
        deduplicate_types=("dinosaur",),
        dedup_radius=20,
        history_file=history_file,
    )
    first = planner.choose(make_frame(), [make_detection(80, 50, "dinosaur")])
    assert first is not None
    assert planner.choose(make_frame(), [make_detection(82, 50, "dinosaur")]) is None

    reloaded = TargetPlanner(
        ("dinosaur",),
        deduplicate_types=("dinosaur",),
        dedup_radius=20,
        history_file=history_file,
    )
    assert reloaded.choose(make_frame(), [make_detection(82, 50, "dinosaur")]) is None
    assert reloaded.choose(make_frame(), [make_detection(130, 50, "dinosaur")]) is not None


def test_hunt_planner_uses_egg_anchor_and_recenters_after_batch() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    exit_button = Detection("map_exit_nest_button", 841, 1295, 1.0)
    forest_button = Detection("forest_recenter_button", 841, 1295, 1.0)
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    planner = HuntPlanner(
        (
            "forest_recenter_button",
            "map_exit_nest_button",
            "hunt_confirm_button",
            "dinosaur",
        ),
        deduplicate_types=("dinosaur",),
        dedup_radius=25,
        recenter_every=2,
        map_settle_frames=1,
        safe_margin=80,
    )

    near = Detection("dinosaur", 500, 820, 0.9)
    far = Detection("dinosaur", 100, 100, 0.99)
    first = planner.choose(frame, [anchor, exit_button, near, far])
    assert first is not None and (first.x, first.y) == (500, 820)

    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    second_dinosaur = Detection("dinosaur", 400, 850, 0.9)
    second = planner.choose(frame, [anchor, exit_button, second_dinosaur])
    assert second is not None and second.type == "dinosaur"
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")

    leave_map = planner.choose(frame, [anchor, exit_button])
    assert leave_map is not None and leave_map.type == "map_exit_nest_button"
    enter_forest = planner.choose(frame, [forest_button])
    assert enter_forest is not None and enter_forest.type == "forest_recenter_button"

    assert planner.choose(frame, [anchor, exit_button, near]) is None
    next_batch = planner.choose(frame, [anchor, exit_button, near])
    assert next_batch is not None and next_batch.type == "dinosaur"


def test_hunt_planner_retries_forest_when_recenter_tap_is_ignored() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("forest_recenter_button", "map_center_egg", "dinosaur"),
        safe_margin=80,
    )
    forest_button = Detection("forest_recenter_button", 841, 1295, 1.0)
    centered_anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 500, 820, 0.9)

    first_attempt = planner.choose(frame, [forest_button])
    assert first_attempt is not None and first_attempt.type == "forest_recenter_button"

    retry = planner.choose(frame, [forest_button])
    assert retry is not None and retry.type == "forest_recenter_button"

    assert planner.choose(frame, [centered_anchor]) is None
    resumed = planner.choose(frame, [centered_anchor, dinosaur])
    assert resumed is not None and resumed.type == "dinosaur"


def test_hunt_planner_accepts_stable_dinosaur_only_recenter_result() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("forest_recenter_button", "map_center_egg", "dinosaur"),
        safe_margin=80,
        map_settle_frames=2,
    )
    forest_button = Detection("forest_recenter_button", 841, 1295, 1.0)
    dinosaur = Detection("dinosaur", 650, 800, 0.9)

    target = planner.choose(frame, [forest_button])
    assert target is not None and target.type == "forest_recenter_button"
    planner.on_action_success("forest_recenter_button")

    assert planner.choose(frame, [dinosaur]) is None
    assert planner.choose(frame, [dinosaur]) is None
    resumed = planner.choose(frame, [dinosaur])
    assert resumed is not None and resumed.type == "dinosaur"


def test_hunt_planner_excludes_dinosaur_on_own_blue_path() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        own_path_radius=90,
        safe_margin=80,
    )
    detections = [
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("own_hunt_path", 500, 820, 0.9),
        Detection("dinosaur", 500, 820, 0.99),
        Detection("dinosaur", 650, 820, 0.8),
    ]
    target = planner.choose(frame, detections)
    assert target is not None and (target.x, target.y) == (650, 820)


def test_hunt_planner_never_clicks_dinosaur_in_bottom_ui() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        safe_margin=80,
        bottom_exclusion_px=180,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    bottom_ui_false_positive = Detection("dinosaur", 225, 1542, 0.99)
    safe_dinosaur = Detection("dinosaur", 650, 1000, 0.80)

    target = planner.choose(
        frame,
        [anchor, bottom_ui_false_positive, safe_dinosaur],
    )
    assert target is not None and (target.x, target.y) == (650, 1000)

    bottom_only_planner = HuntPlanner(
        ("dinosaur",),
        safe_margin=80,
        bottom_exclusion_px=180,
    )
    assert bottom_only_planner.choose(
        frame,
        [anchor, bottom_ui_false_positive],
    ) is None


def test_hunt_planner_waits_when_all_dinosaurs_are_on_own_blue_path() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        own_path_radius=90,
        safe_margin=80,
    )
    detections = [
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("own_hunt_path", 500, 820, 0.9),
        Detection("own_hunt_path", 560, 820, 0.9),
        Detection("dinosaur", 530, 820, 0.99),
    ]
    assert planner.choose(frame, detections) is None


def test_hunt_planner_recenters_after_seconds_without_a_safe_dinosaur() -> None:
    """The stall threshold counts seconds, not frames.

    A frame count buys a different wait on every machine: the same four frames
    were 4.3 seconds when a scan cost 1080ms and 15 seconds when the host was
    busy enough to push it to 3668ms. Seconds hold the wait still.

    The grace period survived the move to supply-driven recentering: a corridor
    full of the bot's own routes clears itself as hunts return, so resetting the
    instant supply dips would trade seconds of waiting for a whole map reload.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        own_path_radius=90,
        stalled_recenter_seconds=10.0,
        safe_margin=80,
        clock=lambda: now[0],
    )
    detections = [
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("mailbox_button", 841, 1210, 0.99),
        Detection("own_hunt_path", 500, 820, 0.9),
        Detection("dinosaur", 500, 820, 0.99),
    ]

    assert planner.choose(frame, detections) is None
    now[0] = 9.0
    assert planner.choose(frame, detections) is None, "still inside the window"

    now[0] = 10.0
    reset = planner.choose(frame, detections)
    assert reset is not None and reset.type == "map_exit_nest_button"
    assert planner.last_recenter_reason() == "low_supply"
    assert planner.last_supply() == 0


def test_hunt_planner_keeps_hunting_once_the_anchor_is_only_predicted() -> None:
    """Tapping a dinosaur must not disqualify the next one.

    `anchor_window` asks whether a tap would push the centre egg off screen.
    That protects the egg, and the egg is not the point: recentering exists to
    restore the supply of reachable dinosaurs, and the egg is only how the bot
    recognises the reset finished. Enforced against an anchor that is *inferred*
    from the previous tap - which is the state for 88% of a measured run - the
    rule threw away 1344 candidates, emptied 22% of all planning cycles, and
    then charged a recenter to refill a map that was never empty.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",), clock=lambda: 0.0)
    # The egg sits off centre, which is the only way the rule can bite: with the
    # anchor exactly centred the projection is a point reflection through the
    # centre, and `screen_margin` has already excluded everything it could
    # reject.
    egg = Detection("map_center_egg", 250, 500, 1.0)
    far = Detection("dinosaur", 700, 1100, 0.95)

    # Measured anchor: the rule applies, and this tap would throw the egg off
    # screen, so it is refused.
    assert planner.choose(frame, [egg, far]) is None
    assert planner.anchor_measured()
    assert planner.last_rejections().get("anchor_window") == 1

    # Same dinosaur, same anchor, but now only predicted. The egg is no longer
    # worth protecting, so the hunt goes ahead.
    planner._anchor_measured = False
    target = planner.choose(frame, [far])

    assert target is not None and target.type == "dinosaur"
    assert "anchor_window" not in planner.last_rejections()
    assert planner.last_supply() == 1


def test_hunt_planner_counts_only_dinosaurs_it_could_actually_tap() -> None:
    """A dinosaur already hunted this map is scenery, not supply.

    It stays on screen and keeps passing every rejection rule, but `choose`
    drops it as a duplicate. Counting it left the map reporting supply while
    planning nothing, so the resupply trigger never fired: a measured run sat
    on `supply=2` through twenty seconds of empty cycles before the blind-stall
    timer had to rescue it. Resetting the map is exactly what should have
    happened, and it is what the operator expected to see.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        deduplicate_types=("dinosaur",),
        dedup_radius=25.0,
        stalled_recenter_seconds=10.0,
        clock=lambda: now[0],
    )
    planner._last_anchor = (450.0, 800.0)
    landmark = Detection("mailbox_button", 841, 1210, 0.99)
    lone = Detection("dinosaur", 450, 640, 0.95)

    first = planner.choose(frame, [landmark, lone])
    assert first is not None and first.type == "dinosaur"
    assert planner.last_supply() == 1

    # The hunt went out; the map still shows the dinosaur that was spent on it.
    planner._awaiting_hunt_button = False
    assert planner.choose(frame, [landmark, lone]) is None
    assert planner.last_supply() == 0, "a spent dinosaur is not supply"

    now[0] = 10.0
    reset = planner.choose(frame, [landmark, lone])

    assert reset is not None and reset.type == "map_exit_nest_button"
    assert planner.last_recenter_reason() == "low_supply"


def test_hunt_planner_skips_dinosaurs_too_far_from_the_viewport_center() -> None:
    """A far tap recenters the map without opening the hunt panel.

    Measured over 161 minutes: taps landing within 300 px of the center opened
    the panel 86% of the time, 300-500 px 69%, and past 500 px only 21%. The
    band beyond 600 px produced 2 hunts out of 31 taps while the failures each
    cost a full verify budget, so the limit removes near-pure waste.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        max_center_distance_px=600.0,
        safe_margin=40,
    )
    planner._last_anchor = (450.0, 800.0)
    landmark = Detection("mailbox_button", 841, 1210, 0.99)
    # 700 px straight up from the center at (450, 800).
    far = Detection("dinosaur", 450, 100, 0.95)
    near = Detection("dinosaur", 450, 500, 0.95)

    assert planner.choose(frame, [landmark, far]) is None
    assert planner.last_supply() == 0, "an unreachable dinosaur is not supply"
    assert planner.last_rejections().get("center_distance") == 1

    chosen = planner.choose(frame, [landmark, near])
    assert chosen is not None and chosen.type == "dinosaur"


def test_hunt_planner_center_distance_limit_can_be_disabled() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        max_center_distance_px=0.0,
        safe_margin=40,
    )
    planner._last_anchor = (450.0, 800.0)
    landmark = Detection("mailbox_button", 841, 1210, 0.99)
    far = Detection("dinosaur", 450, 100, 0.95)

    chosen = planner.choose(frame, [landmark, far])
    assert chosen is not None and chosen.type == "dinosaur"


def test_hunt_planner_prefers_the_tap_that_moves_the_map_least() -> None:
    """Displacement is measured from the screen centre, not from the anchor.

    A tap re-centres the map on the dinosaur, so its distance from the *screen*
    centre is exactly how far the view is about to travel. Ranking from the
    anchor measured the wrong thing the moment the anchor stopped being
    centred - and needed an anchor at all. Cheaper taps mean more hunts before
    the map runs dry and has to be reset.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",), ring_width=75.0, clock=lambda: 0.0)
    planner._last_anchor = (120.0, 300.0)      # stale, far from the screen centre
    planner._anchor_measured = False
    near = Detection("dinosaur", 450, 640, 0.9)    # 160px from the screen centre
    far = Detection("dinosaur", 200, 400, 0.99)    # closer to the stale anchor

    target = planner.choose(frame, [near, far])

    assert target is not None
    assert (target.x, target.y) == (450, 640)


def test_hunt_planner_recenters_when_only_own_paths_remain() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        stalled_recenter_seconds=10.0,
        clock=lambda: now[0],
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    own_path = Detection("own_hunt_path", 500, 820, 0.9)

    assert planner.choose(frame, [anchor, own_path]) is None
    now[0] = 10.0
    reset = planner.choose(frame, [own_path])

    assert reset is not None and reset.type == "map_exit_nest_button"
    assert reset.detection.metadata["detector"] == "map_landmark_fallback"


def test_hunt_planner_escapes_a_stall_with_no_landmark_at_all() -> None:
    """The stall every other guard is blind to, because they all need a landmark.

    Reproduces the measured episode: a batch recenter arms stage 1, which waits
    for the nest or forest button, and the screen shows neither - only dinosaur
    labels. `_choose_map_exit`'s fallback needs an anchor the preceding restart
    cleared, and the recenter timeout in the hunting branch never runs because
    it is guarded on `on_collect_map`. One run held this for 519 seconds across
    two game restarts.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        blind_idle_seconds=20.0,
        clock=lambda: now[0],
    )
    # The state a batch recenter leaves behind, with the anchor cleared by the
    # restart that preceded it.
    planner._recenter_stage = 1
    planner._last_anchor = None
    blind_screen = [
        Detection("dinosaur", 500, 620, 0.94),
        Detection("dinosaur", 300, 700, 0.88),
        Detection("own_hunt_path", 480, 640, 0.8),
    ]

    assert planner.choose(frame, blind_screen) is None
    assert planner.last_stage() == "recenter"
    assert planner.take_blind_escape() is None, "nothing to report yet"

    now[0] = 19.0
    assert planner.choose(frame, blind_screen) is None
    assert planner.take_blind_escape() is None, "still inside the window"

    now[0] = 20.0
    assert planner.choose(frame, blind_screen) is None
    escape = planner.take_blind_escape()

    assert escape is not None
    assert escape["stage"] == "recenter"
    assert escape["escapes"] == 1
    assert escape["seconds"] == pytest.approx(20.0)
    assert planner.take_blind_escape() is None, "reported once, not every cycle"
    assert planner._recenter_stage == 0, "the parked stage is released"

    # Releasing the stage is not enough by itself. Without an anchor the
    # hunting branch reads this same screen as "on the map, anchor lost" and
    # arms the identical recenter again; replaying the measured stall went
    # round that loop ten times. Adopting the frame centre is what lets the
    # next cycle act - on a detected dinosaur, not on a blind coordinate.
    now[0] = 22.0
    recovered = planner.choose(frame, blind_screen)

    assert recovered is not None and recovered.type == "dinosaur"


def test_hunt_planner_blind_escape_re_arms_until_the_stall_ends() -> None:
    """A release that does not reach the cause has to keep reporting.

    Latching after the first escape would hide exactly the episodes worth
    knowing about: the ones where nothing the planner can do is enough and the
    watchdog has to restart the app.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        blind_idle_seconds=20.0,
        clock=lambda: now[0],
    )
    planner._recenter_stage = 1
    planner._last_anchor = None
    # Nothing matched at all, so no release can produce a target.
    blind_screen: list[Detection] = []

    planner.choose(frame, blind_screen)
    for elapsed, expected in ((20.0, 1), (40.0, 2), (60.0, 3)):
        now[0] = elapsed
        assert planner.choose(frame, blind_screen) is None
        escape = planner.take_blind_escape()
        assert escape is not None and escape["escapes"] == expected

    # A cycle that plans something ends the episode, so the next one counts
    # from one again rather than continuing to escalate.
    now[0] = 61.0
    assert planner.choose(frame, [Detection("dinosaur", 500, 620, 0.94)]) is not None
    now[0] = 200.0
    planner.choose(frame, blind_screen)
    now[0] = 220.0
    planner.choose(frame, blind_screen)

    escape = planner.take_blind_escape()
    assert escape is not None and escape["escapes"] == 1


def test_hunt_planner_blind_timer_ignores_deliberate_waits() -> None:
    """A cooldown the planner set itself is a wait, not a stall."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("dinosaur",),
        blind_idle_seconds=20.0,
        capacity_wait_seconds=300.0,
        clock=lambda: now[0],
    )
    capacity_full = [Detection("hunt_capacity_full", 450, 800, 0.99)]

    for elapsed in (0.0, 20.0, 100.0, 280.0):
        now[0] = elapsed
        assert planner.choose(frame, capacity_full) is None
        assert planner.last_stage() == "capacity_wait"
        assert planner.take_blind_escape() is None
    assert planner.last_blind_seconds() == 0.0


def test_hunt_planner_abandons_a_mail_flow_that_stops_advancing() -> None:
    """55 consecutive cycles waited for a mailbox that was never on screen.

    `_resume_hunting_after_mail` releases the flow only when a map landmark
    and a dinosaur are both visible, so a screen with dinosaurs and no landmark
    satisfies neither the stage nor its escape.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("mailbox_button", "dinosaur"),
        mail_stage_timeout_seconds=20.0,
        clock=lambda: now[0],
    )
    planner._mail_stage = 1
    planner._total_hunt_count = 30
    blind_screen = [Detection("dinosaur", 500, 620, 0.94)]

    assert planner.choose(frame, blind_screen) is None
    assert planner.last_stage() == "mail"

    now[0] = 19.0
    assert planner.choose(frame, blind_screen) is None
    assert planner._mail_stage == 1, "still inside the window"

    now[0] = 20.0
    assert planner.choose(frame, blind_screen) is None

    assert planner._mail_stage == 0
    # The counter has to go with the stage: left armed, the next recenter walks
    # straight back into the same flow.
    assert planner._total_hunt_count == 0
    # A stage carrying its own deadline releases itself. Letting the generic
    # blind timer fire here too would report a handled timeout as an
    # unexplained stall, snapshot included.
    assert planner.take_blind_escape() is None


def test_hunt_planner_mail_retry_loop_does_not_hold_off_its_own_deadline() -> None:
    """A re-offered button is a retry, not progress.

    Stage 2 falls back to the mailbox when its own collect-all button is
    missing, without advancing the stage. Counting any returned target as
    progress let that fallback tap the mailbox for 117 measured seconds with
    the deadline never once consulted.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("mailbox_button", "mail_collect_all_button", "dinosaur"),
        mail_stage_timeout_seconds=20.0,
        clock=lambda: now[0],
    )
    planner._mail_stage = 2
    planner._total_hunt_count = 30
    # The collect-all button never appears - an already-emptied mailbox.
    mailbox_only = [Detection("mailbox_button", 841, 1210, 0.99)]

    for elapsed in (0.0, 6.0, 12.0, 18.0):
        now[0] = elapsed
        target = planner.choose(frame, mailbox_only)
        assert target is not None and target.type == "mailbox_button"
        assert planner._mail_stage == 2

    now[0] = 20.0

    assert planner.choose(frame, mailbox_only) is None, (
        "the flow was abandoned, so its buttons stop being pressed"
    )
    assert planner._mail_stage == 0


def test_hunt_planner_mail_timeout_outranks_the_blind_timer() -> None:
    """A longer mail deadline has to survive the generic one, or it is fiction."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("mailbox_button", "dinosaur"),
        mail_stage_timeout_seconds=60.0,
        blind_idle_seconds=20.0,
        clock=lambda: now[0],
    )
    planner._mail_stage = 1
    planner._total_hunt_count = 30

    for elapsed in (0.0, 21.0, 45.0, 59.0):
        now[0] = elapsed
        assert planner.choose(frame, []) is None
        assert planner._mail_stage == 1, f"released early at {elapsed}s"
        assert planner.take_blind_escape() is None

    now[0] = 60.0
    assert planner.choose(frame, []) is None
    assert planner._mail_stage == 0

    # Once mail lets go, the generic timer owns the stall again.
    now[0] = 80.0
    planner.choose(frame, [])
    now[0] = 100.0
    planner.choose(frame, [])

    escape = planner.take_blind_escape()
    assert escape is not None and escape["stage"] == "hunting"


def test_hunt_planner_mail_timeout_measures_progress_not_duration() -> None:
    """A slow mail flow is not a stalled one, so the clock restarts on progress."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("mailbox_button", "mail_collect_all_button", "dinosaur"),
        mail_stage_timeout_seconds=20.0,
        clock=lambda: now[0],
    )
    planner._mail_stage = 1
    mailbox = Detection("mailbox_button", 841, 1210, 0.99)
    collect_all = Detection("mail_collect_all_button", 450, 1200, 0.99)

    now[0] = 15.0
    assert planner.choose(frame, [mailbox]) is not None
    now[0] = 30.0
    assert planner.choose(frame, [collect_all]) is not None

    assert planner._mail_stage == 3, "35 seconds in and the flow is still advancing"


def test_hunt_planner_counts_only_verified_confirmation() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("hunt_confirm_button",))
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)

    assert planner.choose(frame, [confirm]) is not None
    assert planner._total_hunt_count == 0

    planner.on_action_success("hunt_confirm_button")

    assert planner._total_hunt_count == 1


def test_hunt_planner_collects_mail_after_hunt_threshold() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        (
            "duplicate_login_close_button",
            "device_history_confirm_button",
            "startup_offer_dismiss",
            "mail_reward_collect_button",
            "mail_collect_all_button",
            "mail_close_button",
            "mailbox_button",
            "forest_recenter_button",
            "map_exit_nest_button",
            "hunt_confirm_button",
            "dinosaur",
        ),
        recenter_every=1,
        mail_after_hunts=1,
        map_settle_frames=1,
        safe_margin=80,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    exit_button = Detection("map_exit_nest_button", 841, 1295, 1.0)
    dinosaur = Detection("dinosaur", 500, 820, 0.9)
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    forest = Detection("forest_recenter_button", 841, 1295, 1.0)
    mailbox = Detection("mailbox_button", 841, 1210, 1.0)
    collect_all = Detection("mail_collect_all_button", 636, 1165, 1.0)
    reward = Detection("mail_reward_collect_button", 450, 910, 1.0)
    close = Detection("mail_close_button", 450, 1380, 1.0)
    duplicate_login_close = Detection(
        "duplicate_login_close_button",
        449,
        1029,
        1.0,
    )
    device_confirm = Detection("device_history_confirm_button", 333, 900, 1.0)
    offer_dismiss = Detection("startup_offer_dismiss", 800, 800, 1.0)

    # A visible mailbox must remain inert until the hunt threshold is reached.
    assert planner.choose(frame, [anchor, exit_button, mailbox, dinosaur]).type == "dinosaur"  # type: ignore[union-attr]
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    assert planner.choose(frame, [anchor, exit_button]).type == "map_exit_nest_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [forest]).type == "forest_recenter_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [anchor, exit_button, mailbox]) is None
    assert planner.choose(frame, [anchor, exit_button, mailbox]).type == "mailbox_button"  # type: ignore[union-attr]
    # Login and startup overlays preempt mail without advancing its stage.
    interruption = planner.choose(frame, [collect_all, duplicate_login_close])
    assert interruption is not None and interruption.type == "duplicate_login_close_button"
    interruption = planner.choose(frame, [collect_all, device_confirm])
    assert interruption is not None and interruption.type == "device_history_confirm_button"
    assert planner.choose(frame, [collect_all, offer_dismiss]).type == "startup_offer_dismiss"  # type: ignore[union-attr]
    assert planner.choose(frame, [collect_all, close]).type == "mail_collect_all_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [reward, close]).type == "mail_reward_collect_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [close]).type == "mail_close_button"  # type: ignore[union-attr]
    # If BlueStacks ignores the first close tap, re-planning must retry it.
    assert planner.choose(frame, [collect_all, close]).type == "mail_close_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [anchor, exit_button]) is None


def test_hunt_planner_recovers_when_full_mailbox_blocks_confirmation() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        (
            "hunt_confirm_button",
            "hunt_dialog_close_button",
            "mailbox_button",
            "mail_collect_all_button",
            "mail_reward_collect_button",
            "mail_close_button",
            "dinosaur",
        ),
        safe_margin=80,
    )
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    dialog_close = Detection("hunt_dialog_close_button", 628, 1409, 1.0)
    mailbox = Detection("mailbox_button", 841, 1210, 1.0)
    collect_all = Detection("mail_collect_all_button", 636, 1165, 1.0)
    reward = Detection("mail_reward_collect_button", 450, 910, 1.0)
    mail_close = Detection("mail_close_button", 450, 1380, 1.0)

    # The ordinary hunt dialog is unchanged: confirmation still wins while
    # retries remain, even though its close button is also visible.
    chosen = planner.choose(frame, [confirm, dialog_close])
    assert chosen is not None and chosen.type == "hunt_confirm_button"

    failed_target = Target(
        type=confirm.type,
        x=confirm.x,
        y=confirm.y,
        confidence=confirm.confidence,
        detection=confirm,
    )
    planner.on_retry_exhausted(failed_target)
    assert planner.on_retry_exhausted_context(
        failed_target,
        [confirm, dialog_close],
    )

    recovery = planner.choose(frame, [confirm, dialog_close])
    assert recovery is not None and recovery.type == "hunt_dialog_close_button"
    planner.on_action_success("hunt_dialog_close_button")

    assert planner.choose(frame, [mailbox]).type == "mailbox_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [collect_all]).type == "mail_collect_all_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [reward]).type == "mail_reward_collect_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [mail_close]).type == "mail_close_button"  # type: ignore[union-attr]


def test_hunt_planner_does_not_assume_mailbox_full_without_dialog_close() -> None:
    planner = HuntPlanner(("hunt_confirm_button", "hunt_dialog_close_button"))
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    failed_target = Target(
        type=confirm.type,
        x=confirm.x,
        y=confirm.y,
        confidence=confirm.confidence,
        detection=confirm,
    )

    assert not planner.on_retry_exhausted_context(failed_target, [confirm])
    assert "hunt_dialog_close_button" in planner.verification_detection_types(
        "hunt_confirm_button"
    )


def test_hunt_planner_does_not_recover_blocked_confirmation_without_dialog_close() -> None:
    planner = HuntPlanner(("hunt_confirm_button", "hunt_dialog_close_button"))
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    failed_target = Target(
        type=confirm.type,
        x=confirm.x,
        y=confirm.y,
        confidence=confirm.confidence,
        detection=confirm,
    )

    assert not planner.on_blocked_action_context(
        failed_target,
        [confirm],
        attempt=2,
    )


def test_hunt_planner_taps_egg_until_map_is_centered() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        (
            "forest_recenter_button",
            "map_exit_nest_button",
            "map_center_egg",
            "hunt_confirm_button",
            "dinosaur",
        ),
        recenter_every=1,
        map_settle_frames=1,
        safe_margin=80,
    )
    centered_anchor = Detection("map_center_egg", 450, 800, 1.0)
    shifted_anchor = Detection("map_center_egg", 253, 696, 0.9)
    exit_button = Detection("map_exit_nest_button", 841, 1295, 1.0)
    dinosaur = Detection("dinosaur", 500, 820, 0.9)
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    forest = Detection("forest_recenter_button", 841, 1295, 1.0)

    assert planner.choose(frame, [centered_anchor, exit_button, dinosaur]).type == "dinosaur"  # type: ignore[union-attr]
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    assert planner.choose(frame, [centered_anchor, exit_button]).type == "map_exit_nest_button"  # type: ignore[union-attr]
    assert planner.choose(frame, [forest]).type == "forest_recenter_button"  # type: ignore[union-attr]
    recenter = planner.choose(frame, [shifted_anchor, exit_button])
    assert recenter is not None and recenter.type == "map_center_egg"
    assert planner.choose(frame, [centered_anchor, exit_button]) is None


def test_hunt_planner_uses_safe_map_exit_fallback_when_nest_template_is_missing() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("map_exit_nest_button", "dinosaur"))
    mailbox = Detection("mailbox_button", 841, 1210, 0.99)

    target = planner.choose(frame, [mailbox])

    assert target is not None
    assert target.type == "map_exit_nest_button"
    assert (target.x, target.y) == (841, 1295)
    assert target.detection.metadata["detector"] == "map_landmark_fallback"


def test_hunt_planner_counts_return_when_animated_map_landmarks_are_missing() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "hunt_confirm_button", "dinosaur"),
        recenter_every=2,
        map_settle_frames=1,
        safe_margin=80,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    first = Detection("dinosaur", 500, 820, 0.9)
    second = Detection("dinosaur", 600, 820, 0.9)
    third = Detection("dinosaur", 350, 820, 0.9)
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)
    mailbox = Detection("mailbox_button", 841, 1210, 0.99)

    assert planner.choose(frame, [anchor, first]).type == "dinosaur"  # type: ignore[union-attr]
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    # No egg, nest, or mailbox is detected in this animated map frame.
    assert planner.choose(frame, [second]).type == "dinosaur"  # type: ignore[union-attr]
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    # The exact second return starts recentering and must not choose a third hunt.
    assert planner.choose(frame, [third]) is None
    exit_target = planner.choose(frame, [mailbox, third])
    assert exit_target is not None and exit_target.type == "map_exit_nest_button"


def test_hunt_planner_finishes_visible_hunt_control_before_no_available_warning() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("no_available_dinosaurs", "hunt_button", "dinosaur"),
    )
    unavailable = Detection("no_available_dinosaurs", 450, 900, 1.0)
    hunt_button = Detection("hunt_button", 450, 1200, 1.0)
    target = planner.choose(frame, [unavailable, hunt_button])
    assert target is not None and target.type == "hunt_button"


def test_hunt_planner_spreads_targets_away_from_existing_blue_ray() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        own_path_radius=50,
        own_path_angle_degrees=7,
        ring_width=150,
        safe_margin=80,
    )
    detections = [
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("own_hunt_path", 550, 800, 0.9),
        Detection("own_hunt_path", 650, 800, 0.9),
        Detection("own_hunt_path", 750, 800, 0.9),
        Detection("dinosaur", 840, 800, 0.99),
        Detection("dinosaur", 450, 500, 0.8),
    ]
    target = planner.choose(frame, detections)
    assert target is not None and (target.x, target.y) == (450, 500)


def test_hunt_planner_waits_when_concurrent_hunt_capacity_is_full() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("hunt_capacity_full", "dinosaur"),
        capacity_wait_seconds=300,
        safe_margin=80,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    full = Detection("hunt_capacity_full", 815, 240, 0.99)
    dinosaur = Detection("dinosaur", 500, 820, 0.9)
    with patch("dino_bot.planning.time.monotonic", side_effect=[0, 1, 301]):
        assert planner.choose(frame, [anchor, full, dinosaur]) is None
        assert planner.choose(frame, [anchor, dinosaur]) is None
        resumed = planner.choose(frame, [anchor, dinosaur])
    assert resumed is not None and resumed.type == "dinosaur"


def test_template_detector_finds_asset(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    template = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[22:34, 31:43] = template
    assert cv2.imwrite(str(tmp_path / "resource.png"), template)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [
                    {"type": "resource", "file": "resource.png", "threshold": 0.99}
                ],
                "hsv_ranges": [],
            }
        ),
        encoding="utf-8",
    )
    detector = OpenCvDetector(tmp_path / "manifest.json")
    found = detector.detect(Frame(image))
    assert len(found) == 1
    assert found[0].type == "resource"
    assert (found[0].x, found[0].y) == (37, 28)


def test_hsv_detector_finds_blob(tmp_path: Path) -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:40, 30:60] = (0, 255, 0)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [],
                "hsv_ranges": [
                    {
                        "type": "resource",
                        "lower": [50, 200, 200],
                        "upper": [70, 255, 255],
                        "min_area": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    found = OpenCvDetector(tmp_path / "manifest.json").detect(Frame(image))
    assert len(found) == 1
    assert found[0].type == "resource"


def test_hunt_team_availability_detector_only_matches_zero_of_eleven() -> None:
    detector = HuntTeamAvailabilityDetector()

    def team_screen(label: str) -> Frame:
        image = np.full((1600, 900, 3), 30, dtype=np.uint8)
        image[850:1500] = 255
        cv2.putText(
            image,
            label,
            (410, 960),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(image, (592, 1372), (664, 1447), (50, 80, 255), -1)
        return Frame(image)

    unavailable = detector.detect(team_screen("0 / 11"))
    assert len(unavailable) == 1
    assert unavailable[0].type == "no_available_dinosaurs"
    assert (unavailable[0].x, unavailable[0].y) == (628, 1409)
    assert detector.detect(team_screen("11 / 11")) == []


def test_hunt_capacity_detector_only_matches_ten_at_egg_nest() -> None:
    detector = HuntCapacityDetector()
    egg = cv2.imread(str(Path("assets/templates/map-center-egg-anchor.png")))
    assert egg is not None

    def capacity_screen(label: str, *, show_nest: bool = True) -> Frame:
        image = np.full((1600, 900, 3), 30, dtype=np.uint8)
        anchor_x, anchor_y = 500, 700
        if show_nest:
            egg_height, egg_width = egg.shape[:2]
            image[
                anchor_y : anchor_y + egg_height,
                anchor_x : anchor_x + egg_width,
            ] = egg
        cv2.putText(
            image,
            label,
            (anchor_x + 12, anchor_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return Frame(image)

    full = detector.detect(capacity_screen("10/10"))
    assert len(full) == 1 and full[0].type == "hunt_capacity_full"
    assert full[0].metadata["detector"] == "hunt_nest_counter"
    assert full[0].metadata["value"] == "10/10"
    assert detector.detect(capacity_screen("0/10")) == []
    assert detector.detect(capacity_screen("1/10")) == []
    assert detector.detect(capacity_screen("10/10", show_nest=False)) == []


def test_target_too_strong_detector_requires_red_warning_and_close_button() -> None:
    detector = TargetTooStrongDetector()
    image = np.full((1600, 900, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "TARGET TOO STRONG",
        (200, 720),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (592, 1372), (664, 1447), (50, 80, 255), -1)
    found = detector.detect(Frame(image))
    assert len(found) == 1 and found[0].type == "target_too_strong"

    safe = image.copy()
    safe[660:755] = 255
    assert detector.detect(Frame(safe)) == []


def test_template_asset_tool_crops_and_updates_manifest(tmp_path: Path) -> None:
    source = np.zeros((40, 50, 3), dtype=np.uint8)
    source[10:20, 15:30] = (10, 20, 30)
    assert cv2.imwrite(str(tmp_path / "screen.png"), source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"templates": [], "hsv_ranges": []}', encoding="utf-8")
    output = create_template(
        manifest,
        tmp_path / "screen.png",
        (15, 10, 15, 10),
        "resource",
        "Iron Ore",
        0.9,
    )
    cropped = cv2.imread(str(output))
    assert cropped.shape[:2] == (10, 15)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["templates"][0]["file"] == "templates/iron-ore.png"


def test_verifier_accepts_disappeared_target() -> None:
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    result = TargetChangedVerifier().verify(
        make_frame(0), make_frame(255), target, [detection], []
    )
    assert result.success
    assert "disappeared" in result.reason


def test_verifier_rejects_target_still_present() -> None:
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    result = TargetChangedVerifier().verify(
        make_frame(10), make_frame(10), target, [detection], [detection]
    )
    assert not result.success
    assert "still detected" in result.reason


def test_verifier_accepts_target_ui_change() -> None:
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    result = TargetChangedVerifier(pixel_change_threshold=0.08).verify(
        make_frame(0), make_frame(255), target, [detection], [detection]
    )
    assert result.success
    assert "interaction changed UI" in result.reason


def test_verifier_rejects_duplicate_hunt_alert() -> None:
    detection = make_detection(type="dinosaur")
    target = Target("dinosaur", detection.x, detection.y, detection.confidence, detection)
    alert = make_detection(type="duplicate_hunt_alert")
    result = TargetChangedVerifier(
        failure_types=("duplicate_hunt_alert",)
    ).verify(make_frame(), make_frame(255), target, [detection], [alert])
    assert not result.success
    assert "duplicate_hunt_alert" in result.reason


def test_verifier_accepts_expected_next_ui() -> None:
    detection = make_detection(type="dinosaur")
    target = Target("dinosaur", detection.x, detection.y, detection.confidence, detection)
    hunt_button = make_detection(type="hunt_button")
    result = TargetChangedVerifier(
        success_transitions={"dinosaur": ("hunt_button",)}
    ).verify(make_frame(10), make_frame(10), target, [detection], [detection, hunt_button])
    assert result.success
    assert "hunt_button" in result.reason


def test_verifier_reports_pixel_change_when_expected_next_ui_is_missing() -> None:
    """An inert coordinate must be distinguishable from a wrong-screen tap.

    Both fail the same way - the expected successor never appears - so the
    engine's escalate-to-back guard can only tell them apart by
    ``pixel_change``, and it skips the check entirely while that stays None.
    """

    detection = make_detection(type="hatch_egg_pile")
    target = Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )
    verifier = TargetChangedVerifier(
        success_transitions={"hatch_egg_pile": ("hatch_incubator_title",)}
    )

    inert = verifier.verify(
        make_frame(10), make_frame(10), target, [detection], [detection]
    )
    assert not inert.success
    assert "expected next UI not detected" in inert.reason
    assert inert.pixel_change == pytest.approx(0.0)

    wrong_screen = verifier.verify(
        make_frame(0), make_frame(255), target, [detection], [detection]
    )
    assert not wrong_screen.success
    assert wrong_screen.pixel_change is not None
    assert wrong_screen.pixel_change > 0.5


def test_verifier_accepts_expected_frame_structure_without_template_detection() -> None:
    detection = make_detection(type="hatch_claim_button")
    target = Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )
    result = TargetChangedVerifier(
        success_transitions={"hatch_claim_button": ("hatch_incubator_title",)},
        success_frame_predicates={
            "hatch_claim_button": lambda frame: int(frame.image[0, 0, 0]) == 42
        },
    ).verify(
        make_frame(10),
        make_frame(42),
        target,
        [detection],
        [],
    )

    assert result.success
    assert "frame structure" in result.reason


def test_verifier_requires_forest_target_to_disappear_before_dinosaur_success() -> None:
    forest = make_detection(type="forest_recenter_button")
    target = Target(forest.type, forest.x, forest.y, forest.confidence, forest)
    dinosaur = make_detection(x=120, y=70, type="dinosaur")
    verifier = TargetChangedVerifier(
        success_transitions={"forest_recenter_button": ("dinosaur",)},
        success_requires_target_absence=("forest_recenter_button",),
    )

    still_visible = verifier.verify(
        make_frame(10),
        make_frame(20),
        target,
        [forest],
        [forest, dinosaur],
    )
    transitioned = verifier.verify(
        make_frame(10),
        make_frame(20),
        target,
        [forest],
        [dinosaur],
    )

    assert not still_visible.success
    assert "still visible" in still_visible.reason
    assert transitioned.success
    assert verifier.relevant_detection_types(target.type) == frozenset(
        {"forest_recenter_button", "dinosaur"}
    )


def test_verifier_requires_expected_next_ui() -> None:
    detection = make_detection(type="hunt_button")
    target = Target("hunt_button", detection.x, detection.y, detection.confidence, detection)
    result = TargetChangedVerifier(
        success_transitions={"hunt_button": ("hunt_confirm_button",)}
    ).verify(make_frame(10), make_frame(255), target, [detection], [])

    assert not result.success
    assert "expected next UI" in result.reason


def test_verifier_never_accepts_black_frame() -> None:
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    result = TargetChangedVerifier().verify(
        make_frame(255), make_frame(0), target, [detection], []
    )

    assert not result.success
    assert "black" in result.reason


class SequenceCapture:
    def __init__(self, frames: list[Frame]):
        self.frames = frames
        self.index = 0
        self.closed = False

    def capture(self) -> Frame:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class PixelDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        return [make_detection()] if int(frame.image[0, 0, 0]) == 0 else []


class AlwaysFailsVerifier:
    def verify(self, *args, **kwargs) -> VerificationResult:
        return VerificationResult(False, "test failure")


class SequenceVerifier:
    def __init__(self, results: list[VerificationResult]) -> None:
        self.results = results
        self.index = 0

    def verify(self, *args, **kwargs) -> VerificationResult:
        result = self.results[min(self.index, len(self.results) - 1)]
        self.index += 1
        return result


class RecordingEventLog:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.records.append({"e": event, **fields})

    def close(self) -> None:
        return None


def test_engine_runs_complete_feedback_loop() -> None:
    capture = SequenceCapture([make_frame(0, 1), make_frame(255, 2)])
    driver = RecordingActionDriver()
    logger = logging.getLogger("test_engine_success")
    logger.addHandler(logging.NullHandler())
    context = BotContext(
        capture_provider=capture,
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=driver,
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logger,
        click_delay_ms=0,
        idle_delay_ms=0,
        max_actions=1,
    )
    engine = BotEngine(context)
    engine.run()
    assert context.state == BotState.STOPPED
    assert len(driver.actions) == 1
    assert context.last_result is not None and context.last_result.success
    assert capture.closed


def test_engine_stops_when_bounded_planner_reports_complete() -> None:
    class CompletePlanner(TargetPlanner):
        def is_complete(self) -> bool:
            return True

    capture = SequenceCapture([make_frame(0, 1)])
    driver = RecordingActionDriver()
    context = BotContext(
        capture_provider=capture,
        detector=PixelDetector(),
        planner=CompletePlanner(),
        action_driver=driver,
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_planner_complete"),
        idle_delay_ms=0,
    )
    BotEngine(context).run()
    assert driver.actions == []
    assert context.action_count == 0
    assert capture.closed


def test_engine_uses_target_specific_post_action_delay() -> None:
    class HuntConfirmDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            if int(frame.image[0, 0, 0]) == 0:
                return [make_detection(type="hunt_confirm_button")]
            return []

    capture = SequenceCapture([make_frame(0, 1), make_frame(255, 2)])
    logger = logging.getLogger("test_engine_target_delay")
    logger.addHandler(logging.NullHandler())
    context = BotContext(
        capture_provider=capture,
        detector=HuntConfirmDetector(),
        planner=TargetPlanner(("hunt_confirm_button",)),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logger,
        click_delay_ms=0,
        post_action_delays_ms={"hunt_confirm_button": 10_000},
        idle_delay_ms=0,
        max_cycles=1,
        cycle_complete_targets=("hunt_confirm_button",),
    )
    with patch.object(context.stop_event, "wait", return_value=False) as wait:
        BotEngine(context).run()
    wait.assert_called_once_with(0.25)
    assert context.cycle_count == 1
    assert context.action_count == 1


def test_engine_adaptive_verification_finishes_before_timeout() -> None:
    now = [0.0]
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    events = RecordingEventLog()
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(10), make_frame(20)]),
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=RecordingActionDriver(),
        verifier=SequenceVerifier(
            [
                VerificationResult(False, "not ready"),
                VerificationResult(True, "ready"),
            ]
        ),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_adaptive_verify_early_success"),
        click_delay_ms=5000,
        transition_poll_interval_ms=250,
        event_log=events,
        clock=lambda: now[0],
        state=BotState.ACTION,
        frame=make_frame(),
        detections=[detection],
        target=target,
        action=ActionCommand.tap(target.x, target.y),
    )

    def advance(seconds: float) -> bool:
        now[0] += seconds
        return False

    with patch.object(context.stop_event, "wait", side_effect=advance):
        engine = BotEngine(context)
        assert engine.step() == BotState.VERIFY
        assert engine.step() == BotState.VERIFY
        assert engine.step() == BotState.IDLE

    verify_events = [record for record in events.records if record["e"] == "verify"]
    assert [record["phase"] for record in verify_events] == ["pending", "final"]
    assert now[0] == pytest.approx(0.5)


def test_engine_slow_verification_gets_minimum_checks_after_timeout() -> None:
    now = [0.0]
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    events = RecordingEventLog()
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(10)]),
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=RecordingActionDriver(),
        verifier=AlwaysFailsVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_adaptive_verify_minimum_checks"),
        click_delay_ms=100,
        transition_poll_interval_ms=250,
        verification_minimum_checks=2,
        verify_retries=0,
        event_log=events,
        clock=lambda: now[0],
        state=BotState.ACTION,
        frame=make_frame(),
        detections=[detection],
        target=target,
        action=ActionCommand.tap(target.x, target.y),
    )

    def advance(seconds: float) -> bool:
        now[0] += seconds
        return False

    with patch.object(context.stop_event, "wait", side_effect=advance):
        engine = BotEngine(context)
        assert engine.step() == BotState.VERIFY
        assert engine.step() == BotState.VERIFY
        assert engine.step() == BotState.IDLE

    verify_events = [record for record in events.records if record["e"] == "verify"]
    assert [record["phase"] for record in verify_events] == ["pending", "final"]
    assert [record["check"] for record in verify_events] == [1, 2]
    assert now[0] == pytest.approx(0.2)


def test_engine_retries_three_times_then_stops() -> None:
    capture = SequenceCapture([make_frame()])
    driver = RecordingActionDriver()
    logger = logging.getLogger("test_engine_retry")
    logger.addHandler(logging.NullHandler())
    context = BotContext(
        capture_provider=capture,
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=driver,
        verifier=AlwaysFailsVerifier(),
        observer=RuntimeMode(),
        logger=logger,
        click_delay_ms=0,
        idle_delay_ms=0,
        verify_retries=3,
        max_actions=4,
    )
    BotEngine(context).run()
    assert len(driver.actions) == 4
    assert context.state == BotState.STOPPED


def test_engine_short_circuits_repeated_blocked_hunt_confirmation() -> None:
    class BlockedConfirmationDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            return [
                make_detection(80, 50, "hunt_confirm_button"),
                make_detection(100, 70, "hunt_dialog_close_button"),
            ]

    planner = HuntPlanner(
        ("hunt_confirm_button", "hunt_dialog_close_button"),
        safe_margin=0,
    )
    context = BotContext(
        capture_provider=SequenceCapture([make_frame()]),
        detector=BlockedConfirmationDetector(),
        planner=planner,
        action_driver=RecordingActionDriver(),
        verifier=AlwaysFailsVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_blocked_confirmation"),
        click_delay_ms=0,
        idle_delay_ms=0,
        verify_retries=3,
    )
    engine = BotEngine(context)

    for _ in range(30):
        engine.step()
        if context.action_count == 2 and context.state == BotState.IDLE:
            break

    assert context.action_count == 2
    assert context.attempt == 0
    assert context.state == BotState.IDLE
    recovery = planner.choose(make_frame(), BlockedConfirmationDetector().detect(make_frame()))
    assert recovery is not None and recovery.type == "hunt_dialog_close_button"


def test_debug_mode_saves_action_bundle(tmp_path: Path) -> None:
    frame = make_frame()
    detection = make_detection()
    target = Target("resource", 80, 50, 0.95, detection)
    from dino_bot.models import ActionCommand, ActionRecord, utc_now

    record = ActionRecord(
        utc_now(), ActionCommand.tap(80, 50), target, VerificationResult(True, "ok"), 1
    )
    DebugMode(tmp_path, save_images=True).on_action_complete(record, frame, frame)
    event_dirs = list(tmp_path.iterdir())
    assert len(event_dirs) == 1
    assert (event_dirs[0] / "Before.png").exists()
    assert (event_dirs[0] / "After.png").exists()
    payload = json.loads((event_dirs[0] / "debug.json").read_text(encoding="utf-8"))
    assert payload["result"] == "success"


def test_training_mode_prunes_oldest_images(tmp_path: Path) -> None:
    image = make_frame().image
    for index in range(1, 5):
        path = tmp_path / f"{index:06d}.png"
        assert cv2.imwrite(str(path), image)
        time.sleep(0.002)
    TrainingMode(tmp_path, fps=5, max_images=2)
    assert [path.name for path in sorted(tmp_path.glob("*.png"))] == ["000003.png", "000004.png"]


class FakeAdbClient:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def display_size(self) -> tuple[int, int]:
        return (320, 200)

    def run(self, args: list[str]) -> str:
        self.commands.append(args)
        return ""


def test_action_driver_maps_frame_coordinates_to_device() -> None:
    client = FakeAdbClient()
    driver = AdbActionDriver(client)  # type: ignore[arg-type]
    driver.tap(80, 50, make_frame())
    assert client.commands == [["shell", "input", "tap", "160", "100"]]


def test_action_driver_swaps_device_size_for_portrait_frame() -> None:
    client = FakeAdbClient()
    client.display_size = lambda: (1600, 900)  # type: ignore[method-assign]
    driver = AdbActionDriver(client)  # type: ignore[arg-type]
    portrait = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    driver.tap(450, 800, portrait)
    assert client.commands == [["shell", "input", "tap", "450", "800"]]


def test_action_driver_sends_android_back_key() -> None:
    client = FakeAdbClient()
    driver = AdbActionDriver(client)  # type: ignore[arg-type]
    driver.execute(ActionCommand.back(), make_frame())
    assert client.commands == [["shell", "input", "keyevent", "4"]]


class RecordingRestarter:
    def __init__(self) -> None:
        self.restart_count = 0

    def restart(self) -> None:
        self.restart_count += 1


def test_engine_holds_pending_verification_while_frame_is_black() -> None:
    detection = make_detection()
    target = Target("resource", detection.x, detection.y, detection.confidence, detection)
    recovery = BlackScreenRecovery(
        RecordingRestarter(),
        logging.getLogger("test_verify_black_hold"),
        sleeper=lambda _: None,
    )
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(0), make_frame(255)]),
        detector=PixelDetector(),
        planner=TargetPlanner(("resource",)),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_verify_black_hold"),
        idle_delay_ms=0,
        runtime_recovery=recovery,
        state=BotState.VERIFY,
        target=target,
        action=ActionCommand.tap(target.x, target.y),
        before_frame=make_frame(10),
        before_detections=[detection],
    )
    engine = BotEngine(context)

    assert engine.step() == BotState.VERIFY
    assert context.last_result is None
    assert engine.step() == BotState.IDLE
    assert context.last_result is not None and context.last_result.success


def test_black_screen_recovery_ignores_brief_transition() -> None:
    now = [100.0]
    restarter = RecordingRestarter()
    logger = logging.getLogger("test_black_screen_brief")
    recovery = BlackScreenRecovery(
        restarter,
        logger,
        timeout_seconds=45,
        clock=lambda: now[0],
        sleeper=lambda _: None,
    )

    assert not recovery.observe(make_frame(0))
    now[0] += 44
    assert not recovery.observe(make_frame(0))
    assert not recovery.observe(make_frame(20))
    assert restarter.restart_count == 0


def test_black_screen_recovery_restarts_after_timeout_and_honors_cooldown() -> None:
    now = [100.0]
    restarter = RecordingRestarter()
    logger = logging.getLogger("test_black_screen_timeout")
    waits: list[float] = []
    recovery = BlackScreenRecovery(
        restarter,
        logger,
        timeout_seconds=45,
        cooldown_seconds=300,
        launch_wait_seconds=15,
        clock=lambda: now[0],
        sleeper=waits.append,
    )

    assert not recovery.observe(make_frame(0))
    now[0] += 45
    assert recovery.observe(make_frame(0))
    assert restarter.restart_count == 1
    assert waits == [15]

    now[0] += 1
    assert not recovery.observe(make_frame(0))
    now[0] += 46
    assert not recovery.observe(make_frame(0))
    assert restarter.restart_count == 1

    now[0] = 446
    assert recovery.observe(make_frame(0))
    assert restarter.restart_count == 2


def test_manual_game_restart_bypasses_automatic_recovery_cooldown() -> None:
    now = [100.0]
    restarter = RecordingRestarter()
    recovery = BlackScreenRecovery(
        restarter,
        logging.getLogger("test_manual_restart_cooldown"),
        cooldown_seconds=300,
        launch_wait_seconds=0,
        clock=lambda: now[0],
    )

    assert recovery.request_restart("automatic", reason_key="black_screen")
    now[0] += 1
    assert not recovery.request_restart("automatic", reason_key="black_screen")
    assert recovery.request_restart(
        "manual control request",
        reason_key="manual_control",
        bypass_cooldown=True,
    )
    assert restarter.restart_count == 2


def test_adb_app_restarter_only_restarts_configured_game() -> None:
    client = FakeAdbClient()
    restarter = AdbAppRestarter(client, "game.package", "GameActivity")  # type: ignore[arg-type]
    restarter.restart()
    assert client.commands == [
        ["shell", "am", "force-stop", "game.package"],
        ["shell", "am", "start", "-n", "game.package/GameActivity"],
    ]


def test_engine_queues_manual_game_restart_on_bot_thread() -> None:
    class ManualRecovery:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, bool]] = []

        def observe(self, frame: Frame) -> bool:
            return False

        def request_restart(
            self,
            reason: str,
            *,
            reason_key: str,
            bypass_cooldown: bool = False,
        ) -> bool:
            self.requests.append((reason, reason_key, bypass_cooldown))
            return True

    recovery = ManualRecovery()
    planner = HuntPlanner(("dinosaur",))
    planner._awaiting_hunt_button = True
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(255)]),
        detector=PixelDetector(),
        planner=planner,
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_manual_game_restart"),
        runtime_recovery=recovery,
        state=BotState.VERIFY,
        target=Target("resource", 80, 50, 0.9, make_detection()),
        attempt=2,
    )
    engine = BotEngine(context)

    assert engine.request_game_restart()
    assert not engine.request_game_restart()
    assert recovery.requests == []

    assert engine.step() == BotState.IDLE
    assert recovery.requests == [("manual control request", "manual_control", True)]
    assert context.target is None
    assert context.attempt == 0
    assert not planner._awaiting_hunt_button


def test_engine_rejects_manual_game_restart_without_recovery() -> None:
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(255)]),
        detector=PixelDetector(),
        planner=HuntPlanner(("dinosaur",)),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_manual_game_restart_unavailable"),
    )

    assert not BotEngine(context).request_game_restart()


def test_engine_restarts_game_after_closing_duplicate_login_dialog() -> None:
    class DuplicateLoginRecovery:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, bool]] = []

        def observe(self, frame: Frame) -> bool:
            return False

        def request_restart(
            self,
            reason: str,
            *,
            reason_key: str,
            bypass_cooldown: bool = False,
        ) -> bool:
            self.requests.append((reason, reason_key, bypass_cooldown))
            return True

    target_type = "duplicate_login_close_button"
    detection = make_detection(type=target_type)
    target = Target(
        target_type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )
    driver = RecordingActionDriver()
    recovery = DuplicateLoginRecovery()
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(255, 2)]),
        detector=PixelDetector(),
        planner=TargetPlanner((target_type,)),
        action_driver=driver,
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_duplicate_login_restart"),
        click_delay_ms=0,
        runtime_recovery=recovery,
        state=BotState.ACTION,
        frame=make_frame(10, 1),
        detections=[detection],
        target=target,
        action=ActionCommand.tap(target.x, target.y),
    )
    engine = BotEngine(context)

    assert engine.step() == BotState.VERIFY
    assert len(driver.actions) == 1
    assert recovery.requests == []

    assert engine.step() == BotState.IDLE
    assert recovery.requests == [
        ("duplicate login dialog closed", "duplicate_login", True)
    ]
    assert context.target is None
    assert context.action is None


def test_engine_clears_transient_state_after_black_screen_recovery() -> None:
    class ImmediateRecovery:
        def observe(self, frame: Frame) -> bool:
            return True

    capture = SequenceCapture([make_frame(0)])
    planner = HuntPlanner(("dinosaur",))
    planner._awaiting_hunt_button = True
    logger = logging.getLogger("test_engine_runtime_recovery")
    context = BotContext(
        capture_provider=capture,
        detector=PixelDetector(),
        planner=planner,
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logger,
        runtime_recovery=ImmediateRecovery(),
        state=BotState.CAPTURE,
        target=Target("resource", 80, 50, 0.9, make_detection()),
        attempt=2,
    )

    assert BotEngine(context).step() == BotState.IDLE
    assert context.target is None
    assert context.attempt == 0
    assert not planner._awaiting_hunt_button


class RecordingStallSnapshots:
    def __init__(self) -> None:
        self.captures: list[dict[str, object]] = []

    def capture(self, frame, detections, *, seconds, stage, escapes):
        self.captures.append(
            {
                "detections": len(detections),
                "seconds": seconds,
                "stage": stage,
                "escapes": escapes,
            }
        )
        return Path("stall.png")


def test_engine_records_the_frame_a_blind_stall_could_not_act_on() -> None:
    """The event stream can only report what matched, which here is the problem.

    These episodes are caused by a screen the detector has no name for, so the
    frame is the only evidence that can identify it. Without it the diagnosis
    stops at "17 dinosaur labels and no map control", which fits an overlay, a
    zoomed-out view and a missing template equally well.
    """

    class BlindPlanner:
        def __init__(self) -> None:
            self.reported = False

        def choose(self, frame: Frame, detections) -> Target | None:
            return None

        def last_stage(self) -> str:
            return "recenter"

        def take_blind_escape(self):
            if self.reported:
                return None
            self.reported = True
            return {"seconds": 20.0, "stage": "recenter", "escapes": 1}

    snapshots = RecordingStallSnapshots()
    events = RecordingEventLog()
    context = BotContext(
        capture_provider=SequenceCapture([make_frame(0)]),
        detector=PixelDetector(),
        planner=BlindPlanner(),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_blind_stall"),
        idle_delay_ms=0,
        stall_snapshots=snapshots,
        event_log=events,
        state=BotState.CAPTURE,
    )
    engine = BotEngine(context)
    for _ in range(3):
        engine.step()

    assert snapshots.captures == [
        {"detections": 1, "seconds": 20.0, "stage": "recenter", "escapes": 1}
    ]
    stall_events = [record for record in events.records if record["e"] == "blind_stall"]
    assert stall_events == [
        {"e": "blind_stall", "seconds": 20.0, "stage": "recenter", "escapes": 1}
    ]


def test_stall_snapshot_writer_keeps_the_newest_frames_only(tmp_path: Path) -> None:
    now = [0.0]
    writer = StallSnapshotWriter(
        tmp_path / "stalls",
        logging.getLogger("test_stall_writer"),
        limit=2,
        min_interval_seconds=60.0,
        clock=lambda: now[0],
        # Naive, so the local-time filename is the same wherever this runs.
        now=lambda: datetime(2026, 7, 27, 20, int(now[0] // 60), 0),
    )
    frame = Frame(np.zeros((160, 90, 3), dtype=np.uint8))
    detections = [Detection("dinosaur", 45, 80, 0.9), Detection("dinosaur", 20, 30, 0.8)]

    first = writer.capture(frame, detections, seconds=20.0, stage="recenter", escapes=1)
    assert first is not None

    now[0] = 30.0
    assert writer.capture(frame, detections, seconds=40.0, stage="recenter", escapes=2) is None, (
        "an episode re-reports every 20s and must not overwrite itself nine times"
    )

    for minute in (2, 3):
        now[0] = minute * 60.0
        assert writer.capture(
            frame, detections, seconds=20.0, stage="mail", escapes=1
        ) is not None

    written = sorted(path.name for path in (tmp_path / "stalls").glob("stall-*.png"))
    assert written == ["stall-20260727-200200.png", "stall-20260727-200300.png"]
    assert not (tmp_path / "stalls" / "stall-20260727-200000.json").exists()

    sidecar = json.loads(
        (tmp_path / "stalls" / "stall-20260727-200300.json").read_text(encoding="utf-8")
    )
    assert sidecar["stage"] == "mail"
    assert sidecar["blind_seconds"] == 20.0
    # What the detector *did* match is half the evidence: it separates "nothing
    # was on screen" from "the map was there and the buttons were not".
    assert sidecar["detections"] == {"dinosaur": 2}


def test_capacity_snapshot_saves_the_hud_crop_and_the_glyphs(tmp_path: Path) -> None:
    writer = CapacitySnapshotWriter(
        tmp_path / "stalls",
        logging.getLogger("test_capacity_writer"),
        clock=lambda: 0.0,
        now=lambda: datetime(2026, 8, 5, 2, 7, 32),
    )
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    read = CapacityRead(
        count=None,
        text="28?/35O",
        fraction=None,
        region=(10, 239, 110, 258),
        reason="unparsed",
    )

    path = writer.capture(frame, read, stage="cave_navigate", attempts=3)

    assert path is not None and path.name == "capacity-20260805-020732.png"
    # The 19px-tall crop is what the glyph matcher saw; unenlarged it is not
    # something a person can judge, which is the whole point of saving it.
    hud = cv2.imread(str(path.with_name("capacity-20260805-020732-hud.png")))
    assert hud is not None
    assert hud.shape[:2] == ((258 - 239) * HUD_ZOOM, (110 - 10) * HUD_ZOOM)

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["reason"] == "unparsed"
    assert sidecar["glyphs"] == "28?/35O"
    assert sidecar["region"] == [10, 239, 110, 258]
    assert sidecar["stage"] == "cave_navigate"
    assert sidecar["attempts"] == 3
    assert sidecar["frame"] == {"width": 900, "height": 1600}


def test_capacity_snapshot_prunes_whole_episodes_not_half_of_them(
    tmp_path: Path,
) -> None:
    now = [0.0]
    writer = CapacitySnapshotWriter(
        tmp_path / "stalls",
        logging.getLogger("test_capacity_prune"),
        limit=2,
        min_interval_seconds=60.0,
        clock=lambda: now[0],
        now=lambda: datetime(2026, 8, 5, 2, int(now[0] // 60), 0),
    )
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    read = CapacityRead(None, "", None, (10, 239, 110, 258), "unparsed")

    for minute in (0, 1, 2):
        now[0] = minute * 60.0
        assert writer.capture(frame, read, stage="cave_done", attempts=3) is not None

    stalls = tmp_path / "stalls"
    # The -hud companion shares the stem: counted as an episode it would halve
    # the retained set, and missed by the prune it would outlive its frame.
    assert sorted(path.name for path in stalls.glob("capacity-*.png")) == [
        "capacity-20260805-020100-hud.png",
        "capacity-20260805-020100.png",
        "capacity-20260805-020200-hud.png",
        "capacity-20260805-020200.png",
    ]
    assert not (stalls / "capacity-20260805-020000.json").exists()


def test_capacity_snapshot_survives_a_frame_with_no_room_for_the_crop(
    tmp_path: Path,
) -> None:
    writer = CapacitySnapshotWriter(
        tmp_path / "stalls",
        logging.getLogger("test_capacity_empty_crop"),
        now=lambda: datetime(2026, 8, 5, 2, 7, 32),
    )
    read = CapacityRead(None, "", None, (10, 239, 110, 258), "region_outside_frame")

    path = writer.capture(
        Frame(np.zeros((200, 900, 3), dtype=np.uint8)),
        read,
        stage="cave_navigate",
        attempts=1,
    )

    assert path is not None
    assert not path.with_name(f"{path.stem}-hud.png").exists()


def test_stall_snapshot_failure_never_stops_the_run(tmp_path: Path) -> None:
    writer = StallSnapshotWriter(
        tmp_path / "file" / "stalls",
        logging.getLogger("test_stall_writer_failure"),
    )
    tmp_path.joinpath("file").write_text("not a directory", encoding="utf-8")

    result = writer.capture(
        Frame(np.zeros((160, 90, 3), dtype=np.uint8)),
        [],
        seconds=20.0,
        stage="recenter",
        escapes=1,
    )

    assert result is None


def test_template_matches_at_half_resolution_in_reference_coordinates(
    tmp_path: Path,
) -> None:
    """Halving the searched image is a 4x cut that must not move the target.

    Measured on a live 900x1600 frame: the full scan costs 1614ms, and every
    button in it survives being matched at half size - INTER_AREA strips the
    high-frequency noise that fights the match, so confidence holds or improves.
    What cannot shift is where the hit lands, because the coordinate is what
    gets tapped.
    """

    rng = np.random.default_rng(11)
    template = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    image[300:340, 200:240] = template
    assert cv2.imwrite(str(tmp_path / "button.png"), template)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "type": "button",
                        "file": "button.png",
                        "threshold": 0.8,
                        "match_scale": 0.5,
                        "click_offset": [20, 20],
                    }
                ],
                "hsv_ranges": [],
            }
        ),
        encoding="utf-8",
    )

    found = OpenCvDetector(tmp_path / "manifest.json").detect(Frame(image))

    assert len(found) == 1
    assert found[0].type == "button"
    # The click offset lands on the centre of the button, in full-size pixels.
    assert abs(found[0].x - 220) <= 2
    assert abs(found[0].y - 320) <= 2
    assert found[0].bbox is not None
    assert abs(found[0].bbox.width - 40) <= 2, "the box is reported at full size"
    assert found[0].metadata["match_scale"] == 0.5


def test_match_scale_of_one_leaves_a_template_at_full_size(tmp_path: Path) -> None:
    """The dinosaur label is 18x18 and stays whole.

    Halved it produced fifteen distinct hits where full size found ten, and
    the five extras scored 0.54-0.58 at full resolution - too marginal to
    accept sight unseen when a wrong tap costs a failure cooldown.
    """

    rng = np.random.default_rng(12)
    template = rng.integers(0, 256, (18, 18, 3), dtype=np.uint8)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[100:118, 150:168] = template
    assert cv2.imwrite(str(tmp_path / "label.png"), template)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "type": "label",
                        "file": "label.png",
                        "threshold": 0.95,
                        "match_scale": 1.0,
                    }
                ],
                "hsv_ranges": [],
            }
        ),
        encoding="utf-8",
    )

    detector = OpenCvDetector(tmp_path / "manifest.json")
    found = detector.detect(Frame(image))

    assert len(found) == 1
    assert (found[0].x, found[0].y) == (159, 109)
    assert "match_scale" not in found[0].metadata, "full size is the quiet default"


def test_manifest_rejects_a_match_scale_outside_the_unit_range(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    assert cv2.imwrite(
        str(tmp_path / "button.png"),
        rng.integers(0, 256, (10, 10, 3), dtype=np.uint8),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [
                    {"type": "button", "file": "button.png", "match_scale": 1.5}
                ],
                "hsv_ranges": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DetectorAssetError):
        OpenCvDetector(tmp_path / "manifest.json")


def test_planning_scan_leaves_out_screens_the_current_stage_cannot_reach() -> None:
    """The launch dialogs cost a quarter of every scan and cannot appear mid-run.

    Working the map needs none of the login screens, and paying for them on
    every cycle is what put 41% of a hunt's wall clock into planning detection.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",), blocking_types=("duplicate_hunt_alert",))
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 650, 1200, 0.90)

    assert planner.planning_detection_types() is None, "the first cycle sees it all"
    assert planner.choose(frame, [anchor, dinosaur]) is not None

    scoped = planner.planning_detection_types()

    assert scoped is not None
    assert {"dinosaur", "map_center_egg", "own_hunt_path"} <= scoped
    assert "duplicate_hunt_alert" in scoped, "a blocked tap must still be seen"
    assert {"hunt_capacity_full", "target_too_strong"} <= scoped
    assert not scoped & {
        "device_history_confirm_button",
        "duplicate_login_close_button",
        "startup_offer_dismiss",
    }
    assert not scoped & {
        "mail_collect_all_button",
        "mail_reward_collect_button",
        "mail_close_button",
    }, "the mail overlay only exists inside the mail flow"
    assert {"startup_growth_result_back", "startup_auto_battle_close"} <= scoped, (
        "the layout-ratio interrupts cost under a millisecond and stay visible"
    )


def test_planning_scan_widens_again_after_two_cycles_plan_nothing() -> None:
    """A narrow scan is only trustworthy while it keeps finding work."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",), full_scan_after_idle_cycles=2)
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 650, 1200, 0.90)

    assert planner.planning_detection_types() is None
    assert planner.choose(frame, [anchor, dinosaur]) is not None
    assert planner.planning_detection_types() is not None

    assert planner.choose(frame, [anchor]) is None
    assert planner.planning_detection_types() is not None, "one empty cycle is normal"

    assert planner.choose(frame, [anchor]) is None
    assert planner.planning_detection_types() is None, "two in a row widens the scan"


def test_planning_scan_sweeps_everything_on_a_timer() -> None:
    """A run that never stalls still has to re-check the screens it skips."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    now = [0.0]
    planner = HuntPlanner(
        ("dinosaur",),
        full_scan_interval_seconds=30.0,
        clock=lambda: now[0],
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 650, 1200, 0.90)

    assert planner.planning_detection_types() is None
    assert planner.choose(frame, [anchor, dinosaur]) is not None

    now[0] = 29.0
    assert planner.planning_detection_types() is not None

    now[0] = 30.0
    assert planner.planning_detection_types() is None


def test_restarting_the_game_forces_a_full_planning_scan() -> None:
    """A relaunch walks back through the very dialogs a scoped scan omits."""

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",))
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 650, 1200, 0.90)

    assert planner.planning_detection_types() is None
    assert planner.choose(frame, [anchor, dinosaur]) is not None
    assert planner.planning_detection_types() is not None

    planner.reset_workflow()

    assert planner.planning_detection_types() is None


def test_disabling_stage_scoped_scan_always_asks_for_everything() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("dinosaur",), stage_scoped_scan=False)
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    dinosaur = Detection("dinosaur", 650, 1200, 0.90)

    assert planner.planning_detection_types() is None
    assert planner.choose(frame, [anchor, dinosaur]) is not None
    assert planner.planning_detection_types() is None


def test_every_recenter_names_the_reason_that_started_it() -> None:
    """Without the reason, a starved map and a scheduled sweep look identical.

    They are not: the scheduled one is policy, the starved one is throughput
    leaking away, and telling them apart is what the tuning record runs on.
    """

    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "hunt_confirm_button", "dinosaur"),
        recenter_every=1,
        map_settle_frames=1,
        safe_margin=80,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    exit_button = Detection("map_exit_nest_button", 841, 1295, 1.0)
    dinosaur = Detection("dinosaur", 500, 820, 0.9)
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)

    assert planner.choose(frame, [anchor, exit_button, dinosaur]).type == "dinosaur"  # type: ignore[union-attr]
    assert planner.choose(frame, [confirm]).type == "hunt_confirm_button"  # type: ignore[union-attr]
    planner.on_action_success("hunt_confirm_button")
    target = planner.choose(frame, [anchor, exit_button])

    assert target is not None and target.type == "map_exit_nest_button"
    assert planner.last_recenter_reason() == "batch", (
        "a scheduled sweep must not read as a starved map"
    )
