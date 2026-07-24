from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace
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
from dino_bot.detection import (
    CompositeDetector,
    HuntCapacityDetector,
    HuntTeamAvailabilityDetector,
    OpenCvDetector,
    TargetTooStrongDetector,
)
from dino_bot.doctor import platform_supports_capture
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


def test_project_config_uses_short_no_available_verification_delay() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")

    assert config.post_action_delays["no_available_dinosaurs"] == 300
    assert config.post_action_delays["target_too_strong"] == 3000
    assert config.post_action_delays["map_exit_nest_button"] == 2500
    assert config.post_action_delays["forest_recenter_button"] == 3000
    assert config.post_action_delays["mailbox_button"] == 2500
    assert config.planner.anchor_exclusion_radius == 50
    assert config.planner.dinosaur_failure_cooldown_ms == 5_000
    assert config.planner.dinosaur_failure_radius == 80
    assert config.planner.action_cooldowns_ms["target_too_strong"] == 300_000
    assert set(config.verify.success_transitions["hunt_confirm_button"]) == {
        "map_exit_nest_button",
        "map_center_egg",
        "mailbox_button",
    }


def test_config_rejects_negative_anchor_exclusion_radius(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"planner": {"anchor_exclusion_radius": -1}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="anchor_exclusion_radius"):
        load_config(config_file)


@pytest.mark.parametrize(
    "setting",
    ["dinosaur_failure_cooldown_ms", "dinosaur_failure_radius"],
)
def test_config_rejects_negative_dinosaur_failure_cooldown(
    tmp_path: Path,
    setting: str,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"planner": {setting: -1}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=setting):
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
    assert result.transition_poll_interval == 100
    assert result.post_action_delays["dinosaur"] == 300
    assert result.post_action_delays["hunt_button"] == 900
    assert result.post_action_delays["hunt_confirm_button"] == 1200


def test_config_speed_profile_is_the_cli_source_of_truth(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "speed_profiles": {
                    "fast": {
                        "click_delay_ms": 125,
                        "poll_interval_ms": 50,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = apply_run_timing(load_config(config_file), speed="fast")

    assert result.click_delay == 125
    assert result.transition_poll_interval == 50
    assert result.post_action_delays["hunt_button"] == 900


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
            "--poll-interval-ms",
            "75",
            "--status-port",
            "9876",
        ]
    )

    assert args.speed == "fast"
    assert args.hunt_button_delay_ms == 1800
    assert args.poll_interval_ms == 75
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
    monkeypatch.setattr("dino_bot.actions.sys.platform", "win32")

    assert AdbClient._resolve_executable(None) == str(adb)


def test_adb_client_discovers_android_sdk_on_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adb = tmp_path / "AndroidSdk" / "platform-tools" / "adb"
    adb.parent.mkdir(parents=True)
    adb.touch()
    monkeypatch.setenv("ANDROID_SDK_ROOT", str(tmp_path / "AndroidSdk"))
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr("dino_bot.actions.shutil.which", lambda _: None)
    monkeypatch.setattr("dino_bot.actions.sys.platform", "darwin")

    assert AdbClient._resolve_executable(None) == str(adb)


def test_macos_supports_only_adb_capture() -> None:
    windows_config = AppConfig(root=Path("."))
    macos_config = replace(
        windows_config,
        capture=replace(windows_config.capture, backend="adb"),
    )

    assert platform_supports_capture(windows_config, "Windows")
    assert platform_supports_capture(macos_config, "Darwin")
    assert platform_supports_capture(windows_config, "Darwin") is False
    assert platform_supports_capture(macos_config, "Linux") is False


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


def test_hunt_planner_excludes_false_dinosaur_on_center_egg() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        anchor_exclusion_radius=50,
        safe_margin=80,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    egg_false_positive = Detection("dinosaur", 452, 820, 0.99)
    safe_dinosaur = Detection("dinosaur", 510, 820, 0.85)

    target = planner.choose(
        frame,
        [anchor, egg_false_positive, safe_dinosaur],
    )

    assert target is not None and (target.x, target.y) == (510, 820)


def test_hunt_planner_excludes_screen_center_when_map_anchor_is_offset() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("dinosaur",),
        safe_margin=0,
        bottom_exclusion_px=0,
        anchor_exclusion_radius=50,
    )
    offset_anchor = Detection("map_center_egg", 300, 800, 1.0)
    screen_center_false_positive = Detection("dinosaur", 450, 800, 0.99)
    safe_dinosaur = Detection("dinosaur", 600, 800, 0.85)

    target = planner.choose(
        frame,
        [offset_anchor, screen_center_false_positive, safe_dinosaur],
    )

    assert target is not None and (target.x, target.y) == (600, 800)


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


def test_hunt_planner_recenters_after_repeated_frames_without_safe_dinosaur() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        own_path_radius=90,
        stalled_recenter_frames=2,
        safe_margin=80,
    )
    detections = [
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("mailbox_button", 841, 1210, 0.99),
        Detection("own_hunt_path", 500, 820, 0.9),
        Detection("dinosaur", 500, 820, 0.99),
    ]

    assert planner.choose(frame, detections) is None
    reset = planner.choose(frame, detections)
    assert reset is not None and reset.type == "map_exit_nest_button"


def test_hunt_planner_recenters_when_only_own_paths_remain() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("map_exit_nest_button", "dinosaur"),
        stalled_recenter_frames=2,
    )
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    own_path = Detection("own_hunt_path", 500, 820, 0.9)

    assert planner.choose(frame, [anchor, own_path]) is None
    reset = planner.choose(frame, [own_path])

    assert reset is not None and reset.type == "map_exit_nest_button"
    assert reset.detection.metadata["detector"] == "map_landmark_fallback"


def test_hunt_planner_counts_only_verified_confirmation() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(("hunt_confirm_button",))
    confirm = Detection("hunt_confirm_button", 451, 1412, 1.0)

    assert planner.choose(frame, [confirm]) is not None
    assert planner._total_hunt_count == 0

    planner.on_action_success("hunt_confirm_button")

    assert planner._total_hunt_count == 1


def test_hunt_planner_releases_dinosaur_wait_after_failed_selection() -> None:
    planner = HuntPlanner(("dinosaur",))
    planner._awaiting_hunt_button = True
    planner._waited_frames = 3

    planner.on_action_failure("dinosaur")

    assert planner._awaiting_hunt_button is False
    assert planner._waited_frames == 0


def test_hunt_planner_cools_failed_dinosaur_and_restores_anchor() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    anchor = Detection("map_center_egg", 450, 800, 1.0)
    nearest = Detection("dinosaur", 550, 800, 0.95)
    alternative = Detection("dinosaur", 700, 800, 0.90)
    planner = HuntPlanner(
        ("dinosaur",),
        safe_margin=0,
        bottom_exclusion_px=0,
        anchor_exclusion_radius=0,
        dinosaur_failure_cooldown_ms=5_000,
        dinosaur_failure_radius=80,
    )

    first = planner.choose(frame, [anchor, nearest, alternative])

    assert first is not None and (first.x, first.y) == (550, 800)
    assert planner._last_anchor == (350.0, 800.0)

    planner.on_action_failure("dinosaur")
    second = planner.choose(frame, [anchor, nearest, alternative])

    assert second is not None and (second.x, second.y) == (700, 800)
    assert planner._anchor_before_dinosaur == (450.0, 800.0)


def test_hunt_planner_reuses_map_after_verified_confirmation() -> None:
    planner = HuntPlanner(("dinosaur",))
    map_detections = [
        Detection("map_exit_nest_button", 841, 1295, 1.0),
        Detection("map_center_egg", 450, 800, 1.0),
        Detection("dinosaur", 500, 820, 0.9),
    ]

    relevant = planner.verification_detection_types("hunt_confirm_button")

    assert {
        "dinosaur",
        "own_hunt_path",
        "map_exit_nest_button",
        "map_center_egg",
        "hunt_capacity_full",
    } <= relevant
    assert planner.can_reuse_verification_result(
        "hunt_confirm_button",
        map_detections,
    )


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


def test_hunt_planner_prioritizes_no_available_dinosaurs_exception() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("no_available_dinosaurs", "hunt_button", "dinosaur"),
    )
    unavailable = Detection("no_available_dinosaurs", 450, 900, 1.0)
    hunt_button = Detection("hunt_button", 450, 1200, 1.0)
    target = planner.choose(frame, [unavailable, hunt_button])
    assert target is not None and target.type == "no_available_dinosaurs"


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


def test_hunt_planner_applies_verified_action_cooldown() -> None:
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    planner = HuntPlanner(
        ("hunt_button",),
        action_cooldowns_ms={"target_too_strong": 300_000},
    )
    hunt_button = Detection("hunt_button", 450, 1200, 1.0)

    with patch(
        "dino_bot.planning.time.monotonic",
        side_effect=[10, 11, 11, 311, 311],
    ):
        planner.on_action_success("target_too_strong")
        assert planner.choose(frame, [hunt_button]) is None
        assert planner.next_ready_delay_ms() == 299_000
        resumed = planner.choose(frame, [hunt_button])

    assert resumed is not None and resumed.type == "hunt_button"


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


def test_template_detector_can_filter_to_relevant_types(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
    second = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[10:20, 15:25] = first
    image[45:55, 65:75] = second
    assert cv2.imwrite(str(tmp_path / "first.png"), first)
    assert cv2.imwrite(str(tmp_path / "second.png"), second)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "templates": [
                    {"type": "first", "file": "first.png", "threshold": 0.99},
                    {"type": "second", "file": "second.png", "threshold": 0.99},
                ]
            }
        ),
        encoding="utf-8",
    )

    found = OpenCvDetector(tmp_path / "manifest.json").detect_types(
        Frame(image),
        {"second"},
    )

    assert {item.type for item in found} == {"second"}


def test_composite_detector_normalizes_frame_only_once() -> None:
    observed_shapes: list[tuple[int, int]] = []
    observed_images: list[int] = []

    class FixedDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            observed_shapes.append((frame.width, frame.height))
            observed_images.append(id(frame.image))
            return [Detection("fixed", 450, 800, 1.0)]

    detector = CompositeDetector(
        FixedDetector(),
        FixedDetector(),
        reference_size=(900, 1600),
    )

    found = detector.detect(Frame(np.zeros((1920, 1080, 3), dtype=np.uint8)))

    assert observed_shapes == [(900, 1600), (900, 1600)]
    assert len(set(observed_images)) == 1
    assert [(item.x, item.y) for item in found] == [(540, 960), (540, 960)]


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
    egg = cv2.imread(str(Path("assets/templates/map-center-egg.png")))
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


def test_engine_polls_within_target_specific_transition_timeout() -> None:
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


def test_engine_detects_transition_without_repeating_action() -> None:
    class TransitionDetector:
        filtered_calls = 0

        def detect(self, frame: Frame) -> list[Detection]:
            if int(frame.image[0, 0, 0]) == 10:
                return [make_detection(type="hunt_button")]
            return []

        def detect_types(
            self,
            frame: Frame,
            target_types: set[str],
        ) -> list[Detection]:
            self.filtered_calls += 1
            assert "hunt_button" in target_types
            return self.detect(frame)

    now = [0.0]
    capture = SequenceCapture(
        [
            make_frame(10, 1),
            make_frame(10, 2),
            make_frame(255, 3),
        ]
    )
    driver = RecordingActionDriver()
    detector = TransitionDetector()
    context = BotContext(
        capture_provider=capture,
        detector=detector,
        planner=TargetPlanner(("hunt_button",)),
        action_driver=driver,
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_transition_polling"),
        post_action_delays_ms={"hunt_button": 1000},
        transition_poll_interval_ms=100,
        idle_delay_ms=0,
        max_actions=1,
        clock=lambda: now[0],
    )

    def advance(seconds: float) -> bool:
        now[0] += seconds
        return False

    with patch.object(context.stop_event, "wait", side_effect=advance) as wait:
        BotEngine(context).run()

    assert [call.args for call in wait.call_args_list] == [(0.1,), (0.1,)]
    assert len(driver.actions) == 1
    assert detector.filtered_calls == 2
    assert capture.index == 3
    assert context.last_result is not None and context.last_result.success


def test_engine_timeout_starts_after_first_verification_observation() -> None:
    now = [0.0]

    class SlowTransitionDetector:
        calls = 0

        def detect(self, frame: Frame) -> list[Detection]:
            self.calls += 1
            if self.calls == 2:
                now[0] += 2.0
            if self.calls <= 2:
                return [make_detection(type="hunt_button")]
            return []

    capture = SequenceCapture(
        [
            make_frame(10, 1),
            make_frame(10, 2),
            make_frame(255, 3),
        ]
    )
    context = BotContext(
        capture_provider=capture,
        detector=SlowTransitionDetector(),
        planner=TargetPlanner(("hunt_button",)),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_slow_detection_grace"),
        post_action_delays_ms={"hunt_button": 300},
        transition_poll_interval_ms=100,
        idle_delay_ms=0,
        max_actions=1,
        clock=lambda: now[0],
    )

    def advance(seconds: float) -> bool:
        now[0] += seconds
        return False

    with patch.object(context.stop_event, "wait", side_effect=advance):
        BotEngine(context).run()

    assert capture.index == 3
    assert context.last_result is not None and context.last_result.success


def test_engine_reuses_verified_successor_without_full_recapture() -> None:
    class ChainingPlanner(TargetPlanner):
        def can_reuse_verification_result(
            self,
            target_type: str,
            detections: list[Detection],
        ) -> bool:
            return (
                target_type == "dinosaur"
                and any(item.type == "hunt_button" for item in detections)
            )

    class ChainingDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            value = int(frame.image[0, 0, 0])
            if value == 10:
                return [make_detection(type="dinosaur")]
            if value == 20:
                return [make_detection(type="hunt_button")]
            return []

    capture = SequenceCapture(
        [
            make_frame(10, 1),
            make_frame(20, 2),
            make_frame(30, 3),
        ]
    )
    driver = RecordingActionDriver()
    context = BotContext(
        capture_provider=capture,
        detector=ChainingDetector(),
        planner=ChainingPlanner(("dinosaur", "hunt_button")),
        action_driver=driver,
        verifier=TargetChangedVerifier(
            success_transitions={"dinosaur": ("hunt_button",)}
        ),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_reuses_verified_successor"),
        click_delay_ms=0,
        idle_delay_ms=0,
        max_actions=2,
    )

    BotEngine(context).run()

    assert len(driver.actions) == 2
    assert capture.index == 3


def test_action_attempts_reset_when_planner_changes_target_type() -> None:
    frame = make_frame()
    detection = make_detection(type="dinosaur")
    target = Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )
    context = BotContext(
        capture_provider=SequenceCapture([frame]),
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=RecordingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_target_attempt_reset"),
        click_delay_ms=0,
        state=BotState.ACTION,
        frame=frame,
        target=target,
        action=ActionCommand.tap(target.x, target.y),
        attempt=3,
        attempt_target_type="hunt_button",
    )

    BotEngine(context).step()

    assert context.attempt == 1
    assert context.attempt_target_type == "dinosaur"


def test_engine_stop_interrupts_post_action_delay() -> None:
    action_started = threading.Event()

    class SignalingActionDriver:
        def execute(self, action: ActionCommand, frame: Frame) -> None:
            action_started.set()

    frame = make_frame()
    detection = make_detection(type="no_available_dinosaurs")
    target = Target(
        detection.type,
        detection.x,
        detection.y,
        detection.confidence,
        detection,
    )
    context = BotContext(
        capture_provider=SequenceCapture([frame]),
        detector=PixelDetector(),
        planner=TargetPlanner(),
        action_driver=SignalingActionDriver(),
        verifier=TargetChangedVerifier(),
        observer=RuntimeMode(),
        logger=logging.getLogger("test_engine_interruptible_delay"),
        post_action_delays_ms={"no_available_dinosaurs": 300_000},
        state=BotState.ACTION,
        frame=frame,
        target=target,
        action=ActionCommand.tap(target.x, target.y),
    )
    engine = BotEngine(context)
    result: list[BotState] = []
    worker = threading.Thread(target=lambda: result.append(engine.step()))

    worker.start()
    assert action_started.wait(timeout=1)
    engine.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result == [BotState.STOPPED]


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


def test_adb_app_restarter_only_restarts_configured_game() -> None:
    client = FakeAdbClient()
    restarter = AdbAppRestarter(client, "game.package", "GameActivity")  # type: ignore[arg-type]
    restarter.restart()
    assert client.commands == [
        ["shell", "am", "force-stop", "game.package"],
        ["shell", "am", "start", "-n", "game.package/GameActivity"],
    ]


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
