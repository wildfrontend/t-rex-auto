from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from dino_bot.actions import AdbActionDriver, RecordingActionDriver
from dino_bot.assets import create_template
from dino_bot.detection import OpenCvDetector
from dino_bot.engine import BotContext, BotEngine, BotState
from dino_bot.models import BoundingBox, Detection, Frame, Target, VerificationResult
from dino_bot.modes import DebugMode, RuntimeMode, TrainingMode
from dino_bot.planning import TargetPlanner
from dino_bot.verification import TargetChangedVerifier


def make_frame(value: int = 0, sequence: int = 1) -> Frame:
    return Frame(np.full((100, 160, 3), value, dtype=np.uint8), sequence=sequence)


def make_detection(x: int = 80, y: int = 50, type: str = "resource") -> Detection:
    return Detection.from_bbox(type, BoundingBox(x - 5, y - 5, 10, 10), 0.95)


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
        make_frame(), make_frame(), target, [detection], [detection]
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
