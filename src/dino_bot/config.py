"""JSON configuration with validation and path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    backend: Literal["mss", "adb"] = "mss"
    window_titles: tuple[str, ...] = ("BlueStacks App Player", "BlueStacks")
    process_names: tuple[str, ...] = ("HD-Player.exe",)
    viewport: tuple[int, int, int, int] | None = None
    auto_viewport: bool = False
    chrome_insets: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class AdbConfig:
    executable: str | None = None
    serial: str | None = "127.0.0.1:5555"
    connect_on_start: bool = True
    timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    manifest: Path = Path("assets/manifest.json")
    default_threshold: float = 0.85
    nms_iou: float = 0.3


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    target_types: tuple[str, ...] = ("resource",)
    strategy: Literal["nearest_center", "highest_confidence"] = "nearest_center"


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    max_distance: float = 35.0
    pixel_change_threshold: float = 0.08


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    fps: float = 2.0
    max_images: int = 500


@dataclass(frozen=True, slots=True)
class AppConfig:
    root: Path
    mode: Literal["runtime", "debug", "training"] = "runtime"
    debug: bool = False
    capture_fps: float = 10.0
    click_delay: int = 200
    verify_retry: int = 3
    save_debug_image: bool = False
    idle_delay: int = 500
    max_actions: int = 0
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    adb: AdbConfig = field(default_factory=AdbConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def debug_dir(self) -> Path:
        return self.root / "debug"

    @property
    def training_dir(self) -> Path:
        return self.root / "capture"


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")
    return value


def _path_from(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path = "config.json") -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a JSON object")

    root = config_path.parent
    capture_data = _section(data, "capture")
    adb_data = _section(data, "adb")
    detector_data = _section(data, "detector")
    planner_data = _section(data, "planner")
    verify_data = _section(data, "verify")
    training_data = _section(data, "training")

    viewport_raw = capture_data.get("viewport")
    auto_viewport = isinstance(viewport_raw, str) and viewport_raw.lower() == "auto"
    viewport = (
        None
        if viewport_raw is None or auto_viewport
        else tuple(int(v) for v in viewport_raw)
    )
    if viewport is not None and (len(viewport) != 4 or min(viewport[2:]) <= 0):
        raise ConfigError("capture.viewport must be [x, y, width, height]")
    chrome_insets = tuple(int(v) for v in capture_data.get("chrome_insets", [0, 0, 0, 0]))
    if len(chrome_insets) != 4 or min(chrome_insets) < 0:
        raise ConfigError("capture.chrome_insets must be [left, top, right, bottom]")

    mode = str(data.get("mode", "runtime")).lower()
    if data.get("debug", False) and mode == "runtime":
        mode = "debug"
    if mode not in {"runtime", "debug", "training"}:
        raise ConfigError("mode must be runtime, debug, or training")

    backend = str(capture_data.get("backend", "mss")).lower()
    if backend not in {"mss", "adb"}:
        raise ConfigError("capture.backend must be mss or adb")

    strategy = str(planner_data.get("strategy", "nearest_center"))
    if strategy not in {"nearest_center", "highest_confidence"}:
        raise ConfigError("planner.strategy must be nearest_center or highest_confidence")

    config = AppConfig(
        root=root,
        mode=mode,  # type: ignore[arg-type]
        debug=bool(data.get("debug", False)),
        capture_fps=float(data.get("capture_fps", 10)),
        click_delay=int(data.get("click_delay", 200)),
        verify_retry=int(data.get("verify_retry", 3)),
        save_debug_image=bool(data.get("save_debug_image", False)),
        idle_delay=int(data.get("idle_delay", 500)),
        max_actions=int(data.get("max_actions", 0)),
        capture=CaptureConfig(
            backend=backend,  # type: ignore[arg-type]
            window_titles=tuple(capture_data.get("window_titles", ["BlueStacks"])),
            process_names=tuple(capture_data.get("process_names", ["HD-Player.exe"])),
            viewport=viewport,  # type: ignore[arg-type]
            auto_viewport=auto_viewport,
            chrome_insets=chrome_insets,  # type: ignore[arg-type]
        ),
        adb=AdbConfig(
            executable=adb_data.get("executable"),
            serial=adb_data.get("serial"),
            connect_on_start=bool(adb_data.get("connect_on_start", True)),
            timeout=float(adb_data.get("timeout", 5.0)),
        ),
        detector=DetectorConfig(
            manifest=_path_from(root, detector_data.get("manifest", "assets/manifest.json")),
            default_threshold=float(detector_data.get("default_threshold", 0.85)),
            nms_iou=float(detector_data.get("nms_iou", 0.3)),
        ),
        planner=PlannerConfig(
            target_types=tuple(planner_data.get("target_types", ["resource"])),
            strategy=strategy,  # type: ignore[arg-type]
        ),
        verify=VerifyConfig(
            max_distance=float(verify_data.get("max_distance", 35)),
            pixel_change_threshold=float(verify_data.get("pixel_change_threshold", 0.08)),
        ),
        training=TrainingConfig(
            fps=float(training_data.get("fps", 2)),
            max_images=int(training_data.get("max_images", 500)),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.capture_fps <= 0:
        raise ConfigError("capture_fps must be greater than zero")
    if config.click_delay < 0 or config.idle_delay < 0:
        raise ConfigError("delays cannot be negative")
    if config.verify_retry < 0:
        raise ConfigError("verify_retry cannot be negative")
    if config.training.fps <= 0 or config.training.max_images <= 0:
        raise ConfigError("training fps and max_images must be greater than zero")
    if not 0 <= config.detector.default_threshold <= 1:
        raise ConfigError("detector.default_threshold must be between zero and one")
    if not 0 <= config.detector.nms_iou <= 1:
        raise ConfigError("detector.nms_iou must be between zero and one")
