"""JSON configuration with validation and path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    pass


DEFAULT_SPEED_PROFILES: dict[str, dict[str, int]] = {
    "safe": {
        "click_delay_ms": 1500,
        "dinosaur_delay_ms": 1500,
        "hunt_button_delay_ms": 5000,
        "hunt_confirm_delay_ms": 3000,
        "idle_delay_ms": 500,
        "poll_interval_ms": 250,
    },
    "fast": {
        "click_delay_ms": 300,
        "dinosaur_delay_ms": 300,
        "hunt_button_delay_ms": 900,
        "hunt_confirm_delay_ms": 1200,
        "idle_delay_ms": 250,
        "poll_interval_ms": 100,
    },
}


def _default_speed_profiles() -> dict[str, dict[str, int]]:
    return {name: dict(values) for name, values in DEFAULT_SPEED_PROFILES.items()}


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
    serial: str | None = "auto"
    fallback_serial: str | None = "127.0.0.1:5555"
    require_usb: bool = False
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
    blocking_types: tuple[str, ...] = ()
    deduplicate_types: tuple[str, ...] = ()
    dedup_radius: float = 60.0
    history_file: Path | None = None
    history_limit: int = 500
    recenter_every: int = 10
    own_path_radius: float = 90.0
    anchor_exclusion_radius: float = 50.0
    dinosaur_failure_cooldown_ms: int = 5_000
    dinosaur_failure_radius: float = 80.0
    mail_after_hunts: int = 30
    capacity_wait_seconds: float = 300.0
    ring_width: float = 150.0
    own_path_angle_degrees: float = 7.0
    stalled_recenter_frames: int = 8
    bottom_exclusion_px: int = 180
    action_cooldowns_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    max_distance: float = 35.0
    pixel_change_threshold: float = 0.08
    failure_types: tuple[str, ...] = ()
    success_transitions: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    fps: float = 2.0
    max_images: int = 500


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    max_cycles: int = 0
    complete_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    enabled: bool = True
    black_screen_timeout_seconds: float = 45.0
    black_mean_threshold: float = 2.0
    restart_cooldown_seconds: float = 90.0
    launch_wait_seconds: float = 15.0
    package: str = "com.mondayoff.dinomutant"
    activity: str = "com.unity3d.player.UnityPlayerActivity"


@dataclass(frozen=True, slots=True)
class AppConfig:
    root: Path
    mode: Literal["runtime", "debug", "training"] = "runtime"
    debug: bool = False
    capture_fps: float = 10.0
    click_delay: int = 200
    post_action_delays: dict[str, int] = field(default_factory=dict)
    target_actions: dict[str, str] = field(default_factory=dict)
    verify_retry: int = 3
    save_debug_image: bool = False
    idle_delay: int = 500
    transition_poll_interval: int = 250
    speed_profiles: dict[str, dict[str, int]] = field(
        default_factory=_default_speed_profiles
    )
    max_actions: int = 0
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    adb: AdbConfig = field(default_factory=AdbConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

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
    workflow_data = _section(data, "workflow")
    recovery_data = _section(data, "recovery")
    speed_profiles_data = _section(data, "speed_profiles")

    speed_profiles = _default_speed_profiles()
    for profile_name, raw_profile in speed_profiles_data.items():
        if profile_name not in DEFAULT_SPEED_PROFILES:
            raise ConfigError(f"unknown speed profile: {profile_name}")
        if not isinstance(raw_profile, dict):
            raise ConfigError(f"speed_profiles.{profile_name} must be a JSON object")
        speed_profiles[profile_name].update(
            {str(key): int(value) for key, value in raw_profile.items()}
        )

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
        post_action_delays={
            str(target_type): int(delay)
            for target_type, delay in _section(data, "post_action_delays").items()
        },
        target_actions={
            str(target_type): str(action).lower()
            for target_type, action in _section(data, "target_actions").items()
        },
        verify_retry=int(data.get("verify_retry", 3)),
        save_debug_image=bool(data.get("save_debug_image", False)),
        idle_delay=int(data.get("idle_delay", 500)),
        transition_poll_interval=int(data.get("transition_poll_interval", 250)),
        speed_profiles=speed_profiles,
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
            serial=adb_data.get("serial", "auto"),
            fallback_serial=adb_data.get("fallback_serial", "127.0.0.1:5555"),
            require_usb=bool(adb_data.get("require_usb", False)),
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
            blocking_types=tuple(planner_data.get("blocking_types", [])),
            deduplicate_types=tuple(planner_data.get("deduplicate_types", [])),
            dedup_radius=float(planner_data.get("dedup_radius", 60)),
            history_file=(
                _path_from(root, planner_data["history_file"])
                if planner_data.get("history_file")
                else None
            ),
            history_limit=int(planner_data.get("history_limit", 500)),
            recenter_every=int(planner_data.get("recenter_every", 10)),
            own_path_radius=float(planner_data.get("own_path_radius", 90)),
            anchor_exclusion_radius=float(
                planner_data.get("anchor_exclusion_radius", 50)
            ),
            dinosaur_failure_cooldown_ms=int(
                planner_data.get("dinosaur_failure_cooldown_ms", 5_000)
            ),
            dinosaur_failure_radius=float(
                planner_data.get("dinosaur_failure_radius", 80)
            ),
            mail_after_hunts=int(planner_data.get("mail_after_hunts", 30)),
            capacity_wait_seconds=float(
                planner_data.get("capacity_wait_seconds", 300)
            ),
            ring_width=float(planner_data.get("ring_width", 150)),
            own_path_angle_degrees=float(
                planner_data.get("own_path_angle_degrees", 7)
            ),
            stalled_recenter_frames=int(
                planner_data.get("stalled_recenter_frames", 8)
            ),
            bottom_exclusion_px=int(planner_data.get("bottom_exclusion_px", 180)),
            action_cooldowns_ms={
                str(target_type): int(delay)
                for target_type, delay in _section(
                    planner_data, "action_cooldowns_ms"
                ).items()
            },
        ),
        verify=VerifyConfig(
            max_distance=float(verify_data.get("max_distance", 35)),
            pixel_change_threshold=float(verify_data.get("pixel_change_threshold", 0.08)),
            failure_types=tuple(verify_data.get("failure_types", [])),
            success_transitions={
                str(target_type): tuple(str(item) for item in successors)
                for target_type, successors in verify_data.get(
                    "success_transitions", {}
                ).items()
            },
        ),
        training=TrainingConfig(
            fps=float(training_data.get("fps", 2)),
            max_images=int(training_data.get("max_images", 500)),
        ),
        workflow=WorkflowConfig(
            max_cycles=int(workflow_data.get("max_cycles", 0)),
            complete_on=tuple(workflow_data.get("complete_on", [])),
        ),
        recovery=RecoveryConfig(
            enabled=bool(recovery_data.get("enabled", True)),
            black_screen_timeout_seconds=float(
                recovery_data.get("black_screen_timeout_seconds", 45)
            ),
            black_mean_threshold=float(
                recovery_data.get("black_mean_threshold", 2)
            ),
            restart_cooldown_seconds=float(
                recovery_data.get("restart_cooldown_seconds", 90)
            ),
            launch_wait_seconds=float(
                recovery_data.get("launch_wait_seconds", 15)
            ),
            package=str(
                recovery_data.get("package", "com.mondayoff.dinomutant")
            ),
            activity=str(
                recovery_data.get(
                    "activity", "com.unity3d.player.UnityPlayerActivity"
                )
            ),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.capture_fps <= 0:
        raise ConfigError("capture_fps must be greater than zero")
    if config.click_delay < 0 or config.idle_delay < 0:
        raise ConfigError("click_delay and idle_delay cannot be negative")
    if config.transition_poll_interval <= 0:
        raise ConfigError("transition_poll_interval must be greater than zero")
    if any(delay < 0 for delay in config.post_action_delays.values()):
        raise ConfigError("post_action_delays cannot be negative")
    required_profile_keys = frozenset(DEFAULT_SPEED_PROFILES["fast"])
    for profile_name, profile in config.speed_profiles.items():
        unknown = profile.keys() - required_profile_keys
        if unknown:
            raise ConfigError(
                f"speed_profiles.{profile_name} contains unknown keys: "
                f"{', '.join(sorted(unknown))}"
            )
        missing = required_profile_keys - profile.keys()
        if missing:
            raise ConfigError(
                f"speed_profiles.{profile_name} is missing: {', '.join(sorted(missing))}"
            )
        if any(value < 0 for value in profile.values()):
            raise ConfigError(f"speed_profiles.{profile_name} cannot contain negative values")
        if profile["poll_interval_ms"] <= 0:
            raise ConfigError(
                f"speed_profiles.{profile_name}.poll_interval_ms must be greater than zero"
            )
    if any(action not in {"tap", "back"} for action in config.target_actions.values()):
        raise ConfigError("target_actions values must be tap or back")
    if config.verify_retry < 0:
        raise ConfigError("verify_retry cannot be negative")
    if not 1 <= config.training.fps <= 5:
        raise ConfigError("training.fps must be between 1 and 5")
    if not 1 <= config.training.max_images <= 500:
        raise ConfigError("training.max_images must be between 1 and 500")
    if config.workflow.max_cycles < 0:
        raise ConfigError("workflow.max_cycles cannot be negative")
    if config.planner.dedup_radius < 0 or config.planner.history_limit <= 0:
        raise ConfigError("planner dedup_radius/history_limit are invalid")
    if config.planner.recenter_every <= 0:
        raise ConfigError("planner.recenter_every must be greater than zero")
    if config.recovery.black_screen_timeout_seconds <= 0:
        raise ConfigError("recovery.black_screen_timeout_seconds must be greater than zero")
    if not 0 <= config.recovery.black_mean_threshold <= 255:
        raise ConfigError("recovery.black_mean_threshold must be between 0 and 255")
    if config.recovery.restart_cooldown_seconds < 0:
        raise ConfigError("recovery.restart_cooldown_seconds cannot be negative")
    if config.recovery.launch_wait_seconds < 0:
        raise ConfigError("recovery.launch_wait_seconds cannot be negative")
    if not config.recovery.package or not config.recovery.activity:
        raise ConfigError("recovery package/activity cannot be empty")
    if config.planner.own_path_radius < 0:
        raise ConfigError("planner.own_path_radius cannot be negative")
    if config.planner.anchor_exclusion_radius < 0:
        raise ConfigError("planner.anchor_exclusion_radius cannot be negative")
    if config.planner.dinosaur_failure_cooldown_ms < 0:
        raise ConfigError("planner.dinosaur_failure_cooldown_ms cannot be negative")
    if config.planner.dinosaur_failure_radius < 0:
        raise ConfigError("planner.dinosaur_failure_radius cannot be negative")
    if config.planner.mail_after_hunts <= 0:
        raise ConfigError("planner.mail_after_hunts must be greater than zero")
    if config.planner.capacity_wait_seconds < 0:
        raise ConfigError("planner.capacity_wait_seconds cannot be negative")
    if any(delay < 0 for delay in config.planner.action_cooldowns_ms.values()):
        raise ConfigError("planner.action_cooldowns_ms cannot be negative")
    if config.planner.ring_width <= 0:
        raise ConfigError("planner.ring_width must be greater than zero")
    if not 0 <= config.planner.own_path_angle_degrees <= 180:
        raise ConfigError("planner.own_path_angle_degrees must be between 0 and 180")
    if config.planner.stalled_recenter_frames <= 0:
        raise ConfigError("planner.stalled_recenter_frames must be greater than zero")
    if config.planner.bottom_exclusion_px < 0:
        raise ConfigError("planner.bottom_exclusion_px cannot be negative")
    if not 0 <= config.detector.default_threshold <= 1:
        raise ConfigError("detector.default_threshold must be between zero and one")
    if not 0 <= config.detector.nms_iou <= 1:
        raise ConfigError("detector.nms_iou must be between zero and one")
