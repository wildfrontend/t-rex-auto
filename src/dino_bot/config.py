"""JSON configuration with validation and path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .models import ExclusionZone
from .nests import StatUpgradeGuard, default_stat_upgrade_guards


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

EMULATOR_PROFILES: dict[str, dict[str, object]] = {
    "bluestacks": {
        "serial": "127.0.0.1:5555",
        "window_titles": ("BlueStacks App Player", "BlueStacks"),
        "process_names": ("HD-Player.exe",),
    },
    "mumu": {
        "serial": "127.0.0.1:7555",
        "window_titles": ("MuMuPlayer", "MuMu Player", "MuMu模擬器", "MuMu模拟器"),
        "process_names": ("MuMuPlayer.exe", "NemuPlayer.exe", "MuMuNxDevice.exe"),
    },
    "custom": {
        "serial": None,
        "window_titles": (),
        "process_names": (),
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
    mail_failure_limit: int = 3
    capacity_wait_seconds: float = 300.0
    ring_width: float = 150.0
    own_path_angle_degrees: float = 7.0
    stalled_recenter_seconds: float = 10.0
    # Recentering restores the supply of reachable dinosaurs; it is not about
    # where the egg sits. Reset once fewer than this many candidates survive.
    recenter_min_candidates: int = 1
    # Two consecutive empty map observations are enough to prove the current
    # area is exhausted without waiting for the wall-clock stall timeout.
    empty_supply_recenter_frames: int = 2
    # Every other stall guard is written as "leave once the expected control
    # appears", so none of them fire on a screen showing no known control at
    # all. This one is measured from the planner alone.
    blind_idle_seconds: float = 20.0
    mail_stage_timeout_seconds: float = 20.0
    map_settle_frames: int = 2
    map_settle_tolerance_px: float = 20.0
    map_settle_max_frames: int = 12
    # How far from the viewport center a dinosaur may sit and still be worth
    # tapping. Beyond it the tap mostly just recenters the map without opening
    # the hunt panel. 0 disables the limit.
    max_center_distance_px: float = 600.0
    bottom_exclusion_px: int = 180
    exclusion_zones: tuple[ExclusionZone, ...] = ()
    retry_exhausted_cooldown_ms: int = 60_000
    suppression_radius: float = 60.0
    action_cooldowns_ms: dict[str, int] = field(default_factory=dict)
    stage_scoped_scan: bool = True
    full_scan_interval_seconds: float = 30.0
    full_scan_after_idle_cycles: int = 2


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    max_distance: float = 35.0
    pixel_change_threshold: float = 0.08
    minimum_checks: int = 2
    failure_types: tuple[str, ...] = ()
    success_transitions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    success_requires_target_absence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HatchConfig:
    """Auto Hatch feature settings (docs/auto-hatch-plan.md).

    Coordinates live in manifest reference space (a 900-wide layout) and are
    rescaled by frame width at run time, same as ``ExclusionZone``.
    """

    manifest: Path = Path("assets/hatch/manifest.json")
    reference_width: float = 900.0
    # The egg pile is styled dynamically, so it is tapped by coordinate, never
    # matched by template. Calibrate against a snapshot before first use.
    egg_pile: tuple[float, float] = (450.0, 1330.0)
    scroll_vector: tuple[float, float, float, float] = (450.0, 1100.0, 450.0, 500.0)
    scroll_duration_ms: int = 400
    # Ready eggs are ordered at the top. If the visible rows have no hatch
    # label, lower rows do not need scanning. Keep scrolling opt-in only.
    max_scrolls: int = 0
    # Game-imposed incubation cooldown is ~25 minutes; this is only how often
    # the bot re-enters to check, per the plan's rescan rule.
    rescan_interval_seconds: float = 600.0
    # Fixed game stat increments are used as a final OCR/action guard. Update
    # these ranges here when a later game version expands them.
    stat_upgrade_guards: dict[str, StatUpgradeGuard] = field(
        default_factory=default_stat_upgrade_guards
    )
    # OCR values must repeat across complete frames before any parent or
    # candidate tap is allowed. Retries include the initial observations.
    stat_consistent_reads: int = 2
    stat_read_retries: int = 3
    require_home_anchor: bool = True
    home_failure_limit: int = 3
    home_backoff_seconds: float = 30.0
    # Phase C: cull once the cave-view N/350 readout reaches this safety limit.
    # The game capacity remains 350; this lower threshold leaves headroom.
    cull_threshold: int = 330
    # Below the cull line, rerun nest screening every this many newly added
    # dinosaurs, measured from the last completed screening.
    screening_growth_interval: int = 20
    # Slow machines may need several complete detect cycles before the HUD is
    # rendered sharply enough for the N/350 reader.
    capacity_read_retries: int = 2
    # Give the cave map more complete detect cycles to prove that it returned
    # home before entering bounded recovery.
    cave_recenter_checks: int = 3
    # Time without an actionable target before full hatch begins home recovery.
    recovery_timeout_seconds: float = 15.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    fps: float = 2.0
    max_images: int = 500


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    max_cycles: int = 0
    complete_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventLogConfig:
    """Machine-readable event stream settings.

    On by default: the text log alone could not explain any of the failures it
    recorded, and a stream nobody switched on is a stream nobody has when it
    matters.
    """

    enabled: bool = True
    max_bytes: int = 16 * 1024 * 1024
    # Generations kept behind the live file. The cap fills in about 97 minutes,
    # so one generation covered barely three hours and an overnight run lost
    # everything before the last stretch. Generations past the first are
    # gzipped to 6.5%, so twenty of them cost about 21 MB instead of 320 MB.
    backup_count: int = 20


@dataclass(frozen=True, slots=True)
class StallConfig:
    """Evidence written when the planner stalls on an unrecognised screen.

    The planner owns the timer (``planner.blind_idle_seconds``); this owns what
    happens once it fires. On by default, because these episodes are rare
    enough that nobody will have switched it on before the one that matters.
    """

    snapshots_enabled: bool = True
    snapshot_limit: int = 10
    snapshot_min_interval_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    enabled: bool = True
    black_screen_timeout_seconds: float = 45.0
    black_mean_threshold: float = 2.0
    # Restarting the app is the most expensive escape there is - force-stop,
    # relaunch, a launch wait and the whole startup dialog sequence - and a
    # measured run needed two of them to leave one stall. Now that the planner
    # releases its own stages first, this is the backstop rather than the only
    # way out, so it can fire sooner.
    no_hunt_progress_timeout_seconds: float = 90.0
    hunt_progress_suspend_budget_seconds: float = 120.0
    restart_cooldown_seconds: float = 90.0
    launch_wait_seconds: float = 15.0
    # One exhausted retry budget is still a local failure. Repeated exhausted
    # budgets for the same behaviour first unwind its workflow, then restart
    # the game. If restarts themselves do not produce a real milestone, stop
    # instead of force-stopping the app forever.
    action_failure_stage_threshold: int = 2
    action_failure_restart_threshold: int = 3
    max_restarts_without_progress: int = 3
    package: str = "com.mondayoff.dinomutant"
    activity: str = "com.unity3d.player.UnityPlayerActivity"


@dataclass(frozen=True, slots=True)
class AppConfig:
    root: Path
    emulator: Literal["bluestacks", "mumu", "custom"] = "bluestacks"
    mode: Literal["runtime", "debug", "training"] = "runtime"
    debug: bool = False
    capture_fps: float = 10.0
    click_delay: int = 200
    post_action_delays: dict[str, int] = field(default_factory=dict)
    target_actions: dict[str, str] = field(default_factory=dict)
    verify_retry: int = 3
    save_debug_image: bool = False
    idle_delay: int = 500
    # The text log is read back by the control window and the diagnostic
    # bundle, so its size is a latency budget, not just disk.
    log_max_bytes: int = 32 * 1024 * 1024
    # Same reasoning as event_log.backup_count; the text log fills its cap in
    # roughly two hours and compresses to 4.4%.
    log_backup_count: int = 12
    transition_poll_interval: int = 250
    # Set by the CLI speed preset; config-file loads leave it unset.
    timing_profile: str | None = None
    speed_profiles: dict[str, dict[str, int]] = field(
        default_factory=_default_speed_profiles
    )
    max_actions: int = 0
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    adb: AdbConfig = field(default_factory=AdbConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    hatch: HatchConfig = field(default_factory=HatchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    event_log: EventLogConfig = field(default_factory=EventLogConfig)
    stalls: StallConfig = field(default_factory=StallConfig)

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def stalls_dir(self) -> Path:
        return self.root / "logs" / "stalls"

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


def _number_tuple(
    data: dict[str, Any],
    label: str,
    key: str,
    length: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    raw = data.get(key)
    if raw is None:
        return default
    if not isinstance(raw, list) or len(raw) != length:
        raise ConfigError(f"{label} must be a list of {length} numbers")
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must contain numbers") from exc


def _stat_upgrade_guards(data: dict[str, Any]) -> dict[str, StatUpgradeGuard]:
    raw = data.get("stat_upgrade_guards")
    guards = default_stat_upgrade_guards()
    if raw is None:
        return guards
    if not isinstance(raw, dict):
        raise ConfigError("hatch.stat_upgrade_guards must be a JSON object")

    unknown_stats = set(raw) - set(guards)
    if unknown_stats:
        raise ConfigError(
            "hatch.stat_upgrade_guards contains unknown stats: "
            + ", ".join(sorted(str(item) for item in unknown_stats))
        )
    allowed_keys = frozenset(
        {"min_delta", "max_delta", "min_value", "max_value", "multiple_of"}
    )
    for stat, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"hatch.stat_upgrade_guards.{stat} must be a JSON object")
        unknown_keys = set(entry) - allowed_keys
        if unknown_keys:
            raise ConfigError(
                f"hatch.stat_upgrade_guards.{stat} contains unknown keys: "
                + ", ".join(sorted(str(item) for item in unknown_keys))
            )
        default = guards[stat]
        values: dict[str, int | None] = {}
        for key in allowed_keys:
            value = entry.get(key, getattr(default, key))
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigError(
                    f"hatch.stat_upgrade_guards.{stat}.{key} must be an integer or null"
                )
            values[key] = value
        guards[stat] = StatUpgradeGuard(**values)
    return guards


def _exclusion_zones(data: dict[str, Any]) -> tuple[ExclusionZone, ...]:
    raw = data.get("exclusion_zones", [])
    if not isinstance(raw, list):
        raise ConfigError("planner.exclusion_zones must be a JSON array")
    zones: list[ExclusionZone] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"planner.exclusion_zones[{index}] must be a JSON object"
            )
        name = str(entry.get("name") or f"zone_{index}")
        reference_width = float(entry.get("reference_width", 900))
        if reference_width <= 0:
            raise ConfigError(
                f"planner.exclusion_zones[{index}].reference_width must be "
                "greater than zero"
            )
        bounds: list[float] = []
        for axis in ("x", "y"):
            pair = entry.get(axis)
            if not isinstance(pair, list) or len(pair) != 2:
                raise ConfigError(
                    f"planner.exclusion_zones[{index}].{axis} must be [start, end]"
                )
            try:
                low, high = (float(value) for value in pair)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"planner.exclusion_zones[{index}].{axis} must contain numbers"
                ) from exc
            if low < 0:
                raise ConfigError(
                    f"planner.exclusion_zones[{index}].{axis} cannot be negative"
                )
            if low >= high:
                raise ConfigError(
                    f"planner.exclusion_zones[{index}].{axis} start must be "
                    "smaller than end"
                )
            bounds.extend((low, high))
        x0, x1, y0, y1 = bounds
        zones.append(
            ExclusionZone(
                name=name,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                reference_width=reference_width,
            )
        )
    return tuple(zones)


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
    emulator = str(data.get("emulator", "bluestacks")).lower()
    if emulator not in EMULATOR_PROFILES:
        supported = ", ".join(EMULATOR_PROFILES)
        raise ConfigError(f"emulator must be one of: {supported}")
    emulator_profile = EMULATOR_PROFILES[emulator]
    capture_data = _section(data, "capture")
    adb_data = _section(data, "adb")
    detector_data = _section(data, "detector")
    planner_data = _section(data, "planner")
    verify_data = _section(data, "verify")
    hatch_data = _section(data, "hatch")
    training_data = _section(data, "training")
    workflow_data = _section(data, "workflow")
    recovery_data = _section(data, "recovery")
    event_log_data = _section(data, "event_log")
    stalls_data = _section(data, "stalls")
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
        emulator=emulator,  # type: ignore[arg-type]
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
        log_max_bytes=int(data.get("log_max_bytes", 32 * 1024 * 1024)),
        log_backup_count=int(data.get("log_backup_count", 12)),
        transition_poll_interval=int(data.get("transition_poll_interval", 250)),
        speed_profiles=speed_profiles,
        max_actions=int(data.get("max_actions", 0)),
        capture=CaptureConfig(
            backend=backend,  # type: ignore[arg-type]
            window_titles=tuple(
                capture_data.get("window_titles", emulator_profile["window_titles"])
            ),
            process_names=tuple(
                capture_data.get("process_names", emulator_profile["process_names"])
            ),
            viewport=viewport,  # type: ignore[arg-type]
            auto_viewport=auto_viewport,
            chrome_insets=chrome_insets,  # type: ignore[arg-type]
        ),
        adb=AdbConfig(
            executable=adb_data.get("executable"),
            serial=adb_data.get("serial", emulator_profile["serial"]),
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
            mail_failure_limit=int(planner_data.get("mail_failure_limit", 3)),
            capacity_wait_seconds=float(
                planner_data.get("capacity_wait_seconds", 300)
            ),
            ring_width=float(planner_data.get("ring_width", 150)),
            own_path_angle_degrees=float(
                planner_data.get("own_path_angle_degrees", 7)
            ),
            stalled_recenter_seconds=float(
                planner_data.get("stalled_recenter_seconds", 10)
            ),
            recenter_min_candidates=int(
                planner_data.get("recenter_min_candidates", 1)
            ),
            empty_supply_recenter_frames=int(
                planner_data.get("empty_supply_recenter_frames", 2)
            ),
            blind_idle_seconds=float(planner_data.get("blind_idle_seconds", 20)),
            mail_stage_timeout_seconds=float(
                planner_data.get("mail_stage_timeout_seconds", 20)
            ),
            map_settle_frames=int(planner_data.get("map_settle_frames", 2)),
            map_settle_tolerance_px=float(
                planner_data.get("map_settle_tolerance_px", 20)
            ),
            map_settle_max_frames=int(
                planner_data.get("map_settle_max_frames", 12)
            ),
            max_center_distance_px=float(
                planner_data.get("max_center_distance_px", 600)
            ),
            bottom_exclusion_px=int(planner_data.get("bottom_exclusion_px", 180)),
            exclusion_zones=_exclusion_zones(planner_data),
            retry_exhausted_cooldown_ms=int(
                planner_data.get("retry_exhausted_cooldown_ms", 60_000)
            ),
            suppression_radius=float(planner_data.get("suppression_radius", 60)),
            action_cooldowns_ms={
                str(target_type): int(delay)
                for target_type, delay in _section(
                    planner_data, "action_cooldowns_ms"
                ).items()
            },
            stage_scoped_scan=bool(planner_data.get("stage_scoped_scan", True)),
            full_scan_interval_seconds=float(
                planner_data.get("full_scan_interval_seconds", 30)
            ),
            full_scan_after_idle_cycles=int(
                planner_data.get("full_scan_after_idle_cycles", 2)
            ),
        ),
        verify=VerifyConfig(
            max_distance=float(verify_data.get("max_distance", 35)),
            pixel_change_threshold=float(verify_data.get("pixel_change_threshold", 0.08)),
            minimum_checks=int(verify_data.get("minimum_checks", 2)),
            failure_types=tuple(verify_data.get("failure_types", [])),
            success_transitions={
                str(target_type): tuple(str(item) for item in successors)
                for target_type, successors in verify_data.get(
                    "success_transitions", {}
                ).items()
            },
            success_requires_target_absence=tuple(
                str(item)
                for item in verify_data.get(
                    "success_requires_target_absence",
                    [],
                )
            ),
        ),
        hatch=HatchConfig(
            manifest=_path_from(
                root, hatch_data.get("manifest", "assets/hatch/manifest.json")
            ),
            reference_width=float(hatch_data.get("reference_width", 900)),
            egg_pile=_number_tuple(hatch_data, "hatch.egg_pile", "egg_pile", 2, (450.0, 1330.0)),
            scroll_vector=_number_tuple(
                hatch_data,
                "hatch.scroll_vector",
                "scroll_vector",
                4,
                (450.0, 1100.0, 450.0, 500.0),
            ),
            scroll_duration_ms=int(hatch_data.get("scroll_duration_ms", 400)),
            max_scrolls=int(hatch_data.get("max_scrolls", 0)),
            rescan_interval_seconds=float(
                hatch_data.get("rescan_interval_seconds", 600)
            ),
            stat_upgrade_guards=_stat_upgrade_guards(hatch_data),
            stat_consistent_reads=int(hatch_data.get("stat_consistent_reads", 2)),
            stat_read_retries=int(hatch_data.get("stat_read_retries", 3)),
            require_home_anchor=bool(hatch_data.get("require_home_anchor", True)),
            home_failure_limit=int(hatch_data.get("home_failure_limit", 3)),
            home_backoff_seconds=float(hatch_data.get("home_backoff_seconds", 30)),
            cull_threshold=int(hatch_data.get("cull_threshold", 330)),
            screening_growth_interval=int(
                hatch_data.get("screening_growth_interval", 20)
            ),
            capacity_read_retries=int(hatch_data.get("capacity_read_retries", 2)),
            cave_recenter_checks=int(hatch_data.get("cave_recenter_checks", 3)),
            recovery_timeout_seconds=float(
                hatch_data.get("recovery_timeout_seconds", 15)
            ),
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
            no_hunt_progress_timeout_seconds=float(
                recovery_data.get("no_hunt_progress_timeout_seconds", 90)
            ),
            hunt_progress_suspend_budget_seconds=float(
                recovery_data.get("hunt_progress_suspend_budget_seconds", 120)
            ),
            restart_cooldown_seconds=float(
                recovery_data.get("restart_cooldown_seconds", 90)
            ),
            launch_wait_seconds=float(
                recovery_data.get("launch_wait_seconds", 15)
            ),
            action_failure_stage_threshold=int(
                recovery_data.get("action_failure_stage_threshold", 2)
            ),
            action_failure_restart_threshold=int(
                recovery_data.get("action_failure_restart_threshold", 3)
            ),
            max_restarts_without_progress=int(
                recovery_data.get("max_restarts_without_progress", 3)
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
        event_log=EventLogConfig(
            enabled=bool(event_log_data.get("enabled", True)),
            max_bytes=int(event_log_data.get("max_bytes", 16 * 1024 * 1024)),
            backup_count=int(event_log_data.get("backup_count", 20)),
        ),
        stalls=StallConfig(
            snapshots_enabled=bool(stalls_data.get("snapshots_enabled", True)),
            snapshot_limit=int(stalls_data.get("snapshot_limit", 10)),
            snapshot_min_interval_seconds=float(
                stalls_data.get("snapshot_min_interval_seconds", 60)
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
    if any(
        action not in {"tap", "back", "swipe"}
        for action in config.target_actions.values()
    ):
        raise ConfigError("target_actions values must be tap, back, or swipe")
    if config.hatch.reference_width <= 0:
        raise ConfigError("hatch.reference_width must be greater than zero")
    if config.hatch.scroll_duration_ms <= 0:
        raise ConfigError("hatch.scroll_duration_ms must be greater than zero")
    if config.hatch.max_scrolls < 0:
        raise ConfigError("hatch.max_scrolls cannot be negative")
    if config.hatch.rescan_interval_seconds < 0:
        raise ConfigError("hatch.rescan_interval_seconds cannot be negative")
    expected_stats = {"hp", "attack", "speed"}
    if set(config.hatch.stat_upgrade_guards) != expected_stats:
        raise ConfigError(
            "hatch.stat_upgrade_guards must define exactly hp, attack, and speed"
        )
    for stat, guard in config.hatch.stat_upgrade_guards.items():
        for name in (
            "min_delta",
            "max_delta",
            "min_value",
            "max_value",
            "multiple_of",
        ):
            value = getattr(guard, name)
            if value is not None and value < 0:
                raise ConfigError(f"hatch.stat_upgrade_guards.{stat}.{name} cannot be negative")
        if guard.multiple_of == 0:
            raise ConfigError(
                f"hatch.stat_upgrade_guards.{stat}.multiple_of must be greater than zero"
            )
        if (
            guard.min_delta is not None
            and guard.max_delta is not None
            and guard.min_delta > guard.max_delta
        ):
            raise ConfigError(
                f"hatch.stat_upgrade_guards.{stat}.min_delta cannot exceed max_delta"
            )
        if (
            guard.min_value is not None
            and guard.max_value is not None
            and guard.min_value > guard.max_value
        ):
            raise ConfigError(
                f"hatch.stat_upgrade_guards.{stat}.min_value cannot exceed max_value"
            )
    if config.hatch.stat_consistent_reads <= 0:
        raise ConfigError("hatch.stat_consistent_reads must be greater than zero")
    if config.hatch.stat_read_retries < config.hatch.stat_consistent_reads:
        raise ConfigError(
            "hatch.stat_read_retries cannot be less than stat_consistent_reads"
        )
    if config.hatch.capacity_read_retries <= 0:
        raise ConfigError("hatch.capacity_read_retries must be greater than zero")
    if config.hatch.cave_recenter_checks <= 0:
        raise ConfigError("hatch.cave_recenter_checks must be greater than zero")
    if config.hatch.recovery_timeout_seconds <= 0:
        raise ConfigError("hatch.recovery_timeout_seconds must be greater than zero")
    if config.hatch.home_failure_limit <= 0:
        raise ConfigError("hatch.home_failure_limit must be greater than zero")
    if config.hatch.home_backoff_seconds < 0:
        raise ConfigError("hatch.home_backoff_seconds cannot be negative")
    if config.hatch.cull_threshold < 0:
        raise ConfigError("hatch.cull_threshold cannot be negative")
    if config.hatch.screening_growth_interval <= 0:
        raise ConfigError(
            "hatch.screening_growth_interval must be greater than zero"
        )
    if config.verify_retry < 0:
        raise ConfigError("verify_retry cannot be negative")
    if config.verify.minimum_checks <= 0:
        raise ConfigError("verify.minimum_checks must be greater than zero")
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
    if config.recovery.no_hunt_progress_timeout_seconds < 0:
        raise ConfigError(
            "recovery.no_hunt_progress_timeout_seconds cannot be negative"
        )
    if config.recovery.restart_cooldown_seconds < 0:
        raise ConfigError("recovery.restart_cooldown_seconds cannot be negative")
    if config.recovery.launch_wait_seconds < 0:
        raise ConfigError("recovery.launch_wait_seconds cannot be negative")
    if config.recovery.action_failure_stage_threshold <= 0:
        raise ConfigError(
            "recovery.action_failure_stage_threshold must be greater than zero"
        )
    if (
        config.recovery.action_failure_restart_threshold
        <= config.recovery.action_failure_stage_threshold
    ):
        raise ConfigError(
            "recovery.action_failure_restart_threshold must be greater than "
            "action_failure_stage_threshold"
        )
    if config.recovery.max_restarts_without_progress <= 0:
        raise ConfigError(
            "recovery.max_restarts_without_progress must be greater than zero"
        )
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
    if config.planner.mail_failure_limit <= 0:
        raise ConfigError("planner.mail_failure_limit must be greater than zero")
    if config.planner.capacity_wait_seconds < 0:
        raise ConfigError("planner.capacity_wait_seconds cannot be negative")
    if any(delay < 0 for delay in config.planner.action_cooldowns_ms.values()):
        raise ConfigError("planner.action_cooldowns_ms cannot be negative")
    if config.planner.ring_width <= 0:
        raise ConfigError("planner.ring_width must be greater than zero")
    if not 0 <= config.planner.own_path_angle_degrees <= 180:
        raise ConfigError("planner.own_path_angle_degrees must be between 0 and 180")
    if config.planner.stalled_recenter_seconds <= 0:
        raise ConfigError("planner.stalled_recenter_seconds must be greater than zero")
    if config.planner.recenter_min_candidates <= 0:
        raise ConfigError("planner.recenter_min_candidates must be greater than zero")
    if config.planner.empty_supply_recenter_frames <= 0:
        raise ConfigError(
            "planner.empty_supply_recenter_frames must be greater than zero"
        )
    if config.planner.blind_idle_seconds <= 0:
        raise ConfigError("planner.blind_idle_seconds must be greater than zero")
    if config.planner.mail_stage_timeout_seconds <= 0:
        raise ConfigError(
            "planner.mail_stage_timeout_seconds must be greater than zero"
        )
    if config.stalls.snapshot_limit <= 0:
        raise ConfigError("stalls.snapshot_limit must be greater than zero")
    if config.stalls.snapshot_min_interval_seconds < 0:
        raise ConfigError("stalls.snapshot_min_interval_seconds cannot be negative")
    if config.planner.full_scan_interval_seconds < 0:
        raise ConfigError("planner.full_scan_interval_seconds cannot be negative")
    if config.planner.full_scan_after_idle_cycles <= 0:
        raise ConfigError(
            "planner.full_scan_after_idle_cycles must be greater than zero"
        )
    if config.planner.map_settle_frames <= 0:
        raise ConfigError("planner.map_settle_frames must be greater than zero")
    if config.planner.map_settle_tolerance_px < 0:
        raise ConfigError("planner.map_settle_tolerance_px cannot be negative")
    if config.planner.map_settle_max_frames < config.planner.map_settle_frames:
        raise ConfigError(
            "planner.map_settle_max_frames must be at least map_settle_frames"
        )
    if config.planner.max_center_distance_px < 0:
        raise ConfigError("planner.max_center_distance_px cannot be negative")
    if config.planner.bottom_exclusion_px < 0:
        raise ConfigError("planner.bottom_exclusion_px cannot be negative")
    if config.planner.retry_exhausted_cooldown_ms < 0:
        raise ConfigError("planner.retry_exhausted_cooldown_ms cannot be negative")
    if config.planner.suppression_radius < 0:
        raise ConfigError("planner.suppression_radius cannot be negative")
    if config.recovery.hunt_progress_suspend_budget_seconds < 0:
        raise ConfigError(
            "recovery.hunt_progress_suspend_budget_seconds cannot be negative"
        )
    if config.event_log.max_bytes < 0:
        raise ConfigError("event_log.max_bytes cannot be negative")
    if config.event_log.backup_count < 1:
        raise ConfigError("event_log.backup_count must be at least one")
    if config.log_backup_count < 1:
        raise ConfigError("log_backup_count must be at least one")
    if not 0 <= config.detector.default_threshold <= 1:
        raise ConfigError("detector.default_threshold must be between zero and one")
    if not 0 <= config.detector.nms_iou <= 1:
        raise ConfigError("detector.nms_iou must be between zero and one")
