"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2

from .actions import AdbClient
from .application import create_engine
from .assets import AssetToolError, create_template
from .capture import AdbScreencapCapture, MssBlueStacksCapture
from .config import DEFAULT_SPEED_PROFILES, AppConfig, ConfigError, load_config
from .diagnostics import create_diagnostic_bundle, default_diagnostic_output
from .doctor import benchmark_capture, run_checks
from .status import build_runtime_status
from .status_server import LocalStatusServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dino-bot", description="Dino Mutant Bot")
    parser.add_argument("--config", default="config.json", help="path to config.json")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="start the bot loop")
    run.add_argument("--mode", choices=["runtime", "debug", "training"])
    run.add_argument("--max-actions", type=int)
    run.add_argument("--max-cycles", type=int)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--mail-after-hunts", type=int)
    run.add_argument("--speed", choices=sorted(DEFAULT_SPEED_PROFILES))
    run.add_argument("--click-delay-ms", type=int)
    run.add_argument("--dinosaur-delay-ms", type=int)
    run.add_argument("--hunt-button-delay-ms", type=int)
    run.add_argument("--hunt-confirm-delay-ms", type=int)
    run.add_argument("--idle-delay-ms", type=int)
    run.add_argument("--poll-interval-ms", type=int)
    run.add_argument(
        "--status-port",
        type=int,
        default=8765,
        help="read-only localhost status API port; use 0 to disable",
    )
    run.add_argument("--verbose", action="store_true")

    subcommands.add_parser("doctor", help="check configuration and runtime dependencies")

    status = subcommands.add_parser("status", help="show the latest Bot session status")
    status.add_argument("--json", action="store_true", help="print machine-readable JSON")
    status.add_argument("--actions", type=int, default=10, help="recent actions to include")

    diagnostics = subcommands.add_parser(
        "diagnostics",
        help="create a sanitized ZIP bundle for support or Codex analysis",
    )
    diagnostics.add_argument("--output", help="output ZIP path")
    diagnostics.add_argument(
        "--include-screenshot",
        action="store_true",
        help="include one current screenshot after explicit user consent",
    )
    diagnostics.add_argument(
        "--log-lines",
        type=int,
        default=500,
        help="maximum recent raw log lines to include",
    )

    benchmark = subcommands.add_parser("benchmark", help="measure capture throughput")
    benchmark.add_argument("--frames", type=int, default=100)
    benchmark.add_argument("--backend", choices=["mss", "adb"])

    snapshot = subcommands.add_parser("snapshot", help="explicitly save one development screenshot")
    snapshot.add_argument("--output", default="debug/snapshot.png")
    snapshot.add_argument("--backend", choices=["mss", "adb"])

    template = subcommands.add_parser("template", help="crop a detector template from a screenshot")
    template.add_argument("--input", required=True)
    template.add_argument("--roi", nargs=4, required=True, type=int, metavar=("X", "Y", "W", "H"))
    template.add_argument("--type", default="resource")
    template.add_argument("--name", required=True)
    template.add_argument("--threshold", type=float, default=0.85)
    template.add_argument("--click-offset", nargs=2, type=int, metavar=("DX", "DY"))
    return parser


def _load(path: str) -> AppConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


def _capture_once(config: AppConfig):
    if config.capture.backend == "adb":
        adb = AdbClient(config.adb)
        adb.ensure_ready()
        provider = AdbScreencapCapture(adb)
    else:
        provider = MssBlueStacksCapture(
            config.capture.window_titles,
            config.capture.process_names,
            config.capture.viewport,
            config.capture.auto_viewport,
            config.capture.chrome_insets,
        )
    try:
        return provider.capture()
    finally:
        provider.close()


def _create_diagnostics(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    config: AppConfig | None
    config_error: str | None = None
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        config = None
        config_error = str(exc)

    root = config.root if config is not None else config_path.parent
    output = Path(args.output) if args.output else default_diagnostic_output(root)
    checks = run_checks(config) if config is not None else []
    snapshot_png: bytes | None = None
    snapshot_error: str | None = None
    if args.include_screenshot:
        if config is None:
            snapshot_error = "Snapshot unavailable because the configuration could not be loaded."
        else:
            try:
                frame = _capture_once(config)
                encoded, buffer = cv2.imencode(".png", frame.image)
                if not encoded:
                    raise RuntimeError("OpenCV could not encode the screenshot")
                snapshot_png = buffer.tobytes()
            except Exception as exc:  # The failed capture is itself diagnostic evidence.
                snapshot_error = str(exc)

    bundle = create_diagnostic_bundle(
        config,
        output,
        config_path=config_path,
        config_error=config_error,
        checks=checks,
        snapshot_requested=args.include_screenshot,
        snapshot_png=snapshot_png,
        snapshot_error=snapshot_error,
        recent_log_lines=max(1, min(args.log_lines, 5000)),
    )
    print(f"Diagnostic bundle saved: {bundle}")
    if snapshot_error:
        print(f"Snapshot warning: {snapshot_error}", file=sys.stderr)
    return 0


def apply_run_timing(
    config: AppConfig,
    *,
    speed: str | None = None,
    click_delay_ms: int | None = None,
    dinosaur_delay_ms: int | None = None,
    hunt_button_delay_ms: int | None = None,
    hunt_confirm_delay_ms: int | None = None,
    idle_delay_ms: int | None = None,
    poll_interval_ms: int | None = None,
) -> AppConfig:
    """Apply a speed preset, then any explicit terminal overrides."""

    profile = config.speed_profiles.get(speed or "", {})
    click_delay = profile.get("click_delay_ms", config.click_delay)
    idle_delay = profile.get("idle_delay_ms", config.idle_delay)
    poll_interval = profile.get(
        "poll_interval_ms",
        config.transition_poll_interval,
    )
    post_action_delays = dict(config.post_action_delays)
    profile_targets = {
        "dinosaur": profile.get("dinosaur_delay_ms"),
        "hunt_button": profile.get("hunt_button_delay_ms"),
        "hunt_confirm_button": profile.get("hunt_confirm_delay_ms"),
    }
    post_action_delays.update(
        {
            target_type: delay
            for target_type, delay in profile_targets.items()
            if delay is not None
        }
    )

    if click_delay_ms is not None:
        click_delay = max(0, click_delay_ms)
    if idle_delay_ms is not None:
        idle_delay = max(0, idle_delay_ms)
    if poll_interval_ms is not None:
        poll_interval = max(1, poll_interval_ms)
    explicit_targets = {
        "dinosaur": dinosaur_delay_ms,
        "hunt_button": hunt_button_delay_ms,
        "hunt_confirm_button": hunt_confirm_delay_ms,
    }
    post_action_delays.update(
        {
            target_type: max(0, delay)
            for target_type, delay in explicit_targets.items()
            if delay is not None
        }
    )
    return replace(
        config,
        click_delay=click_delay,
        idle_delay=idle_delay,
        transition_poll_interval=poll_interval,
        post_action_delays=post_action_delays,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnostics":
        return _create_diagnostics(args)
    config = _load(args.config)
    if args.command == "doctor":
        checks = run_checks(config)
        for check in checks:
            icon = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
            print(f"[{icon}] {check.name}: {check.detail}")
        return 1 if any(not item.ok and item.required for item in checks) else 0
    if args.command == "status":
        status = build_runtime_status(config.logs_dir, max(0, args.actions))
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(
                f"Bot: {'running' if status['running'] else 'stopped'}"
                f" | stage={status['current_stage']}"
            )
            print(
                f"Hunts: {status['successful_hunts']}"
                f" | mailbox cycles: {status['mailbox_cycles']}"
                f" | actions: {status['total_actions']}"
                f" | verification failures: {status['verification_failures']}"
            )
            print(
                f"Black screens: {status['black_screen_detections']}"
                f" | persisted: {status['black_screen_persisted']}"
                f" | game restarts: {status['game_restarts']}"
            )
        return 0
    if args.command == "benchmark":
        if args.backend:
            config = replace(config, capture=replace(config.capture, backend=args.backend))
        fps, size = benchmark_capture(config, max(1, args.frames))
        status = "PASS" if fps >= config.capture_fps else "FAIL"
        print(
            f"[{status}] {fps:.2f} FPS | {size[0]}x{size[1]} "
            f"| target {config.capture_fps:.2f} FPS"
        )
        return 0 if status == "PASS" else 1
    if args.command == "snapshot":
        if args.backend:
            config = replace(config, capture=replace(config.capture, backend=args.backend))
        frame = _capture_once(config)
        output = Path(args.output)
        if not output.is_absolute():
            output = config.root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame.image):
            raise SystemExit(f"Unable to write {output}")
        print(f"Saved {frame.width}x{frame.height} snapshot to {output}")
        return 0
    if args.command == "template":
        source = Path(args.input)
        if not source.is_absolute():
            source = config.root / source
        try:
            output = create_template(
                config.detector.manifest,
                source,
                tuple(args.roi),
                args.type,
                args.name,
                args.threshold,
                tuple(args.click_offset) if args.click_offset else None,
            )
        except AssetToolError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Template added: {output}")
        return 0
    if args.command == "run":
        if args.mode:
            config = replace(config, mode=args.mode)
        if args.max_actions is not None:
            config = replace(config, max_actions=max(0, args.max_actions))
        if args.max_cycles is not None:
            config = replace(
                config,
                workflow=replace(
                    config.workflow,
                    max_cycles=max(0, args.max_cycles),
                ),
            )
        if args.batch_size is not None:
            config = replace(
                config,
                planner=replace(
                    config.planner,
                    recenter_every=max(1, args.batch_size),
                ),
            )
        if args.mail_after_hunts is not None:
            config = replace(
                config,
                planner=replace(
                    config.planner,
                    mail_after_hunts=max(1, args.mail_after_hunts),
                ),
            )
        config = apply_run_timing(
            config,
            speed=args.speed,
            click_delay_ms=args.click_delay_ms,
            dinosaur_delay_ms=args.dinosaur_delay_ms,
            hunt_button_delay_ms=args.hunt_button_delay_ms,
            hunt_confirm_delay_ms=args.hunt_confirm_delay_ms,
            idle_delay_ms=args.idle_delay_ms,
            poll_interval_ms=args.poll_interval_ms,
        )
        engine = create_engine(config, verbose=args.verbose)
        status_server = None
        if args.status_port > 0:
            status_server = LocalStatusServer(
                config.logs_dir,
                args.status_port,
                control_handlers={"stop": engine.stop},
            )
            try:
                status_server.start()
                engine.context.logger.info(
                    "Status API | %s/status | localhost with allowlisted control",
                    status_server.url,
                )
            except OSError as exc:
                engine.context.logger.warning("Status API | unavailable | %s", exc)
                status_server = None
        try:
            engine.run()
        finally:
            if status_server is not None:
                status_server.stop()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
