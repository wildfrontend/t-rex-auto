"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2

from .actions import AdbClient
from .application import create_engine
from .assets import AssetToolError, create_template
from .capture import AdbScreencapCapture, MssBlueStacksCapture
from .config import AppConfig, ConfigError, load_config
from .doctor import benchmark_capture, run_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dino-bot", description="Dino Mutant Bot")
    parser.add_argument("--config", default="config.json", help="path to config.json")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="start the bot loop")
    run.add_argument("--mode", choices=["runtime", "debug", "training"])
    run.add_argument("--max-actions", type=int)
    run.add_argument("--verbose", action="store_true")

    subcommands.add_parser("doctor", help="check configuration and runtime dependencies")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load(args.config)
    if args.command == "doctor":
        checks = run_checks(config)
        for check in checks:
            icon = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
            print(f"[{icon}] {check.name}: {check.detail}")
        return 1 if any(not item.ok and item.required for item in checks) else 0
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
        engine = create_engine(config, verbose=args.verbose)
        engine.run()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
