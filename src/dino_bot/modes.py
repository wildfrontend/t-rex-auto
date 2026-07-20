"""Runtime, debug, and bounded training-data observers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .models import ActionRecord, Frame


class RuntimeMode:
    def on_frame(self, frame: Frame) -> None:
        pass

    def on_action_complete(self, record: ActionRecord, before: Frame, after: Frame) -> None:
        pass

    def close(self) -> None:
        pass


class DebugMode(RuntimeMode):
    def __init__(self, directory: Path, save_images: bool = True):
        self.directory = directory
        self.save_images = save_images
        self._counter = 0

    def on_action_complete(self, record: ActionRecord, before: Frame, after: Frame) -> None:
        self._counter += 1
        stamp = record.timestamp.astimezone().strftime("%Y%m%d_%H%M%S_%f")
        event_dir = self.directory / f"{stamp}_{self._counter:04d}"
        event_dir.mkdir(parents=True, exist_ok=False)
        if self.save_images:
            if not cv2.imwrite(str(event_dir / "Before.png"), before.image):
                raise OSError("Failed to write debug Before.png")
            if not cv2.imwrite(str(event_dir / "After.png"), after.image):
                raise OSError("Failed to write debug After.png")
        payload = {
            "time": record.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "action": record.action.kind.value,
            "x": record.action.x,
            "y": record.action.y,
            "result": "success" if record.result.success else "failed",
            "reason": record.result.reason,
            "attempt": record.attempt,
            "target": record.target.type if record.target else None,
        }
        (event_dir / "debug.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class TrainingMode(RuntimeMode):
    def __init__(self, directory: Path, fps: float = 2.0, max_images: int = 500):
        self.directory = directory
        self.interval = 1.0 / fps
        self.max_images = max_images
        self._last_saved = 0.0
        self._counter = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = self._images()
        if existing:
            try:
                self._counter = max(int(path.stem) for path in existing if path.stem.isdigit())
            except ValueError:
                self._counter = len(existing)
        self._prune()

    def _images(self) -> list[Path]:
        return sorted(self.directory.glob("*.png"), key=lambda path: path.stat().st_mtime)

    def _prune(self) -> None:
        images = self._images()
        for path in images[: max(0, len(images) - self.max_images)]:
            path.unlink()

    def on_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        if now - self._last_saved < self.interval:
            return
        self._counter += 1
        path = self.directory / f"{self._counter:06d}.png"
        if not cv2.imwrite(str(path), frame.image):
            raise OSError(f"Failed to write training image: {path}")
        self._last_saved = now
        self._prune()


def create_mode(
    mode: str,
    *,
    debug_dir: Path,
    training_dir: Path,
    save_debug_image: bool,
    training_fps: float,
    training_max_images: int,
) -> RuntimeMode:
    if mode == "debug":
        return DebugMode(debug_dir, save_images=save_debug_image)
    if mode == "training":
        return TrainingMode(training_dir, fps=training_fps, max_images=training_max_images)
    return RuntimeMode()
