"""Evidence capture for stalls the planner cannot explain.

A blind stall is defined by absence: the planner has no target and no stage
deadline to point at, and the log records only what the detector matched, which
during such an episode is nothing useful. One run held this state for 519
seconds and the events left behind cannot say what was on screen - only that
17 dinosaur labels and no map control were matched, which fits a background
overlay, a zoomed-out view and a screen with no template alike.

The frame itself settles that. These files are written next to the log rather
than into a diagnostic bundle because the bundle is exported by hand, long
after the screen has moved on.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import cv2

from .models import Detection, Frame, utc_now


class StallSnapshotWriter:
    """Write the frame and detection census behind a blind stall."""

    def __init__(
        self,
        directory: Path,
        logger: logging.Logger,
        *,
        limit: int = 10,
        min_interval_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.directory = directory
        self.logger = logger
        self.limit = max(1, limit)
        # An episode re-reports every `blind_idle_seconds`, so without a floor
        # a three-minute stall would overwrite the whole retained set with nine
        # near-identical frames of itself.
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.clock = clock
        self.now = now
        self._last_written: float | None = None

    def capture(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        *,
        seconds: float,
        stage: str,
        escapes: int,
    ) -> Path | None:
        """Write one stall frame, or return None if it was rate limited."""

        moment = self.clock()
        if (
            self._last_written is not None
            and moment - self._last_written < self.min_interval_seconds
        ):
            return None

        stamp = self.now().astimezone().strftime("%Y%m%d-%H%M%S")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"stall-{stamp}.png"
            if not cv2.imwrite(str(path), frame.image):
                raise OSError(f"cv2 refused to write {path}")
            counts: dict[str, int] = {}
            for item in detections:
                counts[item.type] = counts.get(item.type, 0) + 1
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "captured_at": self.now().astimezone().isoformat(),
                        "blind_seconds": round(float(seconds), 1),
                        "stage": stage,
                        "escapes": escapes,
                        "frame": {"width": frame.width, "height": frame.height},
                        "detections": dict(
                            sorted(counts.items(), key=lambda item: -item[1])
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            # Evidence collection must never be the reason a run stops.
            self.logger.warning("Stall | snapshot failed: %s", exc)
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Stall | no actionable target for %.0fs | stage=%s | saved %s",
            seconds,
            stage or "unknown",
            path.name,
        )
        return path

    def _prune(self) -> None:
        try:
            images = sorted(
                self.directory.glob("stall-*.png"),
                key=lambda path: path.stat().st_mtime,
            )
        except OSError:
            return
        for path in images[: max(0, len(images) - self.limit)]:
            sidecar = path.with_suffix(".json")
            try:
                path.unlink()
                sidecar.unlink(missing_ok=True)
            except OSError:
                continue
