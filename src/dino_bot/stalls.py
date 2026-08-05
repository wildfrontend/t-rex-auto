"""Evidence capture for screens the logs cannot describe.

Two failures share this shape. A blind stall is defined by absence: the planner
has no target and no stage deadline to point at, and the log records only what
the detector matched, which during such an episode is nothing useful. One run
held this state for 519 seconds and the events left behind cannot say what was
on screen - only that 17 dinosaur labels and no map control were matched, which
fits a background overlay, a zoomed-out view and a screen with no template
alike. An unreadable N/350 capacity HUD is the same problem one crop smaller.

The frame itself settles both. These files are written next to the log rather
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
from typing import Any

import cv2

from .cull import CapacityRead
from .models import Detection, Frame, utc_now


class _SnapshotWriter:
    """Rate-limited, size-capped PNG + JSON evidence under one filename stem."""

    prefix = "snapshot"

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

    def _throttled(self, moment: float) -> bool:
        return (
            self._last_written is not None
            and moment - self._last_written < self.min_interval_seconds
        )

    def _write(
        self,
        frame: Frame,
        payload: dict[str, Any],
        *,
        extra_images: dict[str, Any] | None = None,
    ) -> Path | None:
        """Write one frame plus its sidecar, or None when the write failed."""

        stamp = self.now().astimezone().strftime("%Y%m%d-%H%M%S")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{self.prefix}-{stamp}.png"
            if not cv2.imwrite(str(path), frame.image):
                raise OSError(f"cv2 refused to write {path}")
            for suffix, image in (extra_images or {}).items():
                companion = path.with_name(f"{path.stem}-{suffix}.png")
                if not cv2.imwrite(str(companion), image):
                    raise OSError(f"cv2 refused to write {companion}")
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "captured_at": self.now().astimezone().isoformat(),
                        "frame": {"width": frame.width, "height": frame.height},
                        **payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            # Evidence collection must never be the reason a run stops.
            self.logger.warning("%s | snapshot failed: %s", self.prefix, exc)
            return None
        return path

    def _prune(self) -> None:
        try:
            images = sorted(
                self.directory.glob(f"{self.prefix}-*.png"),
                key=lambda path: path.stat().st_mtime,
            )
        except OSError:
            return
        # Companions share the stem, so filter them out before counting or a
        # retained set of N frames would be pruned down to N/2 real episodes.
        images = [path for path in images if path.with_suffix(".json").exists()]
        for path in images[: max(0, len(images) - self.limit)]:
            try:
                for companion in self.directory.glob(f"{path.stem}-*.png"):
                    companion.unlink(missing_ok=True)
                path.unlink()
                path.with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                continue


class StallSnapshotWriter(_SnapshotWriter):
    """Write the frame and detection census behind a blind stall."""

    prefix = "stall"

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
        if self._throttled(moment):
            return None
        counts: dict[str, int] = {}
        for item in detections:
            counts[item.type] = counts.get(item.type, 0) + 1
        path = self._write(
            frame,
            {
                "blind_seconds": round(float(seconds), 1),
                "stage": stage,
                "escapes": escapes,
                "detections": dict(sorted(counts.items(), key=lambda item: -item[1])),
            },
        )
        if path is None:
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


# The HUD crop is 19px tall at the 900-wide reference size, which is legible to
# the glyph matcher and not to a person squinting at a PNG. The companion is
# nearest-neighbour enlarged so the pixels stay honest about what was matched.
HUD_ZOOM = 6


class CapacitySnapshotWriter(_SnapshotWriter):
    """Write the frame behind an unreadable N/350 capacity HUD.

    The event stream cannot distinguish a HUD that is absent, obscured, or
    present but too small for the glyph templates: all three log the same
    "capacity unreadable". Retrying is only the right fix for the first, so
    the crop is saved alongside the frame and the glyphs that were matched.
    """

    prefix = "capacity"

    def capture(
        self,
        frame: Frame,
        read: CapacityRead,
        *,
        stage: str,
        attempts: int,
    ) -> Path | None:
        """Write one unreadable-capacity frame, or None if rate limited."""

        moment = self.clock()
        if self._throttled(moment):
            return None

        x0, y0, x1, y1 = read.region
        crop = frame.image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        companions = {}
        if crop.size:
            companions["hud"] = cv2.resize(
                crop,
                None,
                fx=HUD_ZOOM,
                fy=HUD_ZOOM,
                interpolation=cv2.INTER_NEAREST,
            )
        path = self._write(
            frame,
            {
                "reason": read.reason,
                "glyphs": read.text,
                "fraction": list(read.fraction) if read.fraction else None,
                "region": list(read.region),
                "stage": stage,
                "attempts": attempts,
            },
            extra_images=companions,
        )
        if path is None:
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Hatch cave | capacity unreadable | reason=%s | glyphs=%r | saved %s",
            read.reason,
            read.text,
            path.name,
        )
        return path
