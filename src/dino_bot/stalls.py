"""Evidence capture for screens the logs cannot describe.

Two failures share this shape. A blind stall is defined by absence: the planner
has no target and no stage deadline to point at, and the log records only what
the detector matched, which during such an episode is nothing useful. One run
held this state for 519 seconds and the events left behind cannot say what was
on screen - only that 17 dinosaur labels and no map control were matched, which
fits a background overlay, a zoomed-out view and a screen with no template
alike. An unreadable N/M capacity HUD is the same problem one crop smaller.

The frame itself settles both. These files stay beside the log for the full
local evidence set, while the diagnostic bundle carries a bounded recent
subset so a hand-export still contains the frame that explains the failure.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2

from .cull import CapacityRead
from .models import Detection, Frame, Image, Target, VerificationResult, utc_now


class DigitEvidenceReader(Protocol):
    """Small part of ``DigitReader`` needed for failure evidence."""

    def read(self, image: Image) -> str: ...


class ParentStatsSnapshot(Protocol):
    """Writes evidence when a parent stat read is not trustworthy."""

    def capture(
        self,
        frame: Frame,
        reader: DigitEvidenceReader,
        regions: Sequence[Sequence[tuple[float, float, float, float]]],
        *,
        stage: str,
        side: str,
        attempts: int,
    ) -> Path | None: ...


class EggPileSnapshot(Protocol):
    """Writes calibration evidence after a synthetic egg-pile tap fails."""

    def capture(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        *,
        target_x: int,
        target_y: int,
        measured_base: tuple[float, float] | None,
        proposed_point: tuple[int, int] | None,
        stage: str,
        failures: int,
        attempts: int,
    ) -> Path | None: ...


class HomeRecoverySnapshot(Protocol):
    """Writes evidence when recovery cannot prove it reached centered home."""

    def capture(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        *,
        reason: str,
        stage: str,
        rounds: int,
        forest_trips: int,
        measured_base: tuple[float, float] | None,
        expected_base: tuple[float, float],
    ) -> Path | None: ...


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


class DinosaurFailureSnapshotWriter(_SnapshotWriter):
    """Keep bounded before/after evidence for missed dinosaur taps."""

    prefix = "dinosaur-tap"

    def capture(
        self,
        before: Frame,
        after: Frame,
        target: Target,
        detections: Sequence[Detection],
        result: VerificationResult,
        *,
        attempt: int,
    ) -> Path | None:
        moment = self.clock()
        if self._throttled(moment):
            return None

        annotated_before = before.image.copy()
        annotated_after = after.image.copy()
        radius = max(10, round(20 * before.width / 900.0))
        thickness = max(2, round(4 * before.width / 900.0))
        for image in (annotated_before, annotated_after):
            cv2.circle(image, (target.x, target.y), radius, (0, 0, 255), thickness)
            cv2.line(
                image,
                (target.x - radius, target.y),
                (target.x + radius, target.y),
                (0, 0, 255),
                thickness,
            )
            cv2.line(
                image,
                (target.x, target.y - radius),
                (target.x, target.y + radius),
                (0, 0, 255),
                thickness,
            )

        anchor = target.detection.metadata.get("anchor_bbox")
        if isinstance(anchor, dict):
            try:
                x0 = int(anchor["x"])
                y0 = int(anchor["y"])
                x1 = x0 + int(anchor["width"])
                y1 = y0 + int(anchor["height"])
                cv2.rectangle(
                    annotated_before,
                    (x0, y0),
                    (x1, y1),
                    (255, 255, 0),
                    thickness,
                )
            except (KeyError, TypeError, ValueError):
                anchor = None

        evidence = Frame(
            annotated_before,
            captured_at=before.captured_at,
            source=before.source,
            sequence=before.sequence,
        )
        path = self._write(
            evidence,
            {
                "reason": "dinosaur_tap_failed",
                "attempt": attempt,
                "target": {
                    "x": target.x,
                    "y": target.y,
                    "confidence": round(float(target.confidence), 6),
                    "anchor_bbox": anchor,
                },
                "verification": {
                    "reason": result.reason,
                    "pixel_change": result.pixel_change,
                },
                "after_detections": [item.to_dict() for item in detections],
                "legend": {
                    "red": "tap point",
                    "cyan": "matched Lv anchor",
                },
            },
            extra_images={"after": annotated_after},
        )
        if path is None:
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Hunt calibration | dinosaur tap failed | target=(%d,%d) | saved %s",
            target.x,
            target.y,
            path.name,
        )
        return path


class EggPileSnapshotWriter(_SnapshotWriter):
    """Keep calibration evidence after an egg-pile tap fails.

    The egg pile is intentionally not template-matched because its artwork
    moves with the map.  A failed synthetic tap therefore needs the frame,
    measured base, and proposed point together; a text log alone cannot tell
    whether the point missed the pile or opened an unexpected foreground.
    """

    prefix = "egg-pile"

    def capture(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        *,
        target_x: int,
        target_y: int,
        measured_base: tuple[float, float] | None,
        proposed_point: tuple[int, int] | None,
        stage: str,
        failures: int,
        attempts: int,
    ) -> Path | None:
        """Write one calibration frame, or ``None`` when rate limited."""

        moment = self.clock()
        if self._throttled(moment):
            return None

        annotated = frame.image.copy()
        scale = frame.width / 900.0
        # Red = actual tap, green = proposed recalibrated point, blue = cyan
        # base measurement.  The original frame remains the primary evidence.
        cv2.circle(annotated, (target_x, target_y), max(8, round(18 * scale)), (0, 0, 255), 4)
        if proposed_point is not None:
            cv2.circle(
                annotated,
                proposed_point,
                max(8, round(18 * scale)),
                (0, 255, 0),
                4,
            )
        if measured_base is not None:
            cv2.circle(
                annotated,
                tuple(round(value) for value in measured_base),
                max(8, round(18 * scale)),
                (255, 0, 0),
                4,
            )

        x0 = max(0, round(180 * scale))
        x1 = min(frame.width, round(720 * scale))
        y0 = max(0, round(950 * scale))
        y1 = min(frame.height, round(1590 * scale))
        roi = annotated[y0:y1, x0:x1]
        annotated_frame = Frame(
            annotated,
            captured_at=frame.captured_at,
            source=frame.source,
            sequence=frame.sequence,
        )
        path = self._write(
            annotated_frame,
            {
                "reason": "egg_pile_tap_failed",
                "stage": stage,
                "failures": failures,
                "attempts": attempts,
                "target": {"x": target_x, "y": target_y},
                "measured_base": list(measured_base) if measured_base else None,
                "proposed_point": list(proposed_point) if proposed_point else None,
                "detections": [item.to_dict() for item in detections],
                "legend": {
                    "red": "actual tap",
                    "green": "proposed dynamic point",
                    "blue": "measured cyan base center",
                },
            },
            extra_images={"roi": roi} if roi.size else None,
        )
        if path is None:
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Hatch calibration | egg pile tap failed | target=(%d,%d) | saved %s",
            target_x,
            target_y,
            path.name,
        )
        return path


class HomeRecoverySnapshotWriter(_SnapshotWriter):
    """Keep the frame that a failed return-to-home could not describe.

    Recovery ends by reporting which controls it matched, and the failure it
    reports most often is that it matched none of the ones it needed. Whether
    the map is panned away from home, an unknown overlay is covering it, or a
    template stopped matching produces the same empty event either way. The
    frame separates them; nothing in the log can.
    """

    prefix = "home-recovery"

    def capture(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        *,
        reason: str,
        stage: str,
        rounds: int,
        forest_trips: int,
        measured_base: tuple[float, float] | None,
        expected_base: tuple[float, float],
    ) -> Path | None:
        """Write one recovery-failure frame, or ``None`` when rate limited."""

        moment = self.clock()
        if self._throttled(moment):
            return None

        annotated = frame.image.copy()
        scale = frame.width / 900.0
        radius = max(8, round(18 * scale))
        expected = (
            round(expected_base[0] * scale),
            round(expected_base[1] * scale),
        )
        # Green = where the centred egg-pile base belongs, blue = where it was
        # actually measured. No blue circle means the pile is off-screen or
        # unrecognisable, which is the difference the log cannot express.
        cv2.circle(annotated, expected, radius, (0, 255, 0), 4)
        if measured_base is not None:
            cv2.circle(
                annotated,
                tuple(round(value) for value in measured_base),
                radius,
                (255, 0, 0),
                4,
            )
        annotated_frame = Frame(
            annotated,
            captured_at=frame.captured_at,
            source=frame.source,
            sequence=frame.sequence,
        )
        path = self._write(
            annotated_frame,
            {
                "reason": "home_recovery_failed",
                "recovery_reason": reason,
                "stage": stage,
                "rounds": rounds,
                "forest_trips": forest_trips,
                "measured_base": list(measured_base) if measured_base else None,
                "expected_base": list(expected),
                "detections": [item.to_dict() for item in detections],
                "legend": {
                    "green": "expected centered egg-pile base",
                    "blue": "measured cyan base center",
                },
            },
        )
        if path is None:
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Hatch recovery | centered home unproven | stage=%s | saved %s",
            stage,
            path.name,
        )
        return path


# The HUD crop is 19px tall at the 900-wide reference size, which is legible to
# the glyph matcher and not to a person squinting at a PNG. The companion is
# nearest-neighbour enlarged so the pixels stay honest about what was matched.
HUD_ZOOM = 6


class CapacitySnapshotWriter(_SnapshotWriter):
    """Write the frame behind an unreadable capacity HUD.

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


PARENT_STAT_NAMES = ("hp", "attack", "speed")
PARENT_STATS_ZOOM = 8


class ParentStatsSnapshotWriter(_SnapshotWriter):
    """Keep the frame and every parent-stat crop behind a failed read.

    The parent reader deliberately fails closed, but historically discarded
    the only useful debugging information: which crop was empty or produced
    an uncertain glyph.  This writer stores both the original frame and
    enlarged raw/binarized crops so new evidence can be used to tune the
    regions and digit templates offline.
    """

    prefix = "parent-stats"

    @staticmethod
    def _crop(frame: Frame, region: tuple[float, float, float, float]) -> Image:
        scale = frame.width / 900.0
        x0, y0, x1, y1 = (round(value * scale) for value in region)
        x0 = max(0, min(frame.width, x0))
        x1 = max(x0, min(frame.width, x1))
        y0 = max(0, min(frame.height, y0))
        y1 = max(y0, min(frame.height, y1))
        return frame.image[y0:y1, x0:x1]

    @staticmethod
    def _zoom(image: Image) -> Image:
        if image.size == 0:
            return image
        return cv2.resize(
            image,
            None,
            fx=PARENT_STATS_ZOOM,
            fy=PARENT_STATS_ZOOM,
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _binary(image: Image) -> Image:
        if image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        return ParentStatsSnapshotWriter._zoom(binary)

    def capture(
        self,
        frame: Frame,
        reader: DigitEvidenceReader,
        regions: Sequence[Sequence[tuple[float, float, float, float]]],
        *,
        stage: str,
        side: str,
        attempts: int,
    ) -> Path | None:
        """Write one failure sample, or ``None`` when rate limited."""

        moment = self.clock()
        if self._throttled(moment):
            return None

        readings: list[dict[str, Any]] = []
        companions: dict[str, Image] = {}
        for parent_index, parent_regions in enumerate(regions):
            parent_name = "left" if parent_index == 0 else "right"
            for stat_name, region in zip(
                PARENT_STAT_NAMES,
                parent_regions,
                strict=False,
            ):
                crop = self._crop(frame, region)
                text = reader.read(crop) if crop.size else ""
                key = f"{parent_name}-{stat_name}"
                companions[key] = self._zoom(crop)
                companions[f"{key}-binary"] = self._binary(crop)
                readings.append(
                    {
                        "parent": parent_name,
                        "stat": stat_name,
                        "region_reference": list(region),
                        "raw_glyphs": text,
                        "value": int(text) if text.isdigit() else None,
                        "readable": bool(text) and text.isdigit(),
                    }
                )

        path = self._write(
            frame,
            {
                "reason": "parent_stats_unreadable",
                "stage": stage,
                "failed_side": side,
                "attempts": attempts,
                "reference_width": 900,
                "readings": readings,
            },
            extra_images=companions,
        )
        if path is None:
            return None

        self._last_written = moment
        self._prune()
        self.logger.warning(
            "Hatch stats | parent stats unreadable | stage=%s | side=%s"
            " | attempts=%d | saved %s",
            stage,
            side,
            attempts,
            path.name,
        )
        return path
