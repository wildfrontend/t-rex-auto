"""OpenCV template and HSV range detectors loaded from an asset manifest."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import BoundingBox, Detection, Frame


class DetectorAssetError(ValueError):
    pass


class StartupLayoutGuard:
    """Wrap a startup-only layout detector with two liveness checks.

    These detectors derive confidence from live pixel ratios, so a real dialog
    jitters frame to frame while it animates. A value that repeats bit for bit
    is being computed from static screen furniture, and acting on it once cost
    ten minutes of tapping an empty coordinate.

    The second check is phase: the modals only appear while the app is starting
    up. One misfire arrived after forty-four confirmed hunts, which no amount
    of threshold tuning would have caught.
    """

    def __init__(
        self,
        detector: Any,
        *,
        max_identical_frames: int = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        self.detector = detector
        self.max_identical_frames = max(1, max_identical_frames)
        self.logger = logger
        self.target_type = getattr(detector, "target_type", None)
        self._signature: tuple[tuple[str, int, int, float], ...] | None = None
        self._identical_frames = 0
        self._startup_complete = False

    def on_hunt_completed(self) -> None:
        self._startup_complete = True

    def on_app_restart(self) -> None:
        self._startup_complete = False
        self._signature = None
        self._identical_frames = 0

    def detect(self, frame: Frame) -> list[Detection]:
        if self._startup_complete:
            return []
        results = self.detector.detect(frame)
        signature = tuple(
            (item.type, item.x, item.y, item.confidence) for item in results
        )
        if signature and signature == self._signature:
            self._identical_frames += 1
        else:
            self._identical_frames = 0
        self._signature = signature
        if self._identical_frames >= self.max_identical_frames:
            if self._identical_frames == self.max_identical_frames and self.logger:
                self.logger.warning(
                    "Detect | %s unchanged for %d frames; treating as static",
                    self.target_type or "startup layout",
                    self._identical_frames,
                )
            return []
        return results


@dataclass(frozen=True, slots=True)
class TemplateAsset:
    type: str
    path: Path
    image: np.ndarray
    threshold: float
    click_offset: tuple[int, int] | None = None
    scales: tuple[float, ...] = (1.0,)
    prepared_images: tuple[tuple[float, np.ndarray], ...] = ()
    # Fraction of the reference resolution this template is searched at.
    # matchTemplate costs scale with the searched pixel count, so halving both
    # axes is close to a 4x cut. Confidence survives it - INTER_AREA is a low
    # pass filter, so the noise that fights the match goes with the pixels -
    # but only for templates large enough to keep their shape. Anything that
    # would shrink below a legible size stays at 1.0.
    match_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class HsvRange:
    type: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    min_area: float
    max_area: float


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if candidate.bbox is None:
            kept.append(candidate)
            continue
        if all(
            existing.type != candidate.type
            or existing.bbox is None
            or _iou(existing.bbox, candidate.bbox) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


class OpenCvDetector:
    def __init__(self, manifest: Path, default_threshold: float = 0.85, nms_iou: float = 0.3):
        self.manifest = manifest
        self.default_threshold = default_threshold
        self.nms_iou = nms_iou
        self.templates: list[TemplateAsset] = []
        self.hsv_ranges: list[HsvRange] = []
        self.reference_size: tuple[int, int] | None = None
        self.reload()

    def reload(self) -> None:
        if not self.manifest.exists():
            raise DetectorAssetError(f"Detector manifest not found: {self.manifest}")
        try:
            payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DetectorAssetError(f"Invalid detector manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise DetectorAssetError("Detector manifest root must be an object")
        reference_raw = payload.get("reference_size")
        if reference_raw is not None:
            reference_size = tuple(int(value) for value in reference_raw)
            if len(reference_size) != 2 or min(reference_size) <= 0:
                raise DetectorAssetError("reference_size must be [width, height]")
            self.reference_size = reference_size  # type: ignore[assignment]
        else:
            self.reference_size = None

        templates: list[TemplateAsset] = []
        for raw in payload.get("templates", []):
            path = self.manifest.parent / raw["file"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise DetectorAssetError(f"Cannot read template image: {path}")
            scales = tuple(
                dict.fromkeys(float(value) for value in raw.get("scales", [1.0]))
            )
            if not scales or any(value <= 0 for value in scales):
                raise DetectorAssetError(f"Template scales must be positive: {path}")
            match_scale = float(raw.get("match_scale", 1.0))
            if not 0 < match_scale <= 1.0:
                raise DetectorAssetError(
                    f"Template match_scale must be within (0, 1]: {path}"
                )
            prepared_images: list[tuple[float, np.ndarray]] = []
            for scale in scales:
                # The prepared image carries both scales: `scale` is the size
                # the game draws this control at, `match_scale` the resolution
                # we search for it in.
                effective = scale * match_scale
                if effective == 1.0:
                    prepared_images.append((scale, image))
                    continue
                source_height, source_width = image.shape[:2]
                width = max(1, round(source_width * effective))
                height = max(1, round(source_height * effective))
                interpolation = cv2.INTER_AREA if effective < 1.0 else cv2.INTER_LINEAR
                prepared_images.append(
                    (
                        scale,
                        cv2.resize(
                            image,
                            (width, height),
                            interpolation=interpolation,
                        ),
                    )
                )
            templates.append(
                TemplateAsset(
                    type=str(raw["type"]),
                    path=path,
                    image=image,
                    threshold=float(raw.get("threshold", self.default_threshold)),
                    click_offset=(
                        tuple(int(value) for value in raw["click_offset"])  # type: ignore[arg-type]
                        if "click_offset" in raw
                        else None
                    ),
                    scales=scales,
                    prepared_images=tuple(prepared_images),
                    match_scale=match_scale,
                )
            )

        ranges: list[HsvRange] = []
        for raw in payload.get("hsv_ranges", []):
            ranges.append(
                HsvRange(
                    type=str(raw["type"]),
                    lower=tuple(int(value) for value in raw["lower"]),  # type: ignore[arg-type]
                    upper=tuple(int(value) for value in raw["upper"]),  # type: ignore[arg-type]
                    min_area=float(raw.get("min_area", 20)),
                    max_area=float(raw.get("max_area", float("inf"))),
                )
            )
        self.templates = templates
        self.hsv_ranges = ranges

    @property
    def asset_count(self) -> int:
        return len(self.templates) + len(self.hsv_ranges)

    def detect(self, frame: Frame) -> list[Detection]:
        return self._detect(frame)

    def detect_types(
        self,
        frame: Frame,
        target_types: set[str] | frozenset[str],
    ) -> list[Detection]:
        return self._detect(frame, frozenset(target_types))

    def _detect(
        self,
        frame: Frame,
        target_types: frozenset[str] | None = None,
    ) -> list[Detection]:
        working = frame.image
        scale_x = scale_y = 1.0
        if self.reference_size and (frame.width, frame.height) != self.reference_size:
            reference_width, reference_height = self.reference_size
            working = cv2.resize(
                frame.image,
                (reference_width, reference_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = frame.width / reference_width
            scale_y = frame.height / reference_height
        detections = self._detect_templates(working, target_types)
        detections.extend(self._detect_hsv(working, target_types))
        if scale_x != 1.0 or scale_y != 1.0:
            detections = [self._scale_detection(item, scale_x, scale_y) for item in detections]
        return non_max_suppression(detections, self.nms_iou)

    @staticmethod
    def _scale_detection(item: Detection, scale_x: float, scale_y: float) -> Detection:
        bbox = item.bbox
        scaled_bbox = (
            BoundingBox(
                x=round(bbox.x * scale_x),
                y=round(bbox.y * scale_y),
                width=max(1, round(bbox.width * scale_x)),
                height=max(1, round(bbox.height * scale_y)),
            )
            if bbox
            else None
        )
        return Detection(
            type=item.type,
            x=round(item.x * scale_x),
            y=round(item.y * scale_y),
            confidence=item.confidence,
            bbox=scaled_bbox,
            metadata={**item.metadata, "normalized_from": [scale_x, scale_y]},
        )

    def _detect_templates(
        self,
        image: np.ndarray,
        target_types: frozenset[str] | None = None,
    ) -> list[Detection]:
        results: list[Detection] = []
        # One reduced copy per distinct match_scale, built at most once per
        # scan and shared by every asset that searches at that resolution.
        # The resize itself is under a millisecond; the templates it saves
        # are hundreds.
        searched: dict[float, np.ndarray] = {1.0: image}
        for asset in self.templates:
            if target_types is not None and asset.type not in target_types:
                continue
            haystack = searched.get(asset.match_scale)
            if haystack is None:
                source_height, source_width = image.shape[:2]
                haystack = cv2.resize(
                    image,
                    (
                        max(1, round(source_width * asset.match_scale)),
                        max(1, round(source_height * asset.match_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                searched[asset.match_scale] = haystack
            # Everything below is computed in the reduced space and lifted
            # back to reference coordinates with this factor.
            back = 1.0 / asset.match_scale
            for scale, template in asset.prepared_images:
                height, width = template.shape[:2]
                if haystack.shape[0] < height or haystack.shape[1] < width:
                    continue
                matches = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(matches >= asset.threshold)
                if len(xs) > 2000:
                    scores = matches[ys, xs]
                    top = np.argpartition(scores, -2000)[-2000:]
                    xs, ys = xs[top], ys[top]
                for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
                    bbox = BoundingBox(
                        x=round(x * back),
                        y=round(y * back),
                        width=max(1, round(width * back)),
                        height=max(1, round(height * back)),
                    )
                    metadata = {
                        "detector": "template",
                        "asset": asset.path.name,
                        "template_scale": scale,
                    }
                    if asset.match_scale != 1.0:
                        metadata["match_scale"] = asset.match_scale
                    if asset.click_offset is None:
                        results.append(
                            Detection.from_bbox(
                                asset.type,
                                bbox,
                                float(matches[y, x]),
                                **metadata,
                            )
                        )
                    else:
                        click_x = bbox.x + round(asset.click_offset[0] * scale)
                        click_y = bbox.y + round(asset.click_offset[1] * scale)
                        metadata["anchor_bbox"] = {
                            "x": bbox.x,
                            "y": bbox.y,
                            "width": bbox.width,
                            "height": bbox.height,
                        }
                        results.append(
                            Detection(
                                type=asset.type,
                                x=click_x,
                                y=click_y,
                                confidence=float(matches[y, x]),
                                bbox=bbox,
                                metadata=metadata,
                            )
                        )
        return results

    def _detect_hsv(
        self,
        image: np.ndarray,
        target_types: frozenset[str] | None = None,
    ) -> list[Detection]:
        if not self.hsv_ranges:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        results: list[Detection] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for item in self.hsv_ranges:
            if target_types is not None and item.type not in target_types:
                continue
            mask = cv2.inRange(hsv, np.array(item.lower), np.array(item.upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < item.min_area or area > item.max_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                bbox = BoundingBox(x=x, y=y, width=width, height=height)
                fill_ratio = min(1.0, area / max(1, bbox.area))
                results.append(
                    Detection.from_bbox(
                        item.type,
                        bbox,
                        0.5 + fill_ratio * 0.49,
                        detector="hsv",
                        contour_area=area,
                    )
                )
        return results


class CompositeDetector:
    def __init__(
        self,
        *detectors: Any,
        reference_size: tuple[int, int] | None = None,
    ) -> None:
        self.detectors = detectors
        self.reference_size = reference_size

    def detect(self, frame: Frame) -> list[Detection]:
        return self._detect(frame)

    def detect_types(
        self,
        frame: Frame,
        target_types: set[str] | frozenset[str],
    ) -> list[Detection]:
        return self._detect(frame, frozenset(target_types))

    def on_hunt_completed(self) -> None:
        self._notify("on_hunt_completed")

    def on_app_restart(self) -> None:
        self._notify("on_app_restart")

    def _notify(self, hook: str) -> None:
        for detector in self.detectors:
            callback = getattr(detector, hook, None)
            if callable(callback):
                callback()

    def _detect(
        self,
        frame: Frame,
        target_types: frozenset[str] | None = None,
    ) -> list[Detection]:
        working_frame = frame
        scale_x = scale_y = 1.0
        if self.reference_size and (frame.width, frame.height) != self.reference_size:
            reference_width, reference_height = self.reference_size
            working_frame = Frame(
                cv2.resize(
                    frame.image,
                    self.reference_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
                captured_at=frame.captured_at,
                source=frame.source,
                sequence=frame.sequence,
            )
            scale_x = frame.width / reference_width
            scale_y = frame.height / reference_height

        results: list[Detection] = []
        for detector in self.detectors:
            if target_types is None:
                results.extend(detector.detect(working_frame))
                continue
            detect_types = getattr(detector, "detect_types", None)
            if callable(detect_types):
                results.extend(detect_types(working_frame, target_types))
                continue
            detector_type = getattr(detector, "target_type", None)
            if detector_type is None or detector_type in target_types:
                results.extend(detector.detect(working_frame))
        if scale_x != 1.0 or scale_y != 1.0:
            return [
                OpenCvDetector._scale_detection(item, scale_x, scale_y)
                for item in results
            ]
        return results


# Fraction of a glyph's bounding box that must be enclosed background before it
# can be a "0". Measured across font scales: 0 and 8 sit at 0.18-0.19, the next
# largest (6) only reaches 0.12.
_ZERO_MIN_HOLE_RATIO = 0.15
# Ink coverage of the central band. A "0" is hollow there (0.33-0.40); an "8"
# has its waist stroke crossing (0.83-1.00). Nothing else clears the hole test.
_ZERO_MAX_WAIST = 0.6


def _is_zero_glyph(mask: np.ndarray, glyph: tuple[int, int, int, int, int]) -> bool:
    """Return True when a connected component looks like the digit ``0``.

    Selecting on glyph width alone accepted every leading digit from 2 to 9,
    which turned "some team left" into "no team left" and cancelled thirteen
    consecutive hunts. Two shape measurements separate a zero from the rest:
    the enclosed area rules out everything but 0 and 8, and the hollow centre
    band then rules out the 8.
    """

    x, y, glyph_width, glyph_height, _ = glyph
    if glyph_width <= 0 or glyph_height <= 0:
        return False
    if not 0.4 <= glyph_width / glyph_height <= 1.0:
        return False

    window = mask[y : y + glyph_height, x : x + glyph_width]
    if window.size == 0:
        return False
    background = np.where(window > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background)
    outside = set(labels[0, :]) | set(labels[-1, :])
    outside |= set(labels[:, 0]) | set(labels[:, -1])
    # Anti-aliasing can pinch a small ring into two components, so judge the
    # holes by their combined area rather than by how many there are.
    hole_area = sum(
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if index not in outside and stats[index, cv2.CC_STAT_AREA] >= 3
    )
    if hole_area / (glyph_width * glyph_height) < _ZERO_MIN_HOLE_RATIO:
        return False

    band = window[
        int(glyph_height * 0.40) : int(glyph_height * 0.60),
        int(glyph_width * 0.35) : int(glyph_width * 0.65),
    ]
    if band.size == 0:
        return False
    return float((band > 0).mean()) < _ZERO_MAX_WAIST


class HuntTeamAvailabilityDetector:
    """Detect the fixed-layout ``0 / 11`` hunt-team exhaustion state."""

    def __init__(
        self,
        target_type: str = "no_available_dinosaurs",
        reference_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # The team-selection sheet is white and has a red close button at a
        # fixed location. Requiring both prevents map labels from looking like
        # the team counter.
        if float(image[850:990, 150:750].mean()) < 210:
            return []
        hsv = cv2.cvtColor(image[1360:1455, 580:675], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 100, 120]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([170, 100, 120]), np.array([179, 255, 255]))
        if cv2.countNonZero(red) < 500:
            return []

        x1, y1, x2, y2 = 380, 920, 530, 980
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        mask = np.where(gray < 100, 255, 0).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyphs = sorted(
            (
                (int(x), int(y), int(glyph_width), int(glyph_height), int(area))
                for x, y, glyph_width, glyph_height, area in stats[1:]
                if area >= 20 and glyph_height >= 15
            ),
            key=lambda item: item[0],
        )
        # "0 / 11" is four glyphs, "10 / 11" and "11 / 11" are five. Within the
        # four-glyph shapes only the leading digit separates "no team left"
        # from "some team left", and a width threshold alone accepts every one
        # of 2..9 - which cancelled thirteen consecutive hunts. Identify the
        # zero by its shape instead.
        if len(glyphs) != 4 or not _is_zero_glyph(mask, glyphs[0]):
            return []

        scale_x = frame.width / width
        scale_y = frame.height / height
        bbox = BoundingBox(
            x=round(x1 * scale_x),
            y=round(y1 * scale_y),
            width=max(1, round((x2 - x1) * scale_x)),
            height=max(1, round((y2 - y1) * scale_y)),
        )
        return [
            Detection(
                type=self.target_type,
                x=round(628 * scale_x),
                y=round(1409 * scale_y),
                confidence=0.99,
                bbox=bbox,
                metadata={"detector": "hunt_team_counter", "glyphs": len(glyphs)},
            )
        ]


class HuntCapacityDetector:
    """Detect the map egg nest's fixed ``10/10`` dispatched-team counter."""

    def __init__(
        self,
        target_type: str = "hunt_capacity_full",
        reference_size: tuple[int, int] = (900, 1600),
        anchor_template: str | Path | None = None,
        anchor_threshold: float = 0.65,
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size
        template_path = (
            Path(anchor_template)
            if anchor_template is not None
            else Path(__file__).resolve().parents[2]
            / "assets"
            / "templates"
            / "map-center-egg-anchor.png"
        )
        self.anchor_template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if self.anchor_template is None:
            raise DetectorAssetError(f"Unable to read egg nest template: {template_path}")
        self.anchor_threshold = anchor_threshold

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        matches = cv2.matchTemplate(
            image,
            self.anchor_template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, anchor_confidence, _, (anchor_x, anchor_y) = cv2.minMaxLoc(matches)
        if anchor_confidence < self.anchor_threshold:
            return []

        # The availability label is immediately below/right of the egg nest.
        # Its position follows the nest as the map pans, unlike the top-right
        # capacity label which can be obscured by the hunt dialog.
        x1 = max(0, anchor_x + 5)
        y1 = max(0, anchor_y + 45)
        x2 = min(width, anchor_x + 85)
        y2 = min(height, anchor_y + 95)
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        mask = np.where(gray > 210, 255, 0).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyphs = sorted(
            (
                (int(x), int(y), int(glyph_width), int(glyph_height), int(area))
                for x, y, glyph_width, glyph_height, area in stats[1:]
                if area >= 8 and glyph_height >= 8
            ),
            key=lambda item: item[0],
        )
        # "10/10" is the only five-glyph reading; anything with spare capacity
        # is four. Confirm the shape of each digit rather than its width - at
        # this size a "1" and a "0" differ by a single pixel, which never
        # matched and left this detector silent in every recorded session.
        if len(glyphs) != 5 or [
            _is_zero_glyph(mask, glyph) for glyph in glyphs
        ] != [False, True, False, False, True]:
            return []
        scale_x = frame.width / width
        scale_y = frame.height / height
        bbox = BoundingBox(
            x=round(x1 * scale_x),
            y=round(y1 * scale_y),
            width=max(1, round((x2 - x1) * scale_x)),
            height=max(1, round((y2 - y1) * scale_y)),
        )
        return [
            Detection(
                type=self.target_type,
                x=round(((x1 + x2) / 2) * scale_x),
                y=round(((y1 + y2) / 2) * scale_y),
                confidence=float(anchor_confidence),
                bbox=bbox,
                metadata={
                    "detector": "hunt_nest_counter",
                    "value": "10/10",
                    "anchor_confidence": float(anchor_confidence),
                },
            )
        ]


class TargetTooStrongDetector:
    """Detect the red hunt warning that says the selected target will win."""

    def __init__(
        self,
        target_type: str = "target_too_strong",
        reference_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        # The warning line occupies this fixed band in the hunt-team sheet.
        x1, y1, x2, y2 = 100, 660, 800, 755
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 140, 150]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([170, 140, 150]), np.array([179, 255, 255]))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        if cv2.countNonZero(red) < 80:
            return []
        # Require the team-sheet close button as context so unrelated red map
        # effects cannot trigger a five-minute cooldown.
        close_hsv = cv2.cvtColor(image[1360:1455, 580:675], cv2.COLOR_BGR2HSV)
        close_red = cv2.inRange(
            close_hsv,
            np.array([0, 100, 120]),
            np.array([12, 255, 255]),
        )
        close_red |= cv2.inRange(
            close_hsv,
            np.array([170, 100, 120]),
            np.array([179, 255, 255]),
        )
        if cv2.countNonZero(close_red) < 500:
            return []
        scale_x = frame.width / width
        scale_y = frame.height / height
        return [
            Detection(
                type=self.target_type,
                x=round(628 * scale_x),
                y=round(1409 * scale_y),
                confidence=0.99,
                bbox=BoundingBox(
                    x=round(x1 * scale_x),
                    y=round(y1 * scale_y),
                    width=max(1, round((x2 - x1) * scale_x)),
                    height=max(1, round((y2 - y1) * scale_y)),
                ),
                metadata={"detector": "red_hunt_warning"},
            )
        ]


class StartupGrowthResultDetector:
    """Detect the fixed offline-growth result modal shown after app launch."""

    def __init__(
        self,
        target_type: str = "startup_growth_result_back",
        reference_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # This modal is the only launch screen with a tall white card and a
        # shortcut button along its bottom edge. Older game builds showed two
        # buttons (auto battle + nest); newer builds keep only the centred
        # green "Nest management shortcut" button. Requiring the white card
        # plus a known button layout avoids tapping ordinary map screens.
        white_regions = np.concatenate(
            [
                image[190:275, 200:685].reshape(-1, 3),
                image[300:1120, 135:180].reshape(-1, 3),
                image[300:1120, 720:765].reshape(-1, 3),
            ]
        )
        white_ratio = float(np.mean(np.all(white_regions >= 200, axis=1)))
        if white_ratio < 0.7:
            return []

        cyan = image[1190:1340, 190:425].astype(np.int16)
        cyan_mask = (
            (cyan[:, :, 0] >= 150)
            & (cyan[:, :, 1] >= 120)
            & (cyan[:, :, 0] >= cyan[:, :, 2] + 35)
        )
        legacy_green = image[1190:1340, 475:710].astype(np.int16)
        legacy_green_mask = (
            (legacy_green[:, :, 1] >= 130)
            & (legacy_green[:, :, 1] >= legacy_green[:, :, 0] + 15)
            & (legacy_green[:, :, 1] >= legacy_green[:, :, 2] + 15)
        )
        cyan_ratio = float(np.mean(cyan_mask))
        legacy_green_ratio = float(np.mean(legacy_green_mask))

        # The new centred button is approximately x=330..570, y=1200..1340
        # in the 900x1600 reference layout. Sampling a slightly wider band
        # tolerates the rounded corners and the egg artwork overlapping its
        # top edge while still requiring a substantial green control.
        centred_green = image[1200:1340, 320:580].astype(np.int16)
        centred_green_mask = (
            (centred_green[:, :, 1] >= 120)
            & (centred_green[:, :, 1] >= centred_green[:, :, 0] + 12)
            & (centred_green[:, :, 1] >= centred_green[:, :, 2] + 12)
        )
        centred_green_ratio = float(np.mean(centred_green_mask))

        legacy_layout = cyan_ratio >= 0.2 and legacy_green_ratio >= 0.2
        centred_layout = centred_green_ratio >= 0.25
        if not legacy_layout and not centred_layout:
            return []

        if centred_layout and not legacy_layout:
            shortcut_layout = "centered_nest"
            shortcut_point = (450.0, 1270.0)
            growth_result_point = shortcut_point
            button_ratio = centred_green_ratio
        else:
            shortcut_layout = "legacy_dual"
            shortcut_point = (592.0, 1265.0)
            # Hunt mode uses the left legacy shortcut to dismiss the result;
            # full-hatch mode reads shortcut_point above and chooses the nest.
            growth_result_point = (307.0, 1265.0)
            button_ratio = min(cyan_ratio, legacy_green_ratio)

        scale_x = frame.width / width
        scale_y = frame.height / height
        return [
            Detection(
                type=self.target_type,
                x=round(growth_result_point[0] * scale_x),
                y=round(growth_result_point[1] * scale_y),
                confidence=min(0.99, 0.7 + button_ratio * 0.29),
                bbox=BoundingBox(
                    x=round(125 * scale_x),
                    y=round(180 * scale_y),
                    width=round(650 * scale_x),
                    height=round(1170 * scale_y),
                ),
                metadata={
                    "detector": "startup_growth_result_layout",
                    "white_ratio": white_ratio,
                    "cyan_ratio": cyan_ratio,
                    "green_ratio": legacy_green_ratio,
                    "centered_green_ratio": centred_green_ratio,
                    "shortcut_layout": shortcut_layout,
                    "shortcut_point": shortcut_point,
                },
            )
        ]


class StartupAutoBattleDialogDetector:
    """Detect the auto-battle shortcut modal opened from growth results."""

    def __init__(
        self,
        target_type: str = "startup_auto_battle_close",
        reference_size: tuple[int, int] = (900, 1600),
        min_cyan_ratio: float = 0.4,
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size
        self.min_cyan_ratio = min_cyan_ratio

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        white_regions = np.concatenate(
            [
                image[380:470, 220:680].reshape(-1, 3),
                image[470:1150, 185:235].reshape(-1, 3),
                image[470:1150, 665:715].reshape(-1, 3),
            ]
        )
        white_ratio = float(np.mean(np.all(white_regions >= 200, axis=1)))
        if white_ratio < 0.7:
            return []

        cyan = image[850:980, 345:550].astype(np.int16)
        cyan_mask = (
            (cyan[:, :, 0] >= 150)
            & (cyan[:, :, 1] >= 120)
            & (cyan[:, :, 0] >= cyan[:, :, 2] + 35)
        )
        cyan_ratio = float(np.mean(cyan_mask))
        # A real shortcut button fills roughly 0.42-0.46 of this band. The
        # misfire that deadlocked the bot sat at 0.279, so the gate goes
        # between them - and no higher, or genuine dialogs stop being closed.
        if cyan_ratio < self.min_cyan_ratio:
            return []

        scale_x = frame.width / width
        scale_y = frame.height / height
        return [
            Detection(
                type=self.target_type,
                x=round(50 * scale_x),
                y=round(800 * scale_y),
                confidence=min(0.99, 0.7 + cyan_ratio * 0.29),
                bbox=BoundingBox(
                    x=round(175 * scale_x),
                    y=round(355 * scale_y),
                    width=round(550 * scale_x),
                    height=round(875 * scale_y),
                ),
                metadata={
                    "detector": "startup_auto_battle_layout",
                    "white_ratio": white_ratio,
                    "cyan_ratio": cyan_ratio,
                },
            )
        ]


class HatchAutoplaceDialogDetector:
    """Detect auto-place confirmation variants by their stable modal layout.

    Both the best-attribute and level-order wording contain orange emphasis,
    followed by a cyan Yes button and a red No button. Requiring all three
    regions distinguishes this dialog from parent-selection confirmations.
    """

    def __init__(
        self,
        target_type: str = "hatch_autoplace_notice",
        reference_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size

    def detect(self, frame: Frame) -> list[Detection]:
        width, height = self.reference_size
        image = frame.image
        if (frame.width, frame.height) != self.reference_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv[600:820, 235:665], (0, 70, 120), (18, 255, 255))
        if cv2.countNonZero(orange) < 80:
            return []

        def color_ratio(
            region: np.ndarray,
            predicate: np.ndarray,
        ) -> float:
            return float(np.mean(predicate)) if region.size else 0.0

        yes = image[800:1010, 250:455].astype(np.int16)
        yes_ratio = color_ratio(
            yes,
            (yes[:, :, 0] >= 120)
            & (yes[:, :, 1] >= 120)
            & (yes[:, :, 0] >= yes[:, :, 2] + 25)
            & (yes[:, :, 1] >= yes[:, :, 2] + 25),
        )
        no = image[800:1010, 445:650].astype(np.int16)
        no_ratio = color_ratio(
            no,
            (no[:, :, 2] >= 150)
            & (no[:, :, 2] >= no[:, :, 0] + 35)
            & (no[:, :, 2] >= no[:, :, 1] + 20),
        )
        if yes_ratio < 0.08 or no_ratio < 0.08:
            return []

        scale_x = frame.width / width
        scale_y = frame.height / height
        bbox = BoundingBox(
            x=round(235 * scale_x),
            y=round(580 * scale_y),
            width=round(430 * scale_x),
            height=round(440 * scale_y),
        )
        return [
            Detection(
                type=self.target_type,
                x=round(450 * scale_x),
                y=round(720 * scale_y),
                confidence=min(0.99, 0.8 + min(yes_ratio, no_ratio)),
                bbox=bbox,
                metadata={
                    "detector": "hatch_autoplace_dialog_layout",
                    "yes_ratio": yes_ratio,
                    "no_ratio": no_ratio,
                },
            )
        ]
