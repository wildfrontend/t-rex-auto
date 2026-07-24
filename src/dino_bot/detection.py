"""OpenCV template and HSV range detectors loaded from an asset manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import BoundingBox, Detection, Frame


class DetectorAssetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemplateAsset:
    type: str
    path: Path
    image: np.ndarray
    threshold: float
    click_offset: tuple[int, int] | None = None
    scales: tuple[float, ...] = (1.0,)
    prepared_images: tuple[tuple[float, np.ndarray], ...] = ()


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
            prepared_images: list[tuple[float, np.ndarray]] = []
            for scale in scales:
                if scale == 1.0:
                    prepared_images.append((scale, image))
                    continue
                source_height, source_width = image.shape[:2]
                width = max(1, round(source_width * scale))
                height = max(1, round(source_height * scale))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
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
        for asset in self.templates:
            if target_types is not None and asset.type not in target_types:
                continue
            for scale, template in asset.prepared_images:
                height, width = template.shape[:2]
                if image.shape[0] < height or image.shape[1] < width:
                    continue
                matches = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(matches >= asset.threshold)
                if len(xs) > 2000:
                    scores = matches[ys, xs]
                    top = np.argpartition(scores, -2000)[-2000:]
                    xs, ys = xs[top], ys[top]
                for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
                    bbox = BoundingBox(x=x, y=y, width=width, height=height)
                    metadata = {
                        "detector": "template",
                        "asset": asset.path.name,
                        "template_scale": scale,
                    }
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
                        click_x = x + round(asset.click_offset[0] * scale)
                        click_y = y + round(asset.click_offset[1] * scale)
                        metadata["anchor_bbox"] = {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
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
        # 0 / 11 has four glyphs and its first glyph is wider than a "1".
        # 11 / 11 has five glyphs, so it is intentionally rejected.
        if len(glyphs) != 4 or glyphs[0][2] < 10:
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
        # 10/10 has five glyphs. Confirm both narrow "1" glyphs and both wide
        # zeroes so the available state (0/10) is intentionally accepted.
        if (
            len(glyphs) != 5
            or glyphs[0][2] >= 7
            or glyphs[1][2] < 7
            or glyphs[3][2] >= 7
            or glyphs[4][2] < 7
        ):
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

        # This modal is the only launch screen with a tall white card and two
        # large cyan/green shortcut buttons along its bottom edge. Requiring
        # all three features avoids sending a tap on ordinary map screens.
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
        green = image[1190:1340, 475:710].astype(np.int16)
        green_mask = (
            (green[:, :, 1] >= 130)
            & (green[:, :, 1] >= green[:, :, 0] + 15)
            & (green[:, :, 1] >= green[:, :, 2] + 15)
        )
        cyan_ratio = float(np.mean(cyan_mask))
        green_ratio = float(np.mean(green_mask))
        if cyan_ratio < 0.2 or green_ratio < 0.2:
            return []

        scale_x = frame.width / width
        scale_y = frame.height / height
        return [
            Detection(
                type=self.target_type,
                x=round(307 * scale_x),
                y=round(1265 * scale_y),
                confidence=min(0.99, 0.7 + min(cyan_ratio, green_ratio) * 0.29),
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
                    "green_ratio": green_ratio,
                },
            )
        ]


class StartupAutoBattleDialogDetector:
    """Detect the auto-battle shortcut modal opened from growth results."""

    def __init__(
        self,
        target_type: str = "startup_auto_battle_close",
        reference_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.target_type = target_type
        self.reference_size = reference_size

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
        if cyan_ratio < 0.2:
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
