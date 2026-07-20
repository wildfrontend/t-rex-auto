"""Explicit development helpers for creating OpenCV template assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2


class AssetToolError(ValueError):
    pass


def create_template(
    manifest: Path,
    source: Path,
    roi: tuple[int, int, int, int],
    target_type: str,
    name: str,
    threshold: float,
    click_offset: tuple[int, int] | None = None,
) -> Path:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise AssetToolError(f"Cannot read source screenshot: {source}")
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise AssetToolError("ROI must be non-negative x/y with positive width/height")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise AssetToolError(f"ROI {roi} is outside image {image.shape[1]}x{image.shape[0]}")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    if not safe_name:
        raise AssetToolError("Template name must contain a letter or number")
    template_dir = manifest.parent / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    output = template_dir / f"{safe_name}.png"
    if not cv2.imwrite(str(output), image[y : y + height, x : x + width]):
        raise AssetToolError(f"Cannot write template: {output}")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    templates = payload.setdefault("templates", [])
    entry = {
        "type": target_type,
        "file": f"templates/{output.name}",
        "threshold": threshold,
    }
    if click_offset is not None:
        entry["click_offset"] = list(click_offset)
    templates[:] = [item for item in templates if item.get("file") != entry["file"]]
    templates.append(entry)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
