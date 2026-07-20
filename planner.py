"""Simple ``planner.choose(frame, detections)`` facade."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dino_bot.config import load_config
from dino_bot.interfaces import Planner
from dino_bot.models import Detection, Frame, Target
from dino_bot.planning import HuntPlanner, TargetPlanner

_planner: Planner | None = None


def configure(planner: Planner) -> None:
    global _planner
    _planner = planner


def _default_planner() -> Planner:
    config = load_config(Path(__file__).with_name("config.json"))
    return HuntPlanner(
        config.planner.target_types,
        config.planner.strategy,
        blocking_types=config.planner.blocking_types,
        deduplicate_types=config.planner.deduplicate_types,
        dedup_radius=config.planner.dedup_radius,
        history_file=config.planner.history_file,
        history_limit=config.planner.history_limit,
        recenter_every=config.planner.recenter_every,
        own_path_radius=config.planner.own_path_radius,
        mail_after_hunts=config.planner.mail_after_hunts,
        capacity_wait_seconds=config.planner.capacity_wait_seconds,
        ring_width=config.planner.ring_width,
        own_path_angle_degrees=config.planner.own_path_angle_degrees,
    )


def choose(
    frame: Frame | np.ndarray,
    detections: Sequence[Detection],
) -> Target | None:
    global _planner
    if _planner is None:
        _planner = _default_planner()
    normalized = frame if isinstance(frame, Frame) else Frame(frame)
    return _planner.choose(normalized, detections)

__all__ = ["HuntPlanner", "TargetPlanner", "choose", "configure"]
