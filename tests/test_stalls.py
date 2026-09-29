from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import numpy as np

from dino_bot.config import StallConfig
from dino_bot.full_hatch import HOME_PILE_BASE
from dino_bot.models import Frame
from dino_bot.nest_readout import ATTACK_PARENT_REGIONS
from dino_bot.stalls import (
    EggPileSnapshotWriter,
    HomeRecoverySnapshotWriter,
    ParentStatsSnapshotWriter,
)


class UnreadableReader:
    def read(self, image: np.ndarray) -> str:
        return "?"


def test_stall_evidence_defaults_to_three_recent_groups() -> None:
    assert StallConfig().snapshot_limit == 3


def test_home_recovery_snapshot_prunes_repeated_evidence_to_its_limit(tmp_path) -> None:
    current = [0.0]
    started = datetime(2026, 8, 8, tzinfo=UTC)
    writer = HomeRecoverySnapshotWriter(
        tmp_path,
        logging.getLogger("test-home-recovery-prune"),
        limit=3,
        min_interval_seconds=0,
        clock=lambda: current[0],
        now=lambda: started + timedelta(seconds=current[0]),
    )
    frame = Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))

    for index in range(4):
        current[0] = index * 61.0
        assert writer.capture(
            frame,
            [],
            reason="repeated unknown screen",
            stage="recover_home_exhausted",
            rounds=2,
            forest_trips=1,
            measured_base=None,
            expected_base=HOME_PILE_BASE,
        ) is not None

    primary = [
        path
        for path in tmp_path.glob("home-recovery-*.png")
        if path.with_suffix(".json").exists()
    ]
    assert len(primary) == 3
    assert not (tmp_path / "home-recovery-20260808-080000.png").exists()


def test_parent_stats_snapshot_keeps_full_frame_and_each_crop(tmp_path) -> None:
    now = datetime(2026, 8, 8, 1, 23, 45, tzinfo=UTC)
    writer = ParentStatsSnapshotWriter(
        tmp_path,
        logging.getLogger("test-parent-stats"),
        clock=lambda: 10.0,
        now=lambda: now,
    )
    frame = Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))

    path = writer.capture(
        frame,
        UnreadableReader(),
        ATTACK_PARENT_REGIONS,
        stage="parent_stats_unreadable",
        side="left",
        attempts=4,
    )

    assert path == tmp_path / "parent-stats-20260808-092345.png"
    assert path.exists()
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["reason"] == "parent_stats_unreadable"
    assert payload["failed_side"] == "left"
    assert payload["attempts"] == 4
    assert len(payload["readings"]) == 6
    assert all(item["raw_glyphs"] == "?" for item in payload["readings"])
    assert (tmp_path / "parent-stats-20260808-092345-left-hp.png").exists()
    assert (
        tmp_path / "parent-stats-20260808-092345-right-speed-binary.png"
    ).exists()


def test_parent_stats_snapshot_is_rate_limited(tmp_path) -> None:
    current = [0.0]
    writer = ParentStatsSnapshotWriter(
        tmp_path,
        logging.getLogger("test-parent-stats"),
        clock=lambda: current[0],
        now=lambda: datetime(2026, 8, 8, tzinfo=UTC),
    )
    frame = Frame(np.zeros((1600, 900, 3), dtype=np.uint8))
    reader = UnreadableReader()

    assert writer.capture(
        frame,
        reader,
        ATTACK_PARENT_REGIONS,
        stage="parent_stats_unreadable",
        side="left",
        attempts=1,
    ) is not None
    assert writer.capture(
        frame,
        reader,
        ATTACK_PARENT_REGIONS,
        stage="parent_stats_unreadable",
        side="left",
        attempts=2,
    ) is None

    current[0] = 60.0
    assert writer.capture(
        frame,
        reader,
        ATTACK_PARENT_REGIONS,
        stage="parent_stats_unreadable",
        side="left",
        attempts=3,
    ) is not None


def test_egg_pile_snapshot_marks_points_and_keeps_detection_context(tmp_path) -> None:
    now = datetime(2026, 8, 8, 1, 23, 45, tzinfo=UTC)
    writer = EggPileSnapshotWriter(
        tmp_path,
        logging.getLogger("test-egg-pile"),
        clock=lambda: 10.0,
        now=lambda: now,
    )
    frame = Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))
    detections = []

    path = writer.capture(
        frame,
        detections,
        target_x=457,
        target_y=1279,
        measured_base=(450.0, 1454.0),
        proposed_point=(450, 1354),
        stage="full_hatch:home",
        failures=2,
        attempts=1,
    )

    assert path == tmp_path / "egg-pile-20260808-092345.png"
    assert path.exists()
    assert (tmp_path / "egg-pile-20260808-092345-roi.png").exists()
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["reason"] == "egg_pile_tap_failed"
    assert payload["target"] == {"x": 457, "y": 1279}
    assert payload["measured_base"] == [450.0, 1454.0]
    assert payload["proposed_point"] == [450, 1354]


def test_home_recovery_snapshot_distinguishes_panned_map_from_lost_pile(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 8, 1, 23, 45, tzinfo=UTC)
    writer = HomeRecoverySnapshotWriter(
        tmp_path,
        logging.getLogger("test-home-recovery"),
        clock=lambda: 10.0,
        now=lambda: now,
    )
    frame = Frame(np.full((1600, 900, 3), 255, dtype=np.uint8))

    path = writer.capture(
        frame,
        [],
        reason="hatch started from shifted cave view",
        stage="recover_home_exhausted",
        rounds=2,
        forest_trips=1,
        measured_base=(463.0, 1090.5),
        expected_base=HOME_PILE_BASE,
    )

    assert path == tmp_path / "home-recovery-20260808-092345.png"
    assert path.exists()
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["reason"] == "home_recovery_failed"
    assert payload["recovery_reason"] == "hatch started from shifted cave view"
    assert payload["stage"] == "recover_home_exhausted"
    assert payload["rounds"] == 2
    # Both numbers non-zero says the escape ladder ran and still failed, so
    # the answer is not "wait longer" - it is whatever this frame shows.
    assert payload["forest_trips"] == 1
    assert payload["measured_base"] == [463.0, 1090.5]
    assert payload["expected_base"] == [450, 1455]


def test_home_recovery_snapshot_reports_an_unmeasurable_pile_as_absent(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 8, 1, 23, 45, tzinfo=UTC)
    writer = HomeRecoverySnapshotWriter(
        tmp_path,
        logging.getLogger("test-home-recovery"),
        clock=lambda: 10.0,
        now=lambda: now,
    )

    path = writer.capture(
        Frame(np.zeros((1600, 900, 3), dtype=np.uint8)),
        [],
        reason="retry 2/2 after failed recovery",
        stage="recover_home_exhausted",
        rounds=2,
        forest_trips=1,
        measured_base=None,
        expected_base=HOME_PILE_BASE,
    )

    assert path is not None
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    # A null base is the load-bearing distinction: the pile is off screen or
    # unrecognisable, so no branch of recovery had anything to measure.
    assert payload["measured_base"] is None
