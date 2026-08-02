from __future__ import annotations

from pathlib import Path

from dino_bot.metrics import MetricsStore


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.strip() + "\n")


def test_metrics_persist_verified_hunts_hatches_and_records(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    log = logs / "20260802.log"
    append_log(
        log,
        """
18:00:00 | INFO | Feature | hatch-hunt | full hatch + hunt
18:00:01 | INFO | Bot started | Sense -> Think -> Act
18:00:02 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=0.9
18:00:03 | INFO | Verify | Success | next UI detected: map_exit_nest_button
18:00:04 | INFO | Planning | hatch_claim_button at (330,1242) confidence=1.0
18:00:05 | INFO | Verify | Success | next UI detected: hatch_button
18:00:06 | INFO | Hatch 攻擊特化 | side=left | parent=40/290/1 | pair=40/290/1,40/290/1
18:00:07 | INFO | Hatch 攻擊特化 | side=left | candidates=[] | decision=select row 1 (30/291/1)
18:00:08 | INFO | Planning | hatch_confirm_yes at (366,828) confidence=1.0
18:00:09 | INFO | Verify | Success | next UI detected: hatch_nest_title
18:00:10 | INFO | Hatch auto-place | tag=頂尖 | sort=最佳屬性組合 | completed with confirmation
18:00:11 | INFO | Hatch cave | capacity=301/350 | threshold=320 | cull=False
18:00:12 | WARNING | Verify | Failed | expected next UI not detected
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    first = store.snapshot()
    second = store.snapshot()

    assert first["counters"]["hunt"]["total"] == 1
    assert first["counters"]["hatch"]["total"] == 1
    assert first["counters"]["replacement"]["total"] == 1
    assert first["counters"]["autoplace_top"]["total"] == 1
    assert first["counters"]["verification_failure"]["total"] == 1
    assert first["records"]["hp"]["value"] == 40
    assert first["records"]["attack"]["value"] == 291
    assert first["records"]["speed"]["value"] == 1
    assert second["counters"] == first["counters"], "refresh must be idempotent"


def test_metrics_does_not_count_cave_claim_as_hatched_dinosaur(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    append_log(
        logs / "20260802.log",
        """
19:00:00 | INFO | Feature | hatch-hunt | full hatch + hunt
19:00:01 | INFO | Bot started | Sense -> Think -> Act
19:00:02 | INFO | Planning | hatch_cave_continuous_button at (560,1304) confidence=1.0
19:00:03 | INFO | Verify | Success | next UI detected: hatch_claim_button
19:00:04 | INFO | Planning | hatch_claim_button at (450,1170) confidence=1.0
19:00:05 | INFO | Verify | Success | next UI detected: hatch_home_anchor
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    result = store.snapshot()

    assert result["counters"]["hatch"]["total"] == 0


def test_metrics_incrementally_ingests_appended_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    log = logs / "20260802.log"
    append_log(
        log,
        """
20:00:00 | INFO | Bot started | Sense -> Think -> Act
20:00:01 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
20:00:02 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)
    assert store.snapshot()["counters"]["hunt"]["total"] == 1

    append_log(
        log,
        """
20:00:03 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
20:00:04 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )

    assert store.snapshot()["counters"]["hunt"]["total"] == 2
