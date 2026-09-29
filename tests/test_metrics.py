from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from dino_bot.metrics import MetricsStore


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.strip() + "\n")


def test_metrics_persist_verified_hunts_hatches_and_records(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    today = datetime.now().astimezone().date()
    log = logs / f"{today:%Y%m%d}.log"
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
18:00:07 | INFO | Hatch HP特化 | side=left | parent=2340/143/150 | pair=2340/143/150,10/1/1
18:00:08 | INFO | Planning | hatch_confirm_yes at (366,828) confidence=1.0
18:00:09 | INFO | Verify | Success | next UI detected: hatch_nest_title
18:00:10 | INFO | Hatch auto-place | tag=頂尖 | sort=最佳屬性組合 | completed with confirmation
18:00:10 | INFO | Hatch auto-place | tag=攻擊特化 | sort=攻擊力 | completed with confirmation
18:00:10 | INFO | Hatch auto-place | tag=HP特化 | sort=HP | completed with confirmation
        18:00:11 | INFO | Hatch cave | capacity=301/350 | threshold=330 | cull=False
18:00:12 | WARNING | Verify | Failed | expected next UI not detected
""",
    )
    append_log(
        log,
        "18:00:13 | INFO | Hatch cave | cull completed | before=321/350"
        " | selected=40 | expected_after=281 | result=claim_verified",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    first = store.snapshot()
    second = store.snapshot()

    assert first["counters"]["hunt"]["total"] == 1
    assert first["counters"]["hatch"]["total"] == 1
    assert first["counters"]["replacement"]["total"] == 1
    assert first["counters"]["autoplace_top"]["total"] == 1
    assert first["counters"]["autoplace_attack"]["total"] == 1
    assert first["counters"]["autoplace_hp"]["total"] == 1
    assert first["counters"]["cull_removed"]["total"] == 40
    assert first["counters"]["verification_failure"]["total"] == 1
    assert first["records"]["hp"]["value"] == 2340
    assert first["records"]["attack"]["value"] == 291
    assert first["records"]["speed"]["value"] == 150
    assert second["counters"] == first["counters"], "refresh must be idempotent"


def test_metrics_keeps_secondary_hp_attack_out_of_attack_record(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    today = datetime.now().astimezone().date()
    append_log(
        logs / f"{today:%Y%m%d}.log",
        """
17:23:33 | INFO | Hatch HP特化 | side=left | parent=1410/737/134 | pair=1410/737/134,10/1/1
18:10:17 | INFO | Hatch 攻擊特化 | side=left | parent=1320/158/134 | pair=1320/158/134,10/1/1
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    result = store.snapshot()

    assert result["records"]["hp"]["value"] == 1410
    assert result["records"]["attack"]["value"] == 158
    assert result["records"]["speed"]["value"] == 134
    with closing(sqlite3.connect(store.database)) as connection, connection:
        raw_attacks = [
            json.loads(row[0])["attack"]
            for row in connection.execute(
                "SELECT payload_json FROM metric_events WHERE kind = 'stat_observation'"
            )
        ]
    assert 737 in raw_attacks, "raw evidence remains available for diagnosis"


def test_metrics_keeps_parent_outlier_out_of_record(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    today = datetime.now().astimezone().date()
    # 真實案例:同一隻恐龍讀到攻擊 168,單次 parent 整位數誤讀成 788。每輪的
    # parent 行都早於同輪 candidates,判定不能依賴清單先到。
    append_log(
        logs / f"{today:%Y%m%d}.log",
        """
10:33:16 | INFO | Hatch 攻擊特化 | side=left | parent=1500/788/147 | pair=1500/788/147,1320/168/134
10:33:34 | INFO | Hatch 攻擊特化 | side=right | parent=1320/168/134 | pair=1500/788/147,1320/168/134
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    result = store.snapshot()

    assert result["records"]["attack"]["value"] == 168
    with closing(sqlite3.connect(store.database)) as connection, connection:
        raw_attacks = [
            json.loads(row[0])["attack"]
            for row in connection.execute(
                "SELECT payload_json FROM metric_events WHERE kind = 'stat_observation'"
            )
        ]
    assert 788 in raw_attacks, "raw evidence remains available for diagnosis"


def test_metrics_repairs_legacy_cross_specialization_record(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    today = datetime.now().astimezone().date()
    append_log(
        logs / f"{today:%Y%m%d}.log",
        "18:10:17 | INFO | Hatch 攻擊特化 | side=left | "
        "parent=1320/158/134 | pair=1320/158/134,10/1/1",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)
    store.snapshot()
    bad_payload = json.dumps(
        {"hp": 1410, "attack": 737, "speed": 134, "tag": "HP特化", "role": "parent"},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with closing(sqlite3.connect(store.database)) as connection, connection:
        connection.execute(
            "UPDATE stat_records SET value = 737, payload_json = ? WHERE name = 'attack'",
            (bad_payload,),
        )

    result = store.snapshot()

    assert result["records"]["attack"]["value"] == 158


def test_metrics_does_not_count_cave_claim_as_hatched_dinosaur(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    today = datetime.now().astimezone().date()
    append_log(
        logs / f"{today:%Y%m%d}.log",
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
    today = datetime.now().astimezone().date()
    log = logs / f"{today:%Y%m%d}.log"
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


def test_timeline_groups_today_into_24_hourly_buckets(tmp_path: Path) -> None:
    today = datetime.now().astimezone().date()
    log = tmp_path / "logs" / f"{today:%Y%m%d}.log"
    append_log(
        log,
        """
03:00:00 | INFO | Bot started | Sense -> Think -> Act
03:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
03:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button
03:02:00 | INFO | Planning | hatch_claim_button at (330,1242) confidence=1.0
03:02:01 | INFO | Verify | Success | next UI detected: hatch_button
04:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
04:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", log.parent)

    result = store.snapshot()

    assert result["timeline_date"] == today.isoformat()
    assert result["timeline_granularity"] == "hour"
    assert len(result["timeline"]) == 24
    assert result["timeline"][2] == {
        "hour": 2,
        "label": "02:00",
        "hunt": 0,
        "hatch": 0,
    }
    assert result["timeline"][3] == {
        "hour": 3,
        "label": "03:00",
        "hunt": 1,
        "hatch": 1,
    }
    assert result["timeline"][4]["hunt"] == 1


def test_metrics_reads_plain_and_gzipped_daily_log_generations(tmp_path: Path) -> None:
    today = datetime.now().astimezone().date()
    logs = tmp_path / "logs"
    logs.mkdir()
    with gzip.open(logs / f"{today:%Y%m%d}.2.log.gz", "wt", encoding="utf-8") as stream:
        stream.write(
            "01:00:00 | INFO | Bot started | Sense -> Think -> Act\n"
            "01:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0\n"
            "01:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button\n"
        )
    append_log(
        logs / f"{today:%Y%m%d}.1.log",
        """
02:00:00 | INFO | Bot started | Sense -> Think -> Act
02:01:00 | INFO | Planning | hatch_claim_button at (330,1242) confidence=1.0
02:01:01 | INFO | Verify | Success | next UI detected: hatch_button
""",
    )
    append_log(
        logs / f"{today:%Y%m%d}.log",
        """
03:00:00 | INFO | Bot started | Sense -> Think -> Act
03:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
03:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)

    first = store.snapshot()
    second = store.snapshot()

    assert first["counters"]["hunt"]["today"] == 2
    assert first["counters"]["hatch"]["today"] == 1
    assert first["timeline"][1]["hunt"] == 1
    assert first["timeline"][2]["hatch"] == 1
    assert first["timeline"][3]["hunt"] == 1
    assert second["counters"] == first["counters"]


def test_metrics_deduplicates_events_when_live_log_rolls(tmp_path: Path) -> None:
    today = datetime.now().astimezone().date()
    logs = tmp_path / "logs"
    live = logs / f"{today:%Y%m%d}.log"
    append_log(
        live,
        """
05:00:00 | INFO | Bot started | Sense -> Think -> Act
05:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
05:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )
    store = MetricsStore(tmp_path / "data" / "stats.sqlite3", logs)
    assert store.snapshot()["counters"]["hunt"]["today"] == 1

    first = logs / f"{today:%Y%m%d}.1.log"
    live.replace(first)
    append_log(
        live,
        """
06:00:00 | INFO | Bot started | Sense -> Think -> Act
06:01:00 | INFO | Planning | hunt_confirm_button at (451,1411) confidence=1.0
06:01:01 | INFO | Verify | Success | next UI detected: map_exit_nest_button
""",
    )
    assert store.snapshot()["counters"]["hunt"]["today"] == 2

    with first.open("rb") as source, gzip.open(
        logs / f"{today:%Y%m%d}.2.log.gz", "wb"
    ) as packed:
        packed.write(source.read())
    first.unlink()
    assert store.snapshot()["counters"]["hunt"]["today"] == 2


def test_old_event_details_compact_to_daily_totals_and_records(tmp_path: Path) -> None:
    database = tmp_path / "data" / "stats.sqlite3"
    store = MetricsStore(database, tmp_path / "logs")
    yesterday = datetime.now().astimezone().date() - timedelta(days=1)
    occurred_at = f"{yesterday.isoformat()}T12:00:00"
    stat_payload = {
        "hp": 2340,
        "attack": 3,
        "speed": 1,
        "tag": "HP特化",
        "role": "selected",
    }
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """
            INSERT INTO metric_events(source_key, occurred_at, kind, value, payload_json)
            VALUES('old-hunt', ?, 'hunt', 7, '{}')
            """,
            (occurred_at,),
        )
        connection.execute(
            """
            INSERT INTO metric_events(source_key, occurred_at, kind, value, payload_json)
            VALUES('old-stat', ?, 'stat_observation', 1, ?)
            """,
            (occurred_at, json.dumps(stat_payload, ensure_ascii=False, separators=(",", ":"))),
        )

    result = store.snapshot()

    assert result["counters"]["hunt"] == {"session": 0, "today": 0, "total": 7}
    assert result["records"]["hp"]["value"] == 2340
    with closing(sqlite3.connect(database)) as connection, connection:
        assert connection.execute("SELECT COUNT(*) FROM metric_events").fetchone()[0] == 0
        assert connection.execute(
            "SELECT total FROM daily_totals WHERE day = ? AND kind = 'hunt'",
            (yesterday.isoformat(),),
        ).fetchone()[0] == 7
