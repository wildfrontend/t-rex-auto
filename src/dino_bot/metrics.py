"""Persistent dashboard metrics derived from verified runtime log events."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

_LOG_LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[^|]+) \| (?P<message>.*)$"
)
_PLANNING = re.compile(r"^Planning \| (?P<target>\S+) at \(")
_STATS = re.compile(r"(?P<hp>\d+)/(?P<attack>\d+)/(?P<speed>\d+)")
_PARENT = re.compile(r"^Hatch (?P<tag>攻擊特化|HP特化) \| .*\| parent=")
_SELECTED = re.compile(
    r"^Hatch (?P<tag>攻擊特化|HP特化) \| .*"
    r"decision=select row \d+ \((?P<stats>\d+/\d+/\d+)\)"
)
_CANDIDATES = re.compile(
    r"^Hatch (?P<tag>攻擊特化|HP特化) \| .*\| candidates=\[(?P<stats>[^\]]*)\]"
)
_AUTOPLACE = re.compile(r"^Hatch auto-place \| tag=(?P<tag>[^|]+) \|")

# 一次 parent 讀值可能整個位數誤讀(1->7、6->8),而同批 candidates 是整份清單
# 的掃描結果,會連續多輪讀出一致的值。超過清單最大值這個倍數的 parent 觀測
# 視為 OCR 離群,不納入紀錄。2.0 對真實的世代成長仍有寬裕空間。
_OUTLIER_RATIO = 2.0
_CAVE = re.compile(
    r"^Hatch cave \| capacity=(?P<count>\d+)/(?P<capacity>\d+) \| threshold=(?P<threshold>\d+)"
    r" \| cull=(?P<cull>True|False)"
)
_CULL_COMPLETED = re.compile(
    r"^Hatch cave \| cull completed \| before=(?P<before>\d+)/(?P<capacity>\d+)"
    r" \| selected=(?P<selected>\d+) \| expected_after=(?P<after>\d+)"
    r" \| result=claim_verified$"
)
_FEATURE = re.compile(r"^Feature \| (?P<feature>\S+)")

_COUNTER_KINDS = (
    "hunt",
    "hatch",
    "nest_collect_batch",
    "replacement",
    "autoplace_top",
    "autoplace_mass",
    "cull_removed",
    "verification_failure",
    "game_restart",
)


def _log_identity(path: Path) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"(?P<date>20\d{6})(?:\.(?P<generation>\d+))?\.log(?:\.gz)?",
        path.name,
    )
    if match is None:
        return None
    try:
        date_text = datetime.strptime(match.group("date"), "%Y%m%d").date().isoformat()
    except ValueError:
        return None
    return date_text, int(match.group("generation") or 0)


def _log_date(path: Path) -> str | None:
    identity = _log_identity(path)
    return identity[0] if identity else None


def _log_sort_key(path: Path) -> tuple[str, int]:
    identity = _log_identity(path)
    assert identity is not None
    date_text, generation = identity
    return date_text, -generation


def _file_signature(path: Path) -> str | None:
    """Identify a generation while allowing its live file to keep growing."""

    try:
        with path.open("rb") as stream:
            prefix = stream.read(4096)
    except OSError:
        return None
    return hashlib.sha256(prefix).hexdigest()


def _record_ceiling(name: str) -> int:
    """Upper bound above which a reading is an OCR error, not a real record.

    Attack tops out far below the other stats in practice: production logs peak
    around 181, while a single misread digit yields values several times that
    (168 -> 788).  Keep the bound well clear of real growth but inside the gap
    that whole-digit errors jump into.
    """

    return {"hp": 6000, "attack": 400, "speed": 500}[name]


def _valid_stats(values: tuple[int, int, int]) -> bool:
    hp, attack, speed = values
    # These broad limits reject OCR catastrophes such as 2320 -> 7320 while
    # leaving ample headroom above every value observed in production.
    return 1 <= hp <= 6000 and 1 <= attack <= 1000 and 1 <= speed <= 500


def _is_stat_outlier(
    values: tuple[int, int, int],
    ceilings: dict[str, int],
) -> bool:
    """Reject a reading that towers over the same round's candidate list.

    A parent line is a single OCR pass and can misread a whole digit (168 ->
    788).  The candidates list scans every nest in that round and re-reads the
    same dinosaurs across consecutive rounds, so its maximum is the trustworthy
    bound.  Without a list to compare against, keep the reading.
    """

    for name, value in zip(("hp", "attack", "speed"), values, strict=True):
        ceiling = ceilings.get(name)
        if ceiling and value > ceiling * _OUTLIER_RATIO:
            return True
    return False


def _record_stat_names(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return record fields trusted for this specialization observation.

    Attack and HP rounds deliberately optimize one primary field.  Secondary
    values still remain in the raw observation for diagnosis and tie-breaking,
    but must not become a Dashboard high score: a stable 1/7 OCR error in an
    HP parent's attack once promoted 137 to a fictitious 737 record.
    Speed has no dedicated specialization round, so retain its guarded value
    from either source.
    """

    # 攝入時判定的離群讀值:事件保留供診斷,但任何重建路徑都不得採用。
    if payload.get("outlier"):
        return ()
    tag = payload.get("tag")
    if tag == "HP特化":
        names = ("hp", "speed")
    elif tag == "攻擊特化":
        names = ("attack", "speed")
    else:
        names = ("hp", "attack", "speed")
    # 整位數誤讀(168 -> 788)會落在真實值構不到的區間;事件仍保留供診斷,
    # 但這種讀值不得成為最高紀錄。這道上限對每條重建路徑一致生效。
    return tuple(
        name for name in names if int(payload.get(name, 0)) <= _record_ceiling(name)
    )


class MetricsStore:
    """Incrementally import rotated text logs into a durable SQLite event store."""

    def __init__(self, database: Path, logs_dir: Path) -> None:
        self.database = database
        self.logs_dir = logs_dir
        self._lock = threading.RLock()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        # 目錄可能在執行中被外力移除(例如重新打包);連線前重建,
        # 讓錯誤自我修復而不是每個請求都失敗。
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Transactional connection that is always closed; sqlite3's own
        context manager only commits or rolls back and leaks the file handle."""

        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    path TEXT PRIMARY KEY,
                    offset INTEGER NOT NULL DEFAULT 0,
                    generation INTEGER NOT NULL DEFAULT 0,
                    state_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS metric_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    value INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS daily_totals (
                    day TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    total INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(day, kind)
                );
                CREATE TABLE IF NOT EXISTS stat_records (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_metric_events_time
                    ON metric_events(occurred_at);
                CREATE INDEX IF NOT EXISTS idx_metric_events_kind_time
                    ON metric_events(kind, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_metric_events_identity
                    ON metric_events(occurred_at, kind, value, payload_json);
                """
            )
            self._repair_stat_record_scope(connection)

    def refresh(self) -> None:
        with self._lock:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().astimezone().date().isoformat()
            paths: list[Path] = []
            for path in self.logs_dir.glob("20*.log*"):
                identity = _log_identity(path)
                if path.is_file() and identity is not None and identity[0] == today:
                    paths.append(path)
            paths.sort(key=_log_sort_key)
            with self._session() as connection:
                for path in paths:
                    self._ingest_path(connection, path)
                self._compact_history(connection, today)
                self._repair_stat_record_scope(connection)

    def _compact_history(self, connection: sqlite3.Connection, today: str) -> None:
        """Keep raw events only for today; retain older days as tiny totals."""

        cutoff = f"{today}T00:00:00"
        stat_rows = connection.execute(
            """
            SELECT occurred_at, payload_json FROM metric_events
            WHERE kind = 'stat_observation' AND occurred_at < ?
            """,
            (cutoff,),
        ).fetchall()
        for row in stat_rows:
            payload = json.loads(row["payload_json"])
            self._update_stat_records(
                connection,
                str(row["occurred_at"]),
                payload,
            )

        placeholders = ",".join("?" for _ in _COUNTER_KINDS)
        totals = connection.execute(
            f"""
            SELECT substr(occurred_at, 1, 10) AS day, kind, SUM(value) AS total
            FROM metric_events
            WHERE occurred_at < ? AND kind IN ({placeholders})
            GROUP BY day, kind
            """,
            (cutoff, *_COUNTER_KINDS),
        ).fetchall()
        for row in totals:
            connection.execute(
                """
                INSERT INTO daily_totals(day, kind, total) VALUES(?, ?, ?)
                ON CONFLICT(day, kind) DO UPDATE SET
                    total = daily_totals.total + excluded.total
                """,
                (row["day"], row["kind"], int(row["total"])),
            )
        connection.execute("DELETE FROM metric_events WHERE occurred_at < ?", (cutoff,))

        old_sources = connection.execute("SELECT path FROM sources").fetchall()
        for row in old_sources:
            date_text = _log_date(Path(row["path"]))
            if date_text is not None and date_text < today:
                connection.execute("DELETE FROM sources WHERE path = ?", (row["path"],))

    def _ingest_path(self, connection: sqlite3.Connection, path: Path) -> None:
        row = connection.execute(
            "SELECT offset, generation, state_json FROM sources WHERE path = ?",
            (str(path),),
        ).fetchone()
        offset = int(row["offset"]) if row else 0
        generation = int(row["generation"]) if row else 0
        state: dict[str, Any] = json.loads(row["state_json"]) if row else {}
        signature = _file_signature(path)
        if signature is None:
            return
        previous_signature = state.get("_file_signature")
        try:
            size = path.stat().st_size
        except OSError:
            return
        if previous_signature and previous_signature != signature or size < offset:
            offset = 0
            generation += 1
            state = {}
        date_text = _log_date(path)
        if date_text is None:
            return
        compressed = path.name.endswith(".gz")
        if compressed and previous_signature == signature and offset == size:
            return
        try:
            if compressed:
                # Compressed generations are immutable. If a generation index
                # receives a different archive after rolling, read it from the
                # beginning; semantic event deduplication prevents recounting
                # the same lines under their new filename.
                state = {}
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                    for line_number, line in enumerate(stream):
                        source = f"{path.name}:{generation}:{line_number}"
                        self._ingest_line(
                            connection,
                            source,
                            date_text,
                            line.rstrip("\r\n"),
                            state,
                        )
                offset = size
            else:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while True:
                        line_offset = stream.tell()
                        raw = stream.readline()
                        if not raw:
                            break
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        source = f"{path.name}:{generation}:{line_offset}"
                        self._ingest_line(connection, source, date_text, line, state)
                    offset = stream.tell()
        except (OSError, EOFError):
            return
        state["_file_signature"] = signature
        connection.execute(
            """
            INSERT INTO sources(path, offset, generation, state_json)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                offset = excluded.offset,
                generation = excluded.generation,
                state_json = excluded.state_json
            """,
            (str(path), offset, generation, json.dumps(state, ensure_ascii=False)),
        )

    def _record(
        self,
        connection: sqlite3.Connection,
        source: str,
        occurred_at: str,
        kind: str,
        *,
        value: int = 1,
        payload: dict[str, Any] | None = None,
        suffix: str = "",
    ) -> None:
        payload_json = json.dumps(
            payload or {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        duplicate = connection.execute(
            """
            SELECT 1 FROM metric_events
            WHERE occurred_at = ? AND kind = ? AND value = ? AND payload_json = ?
            LIMIT 1
            """,
            (occurred_at, kind, value, payload_json),
        ).fetchone()
        if duplicate is not None:
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO metric_events(
                source_key, occurred_at, kind, value, payload_json
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (
                source + suffix,
                occurred_at,
                kind,
                value,
                payload_json,
            ),
        )

    def _record_stats(
        self,
        connection: sqlite3.Connection,
        source: str,
        occurred_at: str,
        values: tuple[int, int, int],
        *,
        tag: str,
        role: str,
        suffix: str,
        ceilings: dict[str, int] | None = None,
    ) -> None:
        if not _valid_stats(values):
            return
        hp, attack, speed = values
        payload = {
            "hp": hp,
            "attack": attack,
            "speed": speed,
            "tag": tag,
            "role": role,
        }
        # 離群讀值仍留在 metric_events 供診斷,只是不得登上紀錄榜。標記寫進
        # payload,日後從事件表重建排行榜時才不會把它放回來。
        if _is_stat_outlier(values, ceilings or {}):
            payload["outlier"] = True
        else:
            self._update_stat_records(connection, occurred_at, payload)
        self._record(
            connection,
            source,
            occurred_at,
            "stat_observation",
            payload=payload,
            suffix=suffix,
        )

    @staticmethod
    def _update_stat_records(
        connection: sqlite3.Connection,
        occurred_at: str,
        payload: dict[str, Any],
    ) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        for name in _record_stat_names(payload):
            value = int(payload.get(name, 0))
            connection.execute(
                """
                INSERT INTO stat_records(name, value, occurred_at, payload_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    value = excluded.value,
                    occurred_at = excluded.occurred_at,
                    payload_json = excluded.payload_json
                WHERE excluded.value > stat_records.value
                """,
                (name, value, occurred_at, payload_json),
            )

    @classmethod
    def _repair_stat_record_scope(cls, connection: sqlite3.Connection) -> None:
        """Remove legacy cross-specialization records and rebuild their field."""

        rows = connection.execute(
            "SELECT name, payload_json FROM stat_records"
        ).fetchall()
        removed = False
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["name"] not in _record_stat_names(payload):
                connection.execute(
                    "DELETE FROM stat_records WHERE name = ?",
                    (row["name"],),
                )
                removed = True
        if not removed:
            return
        observations = connection.execute(
            """
            SELECT occurred_at, payload_json FROM metric_events
            WHERE kind = 'stat_observation' ORDER BY occurred_at
            """
        ).fetchall()
        for row in observations:
            cls._update_stat_records(
                connection,
                str(row["occurred_at"]),
                json.loads(row["payload_json"]),
            )

    def _ingest_line(
        self,
        connection: sqlite3.Connection,
        source: str,
        date_text: str,
        line: str,
        state: dict[str, Any],
    ) -> None:
        match = _LOG_LINE.match(line)
        if match is None:
            return
        occurred_at = f"{date_text}T{match.group('time')}"
        message = match.group("message")

        feature = _FEATURE.match(message)
        if feature is not None:
            state["mode"] = feature.group("feature")
        if message.startswith("Bot started |"):
            self._record(
                connection,
                source,
                occurred_at,
                "session_start",
                payload={"mode": state.get("mode", "hunt")},
            )
            state["current_target"] = None
            state["in_cave"] = False
            state["pending_replacement"] = None
            return
        if message.startswith("Bot stopped |"):
            self._record(connection, source, occurred_at, "session_stop")
            return

        planning = _PLANNING.match(message)
        if planning is not None:
            target = planning.group("target")
            state["current_target"] = target
            if target == "hatch_cave_continuous_button":
                state["in_cave"] = True
            return

        if message.startswith("Verify | Success |"):
            target = state.get("current_target")
            if target == "hunt_confirm_button":
                self._record(connection, source, occurred_at, "hunt")
            elif target == "hatch_claim_button":
                if not state.get("in_cave", False):
                    self._record(connection, source, occurred_at, "hatch")
                state["in_cave"] = False
            elif target == "hatch_collect_eggs_button":
                self._record(connection, source, occurred_at, "nest_collect_batch")
            elif target == "hatch_confirm_yes" and state.get("pending_replacement"):
                payload = dict(state["pending_replacement"])
                self._record(
                    connection,
                    source,
                    occurred_at,
                    "replacement",
                    payload=payload,
                )
                state["pending_replacement"] = None
            state["current_target"] = None
            return

        if message.startswith("Verify | Failed |"):
            self._record(
                connection,
                source,
                occurred_at,
                "verification_failure",
                payload={"target": state.get("current_target")},
            )
            state["current_target"] = None
            return

        if message.startswith("Recovery | game restarted;"):
            self._record(connection, source, occurred_at, "game_restart")
            return

        if message.startswith("Hatch | claim confirmed by next unready egg detail"):
            self._record(connection, source, occurred_at, "hatch")
            return

        candidates = _CANDIDATES.match(message)
        if candidates is not None:
            ceilings: dict[str, int] = {}
            for stats_match in _STATS.finditer(candidates.group("stats")):
                for name in ("hp", "attack", "speed"):
                    value = int(stats_match.group(name))
                    if value > ceilings.get(name, 0):
                        ceilings[name] = value
            if ceilings:
                state["stat_ceilings"] = ceilings
            # 這行本身不產生觀測,只為後續 parent 讀值提供離群基準。

        parent = _PARENT.match(message)
        if parent is not None:
            ceilings = state.get("stat_ceilings") or {}
            for index, stats_match in enumerate(_STATS.finditer(message)):
                values = tuple(int(stats_match.group(name)) for name in ("hp", "attack", "speed"))
                self._record_stats(
                    connection,
                    source,
                    occurred_at,
                    values,  # type: ignore[arg-type]
                    tag=parent.group("tag"),
                    role="parent",
                    suffix=f":stat:{index}",
                    ceilings=ceilings,
                )
            return

        selected = _SELECTED.match(message)
        if selected is not None:
            stats_match = _STATS.fullmatch(selected.group("stats"))
            if stats_match is None:
                return
            values = tuple(int(stats_match.group(name)) for name in ("hp", "attack", "speed"))
            if not _valid_stats(values):  # type: ignore[arg-type]
                return
            payload = {
                "hp": values[0],
                "attack": values[1],
                "speed": values[2],
                "tag": selected.group("tag"),
                "role": "selected",
            }
            state["pending_replacement"] = payload
            self._record_stats(
                connection,
                source,
                occurred_at,
                values,  # type: ignore[arg-type]
                tag=selected.group("tag"),
                role="selected",
                suffix=":selected",
            )
            return

        autoplace = _AUTOPLACE.match(message)
        if autoplace is not None and "completed" in message:
            tag = autoplace.group("tag").strip()
            kind = "autoplace_top" if tag == "頂尖" else "autoplace_mass"
            self._record(connection, source, occurred_at, kind, payload={"tag": tag})
            return

        cull_completed = _CULL_COMPLETED.match(message)
        if cull_completed is not None:
            selected = int(cull_completed.group("selected"))
            self._record(
                connection,
                source,
                occurred_at,
                "cull_removed",
                value=selected,
                payload={
                    "before": int(cull_completed.group("before")),
                    "selected": selected,
                    "expected_after": int(cull_completed.group("after")),
                    "result": "claim_verified",
                },
            )
            return

        cave = _CAVE.match(message)
        if cave is not None:
            self._record(
                connection,
                source,
                occurred_at,
                "cull_decision",
                payload={
                    "capacity": int(cave.group("count")),
                    "capacity_limit": int(cave.group("capacity")),
                    "threshold": int(cave.group("threshold")),
                    "cull": cave.group("cull") == "True",
                },
            )

    @staticmethod
    def _counter(
        connection: sqlite3.Connection,
        kind: str,
        since: str | None = None,
    ) -> int:
        query = "SELECT COALESCE(SUM(value), 0) FROM metric_events WHERE kind = ?"
        params: list[Any] = [kind]
        if since is not None:
            query += " AND occurred_at >= ?"
            params.append(since)
        total = int(connection.execute(query, params).fetchone()[0])
        if since is None:
            total += int(
                connection.execute(
                    "SELECT COALESCE(SUM(total), 0) FROM daily_totals WHERE kind = ?",
                    (kind,),
                ).fetchone()[0]
            )
        return total

    def snapshot(self, recent_limit: int = 20) -> dict[str, Any]:
        self.refresh()
        today = datetime.now().astimezone().date().isoformat()
        with self._session() as connection:
            session_row = connection.execute(
                """
                SELECT occurred_at, payload_json FROM metric_events
                WHERE kind = 'session_start'
                ORDER BY occurred_at DESC, id DESC LIMIT 1
                """
            ).fetchone()
            session_since = str(session_row["occurred_at"]) if session_row else None
            session_mode = (
                json.loads(session_row["payload_json"]).get("mode", "hunt")
                if session_row
                else None
            )
            session_counter_since = session_since or f"{today}T00:00:00"
            counters: dict[str, dict[str, int]] = {}
            for kind in _COUNTER_KINDS:
                counters[kind] = {
                    "session": self._counter(connection, kind, session_counter_since),
                    "today": self._counter(connection, kind, f"{today}T00:00:00"),
                    "total": self._counter(connection, kind),
                }

            records: dict[str, dict[str, Any]] = {}
            archived_records = connection.execute(
                "SELECT name, value, occurred_at, payload_json FROM stat_records"
            ).fetchall()
            for row in archived_records:
                payload = json.loads(row["payload_json"])
                records[row["name"]] = {
                    "value": int(row["value"]),
                    "occurred_at": row["occurred_at"],
                    "tag": payload.get("tag"),
                    "role": payload.get("role"),
                }
            rows = connection.execute(
                """
                SELECT occurred_at, payload_json FROM metric_events
                WHERE kind = 'stat_observation' ORDER BY occurred_at
                """
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                for name in _record_stat_names(payload):
                    value = int(payload.get(name, 0))
                    current = records.get(name)
                    if current is None or value > int(current["value"]):
                        records[name] = {
                            "value": value,
                            "occurred_at": row["occurred_at"],
                            "tag": payload.get("tag"),
                            "role": payload.get("role"),
                        }

            recent_rows = connection.execute(
                """
                SELECT occurred_at, kind, value, payload_json
                FROM metric_events
                WHERE kind NOT IN ('stat_observation', 'session_start', 'session_stop')
                ORDER BY occurred_at DESC, id DESC LIMIT ?
                """,
                (max(0, recent_limit),),
            ).fetchall()
            recent = [
                {
                    "occurred_at": row["occurred_at"],
                    "kind": row["kind"],
                    "value": row["value"],
                    "details": json.loads(row["payload_json"]),
                }
                for row in recent_rows
            ]

            timeline_rows = connection.execute(
                """
                SELECT CAST(substr(occurred_at, 12, 2) AS INTEGER) AS hour,
                       kind, SUM(value) AS total
                FROM metric_events
                WHERE kind IN ('hunt', 'hatch')
                  AND substr(occurred_at, 1, 10) = ?
                GROUP BY hour, kind ORDER BY hour, kind
                """,
                (today,),
            ).fetchall()
            by_hour = {
                hour: {"hunt": 0, "hatch": 0}
                for hour in range(24)
            }
            for row in timeline_rows:
                by_hour[int(row["hour"])][row["kind"]] = int(row["total"])

        return {
            "session_started": session_since,
            "session_mode": session_mode,
            "counters": counters,
            "records": records,
            "recent_events": recent,
            "timeline": [
                {"hour": hour, "label": f"{hour:02d}:00", **by_hour[hour]}
                for hour in range(24)
            ],
            "timeline_date": today,
            "timeline_granularity": "hour",
            "database": str(self.database),
            "generated_at": datetime.now().astimezone().isoformat(),
        }
