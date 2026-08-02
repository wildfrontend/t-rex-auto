"""Persistent dashboard metrics derived from verified runtime log events."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
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
_AUTOPLACE = re.compile(r"^Hatch auto-place \| tag=(?P<tag>[^|]+) \|")
_CAVE = re.compile(
    r"^Hatch cave \| capacity=(?P<count>\d+)/350 \| threshold=(?P<threshold>\d+)"
    r" \| cull=(?P<cull>True|False)"
)
_FEATURE = re.compile(r"^Feature \| (?P<feature>\S+)")

_COUNTER_KINDS = (
    "hunt",
    "hatch",
    "nest_collect_batch",
    "replacement",
    "autoplace_top",
    "autoplace_mass",
    "verification_failure",
    "game_restart",
)


def _log_date(path: Path) -> str | None:
    match = re.match(r"(?P<date>20\d{6})\.log", path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group("date"), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _valid_stats(values: tuple[int, int, int]) -> bool:
    hp, attack, speed = values
    # These broad limits reject OCR catastrophes such as 2320 -> 7320 while
    # leaving ample headroom above every value observed in production.
    return 1 <= hp <= 6000 and 1 <= attack <= 1000 and 1 <= speed <= 500


class MetricsStore:
    """Incrementally import rotated text logs into a durable SQLite event store."""

    def __init__(self, database: Path, logs_dir: Path) -> None:
        self.database = database
        self.logs_dir = logs_dir
        self._lock = threading.RLock()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
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
                CREATE INDEX IF NOT EXISTS idx_metric_events_time
                    ON metric_events(occurred_at);
                CREATE INDEX IF NOT EXISTS idx_metric_events_kind_time
                    ON metric_events(kind, occurred_at);
                """
            )

    def refresh(self) -> None:
        with self._lock:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(
                path
                for path in self.logs_dir.glob("20*.log*")
                if path.is_file() and not path.name.endswith(".gz") and _log_date(path)
            )
            with self._connect() as connection:
                for path in paths:
                    self._ingest_path(connection, path)

    def _ingest_path(self, connection: sqlite3.Connection, path: Path) -> None:
        row = connection.execute(
            "SELECT offset, generation, state_json FROM sources WHERE path = ?",
            (str(path),),
        ).fetchone()
        offset = int(row["offset"]) if row else 0
        generation = int(row["generation"]) if row else 0
        state: dict[str, Any] = json.loads(row["state_json"]) if row else {}
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < offset:
            offset = 0
            generation += 1
            state = {}
        date_text = _log_date(path)
        if date_text is None:
            return
        try:
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
        except OSError:
            return
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
                json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")),
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
    ) -> None:
        if not _valid_stats(values):
            return
        hp, attack, speed = values
        self._record(
            connection,
            source,
            occurred_at,
            "stat_observation",
            payload={
                "hp": hp,
                "attack": attack,
                "speed": speed,
                "tag": tag,
                "role": role,
            },
            suffix=suffix,
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

        parent = _PARENT.match(message)
        if parent is not None:
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

        cave = _CAVE.match(message)
        if cave is not None:
            self._record(
                connection,
                source,
                occurred_at,
                "cull_decision",
                payload={
                    "capacity": int(cave.group("count")),
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
        return int(connection.execute(query, params).fetchone()[0])

    def snapshot(self, recent_limit: int = 20) -> dict[str, Any]:
        self.refresh()
        today = datetime.now().astimezone().date().isoformat()
        with self._connect() as connection:
            session_row = connection.execute(
                """
                SELECT occurred_at, payload_json FROM metric_events
                WHERE kind = 'session_start' ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            session_since = str(session_row["occurred_at"]) if session_row else None
            session_mode = (
                json.loads(session_row["payload_json"]).get("mode", "hunt")
                if session_row
                else None
            )
            counters: dict[str, dict[str, int]] = {}
            for kind in _COUNTER_KINDS:
                counters[kind] = {
                    "session": self._counter(connection, kind, session_since),
                    "today": self._counter(connection, kind, f"{today}T00:00:00"),
                    "total": self._counter(connection, kind),
                }

            records: dict[str, dict[str, Any]] = {}
            rows = connection.execute(
                """
                SELECT occurred_at, payload_json FROM metric_events
                WHERE kind = 'stat_observation' ORDER BY occurred_at
                """
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                for name in ("hp", "attack", "speed"):
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
                ORDER BY id DESC LIMIT ?
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
                SELECT substr(occurred_at, 1, 10) AS day, kind, SUM(value) AS total
                FROM metric_events
                WHERE kind IN ('hunt', 'hatch')
                GROUP BY day, kind ORDER BY day DESC LIMIT 14
                """
            ).fetchall()
            by_day: dict[str, dict[str, int]] = {}
            for row in timeline_rows:
                by_day.setdefault(row["day"], {"hunt": 0, "hatch": 0})[row["kind"]] = int(
                    row["total"]
                )

        return {
            "session_started": session_since,
            "session_mode": session_mode,
            "counters": counters,
            "records": records,
            "recent_events": recent,
            "timeline": [
                {"day": day, **values}
                for day, values in sorted(by_day.items())[-7:]
            ],
            "database": str(self.database),
            "generated_at": datetime.now().astimezone().isoformat(),
        }
