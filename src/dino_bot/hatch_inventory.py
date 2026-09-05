"""Persistent local inventory for incubator cooldown boosts."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_BOOST_STOCK = 100
MAX_BOOST_STOCK = 100
BOOST_COST = 1
BOOST_INTERVAL_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class HatchBoostInventory:
    remaining: int
    used_total: int
    enabled: bool
    updated_at: str
    last_used_at: float | None = None
    next_use_at: float = 0.0
    retry_after: float = 0.0
    waiting_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "remaining": self.remaining,
            "used_total": self.used_total,
            "enabled": self.enabled,
            "updated_at": self.updated_at,
            "maximum": MAX_BOOST_STOCK,
            "cost": BOOST_COST,
            "interval_seconds": BOOST_INTERVAL_SECONDS,
            "last_used_at": self.last_used_at,
            "next_use_at": self.next_use_at,
            "next_check_at": max(self.next_use_at, self.retry_after),
            "waiting_reason": self.waiting_reason,
        }


class HatchBoostInventoryStore:
    """Share one transactional stock counter between Bot and Dashboard."""

    def __init__(
        self, database: Path, *, default_stock: int = DEFAULT_BOOST_STOCK,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.database = database
        self.clock = clock
        self.default_stock = self._validate_stock(default_stock)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def snapshot(self) -> HatchBoostInventory:
        with self._session() as connection:
            row = connection.execute(
                "SELECT remaining, used_total, enabled, updated_at, last_used_at, "
                "next_use_at, retry_after, waiting_reason "
                "FROM local_inventory WHERE name = ?",
                ("hatch_cooldown_boost",),
            ).fetchone()
        if row is None:
            self._initialize()
            return self.snapshot()
        return HatchBoostInventory(
            int(row[0]), int(row[1]), bool(row[2]), str(row[3]),
            float(row[4]) if row[4] is not None else None,
            float(row[5]), float(row[6]), str(row[7]),
        )

    def ready_delay_seconds(self) -> float | None:
        inventory = self.snapshot()
        if not inventory.enabled or inventory.remaining <= 0:
            return None
        return max(0.0, max(inventory.next_use_at, inventory.retry_after) - self.clock())

    def defer(self, reason: str, *, seconds: float = 60.0) -> None:
        """Retry an unavailable/uncertain visit without moving the use deadline."""
        with self._session() as connection:
            connection.execute(
                "UPDATE local_inventory SET retry_after = ?, waiting_reason = ? WHERE name = ?",
                (self.clock() + seconds, reason, "hatch_cooldown_boost"),
            )

    def set_remaining(self, remaining: int) -> HatchBoostInventory:
        value = self._validate_stock(remaining)
        updated_at = self._now()
        with self._session() as connection:
            connection.execute(
                "UPDATE local_inventory SET remaining = ?, updated_at = ? WHERE name = ?",
                (value, updated_at, "hatch_cooldown_boost"),
            )
        return self.snapshot()

    def set_enabled(self, enabled: bool) -> HatchBoostInventory:
        if not isinstance(enabled, bool):
            raise ValueError("boost enabled must be a boolean")
        updated_at = self._now()
        with self._session() as connection:
            connection.execute(
                "UPDATE local_inventory SET enabled = ?, updated_at = ? WHERE name = ?",
                (int(enabled), updated_at, "hatch_cooldown_boost"),
            )
        return self.snapshot()

    def consume_one(self, *, only_if_due: bool = False) -> HatchBoostInventory | None:
        """Atomically consume one verified boost, or return None at zero stock."""

        updated_at = self._now()
        now = self.clock()
        with self._session() as connection:
            cursor = connection.execute(
                "UPDATE local_inventory "
                "SET remaining = remaining - 1, used_total = used_total + 1, updated_at = ?, "
                "last_used_at = ?, next_use_at = ?, retry_after = 0, waiting_reason = '' "
                "WHERE name = ? AND remaining > 0 AND (? = 0 OR next_use_at <= ?)",
                (updated_at, now, now + BOOST_INTERVAL_SECONDS,
                 "hatch_cooldown_boost", int(only_if_due), now),
            )
            if cursor.rowcount != 1:
                return None
        return self.snapshot()

    def _initialize(self) -> None:
        with self._session() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS local_inventory ("
                "name TEXT PRIMARY KEY, remaining INTEGER NOT NULL, "
                "used_total INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 0, "
                "updated_at TEXT NOT NULL)"
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(local_inventory)")
            }
            if "enabled" not in columns:
                connection.execute(
                    "ALTER TABLE local_inventory ADD COLUMN enabled INTEGER NOT NULL DEFAULT 0"
                )
            for name, declaration in (
                ("last_used_at", "REAL"),
                ("next_use_at", "REAL NOT NULL DEFAULT 0"),
                ("retry_after", "REAL NOT NULL DEFAULT 0"),
                ("waiting_reason", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE local_inventory ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                "INSERT OR IGNORE INTO local_inventory "
                "(name, remaining, used_total, enabled, updated_at) VALUES (?, ?, 0, 0, ?)",
                ("hatch_cooldown_boost", self.default_stock, self._now()),
            )

    def _connect(self) -> sqlite3.Connection:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.database, timeout=5.0)

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

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _validate_stock(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("boost stock must be an integer")
        if value < 0 or value > MAX_BOOST_STOCK:
            raise ValueError(f"boost stock must be between 0 and {MAX_BOOST_STOCK}")
        return value
