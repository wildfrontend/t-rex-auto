"""Persistent local inventory for incubator cooldown boosts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_BOOST_STOCK = 100
MAX_BOOST_STOCK = 100
BOOST_COST = 1


@dataclass(frozen=True, slots=True)
class HatchBoostInventory:
    remaining: int
    used_total: int
    updated_at: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "remaining": self.remaining,
            "used_total": self.used_total,
            "updated_at": self.updated_at,
            "maximum": MAX_BOOST_STOCK,
            "cost": BOOST_COST,
        }


class HatchBoostInventoryStore:
    """Share one transactional stock counter between Bot and Dashboard."""

    def __init__(self, database: Path, *, default_stock: int = DEFAULT_BOOST_STOCK) -> None:
        self.database = database
        self.default_stock = self._validate_stock(default_stock)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def snapshot(self) -> HatchBoostInventory:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT remaining, used_total, updated_at "
                "FROM local_inventory WHERE name = ?",
                ("hatch_cooldown_boost",),
            ).fetchone()
        if row is None:
            self._initialize()
            return self.snapshot()
        return HatchBoostInventory(int(row[0]), int(row[1]), str(row[2]))

    def set_remaining(self, remaining: int) -> HatchBoostInventory:
        value = self._validate_stock(remaining)
        updated_at = self._now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE local_inventory SET remaining = ?, updated_at = ? WHERE name = ?",
                (value, updated_at, "hatch_cooldown_boost"),
            )
        return self.snapshot()

    def consume_one(self) -> HatchBoostInventory | None:
        """Atomically consume one verified boost, or return None at zero stock."""

        updated_at = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE local_inventory "
                "SET remaining = remaining - 1, used_total = used_total + 1, updated_at = ? "
                "WHERE name = ? AND remaining > 0",
                (updated_at, "hatch_cooldown_boost"),
            )
            if cursor.rowcount != 1:
                return None
        return self.snapshot()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS local_inventory ("
                "name TEXT PRIMARY KEY, remaining INTEGER NOT NULL, "
                "used_total INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO local_inventory "
                "(name, remaining, used_total, updated_at) VALUES (?, ?, 0, ?)",
                ("hatch_cooldown_boost", self.default_stock, self._now()),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=5.0)

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
