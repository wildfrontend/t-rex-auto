from __future__ import annotations

import sqlite3

from dino_bot.hatch_inventory import HatchBoostInventoryStore


def test_boost_inventory_defaults_to_100_and_persists(tmp_path) -> None:
    database = tmp_path / "data" / "stats.sqlite3"
    store = HatchBoostInventoryStore(database)

    assert store.snapshot().remaining == 100
    assert store.snapshot().used_total == 0
    assert not store.snapshot().enabled

    consumed = store.consume_one()
    reopened = HatchBoostInventoryStore(database)

    assert consumed is not None and consumed.remaining == 99
    assert reopened.snapshot().remaining == 99
    assert reopened.snapshot().used_total == 1


def test_boost_inventory_stops_at_zero_and_allows_manual_correction(tmp_path) -> None:
    store = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")

    corrected = store.set_remaining(1)
    consumed = store.consume_one()

    assert corrected.remaining == 1
    assert consumed is not None and consumed.remaining == 0
    assert store.consume_one() is None
    assert store.snapshot().used_total == 1


def test_boost_permission_is_disabled_by_default_and_persists(tmp_path) -> None:
    database = tmp_path / "stats.sqlite3"
    store = HatchBoostInventoryStore(database)

    enabled = store.set_enabled(True)
    reopened = HatchBoostInventoryStore(database)

    assert enabled.enabled
    assert reopened.snapshot().enabled


def test_boost_inventory_rejects_values_outside_local_cap(tmp_path) -> None:
    store = HatchBoostInventoryStore(tmp_path / "stats.sqlite3")

    for value in (-1, 101):
        try:
            store.set_remaining(value)
        except ValueError as exc:
            assert "between 0 and 100" in str(exc)
        else:
            raise AssertionError("invalid stock was accepted")


def test_schedule_persists_across_restart_switch_and_stock_changes(tmp_path) -> None:
    now = [10_000.0]
    database = tmp_path / "stats.sqlite3"
    store = HatchBoostInventoryStore(database, clock=lambda: now[0])
    store.set_enabled(True)
    assert store.ready_delay_seconds() == 0
    used = store.consume_one(only_if_due=True)
    assert used.last_used_at == 10_000
    assert used.next_use_at == 11_800
    assert store.consume_one(only_if_due=True) is None
    now[0] += 900
    reopened = HatchBoostInventoryStore(database, clock=lambda: now[0])
    reopened.set_enabled(False)
    assert reopened.ready_delay_seconds() is None
    reopened.set_remaining(50)
    reopened.set_enabled(True)
    assert reopened.ready_delay_seconds() == 900
    now[0] = 11_799
    assert reopened.ready_delay_seconds() == 1
    now[0] = 11_800
    assert reopened.ready_delay_seconds() == 0
    assert reopened.consume_one(only_if_due=True).remaining == 49


def test_retry_does_not_start_another_thirty_minutes_or_spend_stock(tmp_path) -> None:
    now = [10_000.0]
    store = HatchBoostInventoryStore(tmp_path / "stats.sqlite3", clock=lambda: now[0])
    store.set_enabled(True)
    store.defer("gray bar")
    assert store.ready_delay_seconds() == 60
    assert store.snapshot().next_use_at == 0
    assert store.snapshot().remaining == 100
    now[0] += 60
    assert store.ready_delay_seconds() == 0


def test_schedule_migrates_existing_inventory_and_is_instance_isolated(tmp_path) -> None:
    database = tmp_path / "s9.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE local_inventory (name TEXT PRIMARY KEY, remaining INTEGER, "
            "used_total INTEGER, enabled INTEGER, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO local_inventory VALUES ('hatch_cooldown_boost', 27, 73, 1, 'old')"
        )
    s9 = HatchBoostInventoryStore(database)
    s13 = HatchBoostInventoryStore(tmp_path / "s13.sqlite3")
    assert s9.snapshot().remaining == 27
    assert s9.snapshot().used_total == 73
    assert s9.ready_delay_seconds() == 0
    s9.consume_one(only_if_due=True)
    assert s13.snapshot().last_used_at is None
    assert s13.snapshot().remaining == 100
