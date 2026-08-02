from __future__ import annotations

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
