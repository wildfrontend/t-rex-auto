"""Turn the event stream into throughput figures worth acting on.

``summary.json`` answers "is something broken". It cannot answer "where is the
time going", which is a different question with different evidence: a bot that
hunts half as often as it could reports no errors at all. The waiting branches
are the expensive ones - a capacity cooldown is five minutes of deliberately
doing nothing - and until the planner named them every idle cycle looked alike.

This module reads the events the bundle already ships and reports the shares,
so a tuning decision cites a measurement instead of a hunch. It suggests which
knob a figure points at; it never suggests a value, because the right value
depends on the account and the emulator, not on the code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

HUNT_CONFIRM_TYPE = "hunt_confirm_button"

# A waiting branch below this share of planning cycles is noise: every run
# settles the map and collects mail, and reporting those as tunable would bury
# the one stage that is actually expensive.
_STAGE_SHARE_LIMIT = 0.2
# One rule doing most of the rejecting is a tuning signal. A spread across all
# eight is the planner working as intended on a busy map.
_REJECTION_SHARE_LIMIT = 0.5
_REJECTION_SAMPLE_MINIMUM = 20
_VERIFY_FAILURE_LIMIT = 0.3
_VERIFY_SAMPLE_MINIMUM = 10
# Detection is the per-cycle floor: nothing else in the loop can be faster than
# it. Half a second means the loop reacts at most twice a second.
_DETECT_MS_LIMIT = 400
_MINIMUM_SPAN_SECONDS = 60.0

# Which knob a share points at. Named rather than valued: the stage tells you
# where the time went, not what the number should be.
_STAGE_KNOBS: Mapping[str, str] = {
    "capacity_wait": "planner.capacity_wait_seconds",
    "map_settle": "planner.map_settle_frames, planner.map_settle_max_frames",
    "recenter": "planner.recenter_every, planner.stalled_recenter_frames",
    "mail": "planner.mail_after_hunts",
    # `action_cooldown` is deliberately absent. The game refuses a hunt while
    # the team is hurt, so that wait buys recovery the bot cannot shorten -
    # reporting it as a tunable invites someone to trade it for a run of
    # refused hunts.
    "await_hunt": "HuntPlanner(await_hunt_frames=...) - not exposed in config.json",
    "blocked": "detector thresholds for planner.blocking_types",
}

_REJECTION_KNOBS: Mapping[str, str] = {
    "own_path_marker": "planner.own_path_radius",
    "own_path_angle": "planner.own_path_angle_degrees",
    "screen_center": "planner.anchor_exclusion_radius",
    "anchor_radius": "planner.anchor_exclusion_radius",
    "screen_margin": "planner.bottom_exclusion_px",
    "exclusion_zone": "planner.exclusion_zones",
    "failure_cooldown": (
        "planner.dinosaur_failure_radius, planner.dinosaur_failure_cooldown_ms"
    ),
    "anchor_window": "no config knob - anchor tracking in HuntPlanner",
    "team_status_panel": "no config knob - team sheet guard in HuntPlanner",
}


def _parse(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError:
            # The tail can start mid-line after a size rollover. One unusable
            # record must not cost the whole report.
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


def _clock_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _span_seconds(stamps: Sequence[float]) -> float:
    """Elapsed time across stamps that carry no date.

    Events record the wall clock only, so a run crossing midnight would
    otherwise measure as negative. Every step backwards is a new day.
    """

    if len(stamps) < 2:
        return 0.0
    total = 0.0
    previous = stamps[0]
    for stamp in stamps[1:]:
        step = stamp - previous
        if step < 0:
            step += 86400.0
        total += step
        previous = stamp
    return total


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return round(float(ordered[index]), 1)


def _latency(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "samples": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": round(float(max(values)), 1),
    }


def _verification_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    target = record.get("target")
    if not isinstance(target, Mapping):
        target = {}
    return (
        record.get("c"),
        target.get("type"),
        target.get("x"),
        target.get("y"),
        record.get("attempt"),
    )


def _suggestions(
    stage_counts: Mapping[str, int],
    planning_cycles: int,
    rejections: Mapping[str, int],
    verify_total: int,
    verify_failed: int,
    retry_exhausted: int,
    detect_ms: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    for stage, count in sorted(stage_counts.items(), key=lambda item: -item[1]):
        if stage not in _STAGE_KNOBS or planning_cycles <= 0:
            continue
        share = count / planning_cycles
        if share < _STAGE_SHARE_LIMIT:
            continue
        found.append(
            {
                "code": f"time_spent_in_{stage}",
                "severity": "info",
                "evidence": {
                    "stage": stage,
                    "cycles": count,
                    "planning_cycles": planning_cycles,
                    "share": round(share, 3),
                },
                "tuning": _STAGE_KNOBS[stage],
            }
        )

    total_rejections = sum(rejections.values())
    if total_rejections >= _REJECTION_SAMPLE_MINIMUM:
        rule, count = max(rejections.items(), key=lambda item: item[1])
        share = count / total_rejections
        if share >= _REJECTION_SHARE_LIMIT:
            found.append(
                {
                    "code": "dinosaur_rejections_dominated_by_one_rule",
                    "severity": "info",
                    "evidence": {
                        "rule": rule,
                        "count": count,
                        "total_rejections": total_rejections,
                        "share": round(share, 3),
                    },
                    "tuning": _REJECTION_KNOBS.get(rule, "HuntPlanner rejection rules"),
                }
            )

    if verify_total >= _VERIFY_SAMPLE_MINIMUM:
        failure_rate = verify_failed / verify_total
        if failure_rate >= _VERIFY_FAILURE_LIMIT:
            found.append(
                {
                    "code": "verification_failure_rate_high",
                    "severity": "info",
                    "evidence": {
                        "failed": verify_failed,
                        "total": verify_total,
                        "failure_rate": round(failure_rate, 3),
                        "retry_exhausted": retry_exhausted,
                    },
                    "tuning": (
                        "detector thresholds in assets/manifest.json, "
                        "action.post_action_delays_ms, action.click_delay_ms"
                    ),
                }
            )

    if detect_ms is not None:
        p95 = detect_ms.get("p95")
        if isinstance(p95, (int, float)) and p95 >= _DETECT_MS_LIMIT:
            found.append(
                {
                    "code": "detection_latency_limits_cycle_rate",
                    "severity": "info",
                    "evidence": {"detect_ms_p95": p95, "limit_ms": _DETECT_MS_LIMIT},
                    "tuning": (
                        "template count and scales in assets/manifest.json, "
                        "capture backend in capture.backend"
                    ),
                }
            )

    return found


def summarize_events(lines: Iterable[str]) -> dict[str, Any]:
    """Report where the loop spent its cycles, from the shipped event tail."""

    events = _parse(lines)
    if not events:
        return {"available": False, "reason": "no events recorded"}

    stamps: list[float] = []
    capture_ms: list[float] = []
    detect_ms: list[float] = []
    stage_counts: dict[str, int] = {}
    rejections: dict[str, int] = {}
    action_cycles: set[int] = set()
    planning_cycles = 0
    cycles: set[int] = set()
    hunts_confirmed = 0
    verify_total = 0
    verify_failed = 0
    verify_pending = 0
    verify_checks_total = 0
    retry_exhausted = 0
    recoveries = 0
    sessions = 0

    # v0.2.15 event records did not distinguish an in-progress poll from the
    # final result. The last check for an action is terminal; earlier checks
    # are pending. New records carry an explicit ``phase`` field.
    legacy_terminal: set[int] = set()
    for index, record in enumerate(events):
        if record.get("e") != "verify" or record.get("phase") is not None:
            continue
        next_record = events[index + 1] if index + 1 < len(events) else None
        if (
            not isinstance(next_record, Mapping)
            or next_record.get("e") != "verify"
            or _verification_key(next_record) != _verification_key(record)
        ):
            legacy_terminal.add(index)

    for index, record in enumerate(events):
        stamp = _clock_seconds(record.get("t"))
        if stamp is not None:
            stamps.append(stamp)
        cycle = record.get("c")
        if isinstance(cycle, int):
            cycles.add(cycle)
        kind = record.get("e")

        if kind == "capture":
            value = record.get("ms")
            if isinstance(value, (int, float)):
                capture_ms.append(float(value))
        elif kind == "detect":
            value = record.get("ms")
            if isinstance(value, (int, float)):
                detect_ms.append(float(value))
        elif kind == "plan":
            planning_cycles += 1
            stage = record.get("stage")
            if isinstance(stage, str) and stage:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            reject = record.get("reject")
            if isinstance(reject, Mapping):
                for rule, count in reject.items():
                    if isinstance(count, int):
                        rejections[str(rule)] = rejections.get(str(rule), 0) + count
        elif kind == "action":
            if isinstance(cycle, int):
                action_cycles.add(cycle)
        elif kind == "verify":
            verify_checks_total += 1
            phase = record.get("phase")
            is_pending = phase == "pending" or (
                phase is None
                and index not in legacy_terminal
            )
            if is_pending:
                verify_pending += 1
                continue
            verify_total += 1
            result = record.get("result")
            success = bool(result.get("ok")) if isinstance(result, Mapping) else False
            if not success:
                verify_failed += 1
            target = record.get("target")
            if (
                success
                and isinstance(target, Mapping)
                and target.get("type") == HUNT_CONFIRM_TYPE
            ):
                hunts_confirmed += 1
        elif kind == "retry_exhausted":
            retry_exhausted += 1
        elif kind == "recovery":
            recoveries += 1
        elif kind == "session":
            sessions += 1

    span = _span_seconds(stamps)
    hunts_per_hour = (
        round(hunts_confirmed / span * 3600, 2)
        if span >= _MINIMUM_SPAN_SECONDS
        else None
    )
    detect_latency = _latency(detect_ms)

    return {
        "available": True,
        "events": len(events),
        "cycles": len(cycles),
        "span_seconds": round(span, 1),
        "sessions": sessions,
        "recoveries": recoveries,
        "hunts_confirmed": hunts_confirmed,
        # None when the window is too short to divide by: a 40-second tail
        # showing one hunt is not 90 an hour.
        "hunts_per_hour": hunts_per_hour,
        "planning_cycles": planning_cycles,
        "action_cycles": len(action_cycles),
        "action_rate": (
            round(len(action_cycles) / planning_cycles, 3) if planning_cycles else None
        ),
        "stage_cycles": dict(sorted(stage_counts.items(), key=lambda item: -item[1])),
        "rejections": dict(sorted(rejections.items(), key=lambda item: -item[1])),
        "capture_ms": _latency(capture_ms),
        "detect_ms": detect_latency,
        "verify": {
            "checks_total": verify_checks_total,
            "pending": verify_pending,
            "total": verify_total,
            "failed": verify_failed,
            "failure_rate": (
                round(verify_failed / verify_total, 3) if verify_total else None
            ),
            "retry_exhausted": retry_exhausted,
        },
        "suggestions": _suggestions(
            stage_counts,
            planning_cycles,
            rejections,
            verify_total,
            verify_failed,
            retry_exhausted,
            detect_latency,
        ),
    }
