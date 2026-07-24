"""Runtime recovery for black frames and stalled hunting progress."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from .actions import AdbClient, AdbError
from .models import Detection, Frame, Target


class AppRestarter(Protocol):
    def restart(self) -> None: ...


class AdbAppRestarter:
    def __init__(self, client: AdbClient, package: str, activity: str) -> None:
        self.client = client
        self.package = package
        self.activity = activity

    def restart(self) -> None:
        self.client.run(["shell", "am", "force-stop", self.package])
        self.client.run(
            ["shell", "am", "start", "-n", f"{self.package}/{self.activity}"]
        )


class BlackScreenRecovery:
    """Restarts the game only after a sustained near-black capture."""

    def __init__(
        self,
        restarter: AppRestarter,
        logger: logging.Logger,
        *,
        timeout_seconds: float = 45.0,
        mean_threshold: float = 2.0,
        cooldown_seconds: float = 90.0,
        launch_wait_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.restarter = restarter
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.mean_threshold = mean_threshold
        self.cooldown_seconds = cooldown_seconds
        self.launch_wait_seconds = launch_wait_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._black_since: float | None = None
        self._last_restart_at: float | None = None
        self._is_black = False
        self._deferred_reasons: set[str] = set()

    @property
    def is_black(self) -> bool:
        """Whether the latest captured frame is still near-black."""

        return self._is_black

    def observe(self, frame: Frame) -> bool:
        # Sampling keeps this guard cheap on a 900x1600 frame while still
        # distinguishing a true black render surface from a dark game scene.
        sample = frame.image[::8, ::8]
        mean = float(np.mean(sample))
        now = self.clock()
        if mean > self.mean_threshold:
            self._is_black = False
            if self._black_since is not None:
                self.logger.info(
                    "Recovery | picture returned before timeout | mean=%.2f",
                    mean,
                )
            self._black_since = None
            self._deferred_reasons.discard("black_screen")
            return False

        self._is_black = True
        if self._black_since is None:
            self._black_since = now
            self.logger.warning(
                "Recovery | black screen detected | mean=%.2f | waiting %.0fs",
                mean,
                self.timeout_seconds,
            )
            return False

        black_duration = now - self._black_since
        if black_duration < self.timeout_seconds:
            return False

        restarted = self.request_restart(
            f"black screen persisted {black_duration:.0f}s",
            reason_key="black_screen",
        )
        if restarted:
            self._black_since = None
        return restarted

    def request_restart(self, reason: str, *, reason_key: str) -> bool:
        """Restart the configured app while sharing one cross-cause cooldown."""

        now = self.clock()
        if (
            self._last_restart_at is not None
            and now - self._last_restart_at < self.cooldown_seconds
        ):
            if reason_key not in self._deferred_reasons:
                remaining = self.cooldown_seconds - (now - self._last_restart_at)
                self.logger.warning(
                    "Recovery | restart deferred by cooldown | "
                    "reason=%s | remaining=%.0fs",
                    reason_key,
                    remaining,
                )
                self._deferred_reasons.add(reason_key)
            return False

        self.logger.error("Recovery | %s; restarting game app", reason)
        try:
            self.restarter.restart()
        except AdbError as exc:
            self.logger.error("Recovery | game restart failed: %s", exc)
            return False
        self._last_restart_at = now
        self._deferred_reasons.clear()
        if self.launch_wait_seconds:
            self.logger.info(
                "Recovery | game restarted; waiting %.0fs for launch",
                self.launch_wait_seconds,
            )
            self.sleeper(self.launch_wait_seconds)
        return True


class HuntProgressWatchdog:
    """Restart the app after sustained lack of hunting progress."""

    _PROGRESS_TYPES = frozenset(
        {"dinosaur", "hunt_button", "hunt_max_group_button", "hunt_confirm_button"}
    )
    _EXPECTED_WAIT_TYPES = frozenset(
        {
            "no_available_dinosaurs",
            "target_too_strong",
            "hunt_capacity_full",
            "hunt_team_return_button",
        }
    )
    _SUSPENDED_PREFIXES = ("mail_", "startup_")
    _SUSPENDED_TYPES = frozenset(
        {"duplicate_login_close_button", "device_history_confirm_button"}
    )

    def __init__(
        self,
        runtime_recovery: BlackScreenRecovery,
        logger: logging.Logger,
        *,
        timeout_seconds: float = 180.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime_recovery = runtime_recovery
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self._stalled_since: float | None = None

    def reset(self) -> None:
        self._stalled_since = None

    def observe(
        self,
        detections: Sequence[Detection],
        target: Target | None,
        *,
        cooldown_ms: int = 0,
    ) -> bool:
        if self.timeout_seconds <= 0:
            return False

        visible_types = {item.type for item in detections}
        target_type = target.type if target is not None else None
        observed_types = visible_types | ({target_type} if target_type else set())
        if (
            cooldown_ms > 0
            or observed_types & self._EXPECTED_WAIT_TYPES
            or observed_types & self._SUSPENDED_TYPES
            or any(
                target_type.startswith(prefix)
                for target_type in observed_types
                for prefix in self._SUSPENDED_PREFIXES
            )
        ):
            self.reset()
            return False

        if target_type in self._PROGRESS_TYPES:
            self.reset()
            return False

        now = self.clock()
        if self._stalled_since is None:
            self._stalled_since = now
            self.logger.warning(
                "Recovery | no hunt progress detected; watchdog=%.0fs",
                self.timeout_seconds,
            )
            return False

        stalled_seconds = now - self._stalled_since
        if stalled_seconds < self.timeout_seconds:
            return False

        restarted = self.runtime_recovery.request_restart(
            f"no hunt progress for {stalled_seconds:.0f}s",
            reason_key="no_hunt_progress",
        )
        if restarted:
            self.reset()
        return restarted
