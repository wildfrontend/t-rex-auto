"""Runtime recovery for a Unity app that stops submitting rendered frames."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .actions import AdbClient, AdbError
from .models import Frame


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
        cooldown_seconds: float = 300.0,
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

    def observe(self, frame: Frame) -> bool:
        # Sampling keeps this guard cheap on a 900x1600 frame while still
        # distinguishing a true black render surface from a dark game scene.
        sample = frame.image[::8, ::8]
        mean = float(np.mean(sample))
        now = self.clock()
        if mean > self.mean_threshold:
            if self._black_since is not None:
                self.logger.info(
                    "Recovery | picture returned before timeout | mean=%.2f",
                    mean,
                )
            self._black_since = None
            return False

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

        if (
            self._last_restart_at is not None
            and now - self._last_restart_at < self.cooldown_seconds
        ):
            return False

        self.logger.error(
            "Recovery | black screen persisted %.0fs; restarting game app",
            black_duration,
        )
        self._black_since = None
        try:
            self.restarter.restart()
        except AdbError as exc:
            # Keep the Bot alive if ADB briefly disconnects. A fresh sustained
            # black interval will retry instead of terminating the whole run.
            self.logger.error("Recovery | game restart failed: %s", exc)
            self._black_since = now
            return False
        self._last_restart_at = now
        if self.launch_wait_seconds:
            self.logger.info(
                "Recovery | game restarted; waiting %.0fs for launch",
                self.launch_wait_seconds,
            )
            self.sleeper(self.launch_wait_seconds)
        return True
