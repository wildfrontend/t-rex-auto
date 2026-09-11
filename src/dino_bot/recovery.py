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

    def request_restart(
        self,
        reason: str,
        *,
        reason_key: str,
        bypass_cooldown: bool = False,
    ) -> bool:
        """Restart the configured app while sharing one cross-cause cooldown."""

        now = self.clock()
        if (
            not bypass_cooldown
            and self._last_restart_at is not None
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
        self._black_since = None
        self._is_black = False
        self._deferred_reasons.clear()
        self.logger.info(
            "Recovery | game restarted; waiting %.0fs for launch",
            self.launch_wait_seconds,
        )
        if self.launch_wait_seconds:
            self.sleeper(self.launch_wait_seconds)
        return True


class HuntProgressWatchdog:
    """Restart the app after sustained lack of *completed* hunts.

    Progress is the confirmed-hunt event reported by ``on_hunt_completed``, not
    anything visible on screen. Screen-derived proxies all fail the same way: a
    map always shows dinosaurs, so "a dinosaur is visible" stayed true through
    fourteen minutes of zero hunts, and a stuck ``startup_*`` phantom held the
    suspend list open for the entire deadlock it was causing.

    The one exception is the game refusing a completed attempt - see
    ``_answered_wait``. That is a reply, not a screen state, and it proves the
    whole loop works; a refusal cannot be cleared by restarting the app.
    Everything else only freezes the timer, so a wait that never ends still
    ages out.
    """

    _EXPECTED_WAIT_TYPES = frozenset(
        {
            "no_available_dinosaurs",
            "target_too_strong",
            "hunt_capacity_full",
            "hunt_team_return_button",
        }
    )
    # ``hatch_`` joins these because the combined modes spend legitimate
    # minutes inside screening and placement, where no hunt can complete by
    # definition: s9 restarted five times in ten minutes mid-auto-place, each
    # time one second after a verified tap. The budget ceiling below is what
    # keeps this from becoming the ``startup_`` phantom the docstring warns
    # about - an incubator that never leaves the screen still ages out.
    _SUSPENDED_PREFIXES = ("mail_", "startup_", "hatch_")
    _SUSPENDED_TYPES = frozenset(
        {
            "duplicate_login_close_button",
            "device_history_confirm_button",
            "server_error_restart_button",
        }
    )

    def __init__(
        self,
        runtime_recovery: BlackScreenRecovery,
        logger: logging.Logger,
        *,
        timeout_seconds: float = 180.0,
        suspend_budget_seconds: float = 120.0,
        hatch_suspend_budget_seconds: float = 420.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime_recovery = runtime_recovery
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        # A suspend reason may only mute the watchdog for this long in total
        # before the timer resumes. Without a ceiling, an exempt type that
        # never goes away disables recovery outright.
        self.suspend_budget_seconds = max(0.0, suspend_budget_seconds)
        # Wide enough for a screening pass plus the navigation either side,
        # still far short of the seven-hour blind run this watchdog exists to
        # end.
        self.hatch_suspend_budget_seconds = max(0.0, hatch_suspend_budget_seconds)
        self.clock = clock
        self._stalled_since: float | None = None
        self._suspended_since: float | None = None
        self._warned = False

    def reset(self) -> None:
        self._stalled_since = None
        self._suspended_since = None
        self._warned = False

    def on_hunt_completed(self) -> None:
        """The only event that counts as progress."""

        self.reset()

    def _answered_wait(self, observed_types: set[str]) -> str | None:
        """The game answering "not now" - proof the bot is not stuck at all.

        These are replies to an action the bot completed: the prey is too
        strong, or every dinosaur is still on cooldown. Reaching one means
        capture, detection, tapping and the game's own response all work, so
        the timer restarts rather than merely freezing. Restarting the app
        cannot shorten a cooldown or weaken a target; s9 restarted twice in
        four minutes against exactly this, interrupting live work to fix
        nothing. A genuine deadlock shows none of these types, so the
        watchdog still fires there.
        """

        matched = observed_types & self._EXPECTED_WAIT_TYPES
        return ", ".join(sorted(matched)) if matched else None

    def _suspend_reason(self, observed_types: set[str]) -> str | None:
        matched = observed_types & self._SUSPENDED_TYPES
        if matched:
            return ", ".join(sorted(matched))
        prefixed = sorted(
            observed
            for observed in observed_types
            for prefix in self._SUSPENDED_PREFIXES
            if observed.startswith(prefix)
        )
        if prefixed:
            return ", ".join(prefixed)
        return None

    def observe(
        self,
        detections: Sequence[Detection],
        target: Target | None,
        *,
        cooldown_ms: int = 0,
    ) -> bool:
        if self.timeout_seconds <= 0:
            return False

        now = self.clock()
        if cooldown_ms > 0:
            # The planner set this deadline itself and it expires on its own,
            # so it is a bounded wait rather than an inference that might be
            # wrong. Hold the timer without spending the suspend budget - a
            # legitimate five-minute capacity wait would otherwise outlast the
            # budget and trigger a restart at the moment it was about to end.
            #
            # Clearing the stall origin is what actually holds it. Returning
            # early only skips the check: the elapsed time is measured from
            # `_stalled_since` in wall clock, so a five-minute wait still ages
            # the timer past its timeout and fires the restart three seconds
            # after the wait it was supposed to protect. The bot needs a full
            # timeout of real hunting before a stall is credible again, and
            # restarting the game does not shorten a cooldown anyway.
            self._suspended_since = None
            self._stalled_since = None
            self._warned = False
            return False

        visible_types = {item.type for item in detections}
        target_type = target.type if target is not None else None
        observed_types = visible_types | ({target_type} if target_type else set())

        answered = self._answered_wait(observed_types)
        if answered is not None:
            if self._stalled_since is not None:
                self.logger.info(
                    "Recovery | game answered the hunt attempt; stall timer"
                    " reset | reason=%s",
                    answered,
                )
            self.reset()
            return False

        reason = self._suspend_reason(observed_types)
        if reason is not None:
            if self._suspended_since is None:
                self._suspended_since = now
            suspended_seconds = now - self._suspended_since
            # A hatch phase is long by nature - one measured screening pass ran
            # 155s - while the hunt-side waits this budget was written for stay
            # short. Giving hatch its own ceiling keeps that original limit
            # honest instead of loosening it for everything.
            budget = (
                self.hatch_suspend_budget_seconds
                if any(
                    observed.startswith("hatch_") for observed in observed_types
                )
                else self.suspend_budget_seconds
            )
            if suspended_seconds < budget:
                # Genuine waits are short. Keep the stall timer frozen rather
                # than reset, so a wait that never ends still ages out.
                return False
            self.logger.warning(
                "Recovery | suspend budget exhausted after %.0fs | reason=%s",
                suspended_seconds,
                reason,
            )
        else:
            self._suspended_since = None

        if self._stalled_since is None:
            self._stalled_since = now
            return False

        stalled_seconds = now - self._stalled_since
        if stalled_seconds < self.timeout_seconds:
            if not self._warned and stalled_seconds >= self.timeout_seconds / 2:
                self._warned = True
                self.logger.warning(
                    "Recovery | no confirmed hunt for %.0fs; watchdog=%.0fs",
                    stalled_seconds,
                    self.timeout_seconds,
                )
            return False

        restarted = self.runtime_recovery.request_restart(
            f"no confirmed hunt for {stalled_seconds:.0f}s",
            reason_key="no_hunt_progress",
        )
        if restarted:
            self.reset()
        return restarted
