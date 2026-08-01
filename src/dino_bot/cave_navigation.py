"""Fail-safe cave-view navigation for hatch plan T9.

The calibrated motion is expressed in 900-wide reference coordinates.  It
moves the camera down and left (finger moves up, then right) from the game's
default home view.  The planner never taps the cave: reaching a frame where
the cave template is visible is the terminal read-only state used to read the
N/350 counter.
"""

from __future__ import annotations

from dataclasses import dataclass

CAVE = "hatch_cave"

SWIPE = "swipe"
RESCAN = "rescan"
DONE = "done"
STUCK = "stuck"

DEFAULT_SWIPE_VECTORS: tuple[tuple[int, int, int, int], ...] = (
    (450, 1050, 450, 600),
    (350, 800, 600, 800),
)


@dataclass(frozen=True, slots=True)
class NavigationStep:
    kind: str
    vector: tuple[int, int, int, int] | None = None
    reason: str = ""


class CaveNavigator:
    """Advance through calibrated swipes, retrying only dropped gestures.

    Repeating a successful camera swipe would overshoot the cave, so a swipe
    is retried only when the caller reports that the frame did not move.
    After both successful gestures, a bounded number of detection rescans is
    allowed before returning ``STUCK`` for logging/recovery.
    """

    def __init__(
        self,
        *,
        swipe_vectors: tuple[tuple[int, int, int, int], ...] = DEFAULT_SWIPE_VECTORS,
        max_swipe_failures: int = 2,
        max_rescans: int = 2,
        reference_width: float = 900.0,
    ) -> None:
        if not swipe_vectors:
            raise ValueError("swipe_vectors must not be empty")
        if reference_width <= 0:
            raise ValueError("reference_width must be greater than zero")
        self.swipe_vectors = swipe_vectors
        self.max_swipe_failures = max(0, max_swipe_failures)
        self.max_rescans = max(0, max_rescans)
        self.reference_width = reference_width
        self._swipe_index = 0
        self._swipe_failures = 0
        self._rescans = 0

    def next_step(self, *, cave_visible: bool, frame_width: int = 900) -> NavigationStep:
        if cave_visible:
            return NavigationStep(DONE, reason="cave template visible")
        if self._swipe_index < len(self.swipe_vectors):
            return NavigationStep(
                SWIPE,
                self._scaled(self.swipe_vectors[self._swipe_index], frame_width),
                f"camera move {self._swipe_index + 1}/{len(self.swipe_vectors)}",
            )
        if self._rescans < self.max_rescans:
            self._rescans += 1
            return NavigationStep(RESCAN, reason=f"cave rescan {self._rescans}/{self.max_rescans}")
        return NavigationStep(STUCK, reason="cave absent after calibrated move and rescans")

    def on_swipe_result(self, *, moved: bool) -> None:
        """Record whether the last requested swipe changed the camera frame."""

        if self._swipe_index >= len(self.swipe_vectors):
            return
        if moved:
            self._swipe_index += 1
            self._swipe_failures = 0
            return
        self._swipe_failures += 1
        if self._swipe_failures > self.max_swipe_failures:
            # Exhausted retries are represented by moving directly to the
            # terminal scan phase, which then returns a bounded STUCK result.
            self._swipe_index = len(self.swipe_vectors)
            self._rescans = self.max_rescans

    def _scaled(
        self,
        vector: tuple[int, int, int, int],
        frame_width: int,
    ) -> tuple[int, int, int, int]:
        scale = frame_width / self.reference_width
        return tuple(round(value * scale) for value in vector)  # type: ignore[return-value]
