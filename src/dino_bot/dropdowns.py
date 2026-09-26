"""Dropdown fail-safe convergence (plan §9 T7).

The tag filter and sort-field dropdowns remember their previous selection, so
every visit must drive them to a known state before reading anything. The
screens involved (my-nest, select-dino, auto-place) all use the same widget:
a collapsed header showing the current option, which expands into a vertical
option list when tapped.

This module is pure decision logic over template sightings. The caller crops
option-text templates for the labels it cares about, detects them on the
current frame, tags each hit as inside or outside the header box, and asks
``next_step`` what to do. Convergence is reactive, like HatchPlanner: each
frame independently maps to one action, so dropped taps or an unexpected
initial state simply re-derive the next move instead of derailing a script.

Only the labels the rounds actually target need templates; any unrecognized
current selection just means the header shows no known label, and the answer
to that is always "open the menu and look".
"""

from __future__ import annotations

from dataclasses import dataclass

# Step kinds returned by next_step.
DONE = "done"  # header already shows the target label
OPEN = "open"  # tap the header to expand the menu
SELECT = "select"  # tap the target option in the expanded menu
STUCK = "stuck"  # menu is open but the target is not visible: abort, report


@dataclass(frozen=True, slots=True)
class Sighting:
    """One detected option-label template on the current frame."""

    label: str
    x: int
    y: int
    in_header: bool


@dataclass(frozen=True, slots=True)
class Step:
    kind: str
    point: tuple[int, int] | None = None
    reason: str = ""


def next_step(
    target: str,
    sightings: list[Sighting],
    header_point: tuple[int, int],
) -> Step:
    """Decide the next tap toward "collapsed header shows ``target``".

    Menu sightings (outside the header) mean the dropdown is expanded: tap
    the target if it is visible, otherwise report STUCK — the caller decides
    whether that is a scroll problem or a missing template, and a blind tap
    would land on an arbitrary option. With the menu closed, a header hit on
    the target means done; anything else (another label, or nothing
    recognized) is resolved the same way: open the menu.
    """

    menu = [item for item in sightings if not item.in_header]
    if menu:
        for item in menu:
            if item.label == target:
                return Step(SELECT, (item.x, item.y))
        seen = ",".join(sorted({item.label for item in menu}))
        return Step(STUCK, reason=f"expanded menu lacks {target} (visible: {seen})")
    if any(item.label == target for item in sightings):
        return Step(DONE)
    return Step(OPEN, header_point)


def converged(target: str, sightings: list[Sighting]) -> bool:
    """Whether the dropdown is collapsed and showing ``target``."""

    return next_step(target, sightings, (0, 0)).kind == DONE
