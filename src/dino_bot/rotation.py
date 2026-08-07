"""Shared generation rolling for the daily log and the event stream.

Both writers used to keep exactly one generation, which sounded sufficient
until a seven-hour run was worth reading: the event stream fills its 16 MB cap
in about 97 minutes, so the first five and a half hours had already been
overwritten by the time anyone looked. The text log fills 32 MB in roughly two
hours and loses the same way.

Keeping more generations is only affordable because these files compress
extremely well - the event stream to 6.5% and the text log to 4.4%, measured
on real captures - so twenty generations of events cost about 21 MB on disk
rather than 320 MB.

The newest backup is deliberately left uncompressed. Everything that reads the
history back - the diagnostic bundle's ``events-*.jsonl`` and ``20*.log`` globs
- keeps working unchanged, and the compressed generations do not match those
globs, so they are neither read as binary nor shipped by accident.
"""

from __future__ import annotations

import gzip
import shutil
from contextlib import suppress
from pathlib import Path


def generation_path(
    directory: Path,
    stem: str,
    suffix: str,
    index: int,
) -> Path:
    """Locate one generation. 0 is live, 1 is plain, 2 and beyond are gzipped."""

    if index <= 0:
        return directory / f"{stem}{suffix}"
    if index == 1:
        return directory / f"{stem}.1{suffix}"
    return directory / f"{stem}.{index}{suffix}.gz"


def roll_generations(
    directory: Path,
    stem: str,
    suffix: str,
    backup_count: int,
) -> None:
    """Shift every generation down by one, dropping whatever falls off the end.

    Errors are swallowed the same way the callers already swallow them: losing
    a generation is not a reason to take the bot down mid-run.
    """

    keep = max(1, backup_count)
    with suppress(OSError):
        generation_path(directory, stem, suffix, keep).unlink(missing_ok=True)
    for index in range(keep - 1, 1, -1):
        source = generation_path(directory, stem, suffix, index)
        if not source.exists():
            continue
        with suppress(OSError):
            source.replace(generation_path(directory, stem, suffix, index + 1))
    first = generation_path(directory, stem, suffix, 1)
    if keep >= 2 and first.exists():
        second = generation_path(directory, stem, suffix, 2)
        try:
            with first.open("rb") as raw, gzip.open(second, "wb") as packed:
                shutil.copyfileobj(raw, packed)
            first.unlink(missing_ok=True)
        except OSError:
            # A half-written archive is worse than a missing one: the reader
            # cannot tell truncation from corruption.
            with suppress(OSError):
                second.unlink(missing_ok=True)
    try:
        first.unlink(missing_ok=True)
        generation_path(directory, stem, suffix, 0).replace(first)
    except OSError:
        pass
