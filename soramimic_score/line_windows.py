"""Place adjacent ASR line boundaries at nearby melody rests."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import MelodyNote


def snap_line_windows_to_rests(
    windows: Sequence[tuple[float, float]],
    notes: Sequence[MelodyNote],
) -> tuple[tuple[float, float], ...]:
    """Use a nearby SheetSage rest midpoint for each adjoining Whisper boundary."""
    starts = [float(start) for start, _ in windows]
    ends = [float(end) for _, end in windows]
    rests = []
    for left, right in zip(notes, notes[1:]):
        duration = right.start_sec - left.end_sec
        if duration >= .08:
            rests.append(((left.end_sec + right.start_sec) / 2, duration))

    previous_boundary = -math.inf
    for index in range(len(windows) - 1):
        target = (windows[index][1] + windows[index + 1][0]) / 2
        candidates = [(abs(midpoint - target), -duration, midpoint)
                      for midpoint, duration in rests
                      if abs(midpoint - target) <= .75]
        chosen = min(candidates, default=None)
        if chosen is None:
            previous_boundary = max(previous_boundary, ends[index])
            continue
        boundary = chosen[2]
        if (boundary <= starts[index] or boundary >= ends[index + 1]
                or boundary <= previous_boundary):
            previous_boundary = max(previous_boundary, ends[index])
            continue
        ends[index] = boundary
        starts[index + 1] = boundary
        previous_boundary = boundary
    snapped = tuple(zip(starts, ends, strict=True))
    if any(start > end for start, end in snapped):
        raise ValueError("melody-rest snapping reversed a lyric line")
    if any(right[0] < left[1] for left, right in zip(snapped, snapped[1:])):
        raise ValueError("melody-rest snapping overlapped lyric lines")
    return snapped
