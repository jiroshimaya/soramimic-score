"""Transcript-independent CTC peaks for repeated sung morae."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .japanese import katakana
from .vocalization import VocalizationReattack


def _prominent_peaks(values: Sequence[float]) -> list[int]:
    """Use the same prominence and three-frame separation as Video."""
    floor = min(values)
    prominence = max(.01, max(values) * .03)
    candidates = []
    for index, value in enumerate(values):
        left = values[index - 1] if index else floor
        right = values[index + 1] if index + 1 < len(values) else floor
        if value < left or value <= right:
            continue
        left_base = floor if index == 0 else value
        cursor = index - 1
        while cursor >= 0 and values[cursor] <= value:
            left_base = min(left_base, values[cursor])
            cursor -= 1
        right_base = floor if index + 1 == len(values) else value
        cursor = index + 1
        while cursor < len(values) and values[cursor] <= value:
            right_base = min(right_base, values[cursor])
            cursor += 1
        if value - max(left_base, right_base) >= prominence:
            candidates.append(index)
    selected = []
    for index in sorted(candidates, key=lambda item: values[item], reverse=True):
        if all(abs(index - other) >= 3 for other in selected):
            selected.append(index)
    return sorted(selected)


def decode_repeated_mora_reattacks(log_probs, token_ids: dict[str, int],
                                   mora: str, start: float, end: float,
                                   *, stride: int, rate: int) -> tuple[VocalizationReattack, ...]:
    """Read posterior peaks without conditioning on a requested repeat count."""
    if not 0 <= start < end:
        raise ValueError("CTC window must have positive duration")
    target = tuple(katakana(mora))
    if not target or any(char not in token_ids for char in target):
        return ()
    frame_sec = stride / rate
    first = max(0, math.ceil((start + .5) / frame_sec - 1e-9))
    last = min(len(log_probs), math.ceil((end + .5) / frame_sec - 1e-9))
    if first >= last:
        return ()
    events = []
    for char in set(target):
        probabilities = log_probs[first:last, token_ids[char]].exp().tolist()
        for index in _prominent_peaks(probabilities):
            onset = max(start, (first + index) * frame_sec - .5)
            offset = min(end, onset + frame_sec)
            if offset > onset:
                events.append((onset, offset, char, probabilities[index]))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    result = []
    index = 0
    while index <= len(events) - len(target):
        selected = events[index:index + len(target)]
        if tuple(event[2] for event in selected) == target:
            confidence = math.exp(sum(math.log(max(event[3], 1e-300))
                                      for event in selected) / len(selected))
            result.append(VocalizationReattack(selected[0][0], selected[-1][1],
                                               confidence,
                                               "reazon-kana-ctc-target-posterior"))
            index += len(target)
        else:
            index += 1
    return tuple(result)
