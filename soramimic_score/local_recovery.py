"""Bounded local Whisper retries for likely missing sung lyrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import re
import math
import statistics
from typing import TYPE_CHECKING
import unicodedata

from .japanese import kana_to_moras, katakana

if TYPE_CHECKING:
    from .audio import LyricLine, MelodyNote


def _normalized(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if unicodedata.category(char)[0] not in {"P", "Z"})


def coalesce_repeated_suffix_fragments(lines: Sequence[LyricLine]
                                       ) -> tuple[tuple[LyricLine, ...], tuple[tuple[int, int], ...]]:
    """Join a short Whisper suffix only if another refrain contains that suffix."""
    normalized = [_normalized(line.text) for line in lines]
    candidates = set()
    for index in range(1, len(lines)):
        line, previous = lines[index], lines[index - 1]
        fragment = normalized[index]
        if not re.fullmatch(r"[ぁ-んァ-ヶ一-龯々〆ヵヶー]{2,4}", fragment):
            continue
        if (line.start_sec is None or line.end_sec is None
                or previous.start_sec is None or previous.end_sec is None
                or line.end_sec - line.start_sec > 1.25
                or abs(line.start_sec - previous.end_sec) > .15
                or line.end_sec - previous.start_sec > 6):
            continue
        combined = len(normalized[index - 1]) + len(fragment)
        if any(other not in {index - 1, index} and len(surface) > len(fragment)
               and surface.endswith(fragment)
               and .67 <= combined / len(surface) <= 1.5
               for other, surface in enumerate(normalized)):
            candidates.add(index)
    merged, evidence = [], []
    index = 0
    while index < len(lines):
        if index + 1 in candidates:
            left, right = lines[index:index + 2]
            separator = " " if left.text[-1:].isascii() and right.text[:1].isascii() else ""
            merged.append(replace(left, text=left.text.rstrip() + separator + right.text.lstrip(),
                                  end_sec=right.end_sec))
            evidence.append((index, index + 1))
            index += 2
        else:
            merged.append(lines[index])
            index += 1
    return tuple(merged), tuple(evidence)


_VOCALIZATION_KANA = frozenset("アァイィウゥエェオォラナダファハヤワー")
_LATIN_VOCALIZATION = re.compile(r"wow|la|na|da|fa|ha|ya|a+h*|i+|u+h*|e+|o+h*")
_LATIN_MORAS = {"wow": ("ワ", "ウ"), "la": ("ラ",), "na": ("ナ",),
                "da": ("ダ",), "fa": ("ファ",), "ha": ("ハ",), "ya": ("ヤ",)}


def repeated_vocalization_period(text: str) -> tuple[str, ...] | None:
    """Find a short repeating pure vocalization, excluding ordinary lyrics."""
    normalized = katakana(_normalized(text)).replace("ー", "")
    if not normalized:
        return None
    if re.fullmatch(r"[ァ-ヶ]+", normalized):
        if any(char not in _VOCALIZATION_KANA for char in normalized):
            return None
        moras = list(kana_to_moras(normalized))
    elif normalized.isascii():
        moras = []
        cursor = 0
        while cursor < len(normalized):
            match = _LATIN_VOCALIZATION.match(normalized, cursor)
            if match is None:
                return None
            token = match.group()
            moras.extend(_LATIN_MORAS[token] if token in _LATIN_MORAS else
                         ({"a": "ア", "i": "イ", "u": "ウ",
                           "e": "エ", "o": "オ"}[token[0]],))
            cursor = match.end()
    else:
        return None
    for width in range(1, min(3, len(moras) // 2) + 1):
        if width > 1 and len(moras) < width * 3:
            continue
        if all(mora == moras[index % width] for index, mora in enumerate(moras)):
            return tuple(moras[:width])
    return None


def expand_repeated_vocalization_from_kana(
    line: LyricLine, evidence_text: str, notes: Sequence[MelodyNote],
) -> LyricLine | None:
    """Use KanaWhisper for count only; spell every mora from Whisper's period."""
    period = repeated_vocalization_period(line.text)
    if period is None or line.start_sec is None or line.end_sec is None:
        return None
    source = kana_to_moras(katakana(line.text))
    if not source:
        source = tuple(period[index % len(period)] for index in range(
            max(2, len(_normalized(line.text)) // 2)))
    evidence = kana_to_moras("".join(re.findall(r"[ァ-ヶー]+",
                              katakana(evidence_text).replace("ヲ", "オ"))).replace("ー", ""))
    note_count = sum(line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec
                     for note in notes)
    if not evidence or note_count < 2:
        return None
    gain = len(source) + max(2, math.ceil(len(source) * .25))
    if (len(evidence) < gain or len(evidence) < math.ceil(note_count * .25)
            or len(evidence) > math.ceil(note_count * 1.5)
            or len(evidence) / max(1e-6, line.end_sec - line.start_sec) > 8):
        return None
    similarity = max(
        sum(mora == period[(index + phase) % len(period)]
            for index, mora in enumerate(evidence)) / len(evidence)
        for phase in range(len(period))
    )
    if similarity < .65:
        return None
    return replace(line, text="".join(period[index % len(period)]
                                      for index in range(len(evidence))))


def deficit_windows(lines: Sequence[LyricLine], notes: Sequence[MelodyNote],
                    mora_counts: Sequence[int]) -> tuple[tuple[int, float, float], ...]:
    """Find long note rich lines whose lyric detail is short for this song."""
    if len(lines) != len(mora_counts):
        raise ValueError("one mora count is required for every line")
    rows = []
    for index, (line, count) in enumerate(zip(lines, mora_counts, strict=True)):
        if line.start_sec is None or line.end_sec is None:
            continue
        in_line = [note for note in notes
                   if line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec]
        effective = count + len(re.findall(r"[A-Za-z]+", line.text))
        rows.append((index, line, count, effective, in_line))
    ratios = [len(found) / effective for _, _, _, effective, found in rows
              if effective >= 4 and len(found) >= 2]
    if not ratios:
        return ()
    median = statistics.median(ratios)
    windows = []
    for index, line, count, effective, found in rows:
        if (count < 2 or len(found) < 7 or len(found) / max(effective, 1) < 1.5
                or len(found) - median * effective < 4):
            continue
        for first in range(0, len(found), 24):
            group = found[first:first + 24]
            start = max(line.start_sec, group[0].start_sec - .5)
            end = min(line.end_sec, group[-1].end_sec + .5)
            if end - start >= 1.5 and end - start <= 20:
                windows.append((index, start, end))
    return tuple(windows[:4])


def uncovered_note_windows(lines: Sequence[LyricLine], notes: Sequence[MelodyNote]
                           ) -> tuple[tuple[float, float], ...]:
    """Find long melody runs with no overlapping recognized lyric line."""
    uncovered = [note for note in notes if not any(
        line.start_sec is not None and line.end_sec is not None
        and line.start_sec < note.end_sec and line.end_sec > note.start_sec
        for line in lines)]
    groups: list[list[MelodyNote]] = []
    for note in uncovered:
        crosses_line = bool(groups) and any(
            line.start_sec is not None and line.end_sec is not None
            and line.start_sec < note.start_sec
            and line.end_sec > groups[-1][-1].end_sec
            for line in lines)
        if not groups or note.start_sec - groups[-1][-1].end_sec > .35 or crosses_line:
            groups.append([note])
        else:
            groups[-1].append(note)
    windows = []
    for group in groups:
        start, end = group[0].start_sec, group[-1].end_sec
        if len(group) < 5 or end - start < 1.5 or end - start > 20:
            continue
        left = max((line.end_sec for line in lines if line.end_sec is not None
                    and line.end_sec <= start), default=0.)
        right = min((line.start_sec for line in lines if line.start_sec is not None
                     and line.start_sec >= end), default=end + .3)
        windows.append((max(left, start - .3), min(right, end + .3)))
    return tuple(windows[:4])
