"""Conservative checks for credit-like Whisper text in singing audio."""

from __future__ import annotations

from collections.abc import Sequence
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import LyricLine, MelodyNote


_LABEL = r"(?:作詞|作曲|編曲|原作|監督|制作|製作|出演|翻訳|歌唱|動画制作|イラスト)"
_VALUE = r"[0-9a-zA-Zぁ-んァ-ヶ一-龯々〆ヵヶー@._・]{1,32}"
_SINGLE = re.compile(
    rf"^\s*(?:{_LABEL}|サブタイトル)(?:担当|協力|提供|制作|作成)?"
    rf"(?:\s*[:：/／|｜]\s*|\s+){_VALUE}\s*$"
)
_COMPOUND = re.compile(
    rf"^\s*{_LABEL}(?:\s*[・･/&＆,，、／|｜]\s*{_LABEL})+"
    rf"\s*(?:[:：+＋]|\s)\s*{_VALUE}\s*$"
)
_SUBTITLES = re.compile(r"^\s*(?:字幕(?:制作|製作|提供|翻訳)?|翻訳[・/ ]?字幕)(?:\s*[:：]\s*\S+)?\s*$")
_ENGLISH = re.compile(
    rf"^\s*(?:(?:lyrics|music|arrangement|vocal)\s*[:：]\s*{_VALUE}"
    rf"|subtitles?\s+by\s+{_VALUE})\s*$", re.I,
)


def is_credit_hallucination(text: str) -> bool:
    """Match a whole credit template, leaving lyrics containing role words intact."""
    value = unicodedata.normalize("NFKC", text).strip()
    return bool(_SINGLE.fullmatch(value) or _COMPOUND.fullmatch(value)
                or _SUBTITLES.fullmatch(value) or _ENGLISH.fullmatch(value))


def credit_recovery_windows(line: LyricLine, notes: Sequence[MelodyNote]) -> tuple[tuple[float, float], ...]:
    """Find substantial singing islands inside a credit-like ASR segment."""
    if not is_credit_hallucination(line.text) or line.start_sec is None or line.end_sec is None:
        return ()
    intervals = sorted(
        (max(line.start_sec, note.start_sec), min(line.end_sec, note.end_sec))
        for note in notes
        if note.end_sec > line.start_sec and note.start_sec < line.end_sec
    )
    intervals = [(start, end) for start, end in intervals if end > start]
    if not intervals:
        return ()
    islands = []
    start, end = intervals[0]
    melody_seconds = end - start
    for note_start, note_end in intervals[1:]:
        if note_start - end <= 1.0:
            melody_seconds += max(0.0, note_end - max(note_start, end))
            end = max(end, note_end)
        else:
            if end - start >= 1.5 and melody_seconds >= .5:
                islands.append((start, end))
            start, end = note_start, note_end
            melody_seconds = end - start
    if end - start >= 1.5 and melody_seconds >= .5:
        islands.append((start, end))
    return tuple(islands)


def has_melodic_support(line: LyricLine, notes: Sequence[MelodyNote]) -> bool:
    if line.start_sec is None or line.end_sec is None:
        return False
    duration = max(0.0, line.end_sec - line.start_sec)
    normalized = unicodedata.normalize("NFKC", line.text).strip()
    required = min(duration * .25, len(normalized) * .05)
    overlap = sum(max(0.0, min(line.end_sec, note.end_sec)
                      - max(line.start_sec, note.start_sec)) for note in notes)
    return required > 0 and overlap >= required
