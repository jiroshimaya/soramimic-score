"""Whole-line reconciliation of supplied lyrics against an audio transcript."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
import math
from typing import TYPE_CHECKING
import unicodedata

from .japanese import katakana

if TYPE_CHECKING:
    from .audio import LyricLine


@dataclass(frozen=True)
class LyricAdjustment:
    lines: tuple[LyricLine, ...]
    detail: dict


def _normalize(text: str) -> str:
    return "".join(char for char in katakana(text)
                   if unicodedata.category(char)[0] not in {"P", "Z", "C"})


def adjust_known_lyrics(
    supplied: Sequence[str],
    recognized: Sequence[LyricLine],
    *,
    reading: Callable[[str], str] | None = None,
    min_similarity: float = .65,
    max_group_lines: int = 4,
) -> LyricAdjustment:
    """Keep matched supplied lines verbatim, omit absent lines and add heard lines.

    Recognition determines order and repetition, not the spelling of matched lines.
    Adjacent groups accommodate different ASR line breaks; words within supplied
    lines are never spliced. ``reading`` supplies a linguistic pronunciation, not
    an audio-conditioned reading. Without it, normalized surface text is compared.
    Similarity is a matching heuristic, not a calibrated probability. At least one
    supplied line must match; unrelated transcripts fail instead of replacing all
    input. The caller remains responsible for rejecting ASR hallucinations.
    """
    from .audio import _validate_lines

    if isinstance(supplied, (str, bytes)) or not supplied:
        raise ValueError("supplied lyrics must be a nonempty sequence of lines")
    if any(not isinstance(line, str) or not line.strip() for line in supplied):
        raise ValueError("supplied lyric lines must contain text")
    recognized = _validate_lines(recognized, timed=True)
    if not math.isfinite(min_similarity) or not 0 < min_similarity <= 1:
        raise ValueError("min_similarity must be in (0, 1]")
    if type(max_group_lines) is not int or max_group_lines < 1:
        raise ValueError("max_group_lines must be a positive integer")
    supplied = tuple(supplied)
    convert = reading or (lambda text: text)
    cache: dict[str, str] = {}

    def key(text):
        if text not in cache:
            cache[text] = _normalize(convert(text))
        if not cache[text]:
            raise ValueError("lyric comparison needs nonempty text or pronunciation")
        return cache[text]

    expected = tuple(key(line) for line in supplied)
    heard = tuple(key(line.text) for line in recognized)

    def groups(parts):
        return [(start, end, "".join(parts[start:end]))
                for start in range(len(parts))
                for end in range(start + 1, min(len(parts), start + max_group_lines) + 1)]

    def covered(parts, start, end, matched):
        cursor = 0
        for part in parts[start:end]:
            stop = cursor + len(part)
            if sum(cursor <= index < stop for index in matched) / len(part) < min_similarity:
                return False
            cursor = stop
        return True

    candidates: dict[int, list[tuple[int, int, int, float]]] = {}
    expected_groups = groups(expected)
    for start, end, observed in groups(heard):
        for first, last, wanted in expected_groups:
            matcher = SequenceMatcher(None, wanted, observed, autojunk=False)
            if matcher.real_quick_ratio() < min_similarity or matcher.quick_ratio() < min_similarity:
                continue
            similarity = matcher.ratio()
            if similarity < min_similarity:
                continue
            blocks = matcher.get_matching_blocks()
            left = {i for block in blocks for i in range(block.a, block.a + block.size)}
            right = {i for block in blocks for i in range(block.b, block.b + block.size)}
            if not (covered(expected, first, last, left)
                    and covered(heard, start, end, right)):
                continue
            candidates.setdefault(start, []).append((end, first, last, similarity))

    # Optimize complete ASR coverage, preferring simpler groups on equal scores.
    # Supplied lines may be reused: the recording, rather than the input file,
    # determines how many times a chorus is sung.
    best: list[tuple[float, int]] = [(0.0, 0)] * (len(heard) + 1)
    choices: dict[int, tuple[int, int | None, int | None, float]] = {}
    for start in range(len(heard) - 1, -1, -1):
        best[start] = best[start + 1]
        choices[start] = (start + 1, None, None, 0.0)
        for end, first, last, similarity in candidates.get(start, []):
            score = (
                sum(map(len, heard[start:end])) * similarity + best[end][0],
                best[end][1] - (last - first + end - start - 2),
            )
            if score > best[start]:
                best[start] = score
                choices[start] = (end, first, last, similarity)

    output = []
    decisions = []
    used: set[int] = set()
    cursor = 0
    while cursor < len(recognized):
        end, first, last, similarity = choices[cursor]
        output_start = len(output)
        if first is None:
            output.append(recognized[cursor])
            operation = "add"
            source_indices = []
        else:
            assert last is not None
            # Identical supplied lines are separate occurrences. Consume each
            # once before labelling extra audio occurrences as repetition.
            width = last - first
            for alternative in range(len(supplied) - width + 1):
                if (supplied[alternative:alternative + width] == supplied[first:last]
                        and not any(index in used for index in range(alternative, alternative + width))):
                    first, last = alternative, alternative + width
                    break
            source_indices = list(range(first, last))
            operation = "repeat" if any(index in used for index in source_indices) else "keep"
            for index in source_indices:
                # A merged ASR segment cannot provide per-line acoustic bounds.
                # Leave those unknown for downstream forced alignment to measure.
                output.append(replace(
                    recognized[cursor], text=supplied[index], confidence=None,
                    start_sec=recognized[cursor].start_sec if last - first == 1 else None,
                    end_sec=recognized[end - 1].end_sec if last - first == 1 else None,
                ))
            used.update(source_indices)
        decisions.append({
            "operation": operation, "supplied_line_indices": source_indices,
            "recognized_line_indices": list(range(cursor, end)),
            "output_line_indices": list(range(output_start, len(output))),
            "similarity": similarity if first is not None else None,
        })
        cursor = end
    if not used:
        raise ValueError("No supplied lyric line matched the audio transcript; input was not replaced")
    decisions.extend({"operation": "remove", "supplied_line_indices": [index],
                      "recognized_line_indices": [], "output_line_indices": [], "similarity": None}
                     for index in range(len(supplied)) if index not in used)
    return LyricAdjustment(tuple(output), {
        "mode": "whole-line-audio-adjustment", "supplied_lines": list(supplied),
        "recognized_lines": [asdict(line) for line in recognized],
        "adjusted_lines": [asdict(line) for line in output], "decisions": decisions,
        "min_similarity": min_similarity, "max_group_lines": max_group_lines,
        "confidence_available": False,
        "limitation": "Recognition errors may cause incorrect line removal or addition.",
    })
