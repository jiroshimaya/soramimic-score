"""Dictionary alternatives and conservative acoustic pronunciation selection."""
from __future__ import annotations

import csv
import math

from .audio import ReadingSelection
from .japanese import kana_to_moras, katakana, mora_distance, mora_vowel


def _node_reading(node):
    parts, surfaces = [], []
    while node is not None:
        if node.surface:
            fields = next(csv.reader([node.feature]))
            if fields[0] != "補助記号":
                pronunciation = fields[9] if len(fields) > 9 else "*"
                if pronunciation in {"", "*"}:
                    pronunciation = katakana(node.surface)
                kana = "".join(kana_to_moras(pronunciation))
                if not kana or kana != katakana(pronunciation):
                    raise ValueError(f"Cannot determine Japanese pronunciation: {node.surface!r}")
                parts.append(kana)
                surfaces.append(node.surface)
        node = node.next
    if not parts:
        raise ValueError("No Japanese pronunciation in lyric line")
    return "".join(parts), tuple(surfaces)


def dictionary_candidates(lines):
    """Retain distinct readings from 32 dictionary analyses, regardless of length."""
    import MeCab
    import unidic_lite

    tagger = MeCab.Tagger(f'-d "{unidic_lite.DICDIR}"')
    output = []
    for line in lines:
        primary, surfaces = _node_reading(tagger.parseToNode(line.text))
        candidates = [primary]
        tagger.parseNBestInit(line.text)
        for _ in range(32):
            node = tagger.nextNode()
            if node is None:
                break
            try:
                kana, alternative_surfaces = _node_reading(node)
            except ValueError:
                # An unknown alternative must not invalidate the primary reading.
                continue
            if alternative_surfaces == surfaces and kana not in candidates:
                candidates.append(kana)
        output.append(tuple(candidates))
    return tuple(output)


def dictionary_readings(_path, lines):
    return tuple(ReadingSelection(candidates[0], "unidic-lite", 1.0, candidates,
                                  {"reason": "dictionary", "confidence_available": False})
                 for candidates in dictionary_candidates(lines))


def _comparison_moras(kana):
    output = []
    vowel = None
    for mora in kana_to_moras(kana):
        if mora == "ー" and vowel is not None and vowel in "aiueo":
            mora = dict(zip("aiueo", "アイウエオ"))[vowel]
        mora = {"ヲ": "オ", "ヂ": "ジ", "ヅ": "ズ"}.get(mora, mora)
        vowel = mora_vowel(mora, vowel)
        output.append(mora)
    return tuple(output)


def _substring_distance(candidate, evidence):
    """Match the entire candidate to a context substring; do not normalize by length."""
    previous = [0.0] * (len(evidence) + 1)
    for expected in candidate:
        current = [previous[0] + 1.0]
        for index, observed in enumerate(evidence, 1):
            current.append(min(previous[index] + 1.0, current[-1] + 1.0,
                               previous[index - 1] + mora_distance(expected, observed) / 4))
        previous = current
    return min(previous)


def select_acoustic_reading(candidates, transcripts):
    """Change the dictionary choice only when available audio views agree.

    The 1.0 score is a selection prior, not a calibrated acoustic probability.
    Alternative readings and raw distances are retained as evidence.
    """
    if not candidates:
        raise ValueError("At least one dictionary candidate is required")
    detail = {"transcripts": dict(transcripts), "confidence_available": False,
              "distance": "phonetic-mora-substring", "reason": "no-acoustic-evidence"}
    evidence = {}
    for view, text in transcripts.items():
        try:
            moras = _comparison_moras(text)
        except ValueError:
            continue
        if moras:
            evidence[view] = moras
    distances = []
    for candidate in candidates:
        try:
            moras = _comparison_moras(candidate)
            row = {view: _substring_distance(moras, text) for view, text in evidence.items()}
        except ValueError:
            row = {view: None for view in evidence}
        distances.append(row)
    detail["distances"] = distances
    selected = 0
    if evidence and all(value is not None for value in distances[0].values()):
        totals = [sum(row.values()) if all(value is not None for value in row.values())
                  else math.inf for row in distances]
        best = min(range(len(candidates)), key=lambda index: totals[index])
        tied = sum(math.isclose(total, totals[best], abs_tol=1e-9) for total in totals) > 1
        detail["reason"] = "dictionary-supported"
        if tied:
            detail["reason"] = "ambiguous-evidence"
        elif best != 0:
            gains = [distances[0][view] - distances[best][view] for view in evidence]
            if min(gains) < -1e-9:
                detail["reason"] = "conflicting-evidence"
            elif sum(gains) >= .25:
                selected = best
                detail["reason"] = "acoustic-agreement"
            else:
                detail["reason"] = "weak-evidence"
    return ReadingSelection(candidates[selected], "unidic-kana-whisper", 1.0,
                            tuple(candidates), detail)


def acoustic_windows(start, end, duration):
    """Cover a lyric interval in <=24-second windows without dropping its tail."""
    if not math.isfinite(start + end + duration) or not 0 <= start < end <= duration + .01:
        raise ValueError("Reading window must lie within the audio")
    cursor, stop = max(0.0, start - 1.5), min(duration, end + 1.5)
    result = []
    while cursor < stop:
        last = min(stop, cursor + 24.0)
        result.append((cursor, last))
        cursor = last
    return tuple(result)
