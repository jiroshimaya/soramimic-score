"""Yomi pronunciations, dictionary alternatives and acoustic selection."""
from __future__ import annotations

import csv
from dataclasses import replace
from itertools import islice, product
import math
from threading import Lock
import unicodedata

from .audio import ReadingSelection
from .japanese import _RUBY, kana_to_moras, katakana, mora_distance, mora_vowel


_YOMI_LOCK = Lock()


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
    """Retain distinct readings from eight dictionary analyses, including alternate splits."""
    import MeCab
    import unidic_lite

    tagger = MeCab.Tagger(f'-d "{unidic_lite.DICDIR}"')
    output = []
    for line in lines:
        primary, _ = _node_reading(tagger.parseToNode(line.text))
        candidates = [primary]
        tagger.parseNBestInit(line.text)
        for _ in range(8):
            node = tagger.nextNode()
            if node is None:
                break
            try:
                kana, _ = _node_reading(node)
            except ValueError:
                # An unknown alternative must not invalidate the primary reading.
                continue
            if kana not in candidates:
                candidates.append(kana)
        output.append(tuple(candidates))
    return tuple(output)


def dictionary_readings(_path, lines):
    """Use Yomi first, supplement with UniDic, and retain each candidate's origin.

    Import failures and engine errors propagate instead of silently disabling
    Yomi. UniDic may not read English/numbers that Yomi can pronounce.
    """
    from soramimic_yomi import get_yomi_candidates

    output = []
    for line in lines:
        if _RUBY.search(line.text):
            parts = []
            cursor = 0
            for match in [*_RUBY.finditer(line.text), None]:
                end = match.start() if match is not None else len(line.text)
                plain = line.text[cursor:end]
                if any(not c.isspace() and unicodedata.category(c)[0] not in "PZC"
                       for c in plain):
                    parts.append(dictionary_readings(_path, (replace(line, text=plain),))[0].candidates)
                if match is not None:
                    kana = katakana(match[2])
                    if not kana or "".join(kana_to_moras(kana)) != kana:
                        raise ValueError("Explicit ruby must contain a kana pronunciation")
                    parts.append((kana,))
                    cursor = match.end()
            # Bound the combinatorial generator by order, never by mora count.
            candidates = tuple(dict.fromkeys("".join(row) for row in islice(product(*parts), 32)))
            output.append(ReadingSelection(candidates[0], "explicit-ruby", 1.0, candidates, {
                "reason": "supplied-ruby", "confidence_available": False,
            }))
            continue
        provenance = {}
        # Yomi initializes a process-wide OpenJTalk user dictionary lazily.
        with _YOMI_LOCK:
            yomi_candidates = get_yomi_candidates(line.text, nbest=32)
        for candidate in yomi_candidates:
            kana = katakana(candidate.reading)
            if not kana or "".join(kana_to_moras(kana)) != kana:
                continue
            entry = provenance.setdefault(kana, {
                "kana": kana, "sources": ["soramimic-yomi"], "yomi_candidates": [],
            })
            entry["yomi_candidates"].append(candidate.to_dict())
        yomi_status = "ok" if provenance else "no-pronunciation"
        try:
            alternatives = dictionary_candidates((line,))[0]
            unidic_status = "ok"
        except ValueError:
            alternatives = ()
            unidic_status = "no-pronunciation"
        for kana in alternatives:
            entry = provenance.setdefault(kana, {"kana": kana, "sources": []})
            entry["sources"].append("unidic-lite")
        if not provenance:
            raise ValueError("Neither soramimic-yomi nor UniDic could pronounce the lyric line")
        candidates = tuple(provenance)
        source = provenance[candidates[0]]["sources"][0]
        output.append(ReadingSelection(candidates[0], source, 1.0, candidates, {
            "reason": "dictionary", "confidence_available": False,
            "candidate_provenance": list(provenance.values()),
            "yomi_status": yomi_status, "unidic_status": unidic_status,
        }))
    return tuple(output)


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
    return ReadingSelection(candidates[selected], "kana-whisper", 1.0,
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
