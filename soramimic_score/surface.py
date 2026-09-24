"""Align supplied spelling to an existing performance without rewriting sound.

Surface groups are intentionally many-to-many. They do not invent character or
mora timestamps, and never replace the canonical graph's surface spans.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
from typing import Any
import unicodedata

from .document import ScoreDocument
from .ir import Evidence


@dataclass(frozen=True)
class SurfaceLine:
    text: str
    reading: str = ""
    matching_reading: str | None = None


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        chr(ord(c) + 96) if "ぁ" <= c <= "ゖ" else c
        for c in text
        if not c.isspace() and unicodedata.category(c)[0] not in "PZC"
    )


def _groups(lines, width):
    return {
        start: [
            (end, [normalize_surface(x.text) for x in lines[start:end]],
             [normalize_surface(x.matching_reading or x.reading) for x in lines[start:end]])
            for end in range(start + 1, min(len(lines), start + width) + 1)
        ]
        for start in range(len(lines))
    }


def _similarity(left, right, minimum, coverage):
    a, b = "".join(left), "".join(right)
    if not a or not b:
        return None
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    if matcher.real_quick_ratio() < minimum or matcher.quick_ratio() < minimum:
        return None
    ratio = matcher.ratio()
    if ratio < minimum:
        return None
    blocks = matcher.get_matching_blocks()
    for parts, side in ((left, "a"), (right, "b")):
        covered = {i for block in blocks
                   for i in range(getattr(block, side), getattr(block, side) + block.size)}
        cursor = 0
        for part in parts:
            if not part or sum(i in covered for i in range(cursor, cursor + len(part))) / len(part) < coverage:
                return None
            cursor += len(part)
    return ratio


def align_lyric_surface(
    recognized: Sequence[SurfaceLine], supplied: Sequence[SurfaceLine], *,
    minimum_similarity: float = .72, per_line_coverage: float = .5,
    max_group_lines: int = 4,
) -> dict[str, Any]:
    """Return a display overlay, preserving all recognized lines and input text.

    Low-confidence and ambiguous matches abstain. Unmatched input lines remain in
    ``supplied_lines``/``unused_supplied_indices``; they are not declared unsung.
    Similarity is a heuristic, not a calibrated probability. Readings are used
    only for matching, never substituted into the acoustic performance.
    """
    if (not recognized or not supplied
            or any(not isinstance(x, SurfaceLine) or not isinstance(x.text, str)
                   or not x.text.strip() or not isinstance(x.reading, str)
                   or (x.matching_reading is not None and not isinstance(x.matching_reading, str))
                   for x in (*recognized, *supplied))):
        raise ValueError("nonempty recognized and supplied lyric lines are required")
    if (not math.isfinite(minimum_similarity) or not 0 < minimum_similarity <= 1
            or not math.isfinite(per_line_coverage) or not 0 < per_line_coverage <= 1
            or type(max_group_lines) is not int or not 1 <= max_group_lines <= 8):
        raise ValueError("invalid surface alignment configuration")
    n, m = len(recognized), len(supplied)
    left, right = _groups(recognized, max_group_lines), _groups(supplied, max_group_lines)
    candidates: dict[tuple[int, int], list] = {}
    for i, aa in left.items():
        for j, bb in right.items():
            for end_a, text_a, kana_a in aa:
                for end_b, text_b, kana_b in bb:
                    ts = _similarity(text_a, text_b, minimum_similarity, per_line_coverage)
                    ks = _similarity(kana_a, kana_b, minimum_similarity, per_line_coverage)
                    if ts is None and ks is None:
                        continue
                    evidence = "surface" if (ts or 0) >= (ks or 0) else "reading"
                    ratio = max(ts or 0, ks or 0)
                    weight = (len("".join(text_a)) + len("".join(text_b))) * (2 * ratio - 1)
                    complexity = end_a - i + end_b - j - 2
                    candidates.setdefault((i, j), []).append(
                        (end_a, end_b, weight, complexity, ratio, evidence))
    best: list[list[tuple[float, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = (0.0, 0)
    previous = {}

    def offer(i, j, value, source):
        if best[i][j] is None or value > best[i][j]:
            best[i][j] = value
            previous[i, j] = source

    for i in range(n + 1):
        for j in range(m + 1):
            value = best[i][j]
            if value is None:
                continue
            if i < n:
                offer(i + 1, j, value, (i, j, "retain_asr", None))
            if j < m:
                offer(i, j + 1, value, (i, j, "unused_supplied", None))
            for ea, eb, weight, complexity, ratio, evidence in candidates.get((i, j), []):
                offer(ea, eb, (value[0] + weight, value[1] - complexity),
                      (i, j, "match", (ratio, evidence)))
    decisions = []
    i, j = n, m
    while i or j:
        pi, pj, operation, quality = previous[i, j]
        decisions.append(dict(operation=operation, asr_indices=list(range(pi, i)),
                              supplied_indices=list(range(pj, j)),
                              similarity=quality[0] if quality else None,
                              evidence=quality[1] if quality else None))
        i, j = pi, pj
    decisions.reverse()
    for decision in decisions:
        if decision["operation"] != "match" or decision["evidence"] != "reading":
            continue
        aa = [normalize_surface(recognized[i].matching_reading or recognized[i].reading)
              for i in decision["asr_indices"]]
        selected = "".join(normalize_surface(supplied[j].text) for j in decision["supplied_indices"])
        for options in right.values():
            ambiguous = False
            for _, texts, readings in options:
                if "".join(texts) == selected:
                    continue
                alternative = _similarity(aa, readings, minimum_similarity, per_line_coverage)
                if alternative is not None and alternative >= decision["similarity"] - 1e-9:
                    ambiguous = True
                    break
            if ambiguous:
                decision["operation"] = "ambiguous_retain_asr"
                break
    output = []
    for decision in decisions:
        if not decision["asr_indices"]:
            continue
        originals = [recognized[i] for i in decision["asr_indices"]]
        targets = [supplied[j] for j in decision["supplied_indices"]]
        matched = decision["operation"] == "match"
        output.append(decision | dict(
            original_text="\n".join(x.text for x in originals),
            display_text="\n".join(x.text for x in (targets if matched else originals)),
            acoustic_reading="".join(x.reading for x in originals),
            supplied_reading="".join(x.reading for x in targets) if matched else None,
            pronunciation_equal=(
                normalize_surface("".join(x.matching_reading or x.reading for x in originals))
                == normalize_surface("".join(x.matching_reading or x.reading for x in targets))
            ) if matched else None,
        ))
    return dict(
        schema_version=1, mode="asr-first-surface", groups=output,
        supplied_lines=[x.text for x in supplied],
        unused_supplied_indices=[j for d in decisions if d["operation"] != "match"
                                 for j in d["supplied_indices"]],
        display_text="\n".join(d["display_text"] for d in output), acoustic_changes=False,
        max_group_lines=max_group_lines, minimum_similarity=minimum_similarity,
        per_line_coverage=per_line_coverage, similarity_is_calibrated_probability=False,
    )


def attach_lyric_surface(document: ScoreDocument, overlay: dict[str, Any]) -> ScoreDocument:
    """Attach display provenance without recompiling or changing acoustic fields."""
    evidence = tuple(e for e in document.observations.evidence if e.id != "lyric-surface")
    evidence += (Evidence("lyric-surface", "soramimic_score.surface", "lyric-surface", 0.0, overlay),)
    return replace(document, observations=replace(document.observations, evidence=evidence),
                   score=replace(document.score, evidence=evidence))


def plan_lyric_inputs(
    recognized: Sequence[SurfaceLine], supplied: Sequence[SurfaceLine],
) -> dict[str, Any]:
    """Locate supplied text before selecting pronunciation or final alignment.

    A matched many-to-many group becomes one final lyric interval. Its reading
    must be derived from the supplied text, not the recognition. Unmatched ASR
    lines remain separate and explicitly identified; unused input is retained.
    ``asr_indices`` refer to recognition, ``line_indices`` to final lyric lines.
    This plan contains no invented word/mora times and makes no acoustic claim.
    """
    plan = align_lyric_surface(recognized, supplied)
    groups = []
    for group in plan["groups"]:
        if group["operation"] == "match":
            groups.append(group | {"reading_source": "supplied-lyrics"})
        else:
            for index in group["asr_indices"]:
                line = recognized[index]
                groups.append(group | {
                    "asr_indices": [index], "original_text": line.text,
                    "display_text": line.text, "acoustic_reading": line.reading,
                    "reading_source": "automatic-unmatched",
                })
    for index, group in enumerate(groups):
        group["line_indices"] = [index]
    plan.update(mode="asr-localized-supplied-lyrics", groups=groups,
                display_text="\n".join(g["display_text"] for g in groups))
    return plan


def lyric_surface(document: ScoreDocument) -> dict[str, Any] | None:
    """Return the optional display overlay; the canonical graph remains acoustic."""
    return next((e.detail for e in document.observations.evidence
                 if e.id == "lyric-surface" and e.kind == "lyric-surface"), None)
