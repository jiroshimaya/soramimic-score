"""Expand ASR-rounded vocalizations from explicit acoustic re-attacks.

The caller owns audio inference. Whisper supplies the mora identity while an
unconditioned acoustic pass supplies re-articulation times. SheetSage notes are
used only as pitch/duration coverage that the decoder must sing; their count
never determines how many consonants were performed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from .ir import (
    Boundary,
    Evidence,
    IntermediateRepresentation,
    Mora,
    Phoneme,
    Reading,
    SingingUnit,
    VowelNucleus,
)
from .japanese import SPECIAL_MORAS, phonemes_for_mora


@dataclass(frozen=True)
class VocalizationReattack:
    """One transcript-independent acoustic re-articulation observation."""

    start_sec: float
    end_sec: float
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start_sec + self.end_sec + self.confidence)
                or self.start_sec < 0 or self.end_sec <= self.start_sec
                or not 0 <= self.confidence <= 1):
            raise ValueError(
                "vocalization re-attacks need finite ordered bounds and confidence"
            )
        if not self.source:
            raise ValueError("vocalization re-attacks need a source")


@dataclass(frozen=True)
class VocalizationExpansion:
    """Expanded document plus provenance needed to extend caller segment IDs."""

    document: IntermediateRepresentation
    added_mora_sources: tuple[tuple[str, str], ...] = ()


def _available_prefix(occupied: set[str], stem: str) -> str:
    prefix = stem
    while any(identifier.startswith(prefix) for identifier in occupied):
        prefix = "_" + prefix
    return prefix


def _phoneme_role(symbol: str) -> str:
    if symbol in "aiueo":
        return "vowel"
    if symbol == "N":
        return "nasal"
    if symbol == "q":
        return "closure"
    return "consonant"


def _selected_repeated_units(
    document: IntermediateRepresentation, utterance_id: str,
) -> tuple[Reading, tuple[Mora, ...], tuple[SingingUnit, ...]] | None:
    utterance = next(item for item in document.utterances if item.id == utterance_id)
    reading = next(item for item in document.readings
                   if item.id == utterance.selected_reading_id)
    moras_by_id = {item.id: item for item in document.moras}
    selected_moras = tuple(moras_by_id[identifier] for identifier in reading.mora_ids)
    if not selected_moras:
        return None
    texts = {item.text for item in selected_moras}
    if len(texts) != 1 or next(iter(texts)) in SPECIAL_MORAS:
        return None
    unit_by_mora = {
        unit.mora_ids[0]: unit
        for unit in document.singing_units
        if len(unit.mora_ids) == 1
    }
    if any(mora.id not in unit_by_mora for mora in selected_moras):
        return None
    return (
        reading,
        selected_moras,
        tuple(unit_by_mora[mora.id] for mora in selected_moras),
    )


def _covered_note_ids(
    document: IntermediateRepresentation, window: tuple[float, float],
) -> tuple[str, ...]:
    start_sec, end_sec = window
    return tuple(
        note.id for note in document.note_candidates
        if (note.midi_pitch is not None
            and any("sheetsage" in source.casefold() for source in note.sources)
            and note.end_sec > start_sec and note.start_sec < end_sec)
    )


def expand_repeated_vocalizations(
    document: IntermediateRepresentation,
    line_windows_by_utterance: Mapping[str, tuple[float, float]],
    reattacks_by_utterance: Mapping[str, Sequence[VocalizationReattack]],
) -> VocalizationExpansion:
    """Expand a repeated Whisper mora to independently observed re-attacks.

    There is no repetition cap and SheetSage segmentation does not create morae.
    A line changes only when the caller supplies at least as many acoustic
    re-attacks as Whisper recognized copies of one ordinary mora. Every
    overlapping SheetSage note is recorded as required vocal coverage.
    """
    if not isinstance(document, IntermediateRepresentation):
        raise TypeError("document must be an IntermediateRepresentation")
    document.validate()
    utterance_ids = {item.id for item in document.utterances}
    if set(line_windows_by_utterance) != utterance_ids:
        raise ValueError(
            "vocalization expansion requires one Whisper window per utterance"
        )
    if not set(reattacks_by_utterance) <= utterance_ids:
        raise ValueError("vocalization re-attacks contain an unknown utterance")

    occupied = {
        item.id for collection in (
            document.utterances, document.readings, document.moras,
            document.phonemes, document.vowel_nuclei, document.singing_units,
            document.evidence, document.note_candidates, document.links,
        ) for item in collection
    }
    mora_prefix = _available_prefix(occupied, "repeat-vocalization-mora-")
    phoneme_prefix = _available_prefix(occupied, "repeat-vocalization-phoneme-")
    nucleus_prefix = _available_prefix(occupied, "repeat-vocalization-nucleus-")
    unit_prefix = _available_prefix(occupied, "repeat-vocalization-unit-")
    evidence_prefix = _available_prefix(occupied, "repeat-vocalization-evidence-")

    readings = list(document.readings)
    moras = list(document.moras)
    phonemes = list(document.phonemes)
    nuclei = list(document.vowel_nuclei)
    units = list(document.singing_units)
    evidence = list(document.evidence)
    evidence_by_id = {item.id: item for item in document.evidence}
    added_mora_sources: list[tuple[str, str]] = []
    expansion_index = 0

    for utterance in document.utterances:
        repeated = _selected_repeated_units(document, utterance.id)
        observations = tuple(reattacks_by_utterance.get(utterance.id, ()))
        if repeated is None or not observations:
            continue
        if any(left.start_sec >= right.start_sec
               for left, right in zip(observations, observations[1:])):
            raise ValueError("vocalization re-attacks must be strictly chronological")
        window = line_windows_by_utterance[utterance.id]
        if any(item.start_sec < window[0] or item.start_sec > window[1]
               for item in observations):
            raise ValueError(
                "vocalization re-attack starts must lie in the Whisper window"
            )

        reading, selected_moras, selected_units = repeated
        if len(observations) < len(selected_moras):
            continue
        covered_note_ids = _covered_note_ids(document, window)
        if not covered_note_ids:
            continue

        repeat_text = selected_moras[0].text
        source_mora_id = selected_moras[-1].id
        decision_id = f"{evidence_prefix}{expansion_index}-decision"
        evidence.append(Evidence(
            decision_id,
            "soramimic_score.vocalization",
            "repeated-vocalization-expansion",
            min(item.confidence for item in observations),
            {
                "utterance_id": utterance.id,
                "mora": repeat_text,
                "recognized_repetitions": len(selected_moras),
                "expanded_repetitions": len(observations),
                "whisper_window_sec": list(window),
                "required_note_candidate_ids": list(covered_note_ids),
                "rule": "acoustic-reattacks-with-sheetsage-coverage",
            },
        ))

        new_mora_ids = list(reading.mora_ids)
        for repeat_index, observation in enumerate(observations):
            anchor_id = f"{evidence_prefix}{expansion_index}-reattack-{repeat_index}"
            evidence.append(Evidence(
                anchor_id,
                observation.source,
                "repeated-vocalization-reattack",
                observation.confidence,
                {
                    "time_sec": observation.start_sec,
                    "end_sec": observation.end_sec,
                    "decision_evidence_id": decision_id,
                },
            ))
            boundary_start = Boundary(
                observation.start_sec, observation.confidence, (anchor_id,),
            )
            boundary_end = Boundary(
                observation.end_sec, observation.confidence, (anchor_id,),
            )
            if repeat_index < len(selected_units):
                source_unit = selected_units[repeat_index]
                retained_evidence = tuple(
                    identifier for identifier in source_unit.evidence_ids
                    if evidence_by_id.get(identifier) is None
                    or evidence_by_id[identifier].kind not in {
                        "mora-ctc-anchor", "repeated-vocalization-anchor",
                        "repeated-vocalization-reattack",
                    }
                )
                units = [
                    replace(
                        item,
                        consonant_start=boundary_start,
                        end=boundary_end,
                        evidence_ids=retained_evidence + (decision_id, anchor_id),
                    ) if item.id == source_unit.id else item
                    for item in units
                ]
                nuclei = [
                    replace(item, start=boundary_start, end=boundary_end)
                    if item.singing_unit_id == source_unit.id else item
                    for item in nuclei
                ]
                continue

            mora_id = f"{mora_prefix}{expansion_index}-{repeat_index}"
            unit_id = f"{unit_prefix}{expansion_index}-{repeat_index}"
            new_mora_ids.append(mora_id)
            added_mora_sources.append((mora_id, source_mora_id))
            symbols = phonemes_for_mora(repeat_text)
            phoneme_ids = []
            vowel_phoneme_ids = []
            for symbol_index, symbol in enumerate(symbols):
                phoneme_id = (
                    f"{phoneme_prefix}{expansion_index}-{repeat_index}-{symbol_index}"
                )
                phoneme_ids.append(phoneme_id)
                phonemes.append(Phoneme(
                    phoneme_id, symbol, _phoneme_role(symbol), (mora_id,),
                ))
                if _phoneme_role(symbol) in {"vowel", "nasal"}:
                    vowel_phoneme_ids.append((symbol, phoneme_id))
            nucleus_ids = []
            for nucleus_index, (symbol, phoneme_id) in enumerate(vowel_phoneme_ids):
                nucleus_id = (
                    f"{nucleus_prefix}{expansion_index}-{repeat_index}-{nucleus_index}"
                )
                nucleus_ids.append(nucleus_id)
                nuclei.append(VowelNucleus(
                    nucleus_id, unit_id, symbol, (phoneme_id,),
                    boundary_start, boundary_end,
                ))
            moras.append(Mora(
                mora_id, reading.id, repeat_text, selected_moras[-1].surface_span,
                tuple(phoneme_ids), (unit_id,),
            ))
            source_unit = selected_units[-1]
            units.append(SingingUnit(
                unit_id, (mora_id,), tuple(phoneme_ids), tuple(nucleus_ids),
                "weak", min(source_unit.confidence, observation.confidence),
                boundary_start, boundary_end, (decision_id, anchor_id),
            ))

        readings = [
            replace(item, kana=repeat_text * len(observations),
                    mora_ids=tuple(new_mora_ids))
            if item.id == reading.id else item
            for item in readings
        ]
        expansion_index += 1

    if expansion_index == 0:
        return VocalizationExpansion(document)
    expanded = replace(
        document,
        readings=tuple(readings),
        moras=tuple(moras),
        phonemes=tuple(phonemes),
        vowel_nuclei=tuple(nuclei),
        singing_units=tuple(units),
        evidence=tuple(evidence),
    )
    return VocalizationExpansion(expanded, tuple(added_mora_sources))
