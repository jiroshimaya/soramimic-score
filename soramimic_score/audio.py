"""Model-neutral orchestration from singing audio to a score document.

The heavy models stay behind small callables so applications can decide where
and how they run (locally, in a worker, or through a private inference service).
This module owns the stable data passed between those models and the conversion
into Soramimic Score's canonical JSON document.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ModelConfig

from .alignment import ObservedSingingUnit, build_known_lyrics_document
from .document import ScoreDocument, compile_score
from .ir import Boundary, Evidence, IntermediateRepresentation, NoteCandidate
from .japanese import LyricSpan, ReadingCandidate, kana_to_moras


@dataclass(frozen=True)
class LyricLine:
    """One recognized lyric line on the original audio clock."""

    text: str
    start_sec: float | None = None
    end_sec: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ReadingSelection:
    """The pronunciation selected for one lyric line."""

    kana: str
    source: str
    confidence: float
    candidates: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedMora:
    """One mora located by a text-conditioned acoustic aligner."""

    line_index: int
    mora_index: int
    kana: str
    start_sec: float
    end_sec: float
    confidence: float
    source: str = "mora-alignment"
    vowel_start_sec: float | None = None


@dataclass(frozen=True)
class MelodyNote:
    """One monophonic melody observation on the original audio clock."""

    start_sec: float
    end_sec: float
    midi_pitch: int
    source: str = "melody-transcription"
    confidence: float | None = None


LyricRecognizer = Callable[[Path], Sequence[LyricLine]]
ReadingSelector = Callable[[Path, Sequence[LyricLine]], Sequence[ReadingSelection]]
MoraAligner = Callable[
    [Path, Sequence[LyricLine], Sequence[ReadingSelection]],
    Sequence[AlignedMora],
]
MelodyTranscriber = Callable[[Path], Sequence[MelodyNote]]


@dataclass(frozen=True)
class AudioAdapters:
    """The four model boundaries needed by the audio pipeline.

    ``lyric_recognizer`` is required by the ASR-first audio pipeline.
    ``lyric_reading`` optionally supplies linguistic readings for whole-line
    comparison (otherwise normalized surface text is used). The other adapters
    are required in both modes.
    """

    reading_selector: ReadingSelector
    mora_aligner: MoraAligner
    melody_transcriber: MelodyTranscriber
    lyric_recognizer: LyricRecognizer | None = None
    lyric_reading: Callable[[str], str] | None = None
    reading_refiner: Callable[
        [Path, Sequence[LyricLine], Sequence[ReadingSelection]], Sequence[ReadingSelection]
    ] | None = None


class AudioPipelineError(RuntimeError):
    """An audio adapter failed or returned an inconsistent result."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        super().__init__(f"{stage}: {detail}")


def _run_adapter(stage: str, adapter: Callable[..., Sequence[object]], *args: object):
    try:
        return adapter(*args)
    except AudioPipelineError:
        raise
    except Exception as exc:
        raise AudioPipelineError(stage, str(exc) or type(exc).__name__) from exc


def _confidence(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise AudioPipelineError(label, "confidence must be in [0, 1]")
    return number


def _validate_lines(lines: Sequence[LyricLine], *, timed: bool) -> tuple[LyricLine, ...]:
    result = tuple(lines)
    if not result:
        raise AudioPipelineError("lyrics", "no lyric lines were produced")
    previous_end = 0.0
    normalized = []
    for line in result:
        if not isinstance(line.text, str) or not line.text.strip():
            raise AudioPipelineError("lyrics", "lyric lines must contain text")
        if line.confidence is not None:
            _confidence(line.confidence, "lyrics")
        if timed:
            if line.start_sec is None or line.end_sec is None:
                raise AudioPipelineError("lyrics", "recognized lines need start and end times")
            if (not math.isfinite(line.start_sec + line.end_sec) or line.start_sec < 0
                    or line.start_sec < previous_end - 1e-6
                    or line.end_sec <= max(previous_end, line.start_sec)):
                raise AudioPipelineError(
                    "lyrics", "recognized line times must be finite and non-overlapping",
                )
            if line.start_sec < previous_end:
                line = replace(line, start_sec=previous_end)
            previous_end = line.end_sec
        normalized.append(line)
    return tuple(normalized)


def _validate_readings(
    lines: Sequence[LyricLine], readings: Sequence[ReadingSelection],
) -> tuple[ReadingSelection, ...]:
    result = tuple(readings)
    if len(result) != len(lines):
        raise AudioPipelineError("readings", "one reading is required for every lyric line")
    for item in result:
        if not item.source or not kana_to_moras(item.kana):
            raise AudioPipelineError("readings", "every reading needs kana and a source")
        _confidence(item.confidence, "readings")
        if item.candidates and (item.kana not in item.candidates or any(
            not kana or "".join(kana_to_moras(kana)) != kana for kana in item.candidates
        )):
            raise AudioPipelineError("readings", "candidates must contain the selected kana reading")
    return result


def _validate_moras(
    readings: Sequence[ReadingSelection], aligned: Sequence[AlignedMora],
) -> tuple[AlignedMora, ...]:
    result = tuple(aligned)
    if tuple((item.line_index, item.mora_index) for item in result) != tuple(sorted(
        (item.line_index, item.mora_index) for item in result
    )):
        raise AudioPipelineError("mora alignment", "moras must be in lyric order")
    for line_index, reading in enumerate(readings):
        items = tuple(item for item in result if item.line_index == line_index)
        if tuple(item.mora_index for item in items) != tuple(range(len(items))):
            raise AudioPipelineError("mora alignment", "mora indices must be contiguous")
        if "".join(item.kana for item in items) != "".join(kana_to_moras(reading.kana)):
            raise AudioPipelineError(
                "mora alignment", "aligned moras do not match the selected reading",
            )
    if any(item.line_index < 0 or item.line_index >= len(readings) for item in result):
        raise AudioPipelineError("mora alignment", "a mora refers to an unknown lyric line")
    previous_end = 0.0
    for item in result:
        values = [item.start_sec, item.end_sec, item.confidence]
        if item.vowel_start_sec is not None:
            values.append(item.vowel_start_sec)
        if (not all(math.isfinite(value) for value in values)
                or item.start_sec < previous_end or item.end_sec <= item.start_sec
                or (item.vowel_start_sec is not None
                    and not item.start_sec <= item.vowel_start_sec <= item.end_sec)
                or not item.source):
            raise AudioPipelineError(
                "mora alignment", "mora times must be finite, ordered, and positive",
            )
        _confidence(item.confidence, "mora alignment")
        previous_end = item.end_sec
    return result


def _validate_notes(notes: Sequence[MelodyNote]) -> tuple[MelodyNote, ...]:
    result = tuple(notes)
    if not result:
        raise AudioPipelineError("melody", "no melody notes were produced")
    previous_end = 0.0
    for item in result:
        if (not math.isfinite(item.start_sec + item.end_sec)
                or item.start_sec < previous_end or item.end_sec <= item.start_sec
                or type(item.midi_pitch) is not int or not 0 <= item.midi_pitch <= 127
                or not item.source):
            raise AudioPipelineError(
                "melody", "notes must be finite, monophonic, ordered, and valid MIDI",
            )
        if item.confidence is not None:
            _confidence(item.confidence, "melody")
        previous_end = item.end_sec
    return result


def build_audio_observations(
    lines: Sequence[LyricLine],
    readings: Sequence[ReadingSelection],
    aligned_moras: Sequence[AlignedMora],
    melody_notes: Sequence[MelodyNote],
) -> IntermediateRepresentation:
    """Normalize adapter results into the versioned observation document."""
    lines = _validate_lines(lines, timed=False)
    readings = _validate_readings(lines, readings)
    aligned_moras = _validate_moras(readings, aligned_moras)
    melody_notes = _validate_notes(melody_notes)

    canonical_text = "\n".join(line.text for line in lines)
    spans: list[LyricSpan] = []
    evidence: list[Evidence] = []
    offset = 0
    for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
        evidence_ids = ()
        if reading.candidates or reading.detail:
            evidence_id = f"audio-reading-{index}"
            evidence_ids = (evidence_id,)
            evidence.append(Evidence(
                evidence_id, reading.source, "reading-selection", reading.confidence,
                {**reading.detail, "candidates": list(reading.candidates or (reading.kana,)),
                 "selected": reading.kana},
            ))
        spans.append(LyricSpan(
            line.text,
            (offset, offset + len(line.text)),
            (ReadingCandidate(
                reading.kana, reading.source, reading.confidence, evidence_ids,
            ),),
        ))
        offset += len(line.text) + 1

    observations: list[ObservedSingingUnit] = []
    for item in aligned_moras:
        evidence_id = f"audio-mora-{item.line_index}-{item.mora_index}"
        center = (item.start_sec + item.end_sec) / 2
        evidence.append(Evidence(
            evidence_id,
            item.source,
            "mora-ctc-anchor",
            item.confidence,
            {
                "time_sec": center,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "conditioned_on_text": True,
            },
        ))
        observations.append(ObservedSingingUnit(
            (item.kana,),
            Boundary(item.start_sec, item.confidence, (evidence_id,)),
            (Boundary(item.vowel_start_sec, item.confidence, (evidence_id,))
             if item.vowel_start_sec is not None else None),
            Boundary(item.end_sec, item.confidence, (evidence_id,)),
            item.confidence,
            (evidence_id,),
        ))

    document = build_known_lyrics_document(
        canonical_text, tuple(spans), tuple(observations), tuple(evidence),
    )
    note_evidence: list[Evidence] = []
    candidates: list[NoteCandidate] = []
    for index, item in enumerate(melody_notes):
        evidence_id = f"audio-note-evidence-{index}"
        confidence = item.confidence if item.confidence is not None else 0.0
        note_evidence.append(Evidence(
            evidence_id,
            item.source,
            "model-note",
            confidence,
            {"confidence_available": item.confidence is not None},
        ))
        candidates.append(NoteCandidate(
            f"audio-note-{index}",
            item.start_sec,
            item.end_sec,
            item.midi_pitch,
            confidence,
            (item.source,),
            (evidence_id,),
        ))
    return replace(
        document,
        evidence=document.evidence + tuple(note_evidence),
        note_candidates=tuple(candidates),
    )


def analyze_audio(
    audio_path: str | Path,
    adapters: AudioAdapters | None = None,
    *,
    lyrics: Sequence[str] | None = None,
    model_config: ModelConfig | None = None,
    adjust_lyrics: bool = False,
) -> ScoreDocument:
    """Run configured acoustic adapters and compile one canonical score JSON model.

    Supplied lyrics are aligned as a display overlay after automatic analysis.
    Unmatched input remains unresolved; the input text is never silently removed.
    With ``adjust_lyrics=True``,
    whole input lines may be omitted or repeated and recognized lines added.
    Matched input lines remain verbatim; the original input and decisions are
    retained as evidence. Recognition errors can affect these edits.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if adapters is not None and model_config is not None:
        raise ValueError("Specify adapters or model_config, not both")
    if adjust_lyrics and lyrics is None:
        raise ValueError("adjust_lyrics requires supplied lyrics")
    if adapters is None:
        from .models import prepared_adapters
        if model_config is None:
            raise ValueError("model_config with local SheetSage2 directories is required")
        with prepared_adapters(path, model_config) as prepared:
            return analyze_audio(path, prepared, lyrics=lyrics, adjust_lyrics=adjust_lyrics)

    adjustment = None
    if lyrics is not None:
        if isinstance(lyrics, (str, bytes)):
            raise TypeError("lyrics must be a sequence of lines, not one string")
        _validate_lines(tuple(LyricLine(text) for text in lyrics), timed=False)
    if not adjust_lyrics:
        if adapters.lyric_recognizer is None:
            raise AudioPipelineError(
                "lyrics", "ASR-first analysis requires a recognizer, also with supplied lyrics",
            )
        lines = _validate_lines(
            _run_adapter("lyrics", adapters.lyric_recognizer, path), timed=True,
        )
    else:
        lines = _validate_lines(tuple(LyricLine(text) for text in lyrics), timed=False)
        if adjust_lyrics:
            from .lyrics import adjust_known_lyrics
            if adapters.lyric_recognizer is None:
                raise AudioPipelineError("lyrics", "lyric adjustment requires a recognizer")
            recognized = _validate_lines(
                _run_adapter("lyrics", adapters.lyric_recognizer, path), timed=True,
            )
            adjustment = adjust_known_lyrics(lyrics, recognized, reading=adapters.lyric_reading)
            lines = adjustment.lines

    readings = _validate_readings(
        lines, _run_adapter("readings", adapters.reading_selector, path, lines),
    )
    moras = _validate_moras(
        readings,
        _run_adapter("mora alignment", adapters.mora_aligner, path, lines, readings),
    )
    notes = _validate_notes(
        _run_adapter("melody", adapters.melody_transcriber, path),
    )
    observations = build_audio_observations(lines, readings, moras, notes)
    if adjustment is not None:
        observations = replace(observations, evidence=observations.evidence + (Evidence(
            "audio-lyric-adjustment", "soramimic_score.lyrics", "lyric-adjustment", 0.0,
            adjustment.detail,
        ),))
    line_windows = (
        {f"u{index}": (line.start_sec, line.end_sec)
         for index, line in enumerate(lines)
         if line.start_sec is not None and line.end_sec is not None}
        or None
    )
    result = compile_score(observations, line_windows_by_utterance=line_windows)
    if lyrics is not None and not adjust_lyrics:
        from .surface import SurfaceLine, align_lyric_surface, attach_lyric_surface
        convert = adapters.lyric_reading or (lambda _: "")
        overlay = align_lyric_surface(
            [SurfaceLine(line.text, reading.kana, convert(line.text) or None)
             for line, reading in zip(lines, readings, strict=True)],
            [SurfaceLine(text, convert(text)) for text in lyrics],
        )
        for group in overlay["groups"]:
            group["utterance_ids"] = [f"u{i}" for i in group["asr_indices"]]
            group["start_sec"] = lines[group["asr_indices"][0]].start_sec
            group["end_sec"] = lines[group["asr_indices"][-1]].end_sec
        if adapters.reading_refiner is not None:
            result, reviews = _refine_supplied_readings(
                path, adapters, lines, readings, moras, notes, result, overlay, line_windows,
            )
            overlay["reading_reviews"] = reviews
            overlay["acoustic_changes"] = any(row["status"] == "applied" for row in reviews)
        return attach_lyric_surface(result, overlay)
    return result


def _refine_supplied_readings(path, adapters, lines, readings, moras, notes,
                              result, overlay, line_windows):
    """Recheck only unambiguous one-to-one matches; accept bounded local changes.

    All candidate lengths remain eligible. Rebuilding the graph must preserve the
    audible plan of every other line exactly; otherwise keep the automatic result.
    """
    reviews = []
    matched = []
    for group in overlay["groups"]:
        if group["operation"] != "match":
            continue
        if len(group["asr_indices"]) != 1 or len(group["supplied_indices"]) != 1:
            reviews.append({"asr_indices": group["asr_indices"], "status": "unresolved-group"})
            continue
        matched.append(group)
    if not matched:
        return result, reviews
    requested_lines = tuple(replace(lines[g["asr_indices"][0]], text=g["display_text"])
                            for g in matched)
    baselines = tuple(readings[g["asr_indices"][0]] for g in matched)
    selections = _validate_readings(requested_lines, _run_adapter(
        "reading refinement", adapters.reading_refiner, path, requested_lines, baselines,
    ))
    changed = [i for i, (a, b) in enumerate(zip(baselines, selections, strict=True))
               if a.kana != b.kana]
    aligned_by_request = {}
    alignment_error = False
    if changed:
        changed_readings = tuple(selections[i] for i in changed)
        try:
            local = _validate_moras(changed_readings, _run_adapter(
                "local mora alignment", adapters.mora_aligner, path,
                tuple(requested_lines[i] for i in changed), changed_readings,
            ))
            aligned_by_request = {request: tuple(m for m in local if m.line_index == offset)
                                  for offset, request in enumerate(changed)}
        except AudioPipelineError:
            # A failed optional recheck must not destroy a complete ASR result.
            alignment_error = True
    for request, (group, line, selection) in enumerate(zip(
        matched, requested_lines, selections, strict=True,
    )):
        index = group["asr_indices"][0]
        before = readings[index]
        review = {"asr_indices": [index], "before": before.kana,
                  "proposed": selection.kana, "detail": selection.detail,
                  "status": "unchanged"}
        reviews.append(review)
        if selection.kana == before.kana:
            continue
        if alignment_error:
            review["status"] = "infeasible-local-alignment"
            continue
        aligned = aligned_by_request[request]
        if any(m.start_sec < line.start_sec or m.end_sec > line.end_sec for m in aligned):
            review["status"] = "outside-recognized-window"
            continue
        proposed_readings = tuple(selection if i == index else r for i, r in enumerate(readings))
        proposed_moras = tuple(sorted(
            [m for m in moras if m.line_index != index]
            + [replace(m, line_index=index) for m in aligned],
            key=lambda m: (m.line_index, m.mora_index),
        ))
        try:
            proposed = compile_score(
                build_audio_observations(lines, proposed_readings, proposed_moras, notes),
                line_windows_by_utterance=line_windows,
            )
        except (ValueError, AudioPipelineError):
            review["status"] = "infeasible-local-alignment"
            continue

        def other_plan(doc):
            return [(s.utterance_id, s.kana, s.start_sec, s.end_sec, s.midi_pitch)
                    for s in doc.score.synthesis_plan if s.utterance_id != f"u{index}"]

        if other_plan(proposed) != other_plan(result):
            review["status"] = "would-change-other-lines"
            continue
        if proposed.score.unresolved_unit_ids:
            review["status"] = "unresolved-synthesis"
            continue
        result, readings, moras = proposed, proposed_readings, proposed_moras
        group["original_acoustic_reading"] = group["acoustic_reading"]
        group["acoustic_reading"] = selection.kana
        review["status"] = "applied"
    return result, reviews
