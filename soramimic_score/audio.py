"""Model-neutral orchestration from singing audio to a score document.

The heavy models stay behind small callables so applications can decide where
and how they run (locally, in a worker, or through a private inference service).
This module owns the stable data passed between those models and the conversion
into Soramimic Score's canonical JSON document.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path

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

    ``lyric_recognizer`` may be omitted when ``lyrics`` are supplied to
    :func:`analyze_audio`. The other adapters are required in both modes.
    """

    reading_selector: ReadingSelector
    mora_aligner: MoraAligner
    melody_transcriber: MelodyTranscriber
    lyric_recognizer: LyricRecognizer | None = None


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
    for line in result:
        if not isinstance(line.text, str) or not line.text.strip():
            raise AudioPipelineError("lyrics", "lyric lines must contain text")
        if line.confidence is not None:
            _confidence(line.confidence, "lyrics")
        if timed:
            if line.start_sec is None or line.end_sec is None:
                raise AudioPipelineError("lyrics", "recognized lines need start and end times")
            if (not math.isfinite(line.start_sec + line.end_sec)
                    or line.start_sec < previous_end or line.end_sec <= line.start_sec):
                raise AudioPipelineError(
                    "lyrics", "recognized line times must be finite and non-overlapping",
                )
            previous_end = line.end_sec
    return result


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
    offset = 0
    for line, reading in zip(lines, readings, strict=True):
        spans.append(LyricSpan(
            line.text,
            (offset, offset + len(line.text)),
            (ReadingCandidate(
                reading.kana, reading.source, reading.confidence,
            ),),
        ))
        offset += len(line.text) + 1

    evidence: list[Evidence] = []
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
    adapters: AudioAdapters,
    *,
    lyrics: Sequence[str] | None = None,
) -> ScoreDocument:
    """Run configured acoustic adapters and compile one canonical score JSON model.

    Supplying ``lyrics`` skips automatic lyric recognition. The selected text is
    never rewritten by this function; pronunciation and timing remain separate
    evidence supplied by their adapters.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if lyrics is None:
        if adapters.lyric_recognizer is None:
            raise AudioPipelineError(
                "lyrics", "a lyric recognizer is required when lyrics are not supplied",
            )
        lines = _validate_lines(
            _run_adapter("lyrics", adapters.lyric_recognizer, path), timed=True,
        )
    else:
        if isinstance(lyrics, (str, bytes)):
            raise TypeError("lyrics must be a sequence of lines, not one string")
        lines = _validate_lines(tuple(LyricLine(text) for text in lyrics), timed=False)

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
    line_windows = (
        {f"u{index}": (line.start_sec, line.end_sec)
         for index, line in enumerate(lines)
         if line.start_sec is not None and line.end_sec is not None}
        or None
    )
    return compile_score(observations, line_windows_by_utterance=line_windows)
