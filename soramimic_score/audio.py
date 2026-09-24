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
from .line_windows import snap_line_windows_to_rests
from .note_runs import NoteRunConfig
from .semantic import credit_recovery_windows, has_melodic_support, is_credit_hallucination


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
LyricRecoverer = Callable[[Path, float, float], Sequence[LyricLine]]


@dataclass(frozen=True)
class AudioAdapters:
    """Model boundaries needed by the audio pipeline.

    ``lyric_recognizer`` is required by the ASR-first audio pipeline.
    ``lyric_reading`` optionally supplies linguistic readings for whole-line
    comparison (otherwise normalized surface text is used). ``lyric_recoverer``
    can retry one singing window when Whisper emits a credit template. The
    reading, mora, and melody adapters are required in both modes.
    """

    reading_selector: ReadingSelector
    mora_aligner: MoraAligner
    melody_transcriber: MelodyTranscriber
    lyric_recognizer: LyricRecognizer | None = None
    lyric_reading: Callable[[str], str] | None = None
    lyric_recoverer: LyricRecoverer | None = None


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
    on_progress: Callable[[str], None] | None = None,
    accompaniment_path: Path | None = None,
) -> ScoreDocument:
    """Locate lyrics with ASR, fix pronunciation, then align the final reading.

    Matched supplied text is authoritative for reading selection. Recognition
    locates its interval; it is not an alternative pronunciation. Unmatched
    recognition and unused supplied lines remain explicitly recorded. Optional
    whole-line deletion/completion requires ``adjust_lyrics=True``.
    """
    from .japanese import strip_ruby
    from .surface import SurfaceLine, attach_lyric_surface, plan_lyric_inputs

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if adapters is not None and model_config is not None:
        raise ValueError("Specify adapters or model_config, not both")
    if adjust_lyrics and lyrics is None:
        raise ValueError("adjust_lyrics requires supplied lyrics")
    if adapters is None:
        from .models import prepared_adapters
        from .media import decoded_audio
        if model_config is None:
            raise ValueError("model_config with local SheetSage2 directories is required")
        if on_progress:
            on_progress("音声を読み込んでいます")
        with decoded_audio(path) as prepared_path:
            if on_progress and model_config.separate_vocals:
                on_progress("歌声を分離しています")
            preparation = (prepared_adapters(prepared_path, model_config)
                           if accompaniment_path is None else
                           prepared_adapters(prepared_path, model_config,
                                             accompaniment_path=accompaniment_path))
            with preparation as prepared:
                return analyze_audio(prepared_path, prepared, lyrics=lyrics,
                                     adjust_lyrics=adjust_lyrics, on_progress=on_progress)

    if lyrics is not None:
        if isinstance(lyrics, (str, bytes)):
            raise TypeError("lyrics must be a sequence of lines, not one string")
        _validate_lines(tuple(LyricLine(text) for text in lyrics), timed=False)
    if adapters.lyric_recognizer is None:
        raise AudioPipelineError("lyrics", "ASR-first analysis requires a recognizer")
    if on_progress:
        on_progress("歌詞を認識しています")
    raw_recognized = _validate_lines(
        _run_adapter("lyrics", adapters.lyric_recognizer, path), timed=True,
    )
    if on_progress:
        on_progress("音符と音高を推定しています")
    notes = _validate_notes(_run_adapter("melody", adapters.melody_transcriber, path))
    recognized_lines = []
    semantic_evidence = []
    supplied_surfaces = {strip_ruby(text).strip() for text in lyrics or ()}
    for index, line in enumerate(raw_recognized):
        if not is_credit_hallucination(line.text) or line.text.strip() in supplied_surfaces:
            recognized_lines.append(line)
            continue
        windows = credit_recovery_windows(line, notes)
        recovered = []
        if windows and adapters.lyric_recoverer is not None:
            if on_progress:
                on_progress("歌詞の誤認識区間を再確認しています")
            for start, end in windows:
                try:
                    candidates = _validate_lines(
                        adapters.lyric_recoverer(path, start, end), timed=True,
                    )
                except Exception:
                    continue
                recovered.extend(candidate for candidate in candidates
                                 if candidate.start_sec is not None
                                 and candidate.end_sec is not None
                                 and start <= candidate.start_sec < candidate.end_sec <= end
                                 and not is_credit_hallucination(candidate.text)
                                 and has_melodic_support(candidate, notes))
        recognized_lines.extend(recovered)
        semantic_evidence.append(Evidence(
            f"audio-credit-gate-{index}", "soramimic_score.semantic",
            "lyric-semantic-gate", 0.0,
            {"source_segment_index": index, "surface": line.text,
             "status": "recovered" if recovered else "rejected",
             "recovery_windows": [list(window) for window in windows],
             "recovered_count": len(recovered)},
        ))
    recognized_lines.sort(key=lambda item: (item.start_sec, item.end_sec))
    recognized = _validate_lines(recognized_lines, timed=True)
    adjustment, overlay = None, None
    lines = recognized
    if lyrics is not None:
        if adjust_lyrics:
            from .lyrics import adjust_known_lyrics
            adjustment = adjust_known_lyrics(lyrics, recognized, reading=adapters.lyric_reading)
            lines = adjustment.lines
        else:
            convert = adapters.lyric_reading or (lambda _: "")
            overlay = plan_lyric_inputs(
                [SurfaceLine(strip_ruby(line.text), convert(line.text)) for line in recognized],
                [SurfaceLine(strip_ruby(text), convert(text)) for text in lyrics],
            )
            overlay["supplied_lines"] = list(lyrics)
            prepared = []
            for index, group in enumerate(overlay["groups"]):
                sources = [recognized[i] for i in group["asr_indices"]]
                text = ("\n".join(lyrics[i] for i in group["supplied_indices"])
                        if group["operation"] == "match" else sources[0].text)
                prepared.append(LyricLine(text, sources[0].start_sec, sources[-1].end_sec))
                group.update(utterance_ids=[f"u{index}"], start_sec=sources[0].start_sec,
                             end_sec=sources[-1].end_sec)
            lines = _validate_lines(prepared, timed=True)

    # Only the final text reaches the selector. Its closed candidates contain
    # no pronunciation copied from a different recognized surface.
    if on_progress:
        on_progress("歌詞の読みを確認しています")
    readings = _validate_readings(
        lines, _run_adapter("readings", adapters.reading_selector, path, lines),
    )
    if overlay is not None:
        for group, reading in zip(overlay["groups"], readings, strict=True):
            group["original_acoustic_reading"] = group["acoustic_reading"]
            group["acoustic_reading"] = reading.kana
            group["reading_candidates"] = list(reading.candidates)
        overlay.pop("acoustic_changes", None)
        overlay["readings_fixed_before_alignment"] = True
    lines = tuple(replace(line, text=strip_ruby(line.text)) for line in lines)
    # A failure is reported. Do not fall back to the rejected ASR pronunciation.
    if on_progress:
        on_progress("モーラの時刻を推定しています")
    moras = _validate_moras(
        readings, _run_adapter("mora alignment", adapters.mora_aligner, path, lines, readings),
    )
    if on_progress:
        on_progress("楽譜データを組み立てています")
    observations = build_audio_observations(lines, readings, moras, notes)
    if semantic_evidence:
        observations = replace(observations,
                               evidence=observations.evidence + tuple(semantic_evidence))
    if adjustment is not None:
        observations = replace(observations, evidence=observations.evidence + (Evidence(
            "audio-lyric-adjustment", "soramimic_score.lyrics", "lyric-adjustment", 0.0,
            adjustment.detail,
        ),))
    line_windows = (
        {f"u{index}": (line.start_sec, line.end_sec)
         for index, line in enumerate(lines)
         if line.start_sec is not None and line.end_sec is not None} or None
    )
    if line_windows is not None and len(line_windows) == len(lines):
        snapped = snap_line_windows_to_rests(
            tuple(line_windows[f"u{index}"] for index in range(len(lines))), notes,
        )
        line_windows = {f"u{index}": window
                        for index, window in enumerate(snapped)}
    result = compile_score(
        observations,
        config=NoteRunConfig(whisper_boundary_cost_per_sec2=.1),
        line_windows_by_utterance=line_windows,
    )
    return attach_lyric_surface(result, overlay) if overlay is not None else result
