# Soramimic Score Stage 3 core

`soramimic_score` is the model-free production core shared with Soramimic Video. It
does one job: associate an already selected lyric realization with independent
melody-note candidates, then compile the result into a reversible singing plan.

Soramimic Video owns all audio-model adapters and policy before this boundary:

- Demucs separation;
- normal Whisper surface transcription for unknown lyrics;
- Japanese reading candidates and KanaWhisper reranking;
- ReazonSpeech mora CTC alignment;
- SheetSage2 melody transcription.

This package does not load audio, run those models, select lyric text, or rerun
recognition. Known and automatically transcribed lyrics enter the same Stage 3
path after Soramimic Video has selected their surface and reading.

## Input contract

`IntermediateRepresentation` is the versioned JSON boundary. A fresh Stage 3
input contains:

- canonical utterances, selected readings, morae, and singing units;
- caller-supplied acoustic evidence and mora timing;
- chronological `NoteCandidate` values from the melody adapter;
- no precomputed correspondence links.

`build_known_lyrics_document` is a construction helper for the lyric and mora
side of that document. Despite its historical name, the caller may use it for a
surface transcription selected upstream; the helper does no recognition.

Model-specific raw output stays with its adapter. Only normalized observations,
confidence, IDs, and provenance cross this boundary.

## One production runner

```python
from soramimic_score import VocalizationReattack
from soramimic_score.pipeline import run_stage3_document

run = run_stage3_document(
    observation_document,
    line_windows_by_utterance={"u0": (1.2, 4.8)},  # optional raw Whisper bounds
)

linked_document = run.document
realization = run.realization
note_run_decision = run.note_run
```

Automatic-transcription callers can recover the count of a Whisper-rounded
vocalization from an independent acoustic pass:

```python
run = run_stage3_document(
    observation_document,
    line_windows_by_utterance={"u0": (1.2, 4.8)},
    vocalization_reattacks_by_utterance={
        "u0": (
            VocalizationReattack(1.25, 1.31, 0.8, "raw-kana-ctc"),
            VocalizationReattack(1.62, 1.68, 0.7, "raw-kana-ctc"),
        ),
    },
)
```

Whisper supplies the repeated mora identity; it does not supply the recovered
count. The caller-provided re-attacks determine that count without a fixed cap.
Overlapping SheetSage2 notes determine pitch and duration and are required to
remain sung: notes between re-attacks become continuations rather than new
consonants or discarded tail notes. Each inferred repetition retains its
acoustic source and confidence as explicit evidence. Without re-attacks, normal
lyrics and known-lyrics runs do not change.

`run_stage3_document` validates the immutable observation document, partitions
notes at the midpoint between neighboring utterance CTC ranges, runs the
note-preserving decoder, materializes replayable links and derived note
candidates, and compiles all links together. Existing links are rejected
instead of being silently reused.

The score uses lyric onsets (normally raw mora CTC), selected-lyric segment
identity, and pitched SheetSage2 intervals. Rounded-vocalization recovery uses
caller-supplied acoustic re-attack onsets instead. Automatic-transcription
callers may also supply the raw
Whisper interval for every utterance. If an assigned note begins outside that
interval while its mora CTC onset remains inside, the decoder adds one soft,
quadratic line-ownership cost. A note beginning inside the interval keeps its
complete tail even when it extends past the boundary; a CTC onset that also
crosses the boundary supports the crossing without a confidence threshold.
The score does not use reference/XF notes, F0, note confidence, or singing-unit
intervals. Adjacent same-pitch fragments coalesce only inside one selected
syllable; another CTC syllable keeps the boundary, and a pitch change becomes
an explicit continuation slot. Unpitched candidates remain explicit `rest`
links.

## Correspondence and realization

The linked output retains every raw SheetSage2 candidate unchanged. Coalesced
or split final intervals are additional `NoteCandidate` values with
`note-run-derivation` evidence naming their source candidates. This makes a
saved linked document replayable through `compile_realization` without rerunning
the optimizer.

`compile_realization` exposes three separate layers:

- `canonical`: complete surface text and stable mora IDs;
- `performed`: observation state and selected correspondence;
- `synthesis_plan`: pronunciation allocated to concrete note slots.

Missing correspondence never deletes canonical text. An actual performance
omission requires an explicit `PerformanceOmission` backed by positive evidence.

## Alternate correspondence model

`align_correspondence` and `run_correspondence_document` retain the earlier
semi-Markov correspondence model for controlled comparisons. They are not
called by `run_stage3_document` or by the Soramimic Video production bridge.

## Canonical JSON and CLI

`ScoreDocument` is the single saved document. It wraps the linked observations
and their compiled score so exporters and applications need only one input.

```python
from soramimic_score import compile_score, dump, load

score = compile_score(observation_document)
dump(score, "song.score.json")
same_score = load("song.score.json")
```

Decode a fresh observation document and save one score document:

```sh
python -m soramimic_score \
  --input work/observations.json \
  --output work/song.score.json
```

Compile an already linked observation document without decoding again:

```sh
python -m soramimic_score \
  --input work/correspondence.json \
  --linked-input \
  --output work/song.score.json
```

The package has no model dependency and carries no audio, checkpoints, or song
data. Experimental recognizers, alignment studies, and evaluation harnesses live
under `research/`; they are not alternate production entry points.
