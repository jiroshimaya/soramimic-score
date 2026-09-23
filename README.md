# Soramimic Score

Soramimic Score turns recognized singing observations into one portable,
versioned JSON document. It keeps lyrics, readings, mora timing, melody-note
candidates, alignment decisions, confidence, and provenance instead of
flattening them into an opaque MIDI file.

The current `0.1` package is the model-free production core extracted from
Soramimic Video. Audio loading, automatic lyric recognition, mora CTC, and
melody-model adapters will move into this repository in subsequent releases.
Until then, callers provide their normalized observations and receive the same
JSON contract that future end-to-end analysis will emit.

```python
from soramimic_score import compile_score, dump

score = compile_score(observations)
dump(score, "song.score.json")
```

The JSON root is identified by `"format": "soramimic-score"` and an independent
integer `schema_version`. It contains both the evidence-preserving linked
observations and the consumer-facing `canonical`, `performed`, and
`synthesis_plan` score layers.

The same compiler is available as a command:

```sh
soramimic-score --input observations.json --output song.score.json
```

This repository intentionally contains no audio, songs, complete lyrics, MIDI,
model weights, user uploads, or evaluation truth.
