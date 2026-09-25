import tempfile
import unittest
from pathlib import Path

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             ReadingSelection, analyze_audio)
from soramimic_score.ctc_reattacks import decode_repeated_mora_reattacks
from soramimic_score.vocalization import VocalizationReattack


class ReattackParityTests(unittest.TestCase):
    def test_unconditioned_ctc_peaks_respect_audio_window(self):
        try:
            import torch
        except ImportError:
            self.skipTest("audio dependencies not installed")
        logits = torch.full((100, 2), -10.)
        for frame in (30, 40, 50, 90):
            logits[frame, 1] = -.1
        found = decode_repeated_mora_reattacks(
            logits, {"ラ": 1}, "ラ", 0, .6, stride=320, rate=16000)
        self.assertEqual(len(found), 3)
        self.assertEqual([round(item.start_sec, 2) for item in found], [.1, .3, .5])

    def test_automatic_repeats_reach_note_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "input.wav"
            audio.touch()
            adapters = AudioAdapters(
                lambda _path, _lines: (ReadingSelection("ララ", "test", 1),),
                lambda _path, _lines, _readings: tuple(
                    AlignedMora(0, index, "ラ", index * .2, (index + 1) * .2, .9)
                    for index in range(2)),
                lambda _path: tuple(
                    MelodyNote(index * .2, (index + 1) * .2, 60 + index,
                               "sheetsage2-vocal") for index in range(4)),
                lambda _path: (LyricLine("ララ", 0, .8),),
                vocalization_reattacks=lambda _mora, _start, _end: tuple(
                    VocalizationReattack(index * .15, index * .15 + .02, .9, "ctc")
                    for index in range(4)),
            )
            result = analyze_audio(audio, adapters)
        expansion = [item for item in result.observations.evidence
                     if item.kind == "repeated-vocalization-expansion"]
        self.assertEqual(len(expansion), 1)
        self.assertEqual(expansion[0].detail["expanded_repetitions"], 4)
        self.assertEqual(sum(note.id.startswith("audio-note-")
                             for note in result.observations.note_candidates), 4)
