import tempfile
import unittest
from pathlib import Path

from soramimic_score.audio import (AlignedMora, AudioAdapters, LyricLine,
                                   MelodyNote, ReadingSelection, analyze_audio)
from soramimic_score.local_recovery import (coalesce_repeated_suffix_fragments,
                                            deficit_windows, uncovered_note_windows)


class LocalRecoveryTests(unittest.TestCase):
    def test_short_suffix_joins_only_with_parallel_refrain_evidence(self):
        lines = (LyricLine("君に届け", 0, 2), LyricLine("君に", 3, 4),
                 LyricLine("届け", 4.02, 4.5))
        merged, evidence = coalesce_repeated_suffix_fragments(lines)
        self.assertEqual([line.text for line in merged], ["君に届け", "君に届け"])
        self.assertEqual(evidence, ((1, 2),))
        merged, evidence = coalesce_repeated_suffix_fragments(lines[1:])
        self.assertEqual(merged, lines[1:])
        self.assertFalse(evidence)

    def test_note_rich_short_line_gets_bounded_retry_window(self):
        lines = (LyricLine("ララ", 0, 1.6), LyricLine("カキクケ", 2, 2.8),
                 LyricLine("サシスセ", 3, 3.8))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(8))
        notes += tuple(MelodyNote(2 + i * .2, 2 + (i + 1) * .2, 62)
                       for i in range(4))
        notes += tuple(MelodyNote(3 + i * .2, 3 + (i + 1) * .2, 64)
                       for i in range(4))
        self.assertEqual(deficit_windows(lines, notes, (2, 4, 4)), ((0, 0, 1.6),))

    def test_long_uncovered_note_run_is_retried(self):
        lines = (LyricLine("ア", 0, .4), LyricLine("イ", 3, 3.4))
        notes = tuple(MelodyNote(1 + i * .2, 1 + (i + 1) * .2, 60)
                      for i in range(8))
        self.assertEqual(uncovered_note_windows(lines, notes), ((.7, 2.9),))

    def test_local_retry_recovers_repeated_syllables_without_changing_peers(self):
        lines = (LyricLine("ララ", 0, 1.6), LyricLine("カキクケ", 2, 2.8),
                 LyricLine("サシスセ", 3, 3.8))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(8))
        notes += tuple(MelodyNote(2 + i * .2, 2 + (i + 1) * .2, 62)
                       for i in range(4))
        notes += tuple(MelodyNote(3 + i * .2, 3 + (i + 1) * .2, 64)
                       for i in range(4))
        retries = []

        def retry(_path, start, end):
            retries.append((start, end))
            return (LyricLine("ララララ", 0, 1.6),)

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + (line.end_sec-line.start_sec)*offset/len(line.text),
                                     line.start_sec + (line.end_sec-line.start_sec)*(offset+1)/len(line.text),
                                     .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            document = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
                lyric_reading=lambda text: text, lyric_recoverer=retry,
            ))
        self.assertEqual(retries, [(0, 1.6)])
        self.assertEqual(document.score.canonical_text, "ララララ\nカキクケ\nサシスセ")
        self.assertTrue(any(e.kind == "lyric-local-retry" for e in document.observations.evidence))
