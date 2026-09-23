import tempfile
import unittest
from pathlib import Path

from soramimic_score import (
    AlignedMora,
    AudioAdapters,
    AudioPipelineError,
    LyricLine,
    MelodyNote,
    ReadingSelection,
    analyze_audio,
    build_audio_observations,
)


class AudioPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.audio = Path(self.temporary.name) / "input.wav"
        self.audio.write_bytes(b"test adapter boundary")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _readings(_path, lines):
        table = {"空": "ソラ", "耳": "ミミ"}
        return tuple(ReadingSelection(table[line.text], "test-reading", .9) for line in lines)

    @staticmethod
    def _moras(_path, _lines, readings):
        result = []
        time = 0.0
        for line_index, reading in enumerate(readings):
            for mora_index, kana in enumerate(reading.kana):
                result.append(AlignedMora(
                    line_index, mora_index, kana, time, time + .2, .8,
                    "test-ctc", time + .03,
                ))
                time += .2
        return tuple(result)

    @staticmethod
    def _melody(_path):
        return tuple(
            MelodyNote(index * .2, (index + 1) * .2, 60 + index, "test-melody", .7)
            for index in range(4)
        )

    def test_automatic_lyrics_run_all_four_adapters_to_score(self):
        calls = []

        def recognize(path):
            calls.append(("lyrics", path))
            return (LyricLine("空", 0, .4, .85), LyricLine("耳", .4, .8, .8))

        def readings(path, lines):
            calls.append(("readings", path, tuple(line.text for line in lines)))
            return self._readings(path, lines)

        def align(path, lines, selected):
            calls.append(("moras", path, tuple(item.kana for item in selected)))
            return self._moras(path, lines, selected)

        def melody(path):
            calls.append(("melody", path))
            return self._melody(path)

        score = analyze_audio(
            self.audio,
            AudioAdapters(readings, align, melody, recognize),
        )

        self.assertEqual(score.score.canonical_text, "空\n耳")
        self.assertEqual([item.surface for item in score.observations.utterances], ["空", "耳"])
        self.assertEqual(
            [item.kana for item in score.observations.readings], ["ソラ", "ミミ"],
        )
        self.assertTrue(all(
            any(note.id == f"audio-note-{index}"
                for note in score.observations.note_candidates)
            for index in range(4)
        ))
        self.assertEqual([item[0] for item in calls], ["lyrics", "readings", "moras", "melody"])
        self.assertTrue(any(
            item.kind == "mora-ctc-anchor" and item.source == "test-ctc"
            for item in score.observations.evidence
        ))

    def test_known_lyrics_skip_recognition_without_rewriting_text(self):
        def reject(_path):
            raise AssertionError("known lyrics must skip recognition")

        score = analyze_audio(
            self.audio,
            AudioAdapters(self._readings, self._moras, self._melody, reject),
            lyrics=("空", "耳"),
        )
        self.assertEqual(score.score.canonical_text, "空\n耳")

    def test_observation_builder_marks_uncalibrated_note_confidence(self):
        document = build_audio_observations(
            (LyricLine("空", 0, .4),),
            (ReadingSelection("ソラ", "test-reading", 1),),
            (
                AlignedMora(0, 0, "ソ", 0, .2, .8),
                AlignedMora(0, 1, "ラ", .2, .4, .8),
            ),
            (MelodyNote(0, .4, 60),),
        )
        note = document.note_candidates[0]
        evidence = next(item for item in document.evidence if item.id in note.evidence_ids)
        self.assertEqual(note.confidence, 0)
        self.assertEqual(evidence.detail, {"confidence_available": False})

    def test_known_lyrics_adjustment_is_opt_in_and_keeps_audit_evidence(self):
        score = analyze_audio(
            self.audio,
            AudioAdapters(self._readings, self._moras, self._melody,
                          lambda _: (LyricLine("空", 0, .4), LyricLine("空", .4, .8))),
            lyrics=("耳", "空"), adjust_lyrics=True,
        )
        self.assertEqual(score.score.canonical_text, "空\n空")
        audit = next(item for item in score.observations.evidence if item.kind == "lyric-adjustment")
        self.assertEqual(audit.detail["supplied_lines"], ["耳", "空"])
        self.assertEqual([row["operation"] for row in audit.detail["decisions"]],
                         ["keep", "repeat", "remove"])

    def test_adjustment_requires_known_lyrics_and_a_recognizer(self):
        adapters = AudioAdapters(self._readings, self._moras, self._melody)
        with self.assertRaisesRegex(ValueError, "requires supplied lyrics"):
            analyze_audio(self.audio, adapters, adjust_lyrics=True)
        with self.assertRaisesRegex(AudioPipelineError, "requires a recognizer"):
            analyze_audio(self.audio, adapters, lyrics=("空",), adjust_lyrics=True)

    def test_inconsistent_mora_result_is_rejected_before_compilation(self):
        def bad_moras(_path, _lines, _readings):
            return (AlignedMora(0, 0, "ソ", 0, .2, .8),)

        with self.assertRaisesRegex(AudioPipelineError, "selected reading"):
            analyze_audio(
                self.audio,
                AudioAdapters(self._readings, bad_moras, self._melody),
                lyrics=("空",),
            )

    def test_automatic_mode_requires_timed_nonoverlapping_lines(self):
        adapters = AudioAdapters(
            self._readings,
            self._moras,
            self._melody,
            lambda _path: (LyricLine("空", 0, .4), LyricLine("耳", .3, .8)),
        )
        with self.assertRaisesRegex(AudioPipelineError, "non-overlapping"):
            analyze_audio(self.audio, adapters)


if __name__ == "__main__":
    unittest.main()
