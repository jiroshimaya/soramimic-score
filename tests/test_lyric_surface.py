import copy
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from soramimic_score import (
    AlignedMora, AudioAdapters, AudioPipelineError, LyricLine, MelodyNote, ReadingSelection,
    ScoreDocument, SurfaceLine, align_lyric_surface, analyze_audio, lyric_surface,
)


class SurfaceTests(unittest.TestCase):
    def test_homophone_and_different_line_breaks(self):
        recognized = [SurfaceLine("はしを渡る白い雲", "ハシオワタルシロイクモ")]
        supplied = [SurfaceLine("橋を渡る", "ハシオワタル"), SurfaceLine("白い雲", "シロイクモ")]
        original = copy.deepcopy((recognized, supplied))
        result = align_lyric_surface(recognized, supplied)
        self.assertEqual(result["display_text"], "橋を渡る\n白い雲")
        self.assertEqual(result["groups"][0]["supplied_indices"], [0, 1])
        self.assertEqual((recognized, supplied), original)
        self.assertFalse(result["acoustic_changes"])

    def test_many_recognized_lines_share_one_display_group(self):
        result = align_lyric_surface(
            [SurfaceLine("青い空", "アオイソラ"), SurfaceLine("白い雲", "シロイクモ")],
            [SurfaceLine("青い空白い雲", "アオイソラシロイクモ")],
        )
        self.assertEqual(result["groups"][0]["asr_indices"], [0, 1])

    def test_unrelated_and_ambiguous_abstain(self):
        for source, target, operation in [
            ([SurfaceLine("青い空", "アオイソラ")], [SurfaceLine("電車", "デンシャ")], "retain_asr"),
            ([SurfaceLine("はし", "ハシ")], [SurfaceLine("橋", "ハシ"), SurfaceLine("箸", "ハシ")],
             "ambiguous_retain_asr"),
        ]:
            result = align_lyric_surface(source, target)
            self.assertEqual(result["display_text"], source[0].text)
            self.assertEqual(result["groups"][0]["operation"], operation)
            self.assertEqual(sorted(result["unused_supplied_indices"]), list(range(len(target))))

    def test_repetition_is_not_deleted_or_fabricated(self):
        source = [SurfaceLine("歌おう", "ウタオー")] * 3
        result = align_lyric_surface(source, source[:2])
        self.assertEqual(result["display_text"], "歌おう\n歌おう\n歌おう")
        self.assertEqual([i for g in result["groups"] for i in g["asr_indices"]], [0, 1, 2])
        result = align_lyric_surface(source[:1], [source[0], SurfaceLine("電車", "デンシャ")])
        self.assertEqual(result["unused_supplied_indices"], [1])

    def test_group_cannot_hide_an_unsupported_short_line(self):
        line = SurfaceLine("青い空白い雲", "アオイソラシロイクモ")
        result = align_lyric_surface([line], [line, SurfaceLine("猫", "ネコ")])
        self.assertEqual(result["unused_supplied_indices"], [1])

    def test_matching_reading_does_not_overwrite_sung_reading(self):
        result = align_lyric_surface([SurfaceLine("明日は晴れ", "アスワハレ", "アシタワハレ")],
                                     [SurfaceLine("明日は晴れ", "アシタワハレ")])
        self.assertEqual(result["groups"][0]["acoustic_reading"], "アスワハレ")

    def test_normalization_and_validation(self):
        result = align_lyric_surface([SurfaceLine("hello!")], [SurfaceLine("ＨＥＬＬＯ")])
        self.assertEqual(result["display_text"], "ＨＥＬＬＯ")
        for options in ({"minimum_similarity": math.nan}, {"max_group_lines": True},
                        {"per_line_coverage": 0}):
            with self.assertRaises(ValueError):
                align_lyric_surface([SurfaceLine("空")], [SurfaceLine("空")], **options)
        with self.assertRaises(ValueError):
            align_lyric_surface([], [])


class LocalReadingTests(unittest.TestCase):
    def test_local_refinement_keeps_other_line_and_roundtrips_provenance(self):
        calls = []
        def readings(_path, lines):
            return tuple(ReadingSelection("アス" if l.text == "明日" else "ソラ", "test", 1)
                         for l in lines)
        def align(_path, lines, selected):
            calls.append([l.text for l in lines])
            output = []
            for i, (line, reading) in enumerate(zip(lines, selected)):
                step = (line.end_sec - line.start_sec) / len(reading.kana)
                output.extend(AlignedMora(i, j, kana, line.start_sec + j * step,
                                         line.start_sec + (j + 1) * step, 1)
                              for j, kana in enumerate(reading.kana))
            return output
        def refine(_path, lines, baseline):
            return tuple(ReadingSelection("アシタ", "test-acoustic", 1, ("アス", "アシタ"))
                         if line.text == "明日" else current
                         for line, current in zip(lines, baseline, strict=True))
        adapters = AudioAdapters(
            readings, align,
            lambda _: tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(6)),
            lambda _: (LyricLine("明日", 0, .6), LyricLine("空", .8, 1.2)),
            lambda text: "アシタ" if text == "明日" else "ソラ", refine,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.touch()
            automatic = analyze_audio(path, replace(adapters, reading_refiner=None))
            result = analyze_audio(path, adapters, lyrics=["明日", "空"])
        self.assertEqual(calls[-1], ["明日"])
        self.assertEqual(lyric_surface(result)["reading_reviews"][0]["status"], "applied")
        self.assertTrue(lyric_surface(result)["acoustic_changes"])
        self.assertEqual([s.kana for s in result.score.synthesis_plan if s.utterance_id == "u1"],
                         [s.kana for s in automatic.score.synthesis_plan if s.utterance_id == "u1"])
        self.assertEqual(lyric_surface(ScoreDocument.from_json(result.to_json())), lyric_surface(result))

    def test_touching_float_dust_is_not_a_real_overlap(self):
        from soramimic_score.audio import _validate_lines
        result = _validate_lines([LyricLine("a", 0, 1.0000000000000002),
                                  LyricLine("b", 1, 2)], timed=True)
        self.assertEqual(result[1].start_sec, result[0].end_sec)
        with self.assertRaises(AudioPipelineError):
            _validate_lines([LyricLine("a", 0, 1.01), LyricLine("b", 1, 2)], timed=True)


if __name__ == "__main__":
    unittest.main()
