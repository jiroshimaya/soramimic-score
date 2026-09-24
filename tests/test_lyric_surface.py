import copy
import math
import tempfile
import unittest
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
    def test_supplied_reading_is_fixed_before_the_only_alignment(self):
        calls = []
        def readings(_path, lines):
            calls.append(("readings", [line.text for line in lines]))
            return [ReadingSelection("タチマチ", "test", 1, ("タチマチ",)) for line in lines]
        def align(_path, lines, selected):
            calls.append(("align", [r.kana for r in selected]))
            return [AlignedMora(0, i, kana, i * .2, (i + 1) * .2, 1)
                    for i, kana in enumerate(selected[0].kana)]
        adapters = AudioAdapters(
            readings, align, lambda _: (MelodyNote(0, .8, 60),),
            lambda _: (LyricLine("たつまち", 0, .8),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.touch()
            result = analyze_audio(path, adapters, lyrics=["たちまち"])
        self.assertEqual(calls, [("readings", ["たちまち"]), ("align", ["タチマチ"])])
        self.assertEqual(result.score.canonical_text, "たちまち")
        self.assertEqual([r.kana for r in result.observations.readings], ["タチマチ"])
        overlay = lyric_surface(result)
        self.assertEqual(overlay["groups"][0]["original_text"], "たつまち")
        self.assertTrue(overlay["readings_fixed_before_alignment"])
        self.assertEqual(lyric_surface(ScoreDocument.from_json(result.to_json())), overlay)

    def test_final_alignment_failure_never_restores_the_wrong_reading(self):
        calls = []
        def align(_path, _lines, selected):
            calls.append(selected[0].kana)
            raise ValueError("insufficient frames")
        adapters = AudioAdapters(
            lambda _p, _l: [ReadingSelection("タチマチ", "test", 1)], align,
            lambda _: (MelodyNote(0, .8, 60),),
            lambda _: (LyricLine("たつまち", 0, .8),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.touch()
            with self.assertRaisesRegex(AudioPipelineError, "insufficient frames"):
                analyze_audio(path, adapters, lyrics=["たちまち"])
        self.assertEqual(calls, ["タチマチ"])

    def test_split_recognition_is_joined_before_reading_selection(self):
        from soramimic_score import plan_lyric_inputs
        result = plan_lyric_inputs(
            [SurfaceLine("青い空", "アオイソラ"), SurfaceLine("白い雲", "シロイクモ"),
             SurfaceLine("ねこ", "ネコ")],
            [SurfaceLine("青い空白い雲", "アオイソラシロイクモ"), SurfaceLine("遠い星", "トオイホシ")],
        )
        self.assertEqual(result["groups"][0]["asr_indices"], [0, 1])
        self.assertEqual(result["groups"][0]["line_indices"], [0])
        self.assertEqual(result["groups"][1]["asr_indices"], [2])
        self.assertEqual(result["groups"][1]["line_indices"], [1])
        self.assertEqual(result["groups"][1]["reading_source"], "automatic-unmatched")
        self.assertEqual(result["unused_supplied_indices"], [1])

    def test_touching_float_dust_is_not_a_real_overlap(self):
        from soramimic_score.audio import _validate_lines
        result = _validate_lines([LyricLine("a", 0, 1.0000000000000002),
                                  LyricLine("b", 1, 2)], timed=True)
        self.assertEqual(result[1].start_sec, result[0].end_sec)
        with self.assertRaises(AudioPipelineError):
            _validate_lines([LyricLine("a", 0, 1.01), LyricLine("b", 1, 2)], timed=True)


if __name__ == "__main__":
    unittest.main()
