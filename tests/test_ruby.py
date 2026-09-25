import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from soramimic_score.ruby import ruby_segments


class RubyTests(unittest.TestCase):
    def test_selected_reading_above_kanji_and_not_kana(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("明日", "アシタ"), ("は", "ワ"), ("晴れ", "ハレ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("明日は晴れ", "アシタワハレ"), [
                {"text": "明日", "reading": "あした"},
                {"text": "は", "reading": ""},
                {"text": "晴", "reading": "は"},
                {"text": "れ", "reading": ""},
            ])

    def test_nonstandard_pronunciation_is_localized_to_one_word(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("明日", "アシタ"), ("は", "ワ"), ("晴れ", "ハレ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("明日は晴れ", "アスワハレ"),
                             [{"text": "明日", "reading": "あす"},
                              {"text": "は", "reading": ""},
                              {"text": "晴", "reading": "は"},
                              {"text": "れ", "reading": ""}])

    def test_kana_anchor_locates_two_changed_words(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("明日", "アシタ"), ("は", "ワ"), ("晴れ", "ハレ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("明日は晴れ", "アスワセイレ"), [
                {"text": "明日", "reading": "あす"},
                {"text": "は", "reading": ""},
                {"text": "晴", "reading": "せい"},
                {"text": "れ", "reading": ""},
            ])

    def test_inflected_word_only_marks_kanji(self):
        tokens = [{"surface_form": "歌っ", "pronunciation": "ウタッ"},
                  {"surface_form": "た", "pronunciation": "タ"}]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("歌った", "ウタッタ"), [
                {"text": "歌", "reading": "うた"},
                {"text": "っ", "reading": ""},
                {"text": "た", "reading": ""},
            ])

    def test_latin_and_digits_have_ruby(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("Soramimic", "ソラミー"), (" ", ""), ("100回", "ヒャッカイ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            pieces = ruby_segments("Soramimic 100回", "ソラミーヒャッカイ")
        self.assertEqual("".join(part["text"] for part in pieces), "Soramimic 100回")
        self.assertEqual([part["reading"] for part in pieces if part["reading"]],
                         ["そらみー", "ひゃっかい"])

    def test_video_style_mixed_script_runs_and_silent_separators(self):
        examples = (
            ("夜ノ街", "ヨルノマチ", [
                {"text": "夜", "reading": "よる"},
                {"text": "ノ", "reading": ""},
                {"text": "街", "reading": "まち"},
            ]),
            ("アテル＝参", "アテルサン", [
                {"text": "アテル＝", "reading": ""},
                {"text": "参", "reading": "さん"},
            ]),
            ("少女ケイ", "ショージョケー", [
                {"text": "少女", "reading": "しょーじょ"},
                {"text": "ケイ", "reading": ""},
            ]),
            ("バリッシュ・コノル", "バリッシュコノル", [
                {"text": "バリッシュ・コノル", "reading": ""},
            ]),
        )
        for surface, kana, expected in examples:
            with self.subTest(surface=surface), patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(
                get_tokens=lambda *_args, **_kw: [{"surface_form": surface, "pronunciation": kana}],
            )):
                self.assertEqual(ruby_segments(surface, kana), expected)

    def test_video_style_fallback_stays_with_one_word(self):
        surface, kana = "静カ", "シズケサ"
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(
            get_tokens=lambda *_args, **_kw: [{"surface_form": surface, "pronunciation": kana}],
        )):
            self.assertEqual(ruby_segments(surface, kana), [
                {"text": surface, "reading": "しずけさ"},
            ])

    def test_tokenization_failure_does_not_put_ruby_over_a_whole_line(self):
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(
            get_tokens=lambda *_args, **_kw: (_ for _ in ()).throw(ValueError("unknown")),
        )):
            self.assertEqual(ruby_segments("明日は晴れ", "アシタワハレ"), [
                {"text": "明日は晴れ", "reading": ""},
            ])
