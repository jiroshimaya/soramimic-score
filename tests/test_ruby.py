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
                {"text": "明日", "reading": "アシタ"},
                {"text": "は", "reading": ""},
                {"text": "晴", "reading": "ハ"},
                {"text": "れ", "reading": ""},
            ])

    def test_nonstandard_pronunciation_is_localized_to_one_word(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("明日", "アシタ"), ("は", "ワ"), ("晴れ", "ハレ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("明日は晴れ", "アスワハレ"),
                             [{"text": "明日", "reading": "アス"},
                              {"text": "は", "reading": ""},
                              {"text": "晴", "reading": "ハ"},
                              {"text": "れ", "reading": ""}])

    def test_two_changed_words_do_not_gain_incorrect_ruby(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("明日", "アシタ"), ("は", "ワ"), ("晴れ", "ハレ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("明日は晴れ", "アスワセイレ"),
                             [{"text": "明日は晴れ", "reading": ""}])

    def test_inflected_word_only_marks_kanji(self):
        tokens = [{"surface_form": "歌っ", "pronunciation": "ウタッ"},
                  {"surface_form": "た", "pronunciation": "タ"}]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            self.assertEqual(ruby_segments("歌った", "ウタッタ"), [
                {"text": "歌", "reading": "ウタ"},
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
                         ["ソラミー", "ヒャッカイ"])
