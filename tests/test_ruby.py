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
                {"text": "晴れ", "reading": "ハレ"},
            ])

    def test_nonstandard_pronunciation_is_not_falsely_assigned_to_tokens(self):
        self.assertEqual(ruby_segments("明日は晴れ", "アスワハレ"),
                         [{"text": "明日は晴れ", "reading": "アスワハレ"}])

    def test_latin_and_digits_have_ruby(self):
        tokens = [{"surface_form": s, "pronunciation": p} for s, p in
                  (("Soramimic", "ソラミー"), (" ", ""), ("100回", "ヒャッカイ"))]
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=lambda *_args, **_kw: tokens)):
            pieces = ruby_segments("Soramimic 100回", "ソラミーヒャッカイ")
        self.assertEqual("".join(part["text"] for part in pieces), "Soramimic 100回")
        self.assertEqual([part["reading"] for part in pieces if part["reading"]],
                         ["ソラミー", "ヒャッカイ"])
