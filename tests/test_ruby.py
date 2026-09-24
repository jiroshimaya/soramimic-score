import unittest

from soramimic_score.ruby import ruby_segments


class RubyTests(unittest.TestCase):
    def test_selected_reading_above_kanji_and_not_kana(self):
        self.assertEqual(ruby_segments("明日は晴れ", "アシタワハレ"), [
            {"text": "明日", "reading": "アシタ"},
            {"text": "は", "reading": ""},
            {"text": "晴れ", "reading": "ハレ"},
        ])

    def test_nonstandard_pronunciation_is_not_falsely_assigned_to_tokens(self):
        self.assertEqual(ruby_segments("明日は晴れ", "アスワハレ"),
                         [{"text": "明日は晴れ", "reading": "アスワハレ"}])

    def test_latin_and_digits_have_ruby(self):
        pieces = ruby_segments("Soramimic 100回", "ソラミーヒャッカイ")
        self.assertEqual("".join(part["text"] for part in pieces), "Soramimic 100回")
        self.assertEqual([part["reading"] for part in pieces if part["reading"]],
                         ["ソラミー", "ヒャッカイ"])
