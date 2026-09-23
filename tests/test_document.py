import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from soramimic_score import (
    FORMAT_NAME,
    ScoreDocument,
    ValidationError,
    compile_score,
    dump,
    dumps,
    load,
    loads,
)
from tests.test_pipeline import with_ctc
from tests.test_realization import document
from tests.test_correspondence import _note


class ScoreDocumentTests(unittest.TestCase):
    def score_document(self) -> ScoreDocument:
        observations = with_ctc(document("カ", 1), (0.0,))
        observations = replace(
            observations,
            note_candidates=(_note("n0", 0.0, 0.4),),
        )
        return compile_score(observations)

    def test_round_trip_is_lossless_and_deterministic(self):
        score = self.score_document()
        encoded = dumps(score)
        self.assertTrue(encoded.endswith("\n"))
        self.assertEqual(dumps(loads(encoded)), encoded)
        self.assertEqual(loads(encoded), score)
        self.assertEqual(json.loads(encoded)["format"], FORMAT_NAME)

    def test_file_helpers_use_the_same_contract(self):
        score = self.score_document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "song.score.json"
            self.assertEqual(dump(score, path), path)
            self.assertEqual(load(path), score)

    def test_unknown_schema_is_rejected_without_migration(self):
        raw = self.score_document().to_dict()
        raw["schema_version"] = 2
        with self.assertRaisesRegex(ValidationError, "schema_version"):
            ScoreDocument.from_dict(raw)

    def test_score_and_observation_text_must_match(self):
        score = self.score_document()
        with self.assertRaisesRegex(ValidationError, "canonical_text"):
            replace(score, score=replace(score.score, canonical_text="キ"))


if __name__ == "__main__":
    unittest.main()
