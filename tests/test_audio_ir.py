from dataclasses import replace
import unittest

from soramimic_score import (
    Boundary, Evidence, IntermediateRepresentation, LyricSpan, ReadingCandidate,
    ValidationError, build_known_lyrics_document,
)


class IntermediateRepresentationTests(unittest.TestCase):
    def setUp(self):
        evidence = Evidence("e0", "synthetic-ctc", "phoneme-alignment", 0.9,
                            {"fixture": True})
        boundary = lambda value: Boundary(value, 0.8, ("e0",))
        from soramimic_score import ObservedSingingUnit
        observed = ObservedSingingUnit(("カ",), boundary(0.1), boundary(0.12),
                                       boundary(0.4), 0.9, ("e0",))
        self.document = build_known_lyrics_document(
            "か", (LyricSpan("か", (0, 1),
                             (ReadingCandidate("カ", "synthetic-g2p", 1.0),)),),
            (observed,), (evidence,))

    def test_schema_round_trip_is_lossless(self):
        encoded = self.document.to_json()
        self.assertEqual(IntermediateRepresentation.from_json(encoded), self.document)
        self.assertEqual(IntermediateRepresentation.from_dict(self.document.to_dict()), self.document)

    def test_validation_rejects_dangling_evidence(self):
        unit = replace(self.document.singing_units[0], evidence_ids=("missing",))
        with self.assertRaisesRegex(ValidationError, "missing evidence"):
            replace(self.document, singing_units=(unit,))

    def test_validation_rejects_unsupported_schema(self):
        raw = self.document.to_dict()
        raw["schema_version"] = 2
        with self.assertRaisesRegex(ValidationError, "unsupported schema_version"):
            IntermediateRepresentation.from_dict(raw)

    def test_boundary_confidence_and_provenance_survive_json(self):
        restored = IntermediateRepresentation.from_json(self.document.to_json())
        nucleus = restored.vowel_nuclei[0]
        self.assertEqual(nucleus.start.time_sec, 0.12)
        self.assertEqual(nucleus.start.confidence, 0.8)
        self.assertEqual(nucleus.start.evidence_ids, ("e0",))


if __name__ == "__main__":
    unittest.main()
