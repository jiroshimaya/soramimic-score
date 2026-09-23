from dataclasses import replace
import unittest

from soramimic_score import (
    Boundary,
    CorrespondenceConfig,
    Evidence,
    IntermediateRepresentation,
    LyricSpan,
    NoteCandidate,
    ObservedSingingUnit,
    ReadingCandidate,
    ValidationError,
    align_correspondence,
    boundaryless_correspondence_config,
    build_known_lyrics_document,
)
from soramimic_score.correspondence import _CostModel


KANA = (("か", "カ"), ("き", "キ"), ("く", "ク"), ("け", "ケ"))


def _document(unit_times=(), notes=(), *, extra_evidence=()):
    evidence = [Evidence("ctc", "synthetic-ctc", "phoneme-alignment", 0.9)]
    observations = []
    spans = []
    canonical = "".join(KANA[index][0] for index in range(len(unit_times)))
    for index, (consonant, vowel, end) in enumerate(unit_times):
        evidence_ids = ["ctc"]
        boundary = lambda value: Boundary(value, 0.9, ("ctc",))
        observations.append(ObservedSingingUnit(
            (KANA[index][1],), boundary(consonant), boundary(vowel), boundary(end),
            0.9, tuple(evidence_ids),
        ))
        spans.append(LyricSpan(
            KANA[index][0], (index, index + 1),
            (ReadingCandidate(KANA[index][1], "synthetic", 1.0),),
        ))
    evidence.extend(extra_evidence)
    if unit_times:
        document = build_known_lyrics_document(
            canonical, tuple(spans), tuple(observations), tuple(evidence)
        )
        return replace(document, note_candidates=tuple(notes))
    return IntermediateRepresentation(
        schema_version=1, canonical_text="", evidence=tuple(evidence),
        note_candidates=tuple(notes),
    )


def _note(identifier, start, end, pitch=60, confidence=0.9, sources=("sheetsage2-vocal",),
          evidence_ids=()):
    return NoteCandidate(identifier, start, end, pitch, confidence, sources,
                         tuple(evidence_ids))


def _sokuon_document(*, closure=(0.30, 0.36), notes=(), leading_kana=True,
                     observe_closure=True):
    evidence = (Evidence("ctc", "synthetic-ctc", "phoneme-alignment", 0.9),)
    boundary = lambda value: Boundary(value, 0.9, ("ctc",))
    spans = []
    observations = []
    canonical = ""
    if leading_kana:
        canonical += "か"
        spans.append(LyricSpan("か", (0, 1),
                               (ReadingCandidate("カ", "synthetic", 1.0),)))
        observations.append(ObservedSingingUnit(
            ("カ",), boundary(0.0), boundary(0.05), boundary(closure[0]),
            0.9, ("ctc",),
        ))
    offset = len(canonical)
    canonical += "っ"
    spans.append(LyricSpan("っ", (offset, offset + 1),
                           (ReadingCandidate("ッ", "synthetic", 1.0),)))
    if observe_closure:
        observations.append(ObservedSingingUnit(
            ("ッ",), boundary(closure[0]), None, boundary(closure[1]),
            0.9, ("ctc",),
        ))
    document = build_known_lyrics_document(canonical, tuple(spans), tuple(observations), evidence)
    return replace(document, note_candidates=tuple(notes))


class CorrespondenceOperationTests(unittest.TestCase):
    def test_production_config_has_no_legacy_phone_boundary_weights(self):
        config = boundaryless_correspondence_config()
        self.assertEqual(config, CorrespondenceConfig())
        self.assertEqual(config.vowel_onset_weight, 0.0)
        self.assertEqual(config.interval_overlap_weight, 0.0)
        self.assertEqual(config.boundary_weight, 0.0)

    def test_match(self):
        document = _document(((0.0, 0.10, 0.60),),
                             (_note("n0", 0.10, 0.60),))
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["match"])
        self.assertEqual(result.links[0].singing_unit_ids, ("s0",))
        self.assertEqual(result.links[0].note_candidate_ids, ("n0",))
        self.assertGreater(result.links[0].cost_components.candidate_confidence, 0)

    def test_melisma(self):
        document = _document(
            ((0.0, 0.10, 0.90),),
            (_note("n0", 0.10, 0.45, 60), _note("n1", 0.45, 0.90, 62)),
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["melisma"])
        self.assertEqual(result.links[0].note_candidate_ids, ("n0", "n1"))
        self.assertGreater(result.links[0].cost_components.complexity, 0)

    def test_two_syllables_cannot_stack_on_one_note(self):
        document = _document(
            ((0.0, 0.10, 0.45), (0.40, 0.45, 0.90)),
            (_note("n0", 0.10, 0.90),),
        )
        result = align_correspondence(document)
        self.assertNotIn("stack", {link.operation for link in result.links})
        self.assertEqual(sum(link.operation == "unit_only" for link in result.links), 1)

    def test_one_syllable_split_across_units_can_share_one_note(self):
        evidence = (Evidence("ctc", "synthetic", "alignment", .9),)
        boundary = lambda value: Boundary(value, .9, ("ctc",))
        document = build_known_lyrics_document("かん", (
            LyricSpan("か", (0, 1), (ReadingCandidate("カ", "synthetic", 1),)),
            LyricSpan("ん", (1, 2), (ReadingCandidate("ン", "synthetic", 1),)),
        ), (
            ObservedSingingUnit(("カ",), boundary(0), boundary(.05), boundary(.2), .9,
                                ("ctc",)),
            ObservedSingingUnit(("ン",), boundary(.2), None, boundary(.4), .9,
                                ("ctc",)),
        ), evidence)
        document = replace(document, note_candidates=(_note("n0", .05, .4),))
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["stack"])

    def test_sokuon_normally_attaches_to_preceding_mora_on_one_note(self):
        document = _sokuon_document(notes=(_note("n0", 0.05, 0.36),))
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["sokuon"])
        self.assertEqual(result.links[0].singing_unit_ids, ("s0", "s1"))
        self.assertEqual(result.links[0].note_candidate_ids, ("n0",))
        self.assertEqual(result.links[0].cost_components.complexity, 0)

    def test_unobserved_sokuon_still_attaches_to_observed_preceding_mora(self):
        document = _sokuon_document(
            notes=(_note("n0", 0.05, 0.30),), observe_closure=False,
        )
        self.assertEqual(document.singing_units[-1].status, "unobserved")
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["sokuon"])
        self.assertEqual(result.links[0].singing_unit_ids, ("s0", "s1"))

    def test_sokuon_attaches_after_a_preceding_melisma(self):
        document = _sokuon_document(
            notes=(_note("n0", 0.05, 0.18), _note("n1", 0.18, 0.30)),
            observe_closure=False,
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["sokuon"])
        self.assertEqual(result.links[0].note_candidate_ids, ("n0", "n1"))

    def test_short_sokuon_cannot_take_a_standalone_note(self):
        document = _sokuon_document(
            closure=(0.0, 0.06), notes=(_note("n0", 0.0, 0.06),), leading_kana=False,
        )
        result = align_correspondence(document)
        self.assertEqual({link.operation for link in result.links}, {"unit_only", "note_only"})

    def test_measured_held_sokuon_can_take_one_standalone_note(self):
        document = _sokuon_document(
            closure=(0.0, 0.18), notes=(_note("n0", 0.01, 0.17),), leading_kana=False,
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["match"])

    def test_held_sokuon_can_separate_from_preceding_mora(self):
        document = _sokuon_document(
            closure=(0.30, 0.48),
            notes=(_note("n0", 0.05, 0.29), _note("n1", 0.31, 0.47)),
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["match", "match"])
        self.assertEqual(result.links[1].singing_unit_ids, ("s1",))
        self.assertEqual(result.links[1].note_candidate_ids, ("n1",))

    def test_sokuon_never_becomes_a_standalone_melisma(self):
        document = _sokuon_document(
            closure=(0.0, 0.30),
            notes=(_note("n0", 0.0, 0.15), _note("n1", 0.15, 0.30)),
            leading_kana=False,
        )
        result = align_correspondence(document)
        self.assertNotIn("melisma", {link.operation for link in result.links})

    def test_note_only(self):
        document = _document(notes=(_note("n0", 0.10, 0.40),))
        self.assertEqual([link.operation for link in align_correspondence(document).links],
                         ["note_only"])

    def test_unit_only(self):
        document = _document(((0.0, 0.10, 0.40),))
        self.assertEqual([link.operation for link in align_correspondence(document).links],
                         ["unit_only"])

    def test_rest(self):
        document = _document(notes=(_note("n0", 0.10, 0.40, pitch=None),))
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["rest"])
        self.assertLess(result.links[0].cost, 0.1)


class CorrespondenceEvidenceTests(unittest.TestCase):
    def test_raw_mora_ctc_anchor_selects_the_nearby_note(self):
        anchor = Evidence(
            "kana-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 0.15,
            {"time_sec": 0.60, "start_sec": 0.59, "end_sec": 0.61},
        )
        document = _document(
            ((0.0, 0.10, 0.70),),
            (_note("early", 0.10, 0.60), _note("anchored", 0.60, 0.80)),
            extra_evidence=(anchor,),
        )
        unit = replace(
            document.singing_units[0],
            evidence_ids=document.singing_units[0].evidence_ids + (anchor.id,),
        )
        anchored = replace(document, singing_units=(unit,))
        without_anchor = align_correspondence(document)
        with_anchor = align_correspondence(anchored)
        self.assertEqual(without_anchor.links[0].note_candidate_ids, ("early", "anchored"))
        self.assertEqual(with_anchor.links[-1].note_candidate_ids, ("anchored",))
        self.assertEqual(with_anchor.links[-1].singing_unit_ids, ("s0",))
        self.assertEqual(with_anchor.links[-1].cost_components.ctc_anchor_timing, 0)

    def test_ctc_timing_cost_outweighs_absorbing_an_early_note(self):
        anchor = Evidence(
            "kana-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 0.15,
            {"time_sec": 0.29, "start_sec": 0.28, "end_sec": 0.30},
        )
        document = _document(
            ((0.0, 0.10, 0.50),),
            (_note("early", 0.10, 0.30), _note("near", 0.30, 0.50)),
            extra_evidence=(anchor,),
        )
        unit = replace(
            document.singing_units[0],
            evidence_ids=document.singing_units[0].evidence_ids + (anchor.id,),
        )
        anchored = replace(document, singing_units=(unit,))
        without_ctc_cost = align_correspondence(
            anchored, CorrespondenceConfig(ctc_anchor_weight=0.0)
        )
        result = align_correspondence(anchored)
        self.assertEqual([link.operation for link in without_ctc_cost.links], ["melisma"])
        self.assertEqual([link.operation for link in result.links], ["note_only", "match"])
        self.assertEqual(result.links[-1].note_candidate_ids, ("near",))

    def test_internal_note_omission_costs_one_configured_count(self):
        document = _document(
            ((0.0, 0.05, 0.20), (0.35, 0.40, 0.55)),
            (_note("covered", 0.30, 0.50),),
        )
        weighted = _CostModel(
            document, CorrespondenceConfig(internal_note_omission_weight=0.70)
        ).note_only(document.note_candidates[0], internal=True)
        disabled = _CostModel(
            document, CorrespondenceConfig(internal_note_omission_weight=0.0)
        ).note_only(document.note_candidates[0], internal=True)
        edge = _CostModel(
            document, CorrespondenceConfig(internal_note_omission_weight=0.70)
        ).note_only(document.note_candidates[0])
        self.assertAlmostEqual(weighted.cost - disabled.cost, 0.70)
        self.assertAlmostEqual(edge.cost, disabled.cost)

    def test_internal_omission_count_can_preserve_original_note_timing(self):
        anchors = (
            Evidence("first-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 0.1,
                     {"time_sec": 0.10}),
            Evidence("next-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 0.1,
                     {"time_sec": 0.55}),
        )
        document = _document(
            ((0.0, 0.10, 0.30), (0.30, 0.35, 1.00)),
            (_note("first", 0.10, 0.30),
             _note("original", 0.30, 0.70),
             _note("ctc-near", 0.70, 1.00)),
            extra_evidence=anchors,
        )
        units = tuple(replace(
            unit, evidence_ids=unit.evidence_ids + (anchors[index].id,),
        ) for index, unit in enumerate(document.singing_units))
        document = replace(
            document,
            singing_units=units,
        )
        disabled = align_correspondence(
            document, CorrespondenceConfig(internal_note_omission_weight=0.0)
        )
        weighted = align_correspondence(
            document, CorrespondenceConfig(internal_note_omission_weight=0.70)
        )
        self.assertEqual(
            next(link.operation for link in disabled.links
                 if "original" in link.note_candidate_ids),
            "note_only",
        )
        self.assertEqual(
            next(link.singing_unit_ids for link in weighted.links
                 if "original" in link.note_candidate_ids),
            ("s1",),
        )

    def test_missing_mora_ctc_anchor_preserves_previous_cost(self):
        document = _document(
            ((0.0, 0.10, 0.60),), (_note("n0", 0.10, 0.60),)
        )
        result = align_correspondence(document)
        self.assertEqual(result.links[0].cost_components.ctc_anchor_timing, 0)

    def test_catastrophic_near_zero_ctc_anchor_is_not_timing_evidence(self):
        anchor = Evidence(
            "kana-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 1e-6,
            {"time_sec": 99.0, "start_sec": 98.99, "end_sec": 99.01},
        )
        document = _document(
            ((0.0, 0.10, 0.70),),
            (_note("early", 0.10, 0.60), _note("later", 0.60, 0.80)),
            extra_evidence=(anchor,),
        )
        unit = replace(
            document.singing_units[0],
            evidence_ids=document.singing_units[0].evidence_ids + (anchor.id,),
        )
        result = align_correspondence(replace(document, singing_units=(unit,)))
        self.assertEqual(result.links[0].note_candidate_ids, ("early", "later"))
        self.assertTrue(all(
            link.cost_components.ctc_anchor_timing == 0 for link in result.links
        ))

    def test_stack_cannot_dilute_later_mora_anchor_into_previous_note(self):
        anchors = tuple(Evidence(
            f"kana-ctc-{index}", "synthetic-kana-ctc", "mora-ctc-anchor", 0.1,
            {"time_sec": time_sec},
        ) for index, time_sec in enumerate((0.10, 0.40)))
        document = _document(
            ((0.00, 0.05, 0.15), (0.15, 0.16, 0.25)),
            (_note("n0", 0.00, 0.30, confidence=0.0),
             _note("n1", 0.30, 0.50, confidence=0.0)),
            extra_evidence=anchors,
        )
        units = tuple(replace(
            unit, evidence_ids=unit.evidence_ids + (anchors[index].id,),
        ) for index, unit in enumerate(document.singing_units))
        result = align_correspondence(replace(document, singing_units=units))
        pitched = [link for link in result.links if link.note_candidate_ids]
        self.assertEqual(
            [(link.singing_unit_ids, link.note_candidate_ids) for link in pitched],
            [(("s0",), ("n0",)), (("s1",), ("n1",))],
        )
        disabled = align_correspondence(
            replace(document, singing_units=units),
            CorrespondenceConfig(ctc_anchor_weight=0),
        )
        self.assertNotIn("stack", {link.operation for link in disabled.links})

    def test_melisma_cannot_consume_note_containing_next_mora_anchor(self):
        anchors = tuple(Evidence(
            f"kana-ctc-{index}", "synthetic-kana-ctc", "mora-ctc-anchor", 0.1,
            {"time_sec": time_sec},
        ) for index, time_sec in enumerate((0.10, 0.40)))
        document = _document(
            ((0.00, 0.05, 0.12), (0.38, 0.39, 0.43)),
            (_note("n0", 0.00, 0.20, pitch=60, confidence=0.0),
             _note("n1", 0.20, 0.50, pitch=62, confidence=0.0)),
            extra_evidence=anchors,
        )
        units = tuple(replace(
            unit, evidence_ids=unit.evidence_ids + (anchors[index].id,),
        ) for index, unit in enumerate(document.singing_units))
        anchored = replace(document, singing_units=units)
        result = align_correspondence(anchored)
        self.assertEqual(
            [(link.singing_unit_ids, link.note_candidate_ids)
             for link in result.links if link.note_candidate_ids],
            [(("s0",), ("n0",)), (("s1",), ("n1",))],
        )

    def test_dependent_sokuon_does_not_pull_note_onset_to_closure_peak(self):
        anchor = Evidence(
            "closure-ctc", "synthetic-kana-ctc", "mora-ctc-anchor", 0.9,
            {"time_sec": 0.34},
        )
        document = _sokuon_document(notes=(_note("n0", 0.05, 0.36),))
        closure = replace(
            document.singing_units[-1],
            evidence_ids=document.singing_units[-1].evidence_ids + (anchor.id,),
        )
        document = replace(
            document,
            singing_units=document.singing_units[:-1] + (closure,),
            evidence=document.evidence + (anchor,),
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["sokuon"])
        self.assertEqual(result.links[0].cost_components.ctc_anchor_timing, 0)

    def test_unavailable_note_confidence_does_not_use_legacy_placeholder(self):
        unavailable = Evidence("model", "model", "model-note", 0,
                               {"confidence_available": False})
        for pitched in (True, False):
            with self.subTest(pitched=pitched):
                def document(value):
                    return _document(notes=(
                        _note("n0", 0.1, 0.6, pitch=60 if pitched else None,
                              confidence=value, evidence_ids=("model",)),
                    ), extra_evidence=(unavailable,))
                before = document(0)
                result = align_correspondence(before)
                self.assertEqual(result, align_correspondence(document(0.99)))
                self.assertEqual(before.note_candidates[0].confidence, 0)
                self.assertEqual(before.evidence[-1].confidence, 0)
        measured_zero = _document(notes=(_note("n0", 0.1, 0.6, confidence=0),))
        unknown = _document(notes=(_note("n0", 0.1, 0.6, confidence=0,
                                         evidence_ids=("model",)),),
                            extra_evidence=(unavailable,))
        self.assertGreater(align_correspondence(unknown).total_cost,
                           align_correspondence(measured_zero).total_cost)
        voiced = _document(((0, 0.1, 0.6),), unknown.note_candidates,
                           extra_evidence=(unavailable,))
        changed = replace(voiced, note_candidates=(replace(voiced.note_candidates[0],
                                                           confidence=0.99),))
        self.assertEqual(align_correspondence(voiced), align_correspondence(changed))

    def test_group_confidence_adds_candidate_evidence_without_dilution(self):
        document = _document(((0, 0.1, 0.9),), (
            _note("n0", 0.1, 0.5, confidence=0.8),
            _note("n1", 0.5, 0.9, confidence=0.2),
        ))
        config = CorrespondenceConfig()
        model = _CostModel(document, config)
        single = model.relation(document.singing_units, document.note_candidates[:1], "match")
        group = model.relation(document.singing_units, document.note_candidates, "melisma")
        self.assertIsNotNone(single)
        self.assertIsNotNone(group)
        self.assertAlmostEqual(
            group.components.candidate_confidence - single.components.candidate_confidence,
            config.confidence_weight * (1 - 0.2) / 2,
        )

    def test_weak_periodic_units_do_not_collapse_into_one_runaway_melisma(self):
        document = _document(
            tuple((i * 0.3, i * 0.3 + 0.04, i * 0.3 + 0.28) for i in range(4)),
            tuple(_note(f"n{i}", i * 0.3, i * 0.3 + 0.28, confidence=0)
                  for i in range(4)),
        )
        def weak(boundary):
            return replace(boundary, confidence=0.08) if boundary is not None else None
        document = replace(document,
            singing_units=tuple(replace(u, confidence=0.08, consonant_start=weak(u.consonant_start),
                                        end=weak(u.end)) for u in document.singing_units),
            vowel_nuclei=tuple(replace(v, start=weak(v.start), end=weak(v.end))
                               for v in document.vowel_nuclei),
        )
        result = align_correspondence(document)
        self.assertEqual([link.operation for link in result.links], ["match"] * 4)
        self.assertEqual([link.note_candidate_ids for link in result.links],
                         [(f"n{i}",) for i in range(4)])

    def test_cost_components_are_additive_and_survive_ir_round_trip(self):
        document = _document(((0.0, 0.10, 0.60),),
                             (_note("n0", 0.10, 0.60, 64),))
        result = align_correspondence(document)
        link = result.links[0]
        self.assertAlmostEqual(link.cost, link.cost_components.total)
        applied = result.apply(document)
        restored = IntermediateRepresentation.from_json(applied.to_json())
        self.assertEqual(restored.links, result.links)

    def test_adjacent_vowels_keep_their_separate_overlapping_notes(self):
        document = _document(
            (
                (0.045, 0.132, 0.326),
                (0.326, 0.326, 0.965),
            ),
            (
                _note("n0", 0.100, 0.445, 54),
                _note("n1", 0.445, 1.135, 51),
            ),
        )
        result = align_correspondence(document)
        self.assertEqual(
            [(link.singing_unit_ids, link.note_candidate_ids) for link in result.links],
            [(('s0',), ('n0',)), (('s1',), ('n1',))],
        )

    def test_exact_tie_exposes_alternative_and_low_confidence(self):
        document = _document(((0.0, 0.10, 0.60),),
                             (_note("n0", 0.10, 0.60),))
        config = CorrespondenceConfig(
            ctc_anchor_weight=0, internal_note_omission_weight=0,
            confidence_weight=0, complexity_weight=0, note_only_base=0,
            unit_only_base=0, unobserved_unit_only_base=0, rest_base=0,
        )
        result = align_correspondence(document, config)
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.alternatives)
        self.assertEqual(result.alternatives[0].total_cost, result.total_cost)
        self.assertTrue(all(link.confidence == 0 for link in result.links))

    def test_input_sequences_and_objects_are_not_mutated_or_reordered(self):
        notes = (_note("n0", 0.10, 0.30), _note("n1", 0.30, 0.60))
        document = _document(((0.0, 0.10, 0.60),), notes)
        before = document.to_json()
        result = align_correspondence(document)
        self.assertEqual(document.to_json(), before)
        self.assertEqual(document.note_candidates, notes)
        self.assertEqual(document.links, ())
        self.assertIsNot(result.apply(document), document)

    def test_out_of_order_notes_are_rejected_instead_of_silently_sorted(self):
        with self.assertRaisesRegex(ValueError, "chronological order"):
            align_correspondence(_document(
                ((0.0, 0.10, 0.60),),
                (_note("n1", 0.30, 0.60), _note("n0", 0.10, 0.30)),
            ))

    def test_link_validation_enforces_operation_membership(self):
        document = _document(((0.0, 0.10, 0.60),),
                             (_note("n0", 0.10, 0.60),))
        result = align_correspondence(document)
        invalid = replace(result.links[0], operation="melisma")
        with self.assertRaisesRegex(ValidationError, "invalid membership"):
            replace(document, links=(invalid,))


if __name__ == "__main__":
    unittest.main()
