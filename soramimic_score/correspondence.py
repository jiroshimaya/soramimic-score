"""Explainable many-to-many correspondence for lyric units and melody notes.

The decoder is a semi-Markov dynamic program: each transition consumes an
ordered span of singing units, an ordered span of note candidates, or both.
It reads the versioned IR but never edits or reorders the caller's sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import math
from typing import Sequence

from .ir import (
    IntermediateRepresentation,
    Link,
    LinkCost,
    MINIMUM_USABLE_TIMING_CONFIDENCE,
    NoteCandidate,
    SingingUnit,
    has_usable_timing,
)
from .japanese import kana_to_syllables


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Costs for aligning CTC lyric onsets to SheetSage melody notes."""

    maximum_group_size: int = 4
    maximum_alternatives: int = 3
    ctc_anchor_tolerance_sec: float = 0.10
    ctc_anchor_weight: float = 0.35
    internal_note_omission_weight: float = 0.70
    confidence_weight: float = 0.3122549019607843
    complexity_weight: float = 0.16058823529411764
    note_only_base: float = 0.5352941176470588
    unit_only_base: float = 0.981372549019608
    unobserved_unit_only_base: float = 0.04460784313725491
    rest_base: float = 0.04460784313725491
    minimum_standalone_sokuon_sec: float = 0.10
    minimum_standalone_sokuon_note_coverage: float = 0.65

    def __post_init__(self) -> None:
        if self.maximum_group_size < 2:
            raise ValueError("maximum_group_size must be at least two")
        if self.maximum_alternatives < 1:
            raise ValueError("maximum_alternatives must be positive")
        if self.ctc_anchor_tolerance_sec <= 0:
            raise ValueError("CTC timing tolerance must be positive")
        if (not math.isfinite(self.minimum_standalone_sokuon_sec)
                or self.minimum_standalone_sokuon_sec <= 0):
            raise ValueError("minimum standalone sokuon duration must be positive")
        if (not math.isfinite(self.minimum_standalone_sokuon_note_coverage)
                or not 0 < self.minimum_standalone_sokuon_note_coverage <= 1):
            raise ValueError("minimum standalone sokuon note coverage must be in (0, 1]")
        numeric = (
            self.ctc_anchor_weight, self.internal_note_omission_weight,
            self.confidence_weight, self.complexity_weight, self.note_only_base,
            self.unit_only_base, self.unobserved_unit_only_base, self.rest_base,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("correspondence weights and base costs must be finite and nonnegative")

    # Older callers used these names while the pipeline still accepted measured
    # phone boundaries. The production path now consumes only mora-CTC anchors
    # and melody notes, so they are permanently neutral compatibility views.
    @property
    def vowel_onset_weight(self) -> float:
        return 0.0

    @property
    def interval_overlap_weight(self) -> float:
        return 0.0

    @property
    def boundary_weight(self) -> float:
        return 0.0


def boundaryless_correspondence_config() -> CorrespondenceConfig:
    """Return the retired comparison config for mora-CTC plus SheetSage2 inputs."""
    return CorrespondenceConfig()


@dataclass(frozen=True)
class CorrespondenceAlternative:
    links: tuple[Link, ...]
    total_cost: float
    relative_probability: float


@dataclass(frozen=True)
class CorrespondenceResult:
    links: tuple[Link, ...]
    total_cost: float
    confidence: float
    alternatives: tuple[CorrespondenceAlternative, ...]

    def apply(self, document: IntermediateRepresentation) -> IntermediateRepresentation:
        """Return a validated document with the selected links; keep input immutable."""
        return replace(document, links=self.links)


@dataclass(frozen=True)
class _Step:
    unit_ids: tuple[str, ...]
    note_ids: tuple[str, ...]
    operation: str
    components: LinkCost
    evidence_ids: tuple[str, ...]

    @property
    def cost(self) -> float:
        return self.components.total

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.operation, self.unit_ids, self.note_ids)


@dataclass(frozen=True)
class _Path:
    cost: float
    previous: _Path | None
    step: _Step | None
    tie_key: bytes

    def extend(self, step: _Step) -> _Path:
        encoded = repr(step.signature).encode("utf-8")
        tie_key = hashlib.sha256(self.tie_key + b"\0" + encoded).digest()
        return _Path(self.cost + step.cost, self, step, tie_key)

    def materialize(self) -> tuple[_Step, ...]:
        steps: list[_Step] = []
        current = self
        while current.step is not None:
            steps.append(current.step)
            if current.previous is None:
                raise RuntimeError("correspondence path has a broken backpointer")
            current = current.previous
        steps.reverse()
        return tuple(steps)


class _CostModel:
    def __init__(self, document: IntermediateRepresentation,
                 config: CorrespondenceConfig) -> None:
        self.config = config
        self.evidence = {item.id: item for item in document.evidence}
        self.phonemes = {item.id: item for item in document.phonemes}
        self.units = document.singing_units
        self.moras = {item.id: item for item in document.moras}
        self.unit_index = {item.id: index for index, item in enumerate(self.units)}
        self.notes = document.note_candidates

    def _unit_region(self, unit: SingingUnit) -> tuple[float, float] | None:
        if (not has_usable_timing(unit.consonant_start)
                or not has_usable_timing(unit.end)):
            return None
        return unit.consonant_start.time_sec, unit.end.time_sec

    def _is_sokuon(self, unit: SingingUnit) -> bool:
        symbols = tuple(self.phonemes[phoneme_id].symbol for phoneme_id in unit.phoneme_ids
                        if phoneme_id in self.phonemes)
        return bool(symbols) and all(symbol == "q" for symbol in symbols)

    def _is_sokuon_attachment(self, units: Sequence[SingingUnit], operation: str) -> bool:
        return (operation == "sokuon" and len(units) == 2
                and not self._is_sokuon(units[0]) and self._is_sokuon(units[1]))

    def _standalone_sokuon_supported(
        self, units: Sequence[SingingUnit], notes: Sequence[NoteCandidate], operation: str,
    ) -> bool:
        """Allow an exceptional held sokuon only when a measured note fits it."""
        if (operation != "match" or len(units) != 1 or len(notes) != 1
                or not self._is_sokuon(units[0])):
            return False
        region = self._unit_region(units[0])
        if region is None or not units[0].evidence_ids or not notes[0].sources:
            return False
        unit_duration = region[1] - region[0]
        note_duration = notes[0].end_sec - notes[0].start_sec
        if min(unit_duration, note_duration) < self.config.minimum_standalone_sokuon_sec:
            return False
        intersection = max(0.0, min(region[1], notes[0].end_sec)
                           - max(region[0], notes[0].start_sec))
        return intersection / note_duration >= self.config.minimum_standalone_sokuon_note_coverage

    def _note_confidence(self, note: NoteCandidate) -> float:
        # Schema v1 adapters retain a numeric placeholder when the provider has
        # no confidence measurement. It is not a measured zero. The neutral
        # scoring prior stays internal; never write it back as model evidence.
        if any(self.evidence[eid].detail.get("confidence_available") is False
               for eid in note.evidence_ids if eid in self.evidence):
            return 0.5
        return note.confidence

    def _ctc_anchor(self, unit: SingingUnit) -> tuple[float, float] | None:
        """Return the strongest explicit mora-CTC time attached to one unit.

        A forced-alignment posterior can have a small absolute probability while
        its local peak still gives the best token time.  Confidence is retained
        for provenance and tie-breaking, but does not attenuate timing to zero.
        """
        anchors: list[tuple[float, float, str]] = []
        for evidence_id in unit.evidence_ids:
            item = self.evidence.get(evidence_id)
            if item is None or item.kind != "mora-ctc-anchor":
                continue
            value = item.detail.get("time_sec")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("mora-ctc-anchor evidence needs a finite nonnegative time_sec")
            if item.confidence < MINIMUM_USABLE_TIMING_CONFIDENCE:
                continue
            anchors.append((item.confidence, float(value), item.id))
        if not anchors:
            return None
        confidence, time_sec, _ = max(anchors, key=lambda value: (value[0], -value[1], value[2]))
        return time_sec, confidence

    def _ctc_anchor_cost(
        self, units: Sequence[SingingUnit], notes: Sequence[NoteCandidate], operation: str,
    ) -> float:
        anchor_units = units[:-1] if operation == "sokuon" else units
        timed = tuple(
            (index, anchor)
            for index, unit in enumerate(anchor_units)
            if (anchor := self._ctc_anchor(unit)) is not None
        )
        if not timed:
            return 0.0
        first_start = notes[0].start_sec
        penalties = []
        for index, (time_sec, _confidence) in timed:
            if operation == "stack" and index > 0:
                # Later parts of the same syllable share the first note; their
                # peaks need only lie in that note, not coincide with its onset.
                note = notes[0]
                error = max(note.start_sec - time_sec, 0.0, time_sec - note.end_sec)
            else:
                # A match, the first mora of a stack/sokuon, and a melisma all
                # begin at the first consumed melody note.
                error = abs(time_sec - first_start)
            # A one-note match uses ordinary absolute onset disagreement: a
            # SheetSage note can legitimately begin before the CTC token peak.
            # Multi-note relations use squared disagreement because absorbing
            # an increasingly early run is a larger structural claim. Both
            # remain soft terms under the same fitted CTC weight.
            normalized_error = error / self.config.ctc_anchor_tolerance_sec
            penalties.append(
                normalized_error if operation == "match"
                else normalized_error * normalized_error
            )
        # Each observed mora is independent timing evidence.  Averaging here
        # lets a stack hide a late mora in the preceding note and then skip the
        # note that actually contains its CTC peak.  Sum the per-mora costs, as
        # we already do for candidate confidence, so group width cannot dilute
        # an added anchor.
        return self.config.ctc_anchor_weight * sum(penalties)

    def _stack_hides_anchored_unit(
        self, units: Sequence[SingingUnit], note: NoteCandidate,
    ) -> bool:
        """Reject a stack that skips the note containing a later mora peak.

        A stack remains valid when several token peaks genuinely fall inside
        one melody note.  When a later mora peak falls in another available
        pitched candidate, assigning it to the earlier note would erase the
        very timing distinction supplied by the independent kana CTC.
        """
        for unit in units[1:]:
            anchor = self._ctc_anchor(unit)
            if anchor is None:
                continue
            time_sec, _confidence = anchor
            if note.start_sec <= time_sec < note.end_sec:
                continue
            if any(
                candidate.id != note.id and candidate.midi_pitch is not None
                and candidate.start_sec <= time_sec < candidate.end_sec
                for candidate in self.notes
            ):
                return True
        return False

    def _melisma_hides_following_anchor(
        self, unit: SingingUnit, notes: Sequence[NoteCandidate],
    ) -> bool:
        """Do not let continuation notes consume a later mora's CTC peak."""
        first = self.unit_index[unit.id] + 1
        continuations = notes[1:]
        if not continuations:
            return False
        for later in self.units[first:]:
            anchor = self._ctc_anchor(later)
            if anchor is None:
                continue
            time_sec, _confidence = anchor
            if any(note.start_sec <= time_sec < note.end_sec for note in continuations):
                return True
            if time_sec >= continuations[-1].end_sec:
                break
        return False

    def relation(self, units: Sequence[SingingUnit], notes: Sequence[NoteCandidate],
                 operation: str) -> _Step | None:
        kana = "".join(self.moras[mora_id].text for unit in units for mora_id in unit.mora_ids)
        if len(kana_to_syllables(kana)) > 1:
            return None
        sokuon_attachment = self._is_sokuon_attachment(units, operation)
        if operation == "sokuon" and not sokuon_attachment:
            return None
        observed_required = units[:-1] if sokuon_attachment else units
        if any(unit.status == "unobserved" for unit in observed_required):
            return None
        standalone_sokuon = self._standalone_sokuon_supported(units, notes, operation)
        if (sokuon_attachment and len(notes) > 1
                and self._standalone_sokuon_supported(
                    units[-1:], notes[-1:], "match")):
            return None
        if any(self._is_sokuon(unit) for unit in units) and not (
                sokuon_attachment or standalone_sokuon):
            return None
        if (operation == "stack" and self.config.ctc_anchor_weight > 0
                and self._stack_hides_anchored_unit(units, notes[0])):
            return None
        if (operation == "melisma" and self.config.ctc_anchor_weight > 0
                and self._melisma_hides_following_anchor(units[0], notes)):
            return None
        scored_units = units[:-1] if sokuon_attachment else units
        confidences = tuple(unit.confidence for unit in scored_units) + tuple(
            self._note_confidence(note) for note in notes
        )
        # Keep the one-unit/one-note cost unchanged, but charge each consumed
        # candidate. A group must not dilute an added weak candidate's cost.
        confidence = self.config.confidence_weight * sum(1.0 - c for c in confidences) / 2
        complexity_count = max(0, len(units) - 1) + max(0, len(notes) - 1)
        if sokuon_attachment:
            complexity_count -= 1
        complexity = self.config.complexity_weight * complexity_count
        ctc_anchor = self._ctc_anchor_cost(units, notes, operation)
        components = LinkCost(
            ctc_anchor_timing=ctc_anchor,
            candidate_confidence=confidence,
            complexity=complexity,
        )
        return _Step(tuple(item.id for item in units), tuple(item.id for item in notes),
                     operation, components, self._evidence_ids(units, notes))

    def unit_only(self, unit: SingingUnit) -> _Step:
        base = (self.config.unobserved_unit_only_base if unit.status == "unobserved"
                else self.config.unit_only_base)
        confidence = 0.0 if unit.status == "unobserved" else (
            self.config.confidence_weight * unit.confidence
        )
        # A usable CTC peak is also evidence that the mora should not vanish.
        # Charge one normalized anchor unit when the DP leaves it unassigned;
        # the fitted anchor weight controls both timing disagreement and
        # deletion, without making either outcome illegal.
        missing_anchor = (
            self.config.ctc_anchor_weight if self._ctc_anchor(unit) is not None else 0.0
        )
        return _Step((unit.id,), (), "unit_only", LinkCost(
            ctc_anchor_timing=missing_anchor,
            candidate_confidence=confidence,
            complexity=base,
        ), self._evidence_ids((unit,), ()))

    def note_only(
        self, note: NoteCandidate, *, internal: bool = False,
    ) -> _Step:
        confidence = self._note_confidence(note)
        internal_omission = (
            self.config.internal_note_omission_weight if internal else 0.0
        )
        return _Step((), (note.id,), "note_only", LinkCost(
            candidate_confidence=self.config.confidence_weight * confidence,
            complexity=self.config.note_only_base + internal_omission,
        ), self._evidence_ids((), (note,)))

    def rest(self, note: NoteCandidate) -> _Step:
        confidence = self._note_confidence(note)
        return _Step((), (note.id,), "rest", LinkCost(
            candidate_confidence=self.config.confidence_weight * (1.0 - confidence),
            complexity=self.config.rest_base,
        ), self._evidence_ids((), (note,)))

    def _evidence_ids(self, units: Sequence[SingingUnit],
                      notes: Sequence[NoteCandidate]) -> tuple[str, ...]:
        ids = {evidence_id for unit in units for evidence_id in unit.evidence_ids}
        ids.update(evidence_id for note in notes for evidence_id in note.evidence_ids)
        return tuple(sorted(ids))


def _validate_order(document: IntermediateRepresentation) -> None:
    notes = document.note_candidates
    if any((left.start_sec, left.end_sec) > (right.start_sec, right.end_sec)
           for left, right in zip(notes, notes[1:])):
        raise ValueError("note candidates must be supplied in chronological order")


def _links(path: _Path, prefix: str, path_confidence: float) -> tuple[Link, ...]:
    return tuple(Link(
        id=f"{prefix}{index}",
        singing_unit_ids=step.unit_ids,
        note_candidate_ids=step.note_ids,
        operation=step.operation,
        cost=step.cost,
        confidence=max(0.0, min(1.0, math.exp(-step.cost) * path_confidence)),
        cost_components=step.components,
        evidence_ids=step.evidence_ids,
    ) for index, step in enumerate(path.materialize()))


def _available_link_prefix(document: IntermediateRepresentation, count: int) -> str:
    collections = (
        document.utterances, document.readings, document.moras, document.phonemes,
        document.vowel_nuclei, document.singing_units, document.evidence,
        document.note_candidates,
    )
    occupied = {item.id for collection in collections for item in collection}
    prefix = "correspondence-link-"
    while any(f"{prefix}{index}" in occupied for index in range(count)):
        prefix = f"_{prefix}"
    return prefix


def align_correspondence(
    document: IntermediateRepresentation,
    config: CorrespondenceConfig | None = None,
) -> CorrespondenceResult:
    """Align all units and candidates, returning the best path and alternatives.

    Candidate order is part of the input contract. The function rejects
    out-of-order candidates instead of sorting or modifying them.
    """
    if not isinstance(document, IntermediateRepresentation):
        raise TypeError("document must be an IntermediateRepresentation")
    settings = config or CorrespondenceConfig()
    _validate_order(document)
    units, notes = document.singing_units, document.note_candidates
    model = _CostModel(document, settings)
    keep = settings.maximum_alternatives + 1
    root = _Path(0.0, None, None, hashlib.sha256(b"correspondence-root").digest())
    queue: list[tuple[float, bytes, int, int, _Path]] = [
        (root.cost, root.tie_key, 0, 0, root)
    ]
    accepted: dict[tuple[int, int], set[bytes]] = {}
    finalists: list[_Path] = []
    target = (len(units), len(notes))
    while queue and len(finalists) < keep:
        _, _, unit_index, note_index, path = heapq.heappop(queue)
        state = (unit_index, note_index)
        state_paths = accepted.setdefault(state, set())
        if path.tie_key in state_paths or len(state_paths) >= keep:
            continue
        state_paths.add(path.tie_key)
        if state == target:
            finalists.append(path)
            continue

        transitions: list[tuple[int, int, _Step]] = []
        if unit_index < len(units):
            transitions.append((unit_index + 1, note_index,
                                model.unit_only(units[unit_index])))
        if note_index < len(notes):
            note = notes[note_index]
            transitions.append((unit_index, note_index + 1,
                                model.rest(note) if note.midi_pitch is None
                                else model.note_only(
                                    note,
                                    internal=0 < unit_index < len(units),
                                )))
        if (unit_index < len(units) and note_index < len(notes)
                and notes[note_index].midi_pitch is not None):
            step = model.relation((units[unit_index],), (notes[note_index],), "match")
            if step is not None:
                transitions.append((unit_index + 1, note_index + 1, step))
            for width in range(2, settings.maximum_group_size + 1):
                selected_notes = notes[note_index:note_index + width]
                if (len(selected_notes) == width
                        and all(item.midi_pitch is not None for item in selected_notes)):
                    step = model.relation((units[unit_index],), selected_notes, "melisma")
                    if step is not None:
                        transitions.append((unit_index + 1, note_index + width, step))
                selected_units = units[unit_index:unit_index + width]
                if len(selected_units) == width:
                    step = model.relation(selected_units, (notes[note_index],), "stack")
                    if step is not None:
                        transitions.append((unit_index + width, note_index + 1, step))
            selected_units = units[unit_index:unit_index + 2]
            if len(selected_units) == 2:
                for width in range(1, settings.maximum_group_size + 1):
                    selected_notes = notes[note_index:note_index + width]
                    if (len(selected_notes) == width
                            and all(item.midi_pitch is not None for item in selected_notes)):
                        step = model.relation(selected_units, selected_notes, "sokuon")
                        if step is not None:
                            transitions.append((unit_index + 2, note_index + width, step))
        for next_unit, next_note, step in transitions:
            proposal = path.extend(step)
            heapq.heappush(queue, (proposal.cost, proposal.tie_key,
                                   next_unit, next_note, proposal))

    if not finalists:
        raise RuntimeError("correspondence decoder did not reach the terminal state")
    best = finalists[0]
    if len(finalists) == 1:
        confidence = 1.0
    else:
        margin = max(0.0, finalists[1].cost - best.cost)
        confidence = margin / (margin + 1.0)
    weights = [math.exp(-(item.cost - best.cost)) for item in finalists]
    normalizer = sum(weights)
    alternatives = tuple(
        CorrespondenceAlternative(
            _links(path, f"a{index}-l", math.exp(-(path.cost - best.cost))),
            path.cost,
            weights[index] / normalizer,
        )
        for index, path in enumerate(finalists[1:], 1)
    )
    link_prefix = _available_link_prefix(document, len(best.materialize()))
    return CorrespondenceResult(
        links=_links(best, link_prefix, confidence),
        total_cost=best.cost,
        confidence=confidence,
        alternatives=alternatives,
    )
