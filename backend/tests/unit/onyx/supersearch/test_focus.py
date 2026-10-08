"""Expansion selection never removes originals or silently drops legal conditions."""

import json
from typing import Literal
from unittest.mock import Mock
from uuid import uuid4

import pytest

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite.models import (
    PassageSupport,
    ResearchNeed,
    WorkflowPolicy,
)
from onyx.supersearch.acquisition import SupersearchAcquirer
from onyx.supersearch.dependencies import SupersearchDependencyExpander
from onyx.supersearch.focus import FocusSelection, FocusSubject, focus_subjects
from onyx.supersearch.models import (
    FocusAnswerReview,
    FocusReviewAssessment,
    NeedFocusAssessment,
    NeedFocusDecision,
)
from tests.unit.onyx.supersearch.test_engine import plan, review


def context() -> RunContext:
    return RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )


def item(ledger: EvidenceLedger, citation: int) -> EvidenceItem:
    value = ledger.get(citation)
    assert value is not None
    return value


def original(
    source: CorpusSource, article: str, text: str, *, complete: bool = True
) -> EvidenceItem:
    return evidence_for_chunk(
        source,
        CorpusChunk(
            f"atomic-{article}",
            source.id,
            text,
            int(article),
            int(article),
            ("8917 SAYILI TAŞIMA KANUNU", f"MADDE {article}"),
            {
                "document_type": "kanun",
                "title": "8917 SAYILI TAŞIMA KANUNU",
                "article_closure_complete": complete,
            },
            None,
            None,
            "active",
        ),
    )


def decision(
    subjects: list[FocusSubject],
    ledger: EvidenceLedger,
    *,
    status: Literal["material", "incidental", "pending"] = "incidental",
) -> NeedFocusDecision:
    return NeedFocusDecision(
        assessments=[
            NeedFocusAssessment(
                subject_id=subject.subject_id,
                need_id="clock",
                status=status,
                explanation="Tam özgün hükmün yalnız transit taşıma kapsamı bu başvuru ihtiyacından ayrıdır.",
                witnesses=[
                    PassageSupport(
                        citation=subject.citations[0],
                        quotation=item(ledger, subject.citations[0]).text,
                    )
                ],
            )
            for subject in subjects
        ]
    )


def setup(
    *, complete: bool = True
) -> tuple[EvidenceLedger, list[FocusSubject], FocusSelection]:
    ledger = EvidenceLedger()
    source = CorpusSource(uuid4(), "8917 SAYILI TAŞIMA KANUNU.md", "native-law")
    item = original(
        source,
        "27",
        "**MADDE 27-** Bu madde yalnız transit taşımaya uygulanır.",
        complete=complete,
    )
    item.question_ids = ["clock"]
    ledger.add([item], context())
    return ledger, focus_subjects(ledger, {1}), FocusSelection(ledger)


def test_complete_incidental_units_keep_raw_originals_unchanged() -> None:
    ledger, subjects, selection = setup()
    before = ledger.export()
    assert any(subject.kind == "reference" for subject in subjects)
    selection.apply(subjects, plan(), decision(subjects, ledger), {1}, set())
    assert selection.frontier({1}, plan()) == set()
    assert ledger.export() == before
    assert (
        json.loads(ledger.serialize_records([1], required=[1], max_chars=None))[0][
            "text"
        ]
        == item(ledger, 1).text
    )
    assert all(row["assessments"] for row in selection.audit())


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "undelivered",
        "wrong_witness",
        "blank_witness",
        "missing",
        "duplicate",
        "untrusted",
    ],
)
def test_unproved_exclusion_stays_pending_and_expands(failure: str) -> None:
    ledger, subjects, selection = setup(complete=failure != "partial")
    value = decision(subjects, ledger)
    if failure == "wrong_witness":
        for row in value.assessments:
            row.witnesses[0].quotation = "Source title alone"
    elif failure == "blank_witness":
        for row in value.assessments:
            row.witnesses[0].quotation = " "
    elif failure == "missing":
        value.assessments = []
    elif failure == "duplicate":
        value.assessments *= 2
    elif failure == "untrusted":
        ledger, subjects, selection = setup()
        source_item = item(ledger, 1)
        source_item.metadata["untrusted"] = True
        untrusted = EvidenceLedger()
        untrusted.add([source_item], context())
        ledger, subjects, selection = (
            untrusted,
            focus_subjects(untrusted, {1}),
            FocusSelection(untrusted),
        )
        value = decision(subjects, ledger)
    selection.apply(
        subjects, plan(), value, set() if failure == "undelivered" else {1}, set()
    )
    assert selection.frontier({1}, plan()) == {1}
    assert all(row["status"] == "pending" for row in selection.assessments.values())


def test_pending_reference_cannot_be_hidden_by_incidental_origin_unit() -> None:
    ledger, subjects, selection = setup()
    value = decision(subjects, ledger)
    reference = next(
        row for row in value.assessments if row.subject_id.startswith("reference-")
    )
    reference.status = "pending"
    reference.witnesses = []
    selection.apply(subjects, plan(), value, {1}, set())
    assert selection.frontier({1}, plan()) == {1}
    assert selection.allows(
        1,
        "clock",
        next(subject.edge_key for subject in subjects if subject.kind == "reference"),
    )


def test_material_second_need_and_explicit_read_cannot_be_removed() -> None:
    ledger, subjects, selection = setup()
    two = plan().model_copy(
        update={
            "needs": [
                *plan().needs,
                ResearchNeed(
                    need_id="transit",
                    question="Transit usulü?",
                    governing_source="Taşıma Kanunu",
                    conditions_to_check=["Transit taşıma kapsamı"],
                ),
            ]
        }
    )
    selection.apply(subjects, two, decision(subjects, ledger), {1}, set())
    assert selection.frontier({1}, two) == {1}
    assert not selection.allows(1, "clock")
    assert selection.allows(1, "transit")
    selection.protect({1})
    assert selection.frontier({1}, plan()) == {1}
    own = next(subject for subject in subjects if subject.kind == "reference")
    assert selection.allows(1, "clock", own.edge_key)


def test_native_collect_transfers_retrieval_need_to_material_other_need() -> None:
    ledger, subjects, selection = setup()
    two = plan().model_copy(
        update={
            "needs": [
                *plan().needs,
                ResearchNeed(
                    need_id="transit",
                    question="Transit usulü?",
                    governing_source="Taşıma Kanunu",
                    conditions_to_check=["Transit taşıma kapsamı"],
                ),
            ]
        }
    )
    value = decision(subjects, ledger)
    value.assessments.extend(
        NeedFocusAssessment(
            subject_id=subject.subject_id,
            need_id="transit",
            status="material",
            explanation="Transit kapsamının özgün hükmü bu ayrı ihtiyacı yönetir.",
            witnesses=[PassageSupport(citation=1, quotation=item(ledger, 1).text)],
        )
        for subject in subjects
    )
    selection.apply(subjects, two, value, {1}, set())
    run = context()
    acquirer = SupersearchAcquirer(
        CapabilityRegistry([]), run, ledger, WorkflowPolicy()
    )
    expander = SupersearchDependencyExpander(
        broker=Mock(),
        acquirer=acquirer,
        ledger=ledger,
        context=run,
        source_kinds={item(ledger, 1).source_id: SourceKind.STATUTE},
    )
    expander.configure_focus(selection)
    expander._collect(two, selection.frontier({1}, two))
    assert item(ledger, 1).question_ids == ["clock"]
    assert len(expander.edges) == 1
    assert next(iter(expander.edges.values())).need_ids == ["transit"]


def test_material_provision_always_keeps_its_own_governing_anchor() -> None:
    ledger, subjects, selection = setup()
    value = decision(subjects, ledger)
    for row in value.assessments:
        if row.subject_id.startswith("provision-"):
            row.status = "material"
    selection.apply(subjects, plan(), value, {1}, set())
    own = next(subject for subject in subjects if subject.kind == "reference")
    assert selection.allows(1, "clock", own.edge_key)


@pytest.mark.parametrize("confirmed", [True, False])
def test_independent_review_must_establish_each_exclusion(confirmed: bool) -> None:
    ledger, subjects, selection = setup()
    selection.apply(subjects, plan(), decision(subjects, ledger), {1}, set())
    assessed = FocusAnswerReview(
        **review().model_dump(),
        focus_reviews=[
            FocusReviewAssessment(
                subject_id=subject.subject_id,
                need_id="clock",
                status="nonmaterial" if confirmed else "reopen",
                explanation="Tam özgün kapsam yeniden denetlendi.",
                witnesses=[PassageSupport(citation=1, quotation=item(ledger, 1).text)],
            )
            for subject in subjects
        ],
    )
    defects, reopened = selection.review(assessed, {1})
    assert bool(defects) is not confirmed
    assert reopened == (set() if confirmed else {1})
    assert selection.frontier({1}, plan()) == (set() if confirmed else {1})


def test_native_frontier_does_not_open_excluded_sources_or_expand_incidental_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = context()
    ledger = EvidenceLedger()
    material = CorpusSource(uuid4(), "8917 SAYILI TAŞIMA KANUNU.md", "material-law")
    incidental = CorpusSource(uuid4(), "8917 SAYILI TAŞIMA KANUNU.md", "incidental-law")
    body = original(
        material,
        "27",
        "**MADDE 27-** Başvuru bir yıl içinde yapılır. 8917 sayılı Taşıma Kanununun 98. maddesi yalnız transit işlemlerine uygulanır.",
    )
    unused = original(
        incidental, "77", "**MADDE 77-** Bu madde yalnız transit taşımaya uygulanır."
    )
    for source_item in (body, unused):
        source_item.question_ids = ["clock"]
    ledger.add([body, unused], run)
    subjects = focus_subjects(ledger, {1, 2})
    value = decision(subjects, ledger)
    for row in value.assessments:
        subject = next(
            subject for subject in subjects if subject.subject_id == row.subject_id
        )
        if subject.source_id == str(material.id) and subject.article != "98":
            row.status = "material"
    assert any(
        subject.kind == "reference" and subject.article == "98" for subject in subjects
    )
    selection = FocusSelection(ledger)
    selection.apply(subjects, plan(), value, {1, 2}, set())
    opened: list[str] = []

    def opening(arguments, _context):
        opened.append(arguments["source_id"])
        assert arguments == {"source_id": str(material.id), "start": 0, "limit": 3}
        chunk = CorpusChunk(
            "opening",
            material.id,
            "8917 SAYILI TAŞIMA KANUNU",
            0,
            0,
            ("8917 SAYILI TAŞIMA KANUNU",),
            {"document_type": "kanun", "title": "8917 SAYILI TAŞIMA KANUNU"},
            None,
            None,
            "active",
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical source opening",
            evidence=[evidence_for_chunk(material, chunk)],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_source_range",
                description="opening",
                parameters={"type": "object"},
                handler=opening,
            )
        ]
    )
    acquirer = SupersearchAcquirer(registry, run, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=Mock(), acquirer=acquirer, ledger=ledger, context=run, source_kinds={}
    )
    expander.configure_focus(selection)
    monkeypatch.setattr(expander, "_expand", Mock())
    edges = expander.expand(plan(), frontier=selection.frontier({1, 2}, plan()))
    assert opened == [str(material.id)]
    assert {edge.article for edge in edges} == {"27"}
    assert all(edge.need_ids == ["clock"] for edge in edges)
    assert item(ledger, 2).text == unused.text
