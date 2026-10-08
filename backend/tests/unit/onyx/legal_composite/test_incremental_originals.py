"""Verified source groups survive an unfinished lane without accepting late reads."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.context.search.models import SearchDoc
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    classify_source,
)
from onyx.legal_composite import source_lanes
from onyx.legal_composite.acquisition import CanonicalAcquirer, CanonicalEvidenceStage
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.shared_work import SharedCanonicalCenters
from onyx.legal_composite.source_lanes import SourceLaneBroker
from tests.unit.legal_composite.test_source_lanes import fixture_lane
from tests.unit.onyx.legal_composite.test_review_assessment import original


@pytest.mark.parametrize("withdraw_first", [False, True])
def test_completed_source_is_staged_only_after_revalidation_and_late_source_is_rejected(
    monkeypatch: pytest.MonkeyPatch, withdraw_first: bool
) -> None:
    base, catalogue, first_source = fixture_lane()
    late_source = CorpusSource(uuid4(), "another-rule.md", "file")
    first_record = catalogue.records[0].model_copy(
        update={"prepared": True, "prepared_revision": 1}
    )
    late_record = classify_source(
        late_source, (), opening_texts=("KANUN\nMADDE 1- Başka hüküm.",)
    ).model_copy(update={"prepared": True, "prepared_revision": 1})
    catalogue = catalogue.model_copy(update={"records": (first_record, late_record)})
    lane = SourceLaneBroker(
        base, catalogue, SourceKind.STATUTE, SharedCanonicalCenters()
    )
    items: dict[str, EvidenceItem] = {}
    docs: list[SearchDoc] = []
    for source, name in ((first_source, "completed"), (late_source, "late")):
        item = original(f"Whole canonical {name} original.", name)
        assert item.search_doc is not None
        doc = item.search_doc.model_copy(
            deep=True,
            update={
                "document_id": str(source.id),
                "metadata": {"regulatory_chunk_id": name},
            },
        )
        items[str(source.id)] = item.model_copy(
            deep=True, update={"source_id": str(source.id), "search_doc": doc}
        )
        docs.append(doc)

    ledger = EvidenceLedger()
    context = RunContext(timeout_seconds=20)
    first_committed, post_read_checked = Event(), Event()
    release_late, late_started, late_returned = Event(), Event(), Event()
    first_read = Event()
    stages: list[CanonicalEvidenceStage] = []
    children: list[RunContext] = []
    real_add = ledger.add

    def add(items_to_add: list[EvidenceItem], child: RunContext) -> list[int]:
        numbers = real_add(items_to_add, child)
        if any(item.source_id == str(first_source.id) for item in items_to_add):
            first_committed.set()
        return numbers

    monkeypatch.setattr(ledger, "add", add)
    monkeypatch.setattr(
        source_lanes, "get_session_with_current_tenant", lambda: nullcontext(object())
    )

    def revalidate(_session: object, **kwargs: object) -> dict[UUID, CorpusSource]:
        records = cast(list[SourceClassification], kwargs["recorded"])
        sources = {first_source.id: first_source, late_source.id: late_source}
        if first_read.is_set() and len(records) == 1:
            post_read_checked.set()
            if withdraw_first:
                sources.pop(first_source.id)
        return {
            record.source_id: sources[record.source_id]
            for record in records
            if record.source_id in sources
        }

    monkeypatch.setattr(source_lanes, "revalidate_prepared_sources", revalidate)

    def hydrate(
        _broker: CorpusBroker, originals: list[SearchDoc], _child: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        assert len(originals) == 1
        doc = originals[0]
        if doc.document_id == str(first_source.id):
            first_read.set()
        else:
            late_started.set()
            assert release_late.wait(timeout=5)
            late_returned.set()
        return {(doc.document_id, doc.chunk_ind): [items[doc.document_id]]}

    monkeypatch.setattr(CorpusBroker, "hydrate_search_centers", hydrate)

    def search(_arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        stage = child.services.get("legal_composite_original_stage")
        assert isinstance(stage, CanonicalEvidenceStage)
        stages.append(stage)
        children.append(child)
        result = lane.hydrate_search_centers(docs, child)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Complete canonical search originals.",
            evidence=[item for group in result.values() for item in group],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="search_corpus",
                description="Retrieve exact authorized originals.",
                parameters={
                    "type": "object",
                    "properties": {"expand_query": {"type": "boolean"}},
                    "additionalProperties": False,
                },
                handler=search,
            )
        ]
    )
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="material-rule",
                question="Uygulanabilir hüküm nedir?",
                governing_source="Uygulanabilir kaynak",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    acquirer = CanonicalAcquirer(registry, context, ledger, WorkflowPolicy())
    action = SourceAction(
        need_ids=["material-rule"],
        tool="search_corpus",
        source_kind=SourceKind.STATUTE,
        arguments={},
    )
    with ThreadPoolExecutor(max_workers=1) as coordinator:
        result = coordinator.submit(acquirer.acquire, [action], plan)
        try:
            assert post_read_checked.wait(timeout=3)
            assert late_started.wait(timeout=3)
            if not withdraw_first:
                assert first_committed.wait(timeout=3)
                assert ledger.citation_numbers() == (1,)
                assert not result.done()
            context.research_deadline = 0
            children[0].research_deadline = 0
            with pytest.raises(RunStopped, match="deadline"):
                result.result(timeout=3)
            numbers = [] if withdraw_first else [1]
            assert acquirer.last_receipts[0]["status"] == "truncated"
            assert acquirer.last_receipts[0]["citations"] == numbers
            assert "not proof of corpus absence" in cast(
                str, acquirer.last_receipts[0]["summary"]
            )
            with pytest.raises(RunStopped, match="closed"):
                stages[0].retain([items[str(late_source.id)]])
        finally:
            release_late.set()
        assert late_returned.wait(timeout=3)
    assert ledger.citation_numbers() == (() if withdraw_first else (1,))
    if not withdraw_first:
        retained = ledger.get(1)
        assert retained is not None
        assert retained.text == "Whole canonical completed original."
        assert retained.question_ids == ["material-rule"]
    assert "legal_composite_original_stage" not in context.services


def test_completed_outcome_is_drained_without_waiting_for_expired_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import Future
    from typing import Any

    from onyx.legal_composite import acquisition

    ledger = EvidenceLedger()
    context = RunContext()

    def research_active() -> None:
        if ledger.citation_numbers():
            raise RunStopped("Research deadline reached")

    monkeypatch.setattr(context, "check_research_active", research_active)

    class CompletedExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit(self, execute: object, *args: Any) -> Future[ToolOutcome]:
            future: Future[ToolOutcome] = Future()
            future.set_result(cast(Any, execute)(*args))
            return future

        def shutdown(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(acquisition, "ThreadPoolExecutor", CompletedExecutor)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read an authorized original.",
                parameters={"type": "object", "properties": {}},
                handler=lambda _arguments, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND,
                    summary="Verified before the deadline.",
                    evidence=[original("Completed canonical original.", "completed")],
                ),
            )
        ]
    )
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="rule",
                question="Uygulanabilir hüküm nedir?",
                governing_source="Uygulanabilir kaynak",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    receipts = CanonicalAcquirer(registry, context, ledger, WorkflowPolicy()).acquire(
        [SourceAction(need_ids=["rule"], tool="read_original", arguments={})], plan
    )
    assert receipts[0]["status"] == "found" and receipts[0]["citations"] == [1]
    assert ledger.citation_numbers() == (1,)
