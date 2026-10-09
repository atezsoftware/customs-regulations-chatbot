"""Recover canonical excluded originals without another read or classification call."""

from typing import cast
from unittest.mock import Mock, patch

import pytest

from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.engine import (
    LegalCompositeEngine,
    ModelGateway,
)
from onyx.legal_composite.models import IssueResearchPlan, WorkflowPolicy
from onyx.legal_composite.reviewer import AnswerReviewer
from onyx.legal_composite.selection import (
    SelectionReceipt,
    SourceIdentity,
    SourceSelectionResult,
)
from tests.unit.onyx.legal_composite.test_source_requirements import fixture


def recovery_engine() -> tuple[LegalCompositeEngine, IssueResearchPlan, Mock, Mock]:
    ledger, plan, _, _ = fixture()
    gateway = Mock(spec=ModelGateway)
    context = RunContext()
    acquirer = CanonicalAcquirer(
        CapabilityRegistry([]), context, ledger, WorkflowPolicy()
    )
    tool_io = Mock()
    tool_io.acquire.side_effect = AssertionError("Recovery cannot read sources")
    tool_io.definitions.side_effect = AssertionError("Recovery cannot invoke tools")
    acquirer.acquire = tool_io.acquire
    acquirer.definitions = tool_io.definitions
    engine = LegalCompositeEngine(
        gateway=cast(ModelGateway, gateway),
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
        reviewer=cast(AnswerReviewer, Mock(spec=AnswerReviewer)),
        evidence_context=context,
    )
    engine.plan = plan
    identities = []
    for citation in ledger.citation_numbers():
        item = ledger.get(citation)
        assert item is not None
        identities.append(
            SourceIdentity(
                citation=citation,
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                text_hash=item.text_hash,
            )
        )
    engine.selection = SourceSelectionResult(
        protected_citations=[],
        background_citations=[2],
        rejected_citations=[1],
        retained_citations=[2],
        selection_complete=True,
        gaps=[],
        identities=identities,
        receipts=[
            SelectionReceipt(
                need_id=need,
                citation=1,
                roles=["irrelevant"],
                irrelevance_probability=0.99,
                reason="Earlier relevance judgment.",
                full_original_seen=True,
            )
            for need in ("a", "b")
        ],
        call_id="earlier-selector-call",
    )
    return engine, plan, gateway, tool_io


def test_reconsideration_preserves_rejection_audit_without_tool_or_model_io() -> None:
    engine, plan, gateway, acquirer = recovery_engine()
    assert engine.selection is not None
    before = engine.selection.model_copy(deep=True)
    engine._reconsider_sources([1], plan, None)
    selection = engine.selection
    assert selection is not None
    assert selection.protected_citations == []
    assert selection.background_citations == [1, 2]
    assert selection.retained_citations == [1, 2]
    assert selection.rejected_citations == []
    assert selection.identities == before.identities
    assert selection.call_id == before.call_id
    assert selection.receipts[: len(before.receipts)] == before.receipts
    assert selection.receipts == before.receipts
    recovery = engine.receipts[-1]
    assert recovery["status"] == "reconsidered"
    assert recovery["need_ids"] == ["a", "b"]
    assert recovery["citations"] == [1]
    assert len(engine.ledger.citation_numbers()) == 2
    assert gateway.mock_calls == [] and acquirer.mock_calls == []


def test_focused_reconsideration_preserves_existing_need_binding() -> None:
    engine, plan, gateway, acquirer = recovery_engine()
    item = engine.ledger.get(1)
    assert item is not None and engine.selection is not None
    item.question_ids = ["a"]
    engine.ledger.add([item], RunContext())
    old_count = len(engine.selection.receipts)
    engine._reconsider_sources([1], plan, {"b"})
    stored = engine.ledger.get(1)
    assert stored is not None and set(stored.question_ids) == {"a", "b"}
    assert len(engine.selection.receipts) == old_count
    assert engine.receipts[-1]["need_ids"] == ["b"]
    assert gateway.mock_calls == [] and acquirer.mock_calls == []


def test_unknown_reconsideration_id_rejects_entire_batch_atomically() -> None:
    engine, plan, gateway, acquirer = recovery_engine()
    assert engine.selection is not None
    item = engine.ledger.get(1)
    assert item is not None
    item.question_ids = ["a"]
    engine.ledger.add([item], RunContext())
    before = engine.selection.model_dump(mode="json")
    with pytest.raises(InvalidSourceAction):
        engine._reconsider_sources([1, 999], plan, {"b"})
    assert engine.selection.model_dump(mode="json") == before
    stored = engine.ledger.get(1)
    assert stored is not None and stored.question_ids == ["a"]
    assert gateway.mock_calls == [] and acquirer.mock_calls == []


@pytest.mark.parametrize("defect", ["hash", "document", "chunk", "derived"])
def test_reconsideration_rejects_noncanonical_original(defect: str) -> None:
    engine, plan, gateway, acquirer = recovery_engine()
    assert engine.selection is not None
    item = engine.ledger.get(1)
    assert item is not None and item.search_doc is not None
    before = engine.selection.model_dump(mode="json")
    if defect == "hash":
        item.text = "Changed content without a canonical hash update."
    elif defect == "document":
        item.search_doc.document_id = "another-document"
    elif defect == "chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = "another-chunk"
    else:
        item.metadata["canonical_metadata"] = {"derived": True}
    real_get = engine.ledger.get

    def corrupted_original(number: int) -> EvidenceItem | None:
        return item.model_copy(deep=True) if number == 1 else real_get(number)

    with patch.object(engine.ledger, "get", side_effect=corrupted_original):
        with pytest.raises(InvalidSourceAction):
            engine._reconsider_sources([1], plan, None)
    assert engine.selection.model_dump(mode="json") == before
    assert gateway.mock_calls == [] and acquirer.mock_calls == []


def test_active_requirement_keeps_complete_background_original_mandatory() -> None:
    engine, plan, gateway, acquirer = recovery_engine()
    _, _, requirements, _ = fixture()
    engine._reconsider_sources([1], plan, {"a"})
    engine.requirements.update([requirements[0]], plan, {1, 2})
    payload = engine._payload("A ve B işlemleri?", "", source_phase=False)
    assert payload["required_evidence_numbers"] == [1]
    records = payload["original_evidence"]
    assert isinstance(records, list)
    record = next(
        row for row in records if isinstance(row, dict) and row["citation"] == 1
    )
    item = engine.ledger.get(1)
    assert item is not None and record["text"] == item.text
    assert gateway.mock_calls == [] and acquirer.mock_calls == []
