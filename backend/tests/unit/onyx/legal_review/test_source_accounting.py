"""Source inventories preserve originals and admit only bound canonical continuations."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.legal_review.models import SourceAction
from onyx.legal_review.passages import canonical_evidence_view
from onyx.legal_review.source_accounting import (
    SourceAccountant,
    SourceAssessment,
    SourceInventory,
)
from tests.unit.onyx.legal_review.test_engine import original, plan


def assessment(**updates: Any) -> SourceAssessment:
    values = {
        "slot": "s0001",
        "disposition": "material_limitation",
        "reason": "The challenge may restrict the requested outcome.",
        "passage_roles": ["quoted_rule"],
        "established_effect": None,
        "supports": [{"citation": 1, "span_number": 1}],
        "missing_effect": "The disposition and temporal scope are unread.",
        "content_status": "operative_effect_missing",
        "requested_read": {
            "tool": "read_source_range",
            "arguments": {"source_id": "source-1", "start": 0, "limit": 20},
            "issue_ids": ["i1"],
        },
    }
    values.update(updates)
    return SourceAssessment.model_validate(values)


def state(ledger: EvidenceLedger) -> dict[str, Any]:
    return {
        "request": "What conditions govern this permit?",
        "plan": plan().model_dump(mode="json"),
        "original_evidence": canonical_evidence_view(ledger),
        "tools": [
            {
                "function": {
                    "name": "read_source_range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "start": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["source_id"],
                        "additionalProperties": False,
                    },
                }
            }
        ],
    }


def test_inventory_caches_only_unchanged_sources_and_questions() -> None:
    ledger = EvidenceLedger()
    context = RunContext()
    ledger.add([original()], context)
    gateway = Mock()
    gateway.complete.return_value = SourceInventory(source_assessments=[assessment()])
    accountant = SourceAccountant(gateway, ledger)
    packet = state(ledger)
    first = accountant.scan(packet)
    assert first[0]["source_id"] == "source-1"
    selected = SourceAction.model_validate(first[0]["requested_read"])
    assert selected.arguments["source_id"] == "source-1"
    assert accountant.scan(packet) == first
    assert gateway.complete.call_count == 1
    extra = original()
    extra.chunk_id = "chunk-2"
    assert extra.search_doc is not None
    extra.search_doc = extra.search_doc.model_copy(
        update={"metadata": {"regulatory_chunk_id": "chunk-2"}}
    )
    ledger.add([extra], context)
    accountant.scan(state(ledger))
    assert gateway.complete.call_count == 2
    packet = state(ledger)
    packet["request"] = "A newly requested material consequence"
    accountant.scan(packet)
    assert gateway.complete.call_count == 3


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "wrong_source", "wrong_issue", "wrong_passage"]
)
def test_unbound_or_incomplete_source_inventories_are_never_admitted(
    mutation: str,
) -> None:
    ledger = EvidenceLedger()
    ledger.add([original()], RunContext())
    item = assessment()
    assert item.requested_read is not None
    rows = [item]
    if mutation == "missing":
        rows = []
    if mutation == "duplicate":
        rows = [item, item]
    if mutation == "wrong_source":
        item.requested_read.arguments["source_id"] = "other-source"
    if mutation == "wrong_issue":
        item.requested_read.issue_ids = ["unknown"]
    if mutation == "wrong_passage":
        item.supports[0] = item.supports[0].model_copy(update={"citation": 10})
    gateway = Mock()
    gateway.complete.return_value = SourceInventory(source_assessments=rows)
    accountant = SourceAccountant(gateway, ledger)
    with pytest.raises(ValueError):
        accountant.scan(state(ledger))
    assert accountant._assessments == {}


def test_a_quoted_rule_cannot_count_as_the_sources_own_operative_effect() -> None:
    with pytest.raises(ValidationError, match="quoted rule"):
        assessment(
            content_status="operative_effect_read",
            requested_read=None,
            established_effect="The quoted rule remains applicable.",
        )
    with pytest.raises(ValidationError):
        assessment(
            requested_read={
                "tool": "search_corpus",
                "arguments": {"query": "repeat"},
                "issue_ids": ["i1"],
            }
        )


def test_engine_executes_selected_read_rescans_new_original_and_does_not_repeat() -> (
    None
):
    from tests.unit.onyx.legal_review.test_engine import engine as make_engine
    from tests.unit.onyx.legal_review.test_engine import reading

    engine, _, _ = make_engine([], [])
    engine.plan = plan()
    engine.ledger.add([original()], engine.context)
    engine._accept_reading(reading())
    unresolved = assessment().model_dump(mode="json")
    unresolved.update(source_id="source-1", citations=[1])
    resolved = dict(
        unresolved,
        requested_read=None,
        content_status="operative_effect_read",
        missing_effect=None,
    )
    accountant = Mock()
    accountant.scan.side_effect = [[unresolved], [resolved], [resolved]]
    engine.source_accountant = accountant

    def acquire(actions: list[SourceAction], **_: Any) -> int:
        assert len(actions) == 1
        engine.acquirer.receipts.append(
            {"tool": actions[0].tool, "arguments": actions[0].arguments}
        )
        extra = original()
        extra.chunk_id = "chunk-2"
        assert extra.search_doc is not None
        extra.search_doc = extra.search_doc.model_copy(
            update={"metadata": {"regulatory_chunk_id": "chunk-2"}}
        )
        engine.ledger.add([extra], engine.context)
        return 1

    with patch.object(engine, "_acquire", side_effect=acquire) as execute:
        engine._account_sources("request", "")
        assert accountant.scan.call_count == 2
        engine._account_sources("request", "")
        assert execute.call_count == 1
        accountant.scan.side_effect = None
        accountant.scan.return_value = [unresolved]
        engine._account_sources("request", "")
        assert execute.call_count == 1
    assert "disposition" in " ".join(engine.issue_closures()[0].reasons)


def test_a_quoted_supporting_rule_does_not_claim_the_containing_sources_limiting_effect() -> (
    None
):
    row = assessment(
        disposition="supports_existing_finding",
        content_status="operative_effect_read",
        requested_read=None,
        established_effect="This source quotes the document condition.",
        missing_effect=None,
    )
    assert row.passage_roles == ["quoted_rule"]
    assert row.requested_read is None
