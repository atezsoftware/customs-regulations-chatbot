"""Related-source range cursors remain navigation with canonical receipt bindings."""

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from tests.unit.onyx.asv3.test_experimental_workflow import experimental_context
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review, seen
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def range_receipt(
    context: RunContext,
    ledger: EvidenceLedger,
    *,
    start: int = 0,
    next_position: int = 30,
    has_more: bool = True,
    call_id: str = "bounded-range",
) -> ToolReceipt:
    items = [
        original(
            f"source-position-{position}",
            f"Recorded original at position {position}.",
            source="decision",
            metadata={"position": position},
        )
        for position in (start, next_position - 1)
    ]
    for item in items:
        assert item.chunk_id is not None
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    return ToolReceipt(
        call=CapabilityCall(
            name="read_source_range",
            arguments={"source_id": "decision", "start": start},
            call_id=call_id,
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL if has_more else OutcomeStatus.FOUND,
            summary="Private provider information must not enter the projection.",
            data={
                "has_more": has_more,
                "next_position": next_position,
                "source_name": "Unverified competing title",
                "arbitrary_payload": "Do not copy this receipt field.",
            },
        ),
        elapsed_seconds=0,
        evidence_ids=ledger.add(items, context),
    )


@pytest.mark.parametrize("assessed", [False, True])
def test_partial_range_cursor_is_visible_without_approving_or_resetting_a_review(
    assessed: bool,
) -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    if assessed:
        deliver(ledger, "own-assessment", [2])
        reviews.apply([review()], "own-assessment", context, ledger)
    receipt = range_receipt(context, ledger)
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, n) for n in [1, *receipt.evidence_ids]],
    ).model_copy(update={"receipts": [receipt]})
    before_view, before_ledger = current.model_dump(), ledger.export()
    before_reviews, before_budget = reviews.export(), context.budget.snapshot()
    llm = model()

    prompt, _, _ = ResearchModel(
        llm, context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    row = payload["related_source_reviews"]["reviews"][0]
    cursor = row["source_range_read"]
    assert row["status"] == ("examined" if assessed else "pending")
    assert cursor["status"] == "partial"
    assert cursor["start"] == 0
    assert cursor["has_more"] is True
    assert cursor["next_position"] == 30
    assert cursor["receipt_id"] == "bounded-range"
    assert "not holding or applicability approval" in cursor["notice"]
    assert set(cursor) == {
        "status",
        "start",
        "has_more",
        "next_position",
        "receipt_id",
        "notice",
    }
    assert "Private provider information" not in json.dumps(cursor)
    assert "Unverified competing title" not in json.dumps(cursor)
    assert current.model_dump() == before_view
    assert ledger.export() == before_ledger
    assert reviews.export() == before_reviews
    assert context.budget.snapshot() == before_budget
    llm.invoke.assert_not_called()


def test_latest_acquired_page_retains_exact_start_without_claiming_complete_legal_review() -> (
    None
):
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    older = range_receipt(context, ledger)
    latest = range_receipt(
        context,
        ledger,
        start=30,
        next_position=34,
        has_more=False,
        call_id="continued-range",
    )
    state = reviews.view(context, ledger, set(ledger.citation_numbers()))
    projected = ResearchModel._related_source_range_continuations(
        state, [older, latest], ledger
    )
    rows = cast(list[dict[str, JsonValue]], projected["reviews"])
    cursor = cast(dict[str, JsonValue], rows[0]["source_range_read"])
    assert rows[0]["status"] == "pending"
    assert cursor["status"] == "found"
    assert cursor["start"] == 30
    assert cursor["has_more"] is False
    assert cursor["next_position"] == 34
    assert cursor["receipt_id"] == "continued-range"
    assert "has_more=false" in str(cursor["notice"])
    assert "complete" not in cursor
    assert (
        "source_range_read" not in cast(list[dict[str, JsonValue]], state["reviews"])[0]
    )


@pytest.mark.parametrize(
    "failure", ["wrong_source", "foreign_evidence", "forged_cursor", "failed_read"]
)
def test_unvalidated_acquisition_cannot_supply_a_related_source_cursor(
    failure: str,
) -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    receipt = range_receipt(context, ledger)
    if failure == "wrong_source":
        receipt.call.arguments["source_id"] = "another-source"
    elif failure == "foreign_evidence":
        receipt.evidence_ids.append(3)
    elif failure == "forged_cursor":
        receipt.outcome.data["next_position"] = 999
    else:
        receipt.outcome.status = OutcomeStatus.ERROR
    state = reviews.view(context, ledger, set(ledger.citation_numbers()))
    assert (
        ResearchModel._related_source_range_continuations(state, [receipt], ledger)
        == state
    )


def test_another_tasks_lead_is_not_enriched_by_shared_source_originals() -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    receipt = range_receipt(context, ledger)
    child = context.child()
    child.services["task_id"] = "independent-owner"
    state = reviews.view(child, ledger, set(ledger.citation_numbers()))
    assert state["reviews"] == []
    assert (
        ResearchModel._related_source_range_continuations(state, [receipt], ledger)
        == state
    )


@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_legacy_context_does_not_gain_a_source_range_projection(profile: str) -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    receipt = range_receipt(context, ledger)
    context.services["research_profile"] = profile
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, n) for n in [1, *receipt.evidence_ids]],
    ).model_copy(update={"receipts": [receipt]})
    prompt, _, _ = ResearchModel(
        model(), context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    row = payload["related_source_reviews"]["reviews"][0]
    assert "source_range_read" not in row


def test_opaque_receipt_id_stays_out_of_the_navigation_projection() -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    receipt = range_receipt(context, ledger, call_id="__thought__" + "opaque" * 100)
    state = reviews.view(context, ledger, set(ledger.citation_numbers()))
    projected = ResearchModel._related_source_range_continuations(
        state, [receipt], ledger
    )
    row = cast(list[dict[str, JsonValue]], projected["reviews"])[0]
    cursor = cast(dict[str, JsonValue], row["source_range_read"])
    assert "receipt_id" not in cursor
    assert "__thought__" not in json.dumps(projected)
