"""Cheap fresh delivery may revalidate only a cached physical-delivery defect."""

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import HarnessView, OutcomeStatus
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import ToolMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response, turn
from tests.unit.onyx.asv3.test_parallel_terminal_delivery_retry import (
    BODY,
    REQUEST,
    begin,
    setup,
    terminal,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


@pytest.mark.parametrize("name", ["submit_answer", "submit_partial_answer"])
def test_actual_fresh_call_revalidates_owned_stale_delivery_failure(name: str) -> None:
    harness, context, ledger, reviews, handled = setup(parallel=False)
    context.services.update(
        asv3_exp_cheap=True,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
    )
    initial = terminal("first-terminal", name=name)
    if name == "submit_partial_answer":
        # The serial partial terminal has no basis field.
        partial_spec = harness.registry.get(name)
        assert partial_spec is not None
        partial_spec.parameters.update(
            properties={"answer": {"type": "string", "minLength": 1}},
            required=["answer"],
        )
        initial.arguments.pop("basis")
    begin(context, ledger)
    first = harness._dispatch([initial])[0]
    assert first.outcome.status == OutcomeStatus.INVALID
    error = cast(
        dict[str, JsonValue], first.outcome.data["related_source_review_error"]
    )
    assert error["code"] == "not_fully_delivered" and handled == []
    before = reviews.export()
    originals = [full_record(ledger, number) for number in ledger.citation_numbers()]
    current = HarnessView(
        request=REQUEST,
        questions=[REQUEST],
        facts=[],
        receipts=[],
        evidence=[],
        tools=harness.registry.definitions(context),
        turns=[turn("prior-read", originals)],
        original_evidence=originals,
        draft_to_repair=BODY,
        publication_gap={"summary": first.outcome.summary, **first.outcome.data},
    )
    selected = model(limit=1_000_000)
    selected.invoke.return_value = native_response(
        "submit_retained_answer"
        if name == "submit_answer"
        else "submit_retained_partial_answer",
        json.dumps({"_related_source_reviews": [review()]}),
        call_id="fresh-terminal",
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter._cheap_fresh_repair(current)
    assert decision is not None
    assert selected.invoke.call_count == 1
    assert not any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}
    assert (
        ledger.delivery_flow(adapter.last_call_id or "")
        == LLMFlow.ASV3_ANSWER_REPAIR.value
    )
    assert decision.calls[0].arguments == first.call.arguments
    accepted = harness._dispatch(decision.calls)[0]
    assert accepted.outcome.status == OutcomeStatus.FOUND
    assert handled == [BODY]
    assert reviews.export() != before


@pytest.mark.parametrize("partial", [True, False])
def test_partial_or_only_historical_original_delivery_does_not_clear_failure(
    partial: bool,
) -> None:
    harness, context, ledger, _, handled = setup(parallel=False)
    context.services.update(
        asv3_exp_cheap=True, asv3_workflow_variant=ASV3_TUNED_VARIANT
    )
    begin(context, ledger)
    harness._dispatch([terminal("first-terminal")])
    deliver(ledger, "historical-full", [1, 2])
    records = [full_record(ledger, 1)]
    if partial:
        record = full_record(ledger, 2)
        record["text"] = str(record["text"])[:10]
        records.append(record)
    ledger.record_delivery("current", "asv3_answer_repair", records)
    context.services["last_model_call_id"] = "current"
    blocked = harness._dispatch([terminal("retry-terminal")])[0]
    assert blocked.outcome.summary.startswith("Repeated failed call:")
    assert handled == []


@pytest.mark.parametrize(
    "changes",
    [
        {"gap": "The interaction remains unresolved."},
        {"witnesses": []},
        {"witnesses": [{"citation": 2, "start_char": 0, "end_char": 500}]},
    ],
)
def test_semantic_or_metadata_invalidity_remains_cached_after_full_delivery(
    changes: dict[str, JsonValue],
) -> None:
    harness, context, ledger, _, handled = setup(parallel=False)
    context.services.update(
        asv3_exp_cheap=True, asv3_workflow_variant=ASV3_TUNED_VARIANT
    )
    begin(context, ledger)
    assessment = review(**changes)
    first = harness._dispatch([terminal("first-terminal", assessment=assessment)])[0]
    assert first.outcome.status == OutcomeStatus.INVALID
    deliver(ledger, "after", [1, 2, 3])
    context.services["last_model_call_id"] = "after"
    second = harness._dispatch([terminal("retry-terminal", assessment=assessment)])[0]
    assert second.outcome.summary.startswith("Repeated failed call:")
    assert handled == []


def test_plain_serial_experimental_failure_cache_keeps_baseline_bytes() -> None:
    harness, context, ledger, _, handled = setup(parallel=False)
    begin(context, ledger)
    first = harness._dispatch([terminal("first-terminal")])[0]
    assert first.outcome.status == OutcomeStatus.INVALID
    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    blocked = harness._dispatch([terminal("retry-terminal")])[0]
    assert blocked.outcome.model_dump(mode="json") == {
        "status": "invalid",
        "summary": "Repeated failed call: change the arguments or method",
        "evidence": [],
        "artifacts": [],
        "data": {},
        "original_reads": [],
    }
    assert handled == []
