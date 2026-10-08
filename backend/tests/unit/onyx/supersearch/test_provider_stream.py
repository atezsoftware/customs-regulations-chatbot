"""Complete JSON delivery and cancellation while a selected provider is streaming."""

import json
from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.llm.model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoice,
    Usage,
)
from onyx.supersearch import gateway
from onyx.supersearch.models import WriterDecision
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.supersearch.test_canonical_runtime import (
    selected_llm,
    structured_stream,
)
from tests.unit.onyx.supersearch.test_engine import ANSWER, original


def stream_gateway(
    monkeypatch: pytest.MonkeyPatch,
    chunks: Iterator[ModelResponseStream],
    context: RunContext,
) -> tuple[gateway.SelectedModelGateway, EvidenceLedger, list[ModelResponse]]:
    llm = selected_llm()
    cast(MagicMock, llm).stream.return_value = chunks
    span = SimpleNamespace(span_data=SimpleNamespace(model_config={}))
    recorded: list[ModelResponse] = []
    monkeypatch.setattr(
        gateway, "llm_generation_span", lambda **_kwargs: nullcontext(span)
    )
    monkeypatch.setattr(
        gateway,
        "record_llm_response",
        lambda _span, response: recorded.append(response),
    )
    ledger = EvidenceLedger()
    ledger.add([original()], context)
    return (
        gateway.SelectedModelGateway(llm=llm, ledger=ledger, context=context),
        ledger,
        recorded,
    )


def complete(
    selected: gateway.SelectedModelGateway, ledger: EvidenceLedger
) -> WriterDecision:
    return selected.complete(
        "Original PC evidence only",
        {"original_evidence": json.loads(ledger.serialize_records([1]))},
        WriterDecision,
        LLMFlow.SUPERSEARCH_REVIEW,
        True,
    )


def test_private_stream_assembly_retains_finish_marker_and_final_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = structured_stream(
        json.dumps({"answer": ANSWER, "unresolved_need_ids": [], "actions": []})
    )
    chunks[0].choice.delta.reasoning_content = "private reasoning"
    usage = Usage(
        completion_tokens=15,
        prompt_tokens=90,
        total_tokens=105,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        reasoning_tokens=5,
    )
    chunks.append(
        ModelResponseStream(
            id="fixture-response",
            created="2026-10-09",
            choice=StreamingChoice(),
            usage=usage,
        )
    )
    selected, ledger, recorded = stream_gateway(monkeypatch, iter(chunks), RunContext())
    assert complete(selected, ledger).answer == ANSWER
    assert selected.last_delivered_citations == {1}
    assert recorded[0].usage == usage
    assert recorded[0].choice.finish_reason == "stop"
    assert recorded[0].choice.message.reasoning_content == "private reasoning"


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_stop_during_provider_generation_closes_stream_without_delivering_draft(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: bool,
) -> None:
    stopped, closed = False, False

    def chunks() -> Iterator[ModelResponseStream]:
        nonlocal stopped, closed
        try:
            yield ModelResponseStream(
                id="fixture",
                created="2026-10-09",
                choice=StreamingChoice(delta=Delta(content='{"answer":"')),
            )
            stopped = True
            yield ModelResponseStream(
                id="fixture",
                created="2026-10-09",
                choice=StreamingChoice(delta=Delta(content="unpublished draft")),
            )
            raise AssertionError("Cancelled provider iterator must not be drained")
        finally:
            closed = True
            if cleanup_error:
                raise RuntimeError("Provider transport cleanup fixture")

    context = RunContext(
        cancelled=lambda: stopped, budget=SharedBudget(unlimited_execution=True)
    )
    selected, ledger, recorded = stream_gateway(monkeypatch, chunks(), context)
    with pytest.raises(RunStopped, match="cancel"):
        complete(selected, ledger)
    assert closed
    assert not recorded
    assert selected.last_call_id is None and not selected.last_delivered_citations
    assert ledger.citation_numbers() == (1,)


@pytest.mark.parametrize(
    "failure", ["missing_finish", "length", "tool_call", "other_choice"]
)
def test_incomplete_or_undeclared_provider_output_never_counts_as_original_delivery(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    chunks = structured_stream(
        json.dumps({"answer": ANSWER, "unresolved_need_ids": [], "actions": []})
    )
    if failure == "missing_finish":
        chunks[-1].choice.finish_reason = None
    elif failure == "length":
        chunks[-1].choice.finish_reason = "length"
    elif failure == "tool_call":
        chunks[-1].choice.delta.tool_calls = [
            ChatCompletionDeltaToolCall(id="unexpected")
        ]
    else:
        chunks[-1].choice.index = 1
    selected, ledger, _recorded = stream_gateway(
        monkeypatch, iter(chunks), RunContext()
    )
    with pytest.raises(RunStopped):
        complete(selected, ledger)
    assert selected.last_call_id is None and not selected.last_delivered_citations
