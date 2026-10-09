import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import CallReservation, WorkflowBudget
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import IssueResearchStep, WorkflowPolicy
from onyx.llm.cost import ModelPrice
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow

SECRET_INPUT = "private-source-passage-7f394b"
SECRET_CONTEXT = "private-error-context-12a349"
SECRET_LOCATION = "private-citation-id-83216c"
SECRET_TYPE = "private-custom-error-49af10"
SECRET_MODEL = "PrivateProviderPayloadModel-20c65e"


class AdversarialResponse(BaseModel):
    private_value: str = Field(alias=SECRET_LOCATION)

    @field_validator("private_value")
    @classmethod
    def reject_private_value(cls, _value: str) -> str:
        raise PydanticCustomError(
            SECRET_TYPE, "Private context: {detail}", {"detail": SECRET_CONTEXT}
        )


AdversarialResponse.__name__ = SECRET_MODEL


@pytest.fixture
def harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]]:
    traces: list[dict[str, JsonValue]] = []

    @contextmanager
    def capture(
        operation: str,
        attributes: dict[str, JsonValue],
        *,
        summary: str | None = None,
    ) -> Iterator[Mock]:
        step = Mock()
        yield step
        traces.append(
            {
                "operation": operation,
                "attributes": attributes,
                "summary": summary,
                "output": step.output_value,
            }
        )

    monkeypatch.setattr("onyx.legal_composite.gateway.graph_step", capture)
    monkeypatch.setattr(
        "onyx.legal_composite.gateway._priced_model",
        lambda *_args: ModelPrice(
            model="local-only",
            provider="local-only",
            input_per_mtok=0.1,
            output_per_mtok=0.5,
            cache_per_mtok=None,
        ),
    )
    monkeypatch.setattr(
        "onyx.legal_composite.gateway.check_number_of_tokens",
        lambda text: len(text) // 4,
    )
    model = Mock(spec=LLM)
    model.config = LLMConfig(
        model_name="local-only",
        model_provider="local-only",
        temperature=0,
        max_input_tokens=32_000,
    )
    ledger = Mock(spec=EvidenceLedger)
    ledger.completely_delivered.return_value = {1}
    gateway = BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
        share_draft_context=True,
    )
    return gateway, ledger, traces


def complete(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    response_type: type[BaseModel] = IssueResearchStep,
) -> BaseModel:
    gateway, _, _ = harness
    response = ModelResponse(
        id="local-response",
        created="0",
        choice=Choice(message=Message(content=content), finish_reason="stop"),
    )
    reservation = CallReservation(
        call_id="local-reservation",
        input_tokens=100,
        output_tokens=100,
        estimated_cost_usd=0.001,
        input_price_per_million=0.1,
        output_price_per_million=0.5,
        timeout_seconds=5,
    )
    monkeypatch.setattr(gateway, "_generate", lambda *_args: (reservation, response))
    return gateway.complete(
        "Synthetic source-reading protocol",
        {"original_evidence": [{"citation": 1, "text": SECRET_INPUT}]},
        response_type,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )


def validation_trace(traces: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    records = [
        trace
        for trace in traces
        if trace["operation"] == "legal_composite.typed_validation"
    ]
    assert len(records) == 1
    return records[0]


def assert_source_free(traces: list[dict[str, JsonValue]]) -> None:
    encoded = json.dumps(traces)
    for secret in (
        SECRET_INPUT,
        SECRET_CONTEXT,
        SECRET_LOCATION,
        SECRET_TYPE,
        SECRET_MODEL,
    ):
        assert secret not in encoded
    summary = validation_trace(traces)["summary"]
    assert isinstance(summary, str) and len(summary) <= 180
    assert all("=" in word for word in summary.split())


def test_schema_failure_reports_only_allowlisted_top_fields_and_error_types(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = json.dumps(
        {
            "actions": SECRET_INPUT,
            "ready_to_answer": SECRET_INPUT,
            "remaining_gaps": [],
            "issue_gaps": {SECRET_LOCATION: [123]},
            SECRET_LOCATION: SECRET_CONTEXT,
        }
    )
    with pytest.raises(
        RunStopped, match="model response failed the workflow schema"
    ) as caught:
        complete(harness, monkeypatch, content)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.__cause__.error_count() == 4
    gateway, ledger, traces = harness
    assert gateway.last_call_id == "local-reservation"
    ledger.record_delivery.assert_called_once()
    record = validation_trace(traces)
    assert record["attributes"] == {
        "model_kind": "research_step",
        "validated": 0,
        "errors": 4,
    }
    output = cast(dict[str, JsonValue], record["output"])
    assert output["error_type_counts"] == {
        "list_type": 1,
        "bool_type": 1,
        "string_type": 1,
        "extra_forbidden": 1,
    }
    assert output["field_counts"] == {
        "actions": 1,
        "ready_to_answer": 1,
        "issue_gaps": 1,
        "other": 1,
    }
    assert_source_free(traces)


def test_custom_error_kind_context_model_name_and_location_remain_private(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RunStopped) as caught:
        complete(
            harness,
            monkeypatch,
            json.dumps({SECRET_LOCATION: SECRET_INPUT}),
            AdversarialResponse,
        )
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    actual = cause.errors()[0]
    assert actual["type"] == SECRET_TYPE
    assert actual["loc"] == (SECRET_LOCATION,)
    assert actual["ctx"] == {"detail": SECRET_CONTEXT}
    assert actual["input"] == SECRET_INPUT
    record = validation_trace(harness[2])
    assert record["output"] == {
        "model_kind": "other",
        "validated": 0,
        "errors": 1,
        "error_type_counts": {"other_error": 1},
        "field_counts": {"other": 1},
    }
    assert_source_free(harness[2])


def test_success_records_validated_schema_and_preserves_original_delivery(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = complete(
        harness,
        monkeypatch,
        json.dumps({"actions": [], "ready_to_answer": True, "remaining_gaps": []}),
    )
    assert isinstance(result, IssueResearchStep) and result.ready_to_answer
    gateway, ledger, traces = harness
    ledger.record_delivery.assert_called_once_with(
        "local-reservation",
        LLMFlow.LEGAL_COMPOSITE_RESEARCH.value,
        [{"citation": 1, "text": SECRET_INPUT}],
    )
    assert gateway.last_delivered_citations == {1}
    record = validation_trace(traces)
    assert record["summary"] == "model_kind=research_step validated=1 errors=0"
    assert record["output"] == {
        "model_kind": "research_step",
        "validated": 1,
        "errors": 0,
        "error_type_counts": {},
        "field_counts": {},
    }
    assert_source_free(traces)


def test_invalid_json_counts_root_without_exporting_the_private_input(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RunStopped):
        complete(harness, monkeypatch, '{"' + SECRET_INPUT + '":')
    record = validation_trace(harness[2])
    assert record["output"] == {
        "model_kind": "research_step",
        "validated": 0,
        "errors": 1,
        "error_type_counts": {"json_invalid": 1},
        "field_counts": {"root": 1},
    }
    assert_source_free(harness[2])


@pytest.mark.parametrize("valid", [True, False])
def test_default_gateway_does_not_add_validation_events(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, JsonValue]]],
    monkeypatch: pytest.MonkeyPatch,
    valid: bool,
) -> None:
    opted_in, ledger, traces = harness
    gateway = BudgetedGateway(
        selected_llm=opted_in.selected_llm,
        research_llm=opted_in.research_llm,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
    )
    assert gateway.share_draft_context is False
    legacy_harness = (gateway, ledger, traces)
    content = json.dumps({"actions": [], "ready_to_answer": True, "remaining_gaps": []})
    if valid:
        result = complete(legacy_harness, monkeypatch, content)
        assert isinstance(result, IssueResearchStep) and result.ready_to_answer
    else:
        with pytest.raises(
            RunStopped, match="model response failed the workflow schema"
        ):
            complete(legacy_harness, monkeypatch, "{}")
    assert not any(
        trace["operation"] == "legal_composite.typed_validation" for trace in traces
    )
    ledger.record_delivery.assert_called_once()
