"""Exercise the numbered reading transport through the real LC gateway and engine."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite import gateway as gateway_module
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine, SourceAcquirer
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import (
    IssueResearchPlan,
    IssueResearchStep,
    ResearchStep,
    SourceRequirement,
    SpanSupport,
    WorkflowPolicy,
)
from onyx.legal_composite.prompts import COMMON, RESEARCH_PROMPT
from onyx.legal_composite.reading_evidence import IssueReadingResponse
from onyx.legal_composite.reading_prompt import NUMBERED_READING_PROMPT
from onyx.legal_composite.reviewer import AnswerReviewer
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow
from tests.unit.legal_composite.test_gateway import model as model
from tests.unit.legal_composite.test_query_repair import raw_plan
from tests.unit.onyx.legal_composite.test_reading_evidence import (
    OTHER_RULE,
    RULE,
    evidence,
    requirement,
    response,
)
from tests.unit.onyx.legal_composite.test_span_support import plan


def gateway(
    model: Mock,
    ledger: EvidenceLedger,
    *,
    context_tokens: int = 32_000,
    share_draft_context: bool = True,
) -> BudgetedGateway:
    return BudgetedGateway(
        selected_llm=model,
        research_llm=model,
        budget=WorkflowBudget(WorkflowPolicy(max_context_tokens=context_tokens)),
        ledger=ledger,
        share_draft_context=share_draft_context,
    )


def provider_response(model: Mock, value: BaseModel | dict[str, Any]) -> ModelResponse:
    content = (
        value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value)
    )
    return model.invoke.return_value.model_copy(
        deep=True,
        update={
            "choice": Choice(message=Message(content=content), finish_reason="stop")
        },
    )


def sent_body(model: Mock, index: int = -1) -> dict[str, JsonValue]:
    content = model.invoke.call_args_list[index].args[0][1].content
    assert isinstance(content, str)
    return cast(dict[str, JsonValue], json.loads(content))


def engine(
    actual_gateway: BudgetedGateway,
    ledger: EvidenceLedger,
    *,
    numbered: bool = True,
    semantic: bool = True,
) -> tuple[LegalCompositeEngine, Mock]:
    acquirer = Mock(spec=SourceAcquirer)
    acquirer.definitions.return_value = []
    instance = LegalCompositeEngine(
        gateway=actual_gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=actual_gateway.budget.policy,
        check_active=lambda: None,
        research_available=actual_gateway.budget.research_available,
        reviewer=Mock(spec=AnswerReviewer) if semantic else None,
        use_numbered_reading_supports=numbered,
    )
    instance.plan = plan()
    return instance, acquirer


def test_actual_gateway_fits_before_manifest_and_omitted_original_cannot_bind(
    model: Mock,
) -> None:
    ledger, records, _prior = evidence(RULE, "A" * 60_000)
    actual = gateway(model, ledger, context_tokens=5_000)
    model.invoke.return_value = provider_response(model, response(requirement()))
    payload: dict[str, JsonValue] = {
        "original_evidence": cast(list[JsonValue], records),
        "required_evidence_numbers": [1],
    }
    frozen = deepcopy(payload)
    value = actual.complete(
        "Read the exact numbered original supports.",
        payload,
        IssueReadingResponse,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    sent = sent_body(model)
    assert sent["original_evidence"] == [records[0]]
    assert sent["omitted_original_ids"] == [2]
    assert actual.last_delivered_citations == {1}
    manifest = actual.last_reading_manifest
    assert manifest is not None and manifest.call_id == actual.last_call_id
    assert {entry.citation for entry in manifest.witnesses} == {1}
    assert ledger.completely_delivered(manifest.call_id) == {1}
    assert model.invoke.call_args.kwargs["max_tokens"] == 6_144
    assert model.invoke.call_count == 1 and payload == frozen
    instance, acquirer = engine(actual, ledger)
    resolved = instance._resolve_source_reading(value)
    assert isinstance(resolved, IssueResearchStep)
    assert resolved.requirements[0].supports[0].quotation == RULE
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        instance._resolve_source_reading(response(requirement(citation=2)))
    assert model.invoke.call_count == 1
    acquirer.acquire.assert_not_called()


def test_numbered_response_has_existing_capacity_and_scalar_research_telemetry(
    model: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, records, _prior = evidence(RULE)
    actual = gateway(model, ledger)
    rows: list[dict[str, Any]] = []

    @contextmanager
    def capture(
        operation: str, _inputs: Any = None, *, summary: str | None = None
    ) -> Iterator[Mock]:
        step = Mock()
        step.output_value = None
        step.summary = summary
        yield step
        rows.append(
            {
                "operation": operation,
                "summary": step.summary,
                "output": step.output_value,
            }
        )

    monkeypatch.setattr(gateway_module, "graph_step", capture)
    model.invoke.return_value = provider_response(model, response(requirement()))
    actual.complete(
        NUMBERED_READING_PROMPT,
        {"original_evidence": cast(list[JsonValue], records)},
        IssueReadingResponse,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    validation = [
        row for row in rows if row["operation"] == "legal_composite.typed_validation"
    ]
    assert len(validation) == 1
    assert validation[0]["summary"] == "model_kind=research_step validated=1 errors=0"
    assert "approval_rule" not in repr(validation) and RULE not in repr(validation)
    assert model.invoke.call_args.kwargs["max_tokens"] == 6_144
    assert actual.budget.snapshot()["model_calls"] == 1


def test_actual_engine_reads_once_and_preserves_all_original_and_requirement_fields(
    model: Mock,
) -> None:
    ledger, _records, _prior = evidence(RULE, OTHER_RULE)
    actual = gateway(model, ledger)
    instance, acquirer = engine(actual, ledger)
    proposed = response(requirement())
    proposed.requirements[0].missing_user_facts = ["approval status"]
    model.invoke.return_value = provider_response(model, proposed)
    payload = instance._payload("Apply both fictional conditions to my facts.", "")
    frozen = deepcopy(payload)
    value = instance._complete_source_reading(payload)
    assert isinstance(value, IssueReadingResponse)
    resolved = instance._resolve_source_reading(value)
    assert isinstance(resolved, IssueResearchStep)
    assert isinstance(instance.plan, IssueResearchPlan)
    instance._validate_gap_scope(resolved, instance.plan)
    instance.requirements.update(
        resolved.requirements, instance.plan, actual.last_delivered_citations
    )
    recorded = instance.requirements.records()[0]
    proposed_fields = proposed.requirements[0].model_dump(
        exclude={"supports"}, mode="json"
    )
    assert recorded.model_dump(exclude={"supports"}, mode="json") == proposed_fields
    assert recorded.supports[0].quotation == RULE
    assert recorded.supports[0].span_id is not None
    originals = sent_body(model)["original_evidence"]
    assert isinstance(originals, list) and len(originals) == 2
    for row in originals:
        assert isinstance(row, dict) and isinstance(row["citation"], int)
        source = ledger.get(row["citation"])
        assert source is not None
        assert row["text"] == source.text and row["text_hash"] == source.text_hash
        assert (
            row["source_id"] == source.source_id and row["chunk_id"] == source.chunk_id
        )
        spans = row["witness_spans"]
        assert isinstance(spans, list) and all(
            isinstance(span, dict) and "span_number" in span for span in spans
        )
    assert model.invoke.call_args.args[0][0].content.startswith(NUMBERED_READING_PROMPT)
    assert model.invoke.call_count == 1 and payload == frozen
    acquirer.acquire.assert_not_called()


def test_bad_pair_cannot_partially_change_active_requirement_ledger_or_retry(
    model: Mock,
) -> None:
    ledger, _records, _prior = evidence(RULE)
    actual = gateway(model, ledger)
    instance, acquirer = engine(actual, ledger)
    assert instance.plan is not None
    existing = SourceRequirement(
        requirement_id="existing",
        need_id="approval",
        dimension="procedure",
        rule="Existing exact supported rule.",
        application="Retain its actual source.",
        supports=[SpanSupport(citation=1, quotation=RULE)],
    )
    instance.requirements.update([existing], instance.plan, {1})
    frozen = deepcopy(instance.requirements.export())
    proposed = response(
        requirement(identity="new-valid"),
        requirement(span_number=999, identity="invalid"),
    )
    model.invoke.return_value = provider_response(model, proposed)
    value = instance._complete_source_reading(
        instance._payload("Apply the actual source.", "")
    )
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        step = instance._resolve_source_reading(value)
        assert isinstance(step, IssueResearchStep)
        instance.requirements.update(
            step.requirements, instance.plan, actual.last_delivered_citations
        )
    assert instance.requirements.export() == frozen
    assert model.invoke.call_count == actual.budget.snapshot()["model_calls"] == 1
    acquirer.acquire.assert_not_called()


@pytest.mark.parametrize("semantic,numbered", [(True, False), (False, True)])
def test_old_engine_reading_path_keeps_type_prompt_defaults_and_unnumbered_originals(
    model: Mock, semantic: bool, numbered: bool
) -> None:
    ledger, _records, _prior = evidence(RULE)
    actual = gateway(model, ledger)
    instance, acquirer = engine(actual, ledger, numbered=numbered, semantic=semantic)
    legacy = (
        IssueResearchStep(actions=[], ready_to_answer=True, remaining_gaps=[])
        if semantic
        else ResearchStep(actions=[], ready_to_answer=True, remaining_gaps=[])
    )
    model.invoke.return_value = provider_response(model, legacy)
    value = instance._complete_source_reading(
        instance._payload("Use the actual source.", "")
    )
    assert type(value) is type(legacy) and value.model_dump(
        mode="json"
    ) == legacy.model_dump(mode="json")
    assert instance._resolve_source_reading(value) is value
    assert actual.last_reading_manifest is None
    assert model.invoke.call_args.args[0][0].content.startswith(RESEARCH_PROMPT)
    originals = sent_body(model)["original_evidence"]
    assert isinstance(originals, list)
    for row in originals:
        assert isinstance(row, dict)
        for span in row.get("witness_spans", []):
            assert isinstance(span, dict) and "span_number" not in span
    assert model.invoke.call_count == 1
    acquirer.acquire.assert_not_called()


@pytest.mark.parametrize("invalid_compression", [False, True])
def test_parent_query_compression_keeps_parent_delivery_and_does_not_reuse_prior_manifest(
    model: Mock, invalid_compression: bool
) -> None:
    ledger, records, _prior = evidence(RULE)
    actual = gateway(model, ledger)
    model.invoke.return_value = provider_response(model, response(requirement()))
    actual.complete(
        "Read",
        {"original_evidence": cast(list[JsonValue], records)},
        IssueReadingResponse,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    prior = actual.last_reading_manifest
    assert prior is not None
    oversized = raw_plan()
    model.invoke.side_effect = [
        provider_response(model, oversized),
        provider_response(
            model, {"query": "" if invalid_compression else "refund replacement terms"}
        ),
    ]
    if invalid_compression:
        with pytest.raises(RunStopped, match="workflow schema"):
            actual.complete(
                "Plan",
                {"original_evidence": cast(list[JsonValue], records)},
                IssueResearchPlan,
                LLMFlow.LEGAL_COMPOSITE_RESEARCH,
            )
    else:
        accepted = actual.complete(
            "Plan",
            {"original_evidence": cast(list[JsonValue], records)},
            IssueResearchPlan,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
        assert accepted.discovery_query == "refund replacement terms"
    assert model.invoke.call_count == 3
    assert actual.last_call_id is not None and actual.last_call_id != prior.call_id
    assert (
        actual.last_delivered_citations
        == ledger.completely_delivered(actual.last_call_id)
        == {1}
    )
    assert actual.last_reading_manifest is None
    assert model.invoke.call_args.kwargs["max_tokens"] == 2_048
    assert actual.budget.snapshot()["model_calls"] == 3


def test_new_prompt_changes_only_the_two_selector_regions_and_preserves_substantive_guidance() -> (
    None
):
    selector = "Complete original_evidence contains witness_spans:"
    prefix, _legacy_selector = COMMON.split(selector, 1)
    assert RESEARCH_PROMPT.startswith(COMMON)
    assert NUMBERED_READING_PROMPT.startswith(prefix)
    content = RESEARCH_PROMPT[len(COMMON) :]
    reading_start = "\nUse the issue plan, full request and exact delivered originals"
    expected = content.replace(
        "Use provided span_id references instead of retyping original quotations.",
        "Use provided citation/span_number pairs instead of retyping original quotations.",
        1,
    )
    actual_content = NUMBERED_READING_PROMPT[
        NUMBERED_READING_PROMPT.index(reading_start) :
    ]
    assert actual_content == expected
    assert "provided citation and set span_id" in COMMON
    assert "Use provided span_id references" in RESEARCH_PROMPT
    assert "Use provided span_id references" not in NUMBERED_READING_PROMPT
    assert "return only citation and span_number" in NUMBERED_READING_PROMPT
