"""Real structured parsing prevents the live planner's omitted-query shape from passing."""

import json
from collections.abc import Iterator
from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.models import RunContext, SharedBudget
from onyx.legal_review.drafting import (
    ClaimApplication,
    GeneratedBlock,
    GeneratedClaim,
    GeneratedDraft,
)
from onyx.legal_review.gateway import GeminiGateway
from onyx.legal_review.models import (
    InitialDiscoveryPlan,
    LegalDimension,
    WorkflowPolicy,
)
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.prompts.legal_review.prompts import PLAN_PROMPT
from onyx.regulatory.structured_llm import StructuredOutputValidationError
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_review.test_engine import RULE, draft, engine, plan


def omitted_queries() -> dict[str, JsonValue]:
    return {
        "language": "tr",
        "issues": [
            {
                "issue_id": f"i{index}",
                "question": "Başvuru şartı nedir?",
                "requested_outcome": "Başvuru şartını açıklamak",
            }
            for index in range(1, 4)
        ],
    }


def valid_plan() -> InitialDiscoveryPlan:
    return InitialDiscoveryPlan.model_validate(
        {
            **plan().model_dump(),
            "requested_outcomes": [
                {"request": "Application condition", "issue_ids": ["i1"]}
            ],
            "discovery_queries": [
                {"query": "Başvuru şartı belge ibrazı", "issue_ids": ["i1"]}
            ],
        }
    )


def response(value: dict[str, JsonValue]) -> ModelResponse:
    return ModelResponse(
        id="fixture-response",
        created="2026-10-09T00:00:00Z",
        choice=Choice(message=Message(content=json.dumps(value, ensure_ascii=False))),
    )


def generated_draft() -> GeneratedDraft:
    fixture = draft()
    return GeneratedDraft(
        blocks=[
            GeneratedBlock(
                block_id="answer",
                text=fixture.answer,
                claims=[
                    GeneratedClaim(
                        claim_id=claim.claim_id,
                        issue_ids=claim.issue_ids,
                        supports=claim.supports,
                        application=ClaimApplication(
                            source_conditions="The original requires the document.",
                            fact_application="The question requests that condition.",
                            remaining_uncertainty=None,
                        ),
                    )
                    for claim in fixture.claims
                ],
            )
        ],
        unresolved_issue_ids=fixture.unresolved_issue_ids,
    )


def reading_response() -> dict[str, JsonValue]:
    return {
        "requirements": [
            {
                "requirement_id": "r1",
                "rule": RULE,
                "supports": [{"citation": 1, "span_number": 1}],
            }
        ],
        "dimensions": [
            {
                "issue_id": "i1",
                "dimension": dimension.value,
                "status": "addressed"
                if dimension is LegalDimension.LEGAL_BASIS
                else "not_applicable",
                "reason": "Başvuru için belgenin ibrazı gerekir."
                if dimension is LegalDimension.LEGAL_BASIS
                else "Bu boyutun istenen başvuru şartına etkisi yoktur.",
                "requirement_ids": ["r1"]
                if dimension is LegalDimension.LEGAL_BASIS
                else [],
            }
            for dimension in LegalDimension
        ],
    }


def gateway(llm: MagicMock, context: RunContext) -> GeminiGateway:
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=200_000,
    )
    return GeminiGateway(
        llm=cast(LLM, llm), context=context, policy=WorkflowPolicy(), token_counter=len
    )


@pytest.fixture(autouse=True)
def no_external_tracing() -> Iterator[None]:
    with (
        patch(
            "onyx.regulatory.structured_llm.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.regulatory.structured_llm.record_llm_response"),
    ):
        yield


def test_live_omitted_query_shape_fails_real_structured_generation() -> None:
    llm = MagicMock(spec=LLM)
    llm.invoke.return_value = response(omitted_queries())
    context = RunContext(budget=SharedBudget(max_decisions=32))
    with pytest.raises(StructuredOutputValidationError, match="discovery_queries"):
        gateway(llm, context).complete(
            PLAN_PROMPT,
            {"request": "Başvuru şartı nedir?"},
            InitialDiscoveryPlan,
            LLMFlow.LEGAL_REVIEW_PLANNER,
        )
    assert llm.invoke.call_count == 1
    assert context.budget.snapshot()["decisions"] == 1
    schema = llm.invoke.call_args.kwargs["structured_response_format"]["json_schema"][
        "schema"
    ]
    assert "discovery_queries" in schema["required"]
    assert schema["properties"]["discovery_queries"]["minItems"] == 1


@pytest.mark.parametrize(
    "queries",
    [
        [],
        [{"query": " ", "issue_ids": ["i1"]}],
        [{"query": "q" * 601, "issue_ids": ["i1"]}],
        [{"query": "Başvuru", "issue_ids": ["unknown"]}],
    ],
)
def test_empty_unbounded_or_unbound_initial_queries_cannot_validate(
    queries: list[JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        InitialDiscoveryPlan.model_validate(
            {**plan().model_dump(), "discovery_queries": queries}
        )


def test_one_global_query_validates_with_31_issues_and_no_per_issue_query_fields() -> (
    None
):
    data = omitted_queries()
    data["issues"] = [
        {
            "issue_id": f"i{index}",
            "question": "Başvuru şartı nedir?",
            "requested_outcome": f"Sonuç {index}",
            "material_reason": "Resolve eligibility.",
            "closure_criteria": ["Establish eligibility conditions."],
        }
        for index in range(31)
    ]
    data["requested_outcomes"] = [
        {"request": "Eligibility", "issue_ids": [f"i{index}" for index in range(31)]}
    ]
    data["discovery_queries"] = [
        {
            "query": "Başvuru şartı belge ibrazı",
            "issue_ids": [f"i{index}" for index in range(31)],
        }
    ]
    result = InitialDiscoveryPlan.model_validate(data)
    assert len(result.issues) == 31 and len(result.discovery_queries) == 1
    assert all(issue.research_queries == [] for issue in result.issues)


def test_initial_schema_correction_is_separately_metered_and_can_resume_research() -> (
    None
):
    workflow, _, _ = engine([], ["pass", "pass"])
    llm = MagicMock(spec=LLM)
    llm.invoke.side_effect = [
        response(omitted_queries()),
        response(valid_plan().model_dump(mode="json")),
        response(reading_response()),
        response(generated_draft().model_dump(mode="json")),
    ]
    workflow.gateway = gateway(llm, workflow.context)
    result = workflow.run("Başvuru şartı nedir?", "Verilen olgular")
    assert result.answer is not None
    assert llm.invoke.call_count == 4
    assert workflow.context.budget.snapshot()["decisions"] == 4
    corrected_state = json.loads(llm.invoke.call_args_list[1].args[0][1].content)
    assert corrected_state["planning_schema_correction"] is True
    assert corrected_state["request"] == "Başvuru şartı nedir?"
    writer_schema = llm.invoke.call_args.kwargs["structured_response_format"][
        "json_schema"
    ]["schema"]
    assert "blocks" in writer_schema["required"]
    assert "answer" not in writer_schema["properties"]


def test_repeated_initial_schema_failure_stops_after_two_admitted_calls() -> None:
    workflow, _, _ = engine([], [])
    llm = MagicMock(spec=LLM)
    llm.invoke.return_value = response(omitted_queries())
    workflow.gateway = gateway(llm, workflow.context)
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "unavailable"
    assert llm.invoke.call_count == 2
    assert workflow.context.budget.snapshot()["decisions"] == 2
    assert workflow.acquirer.searches == 0


def test_provider_failure_never_triggers_schema_correction() -> None:
    workflow, _, _ = engine([], [])
    llm = MagicMock(spec=LLM)
    llm.invoke.side_effect = RuntimeError("provider unavailable")
    workflow.gateway = gateway(llm, workflow.context)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        workflow.run("Başvuru şartı nedir?", "")
    assert llm.invoke.call_count == 1
    assert workflow.context.budget.snapshot()["decisions"] == 1


def test_outcome_coverage_can_group_requests_without_losing_issue_discovery() -> None:
    data = valid_plan().model_dump()
    data["requested_outcomes"] = [
        {"request": "Which condition applies?", "issue_ids": ["i1"]},
        {"request": "How does that condition affect release?", "issue_ids": ["i1"]},
    ]
    grouped = InitialDiscoveryPlan.model_validate(data)
    assert len(grouped.issues) == 1 and len(grouped.requested_outcomes) == 2
    data["requested_outcomes"][0]["issue_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="exactly the planned issues"):
        InitialDiscoveryPlan.model_validate(data)


def test_initial_issue_cannot_omit_materiality_or_closure_criteria() -> None:
    for field in ("material_reason", "closure_criteria"):
        data = valid_plan().model_dump()
        data["issues"][0].pop(field)
        with pytest.raises(ValidationError, match=field):
            InitialDiscoveryPlan.model_validate(data)


def test_declared_issue_cannot_silently_miss_initial_discovery() -> None:
    data = valid_plan().model_dump()
    data["issues"].append({**data["issues"][0], "issue_id": "i2"})
    data["requested_outcomes"].append(
        {"request": "Another outcome", "issue_ids": ["i2"]}
    )
    with pytest.raises(ValidationError, match="covered by initial discovery"):
        InitialDiscoveryPlan.model_validate(data)
