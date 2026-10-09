"""Exercise the replacement selection and checkpoint fences at their real boundaries."""

import copy
import math
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from onyx.asv3 import runtime
from onyx.asv3.jev_answer_review import GuardrailsV2ReviewOutcome
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import ASv3WorkflowSelection, RunContext
from onyx.asv3.shared_reads import SharedReads
from onyx.asv3.workflow_variant import (
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_GUARDRAILS_V3_POLICY,
    ASV3_GUARDRAILS_V3_VARIANT,
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_POLICY,
    ASV3_TUNED_VARIANT,
    checkpoint_variant_fields,
    resolve_asv3_workflow,
    validate_asv3_variant_resume,
)
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.prompts.asv3.tuned import TUNED_PROMPT_VERSION
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.asv3.test_runtime import response, setup_run

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_only_legacy_parallel_selection_is_replaced(profile: str) -> None:
    ordinary = resolve_asv3_workflow(profile, False)
    assert ordinary.research_profile == profile
    assert ordinary.parallel_research is False
    assert ordinary.workflow_variant == ASV3_STANDARD_VARIANT
    assert not ordinary.selected_model_only
    assert checkpoint_variant_fields(ordinary.workflow_variant) == {}
    tuned = resolve_asv3_workflow("experimental", True)
    assert tuned.research_profile == "normal"
    assert tuned.parallel_research is False
    assert tuned.workflow_variant == ASV3_TUNED_VARIANT
    assert tuned.selected_model_only


def test_workflow_selection_is_strict_and_immutable() -> None:
    selected = resolve_asv3_workflow("experimental", True)
    with pytest.raises(ValidationError):
        selected.parallel_research = True
    with pytest.raises(ValidationError):
        ASv3WorkflowSelection.model_validate(
            {"research_profile": "normal", "parallel_research": 0}
        )


def test_guarded_experimental_requires_its_explicit_selection() -> None:
    guarded = resolve_asv3_workflow("normal", False, guarded_experimental=True)
    assert guarded.research_profile == "normal"
    assert guarded.parallel_research is False
    assert guarded.workflow_variant == ASV3_GUARDED_EXPERIMENTAL_VARIANT
    assert guarded.uses_guardrails
    assert not guarded.selected_model_only

    with pytest.raises(ValueError, match="normal profile"):
        resolve_asv3_workflow("experimental", False, guarded_experimental=True)
    with pytest.raises(ValueError, match="without parallel"):
        resolve_asv3_workflow("normal", True, guarded_experimental=True)


def test_guarded_checkpoint_cannot_cross_resume_legacy_variants() -> None:
    guarded = {
        **checkpoint_variant_fields(ASV3_GUARDED_EXPERIMENTAL_VARIANT),
        "research_profile": "normal",
        "parallel_research": False,
    }
    validate_asv3_variant_resume(ASV3_GUARDED_EXPERIMENTAL_VARIANT, guarded)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(ASV3_STANDARD_VARIANT, guarded)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(
            ASV3_GUARDED_EXPERIMENTAL_VARIANT,
            checkpoint_variant_fields(ASV3_TUNED_VARIANT),
        )


def test_guardrails_v2_has_an_isolated_variant_and_checkpoint_policy() -> None:
    selected = resolve_asv3_workflow("normal", False, guardrails_v2=True)

    assert selected.research_profile == "normal"
    assert selected.parallel_research is False
    assert selected.workflow_variant == "asv3_guardrails_v2"
    assert selected.uses_guardrails_v2
    assert not selected.uses_guardrails
    checkpoint = {
        **checkpoint_variant_fields(selected.workflow_variant),
        "research_profile": "normal",
        "parallel_research": False,
    }
    assert checkpoint["asv3_workflow_policy"] == "jev-gemini-review-v1"
    validate_asv3_variant_resume(selected.workflow_variant, checkpoint)

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_asv3_workflow(
            "normal",
            False,
            guarded_experimental=True,
            guardrails_v2=True,
        )
    with pytest.raises(ValueError, match="normal profile"):
        resolve_asv3_workflow("experimental", False, guardrails_v2=True)
    with pytest.raises(ValueError, match="without parallel"):
        resolve_asv3_workflow("normal", True, guardrails_v2=True)


def test_guardrails_v3_has_an_isolated_variant_and_checkpoint_policy() -> None:
    selected = resolve_asv3_workflow("normal", False, guardrails_v3=True)

    assert selected.research_profile == "normal"
    assert selected.parallel_research is False
    assert selected.workflow_variant == ASV3_GUARDRAILS_V3_VARIANT
    assert selected.uses_guardrails_v3
    assert not selected.uses_guardrails
    assert not selected.uses_guardrails_v2
    checkpoint = {
        **checkpoint_variant_fields(selected.workflow_variant),
        "research_profile": "normal",
        "parallel_research": False,
    }
    assert checkpoint["asv3_workflow_policy"] == ASV3_GUARDRAILS_V3_POLICY
    validate_asv3_variant_resume(selected.workflow_variant, checkpoint)

    for incompatible in (
        {"guarded_experimental": True},
        {"guardrails_v2": True},
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_asv3_workflow("normal", False, guardrails_v3=True, **incompatible)
    with pytest.raises(ValueError, match="normal profile"):
        resolve_asv3_workflow("experimental", False, guardrails_v3=True)
    with pytest.raises(ValueError, match="without parallel"):
        resolve_asv3_workflow("normal", True, guardrails_v3=True)


def test_guardrails_v2_request_validation_is_independent_from_v1() -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guardrails_v2": True,
        }
    )
    assert request.asv3_guardrails_v2 is True
    assert request.asv3_guarded_experimental is False

    with pytest.raises(ValidationError, match="mutually exclusive"):
        SendMessageRequest.model_validate(
            {
                "message": "İlgili mevzuat nedir?",
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_guarded_experimental": True,
                "asv3_guardrails_v2": True,
            }
        )


def test_guardrails_v3_request_validation_is_independent_from_v1_and_v2() -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guardrails_v3": True,
        }
    )
    assert request.asv3_guardrails_v3 is True
    assert request.asv3_guarded_experimental is False
    assert request.asv3_guardrails_v2 is False

    for conflicting_flag in (
        "asv3_guarded_experimental",
        "asv3_guardrails_v2",
    ):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SendMessageRequest.model_validate(
                {
                    "message": "İlgili mevzuat nedir?",
                    "atez_search_v3": True,
                    "asv3_research_profile": "normal",
                    "asv3_guardrails_v3": True,
                    conflicting_flag: True,
                }
            )
    with pytest.raises(ValidationError, match="normal non-parallel"):
        SendMessageRequest.model_validate(
            {
                "message": "İlgili mevzuat nedir?",
                "atez_search_v3": True,
                "asv3_research_profile": "experimental",
                "asv3_guardrails_v3": True,
            }
        )
    with pytest.raises(ValidationError, match="normal non-parallel"):
        SendMessageRequest.model_validate(
            {
                "message": "İlgili mevzuat nedir?",
                "atez_search_v3": True,
                "asv3_research_profile": "experimental",
                "asv3_guardrails_v2": True,
            }
        )


@pytest.mark.parametrize(
    "variant,profile,legacy_payload",
    [
        (ASV3_STANDARD_VARIANT, "normal", True),
        (ASV3_STANDARD_VARIANT, "deep", True),
        (ASV3_STANDARD_VARIANT, "experimental", True),
        (ASV3_TUNED_VARIANT, "normal", True),
        (ASV3_GUARDED_EXPERIMENTAL_VARIANT, "normal", False),
    ],
)
def test_native_runtime_marks_only_legacy_search_payloads(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    profile: str,
    legacy_payload: bool,
) -> None:
    kwargs, _broker, selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile=profile, workflow_variant=variant)
    contexts: list[RunContext] = []

    class SetupObserved(Exception):
        pass

    def observe_model(_llm: LLM, context: RunContext, **_options: Any) -> ResearchModel:
        contexts.append(context)
        raise SetupObserved

    monkeypatch.setattr(runtime, "ResearchModel", observe_model)
    with pytest.raises(SetupObserved):
        runtime.run_asv3_loop(**kwargs)

    context = contexts[0]
    assert context.services["asv3_legacy_search_payload"] is legacy_payload
    assert context.child().services["asv3_legacy_search_payload"] is legacy_payload
    assert context.services["research_profile"] == profile
    if variant == ASV3_STANDARD_VARIANT:
        assert "asv3_workflow_variant" not in context.services
    else:
        assert context.services["asv3_workflow_variant"] == variant
    if legacy_payload:
        assert context.budget.unlimited_execution is True
        assert math.isinf(context.deadline)
        assert "provider_max_attempts" not in context.services
        assert "provider_compatibility_attempts" not in context.services
    else:
        assert context.budget.unlimited_execution is False
        assert math.isfinite(context.deadline)
        assert context.services["provider_max_attempts"] == 2
        assert context.services["provider_compatibility_attempts"] == 1
    selected.invoke.assert_not_called()


@pytest.mark.parametrize(
    "saved",
    [
        {},
        {"research_profile": "normal", "parallel_research": False},
        {"research_profile": "experimental", "parallel_research": True},
        {"asv3_workflow_variant": "asv3_tuned"},
        {
            "asv3_workflow_variant": "asv3_tuned",
            "asv3_workflow_policy": ASV3_TUNED_POLICY,
            "research_profile": "normal",
            "parallel_research": 0,
        },
    ],
)
def test_tuned_resume_requires_its_exact_variant_policy(saved: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_asv3_variant_resume(ASV3_TUNED_VARIANT, saved)


def test_standard_keeps_historical_checkpoints_but_rejects_tuned() -> None:
    validate_asv3_variant_resume("standard", {"research_profile": "normal"})
    validate_asv3_variant_resume("standard", None)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(
            "standard", checkpoint_variant_fields("asv3_tuned")
        )


def test_tuned_runtime_uses_normal_tools_and_selected_model_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    secondary = MagicMock(spec=LLM)
    secondary.invoke.side_effect = AssertionError("Lite handoff must not run")
    search = MagicMock(spec=SearchTool)
    fork = MagicMock(spec=SearchTool)
    search.llm = secondary
    search.fork_for_independent_context.return_value = fork
    contexts = []
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        assert llm is selected
        assert options.get("research_llm") is None
        contexts.append(context)
        return original_model(llm, context, **options)

    def make_broker(_user: Any, scope: Any, **options: Any) -> Any:
        assert options["vision_llm"] is selected
        assert options["allow_numbered_title_fallback"] is True
        broker.scope = scope
        return broker

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    monkeypatch.setattr(runtime, "CorpusBroker", make_broker)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant=ASV3_TUNED_VARIANT,
        research_llm=secondary,
        tools=[search],
    )
    runtime.run_asv3_loop(**kwargs)
    assert selected.invoke.call_count == 2
    secondary.invoke.assert_not_called()
    assert fork.llm is selected
    assert search.llm is secondary
    context = contexts[0]
    assert context.services["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    assert context.services["research_profile"] == "normal"
    assert context.services["experimental_parallel"] is False
    assert context.services["independent_question_mode"] is False
    assert not {
        "authority_requirements",
        "parallel_execution_slots",
        "parallel_query_embeddings",
    }.intersection(context.services)
    assert isinstance(context.services["legal_source_reviews"], LegalSourceReviews)
    assert isinstance(context.services["shared_reads"], SharedReads)
    for invocation in selected.invoke.call_args_list:
        names = {tool["function"]["name"] for tool in invocation.kwargs["tools"]}
        assert not {
            "research_questions",
            "assemble_answers",
            "commit_retained_answer",
        }.intersection(names)
    assert checkpoints[-1]["prompt_version"] == TUNED_PROMPT_VERSION
    assert checkpoints[-1]["research_profile"] == "normal"
    assert checkpoints[-1]["parallel_research"] is False
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    assert checkpoints[-1]["asv3_workflow_policy"] == ASV3_TUNED_POLICY


def test_guarded_runtime_has_finite_cost_and_latency_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, _selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    contexts = []
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        contexts.append(context)
        return original_model(llm, context, **options)

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant=ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    )
    runtime.run_asv3_loop(**kwargs)

    context = contexts[0]
    assert math.isfinite(context.deadline)
    remaining_total_seconds = context.deadline - time.monotonic()
    remaining_research_seconds = context.research_deadline - time.monotonic()
    assert 1798 <= remaining_total_seconds <= 1800
    assert 1768 <= remaining_research_seconds <= 1770
    assert context.budget.unlimited_execution is False
    assert context.budget.limits["tools"] == 24
    assert context.budget.limits["decisions"] == 32
    assert context.services["provider_max_attempts"] == 2
    assert context.services["asv3_workflow_variant"] == (
        ASV3_GUARDED_EXPERIMENTAL_VARIANT
    )


def test_guardrails_v3_runtime_reserves_time_for_final_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, _selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    contexts = []
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        contexts.append(context)
        return original_model(llm, context, **options)

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant=ASV3_GUARDRAILS_V3_VARIANT,
    )
    runtime.run_asv3_loop(**kwargs)

    context = contexts[0]
    remaining_total_seconds = context.deadline - time.monotonic()
    remaining_research_seconds = context.research_deadline - time.monotonic()
    assert 1798 <= remaining_total_seconds <= 1800
    assert 1678 <= remaining_research_seconds <= 1680
    assert context.budget.unlimited_execution is False
    assert context.budget.limits["tools"] == 24
    assert context.budget.limits["decisions"] == 32
    assert context.services["asv3_workflow_variant"] == ASV3_GUARDRAILS_V3_VARIANT


def test_guarded_runtime_keeps_source_tools_cheap_and_coordinator_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    secondary = MagicMock(spec=LLM)
    secondary.invoke.side_effect = AssertionError("Coordinator must not use Lite")
    search = MagicMock(spec=SearchTool)
    fork = MagicMock(spec=SearchTool)
    search.llm = secondary
    search.fork_for_independent_context.return_value = fork
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        assert llm is selected
        assert options.get("research_llm") is None
        return original_model(llm, context, **options)

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant=ASV3_GUARDED_EXPERIMENTAL_VARIANT,
        research_llm=secondary,
        tools=[search],
    )
    runtime.run_asv3_loop(**kwargs)

    assert selected.invoke.call_count == 2
    secondary.invoke.assert_not_called()
    assert fork.llm is secondary


def test_guardrails_v2_runtime_reviews_only_the_final_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, _selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    contexts = []
    review_calls: list[dict[str, Any]] = []
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        contexts.append(context)
        return original_model(llm, context, **options)

    def review(**options: Any) -> GuardrailsV2ReviewOutcome:
        review_calls.append(options)
        return GuardrailsV2ReviewOutcome(
            answer="JEV sonrası düzeltilmiş tamir [1] ve değiştirme [2].",
            review_completed=True,
            repair_requested=True,
            repair_applied=True,
            defects=["condition_loss"],
            review_input_tokens=120,
            review_output_tokens=5,
            repair_input_tokens=80,
            repair_output_tokens=12,
        )

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    monkeypatch.setattr(runtime, "review_and_repair_answer", review)
    repair_llm = MagicMock(spec=LLM)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant="asv3_guardrails_v2",
        repair_llm=repair_llm,
    )

    runtime.run_asv3_loop(**kwargs)

    assert len(review_calls) == 1
    call = review_calls[0]
    assert call["repair_llm"] is repair_llm
    assert call["candidate_answer"].startswith("Tamir sonucu [")
    assert set(item.citation for item in call["evidence"]) == {1, 2}
    published = kwargs["state_container"].get_answer_tokens()
    assert published is not None
    assert published.startswith("JEV sonrası düzeltilmiş tamir")
    assert "https://example.test/law-0" in published
    assert "https://example.test/law-1" in published
    assert contexts[0].services["guardrails_v2_review"]["repair_applied"] is True
    assert checkpoints[-1]["guardrails_v2_review"]["repair_requested"] is True
    assert checkpoints[-1]["guardrails_v2_review"]["repair_applied"] is True
    assert checkpoints[-1]["guardrails_v2_review"]["review_input_tokens"] == 120
    assert checkpoints[-1]["guardrails_v2_review"]["repair_output_tokens"] == 12


def test_actual_runtime_resume_fences_before_saved_mode_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message="Merhaba", token_count=1, message_type=MessageType.USER
        )
    ]
    selected.invoke.side_effect = lambda **_args: response(
        calls=[
            (
                "submit_answer",
                {"answer": "Merhaba!", "basis": "conversation", "_language": "tr"},
            )
        ]
    )
    runtime.run_asv3_loop(**kwargs)
    tuned = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: tuned)
    kwargs["resume_message_id"] = 2
    selected.reset_mock()
    runtime.run_asv3_loop(**kwargs)
    assert selected.invoke.call_count == 1
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    kwargs["workflow_variant"] = ASV3_STANDARD_VARIANT
    selected.reset_mock()
    with pytest.raises(ValueError, match="same workflow variant"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
    old_parallel = copy.deepcopy(tuned)
    old_parallel.pop("asv3_workflow_variant")
    old_parallel.pop("asv3_workflow_policy")
    old_parallel.update(research_profile="experimental", parallel_research=True)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: old_parallel)
    kwargs["workflow_variant"] = ASV3_TUNED_VARIANT
    with pytest.raises(ValueError, match="retired parallel"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
    kwargs["workflow_variant"] = ASV3_STANDARD_VARIANT
    with pytest.raises(ValueError, match="retired parallel"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
