"""Exercise the replacement selection and checkpoint fences at their real boundaries."""

import copy
import json
import math
import time
from types import SimpleNamespace
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
    ASV3_EXP_CHEAP_POLICY,
    ASV3_EXP_CHEAP_VARIANT,
    ASV3_EXP_CROPPED_POLICY,
    ASV3_EXP_CROPPED_VARIANT,
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
from tests.unit.onyx.asv3.test_cropped_focused_repair_runtime import (
    priced_scripted_provider as priced_scripted_provider,
)
from tests.unit.onyx.asv3.test_runtime import response, setup_run

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


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


def test_cheap_selection_keeps_a_distinct_tuned_copy_identity() -> None:
    selected = resolve_asv3_workflow("normal", False, exp_cheap=True)

    assert selected.workflow_variant == ASV3_EXP_CHEAP_VARIANT
    assert selected.research_profile == "normal"
    assert selected.parallel_research is False
    assert selected.selected_model_only
    assert not selected.uses_guardrails
    assert not selected.uses_guardrails_v2
    assert not selected.uses_guardrails_v3
    checkpoint = {
        **checkpoint_variant_fields(selected.workflow_variant),
        "research_profile": "normal",
        "parallel_research": False,
    }
    assert checkpoint["asv3_workflow_policy"] == ASV3_EXP_CHEAP_POLICY
    validate_asv3_variant_resume(selected.workflow_variant, checkpoint)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(ASV3_TUNED_VARIANT, checkpoint)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(
            ASV3_EXP_CHEAP_VARIANT,
            {
                **checkpoint_variant_fields(ASV3_TUNED_VARIANT),
                "research_profile": "normal",
                "parallel_research": False,
            },
        )


@pytest.mark.parametrize(
    ("profile", "parallel", "guarded", "v2", "v3"),
    [
        ("normal", True, False, False, False),
        ("experimental", False, False, False, False),
        ("deep", False, False, False, False),
        ("normal", False, True, False, False),
        ("normal", False, False, True, False),
        ("normal", False, False, False, True),
    ],
)
def test_cheap_resolver_cannot_bypass_profile_or_variant_fences(
    profile: str, parallel: bool, guarded: bool, v2: bool, v3: bool
) -> None:
    with pytest.raises(ValueError):
        resolve_asv3_workflow(
            profile,
            parallel,
            guarded_experimental=guarded,
            guardrails_v2=v2,
            guardrails_v3=v3,
            exp_cheap=True,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"asv3_workflow_policy": ASV3_TUNED_POLICY},
        {"asv3_workflow_policy": None},
        {"research_profile": "experimental"},
        {"parallel_research": 0},
        {"parallel_research": True},
    ],
)
def test_cheap_checkpoint_requires_its_exact_policy(changed: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_asv3_variant_resume(
            ASV3_EXP_CHEAP_VARIANT,
            {
                **checkpoint_variant_fields(ASV3_EXP_CHEAP_VARIANT),
                "research_profile": "normal",
                "parallel_research": False,
                **changed,
            },
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


@pytest.mark.parametrize(
    "variant", [ASV3_TUNED_VARIANT, ASV3_EXP_CHEAP_VARIANT, ASV3_EXP_CROPPED_VARIANT]
)
def test_tuned_runtime_uses_normal_tools_and_selected_model_everywhere(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
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
        workflow_variant=variant,
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
    assert context.services.get("asv3_exp_cheap", False) is (
        variant == ASV3_EXP_CHEAP_VARIANT
    )
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
    assert checkpoints[-1]["asv3_workflow_variant"] == variant
    assert checkpoints[-1]["asv3_workflow_policy"] == (
        ASV3_EXP_CHEAP_POLICY
        if variant == ASV3_EXP_CHEAP_VARIANT
        else ASV3_EXP_CROPPED_POLICY
        if variant == ASV3_EXP_CROPPED_VARIANT
        else ASV3_TUNED_POLICY
    )


def test_cropped_adds_scoped_bindings_while_preserving_history_and_provider_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    contexts: list[RunContext] = []
    original_model = runtime.ResearchModel
    original_context = runtime.RunContext
    shared_inputs: dict[str, Any] = {}
    history = [
        ChatMessageSimple(
            message="Önceki soru", token_count=2, message_type=MessageType.USER
        ),
        ChatMessageSimple(
            message="Önceki cevap", token_count=2, message_type=MessageType.ASSISTANT
        ),
        ChatMessageSimple(
            message="Merhaba", token_count=1, message_type=MessageType.USER
        ),
    ]

    def make_context(**options: Any) -> RunContext:
        options.setdefault("run_id", "00000000-0000-4000-8000-000000000001")
        return original_context(**options)

    def observe_model(llm: LLM, context: RunContext, **options: Any) -> ResearchModel:
        contexts.append(context)
        assert options["research_llm"] is None
        return original_model(llm, context, **options)

    monkeypatch.setattr(runtime, "RunContext", make_context)
    monkeypatch.setattr(runtime, "ResearchModel", observe_model)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    for variant in (
        ASV3_TUNED_VARIANT,
        ASV3_EXP_CHEAP_VARIANT,
        ASV3_EXP_CROPPED_VARIANT,
    ):
        kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
        kwargs.pop("test_language")
        if not shared_inputs:
            shared_inputs = {
                name: kwargs[name]
                for name in (
                    "llm",
                    "user",
                    "filters",
                    "user_identity",
                    "chat_session_id",
                    "user_message_id",
                    "assistant_message_id",
                    "reasoning_effort",
                    "include_citations",
                )
            }
        kwargs.update(shared_inputs)
        kwargs.update(research_profile="normal", workflow_variant=variant)
        kwargs["simple_chat_history"] = copy.deepcopy(history)
        selected = shared_inputs["llm"]
        selected.reset_mock()
        selected.invoke.side_effect = lambda **_arguments: response(
            calls=[
                (
                    "submit_answer",
                    {"answer": "Merhaba!", "basis": "conversation", "_language": "tr"},
                )
            ]
        )
        runtime.run_asv3_loop(**kwargs)
        assert selected.invoke.call_count == 1
        requests.append(copy.deepcopy(selected.invoke.call_args.kwargs))
        assert checkpoints[-1]["prompt_version"] == TUNED_PROMPT_VERSION
        assert checkpoints[-1]["asv3_workflow_variant"] == variant

    assert requests[0] == requests[1]

    def canonical_content(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        if isinstance(value, list):
            return [canonical_content(item) for item in value]
        if isinstance(value, dict):
            return {
                key: canonical_content(item) if key in {"content", "text"} else item
                for key, item in value.items()
            }
        return value

    # Cropped may serialize JSON whitespace compactly; semantic inputs are exact.
    logical_requests = [
        {
            **request,
            "prompt": [
                canonical_content(message.model_dump(mode="json"))
                for message in request["prompt"]
            ],
        }
        for request in (requests[0], requests[2])
    ]
    tuned_requested = logical_requests[0].pop("max_tokens")
    cropped_allowed = logical_requests[1].pop("max_tokens")
    assert isinstance(tuned_requested, int)
    assert isinstance(cropped_allowed, int)
    assert 2048 <= cropped_allowed <= tuned_requested
    tuned_prompt = logical_requests[0].pop("prompt")
    cropped_prompt = logical_requests[1].pop("prompt")
    logical_requests[0].pop("tools")
    logical_requests[1].pop("tools")
    assert logical_requests[0] == logical_requests[1]
    for message in history:
        assert message.message in str(tuned_prompt)
        assert message.message in str(cropped_prompt)
    assert "question_bindings" in str(cropped_prompt)
    assert "cropped_completion" in str(cropped_prompt)
    assert "question_bindings" not in str(tuned_prompt)
    assert contexts[2].services.get("asv3_exp_cheap", False) is False
    assert "cheap_authority_policy" not in contexts[2].services
    assert "cheap_source_review_policy" not in contexts[2].services
    for context in contexts:
        assert context.services["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
        from onyx.asv3.search_adapter import guarded_retrieval_overrides

        retrieval = guarded_retrieval_overrides(
            context.services["asv3_workflow_variant"]
        )
        assert retrieval is not None
        assert retrieval.per_lane_num_hits == 256
        assert retrieval.rerank_candidate_limit == 384
        assert retrieval.regulatory_rerank_candidate_limit == 384
        assert retrieval.max_llm_chunks == 50
        assert context.services["asv3_legacy_search_payload"] is True
        assert context.services["experimental_parallel"] is False
        assert context.services["independent_question_mode"] is False
        assert context.budget.unlimited_execution is True
        assert math.isinf(context.deadline)
        assert isinstance(context.services["legal_source_reviews"], LegalSourceReviews)
        assert isinstance(context.services["shared_reads"], SharedReads)


def test_cheap_runtime_binds_only_the_selected_model_cache_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.llm import multi_llm

    kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CHEAP_VARIANT)
    bound = MagicMock(spec=LLM)
    bound.config = selected.config.model_copy()
    bound.invoke.side_effect = selected.invoke.side_effect
    selected.with_asv3_exp_cheap_cache = MagicMock(return_value=bound)

    class CacheBindable:
        pass

    selected.__class__ = CacheBindable
    monkeypatch.setattr(multi_llm, "LitellmLLM", CacheBindable)

    runtime.run_asv3_loop(**kwargs)

    selected.with_asv3_exp_cheap_cache.assert_called_once_with()
    selected.invoke.assert_not_called()
    assert bound.invoke.call_count == 2
    assert bound.config == selected.config
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_EXP_CHEAP_VARIANT
    assert checkpoints[-1]["prompt_version"] == TUNED_PROMPT_VERSION


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
    kwargs, _broker, _selected, checkpoints, _queue = setup_run(monkeypatch)
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
    assert checkpoints[-1]["candidate_audit"]["version"] == 1
    assert checkpoints[-1]["candidate_audit"]["records"] == []
    review = context.services["guardrails_v3_review"]
    assert review["review_completed"] is False
    assert review["failure_reason"] in {
        "jev_credential_unavailable",
        "outcome_map_missing",
        "review_packet_too_large",
    }
    assert checkpoints[-1]["guardrails_v3_review"] == review


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


@pytest.mark.parametrize(
    "variant", [ASV3_TUNED_VARIANT, ASV3_EXP_CHEAP_VARIANT, ASV3_EXP_CROPPED_VARIANT]
)
def test_actual_runtime_resume_fences_before_saved_mode_override(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
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
    assert checkpoints[-1]["asv3_workflow_variant"] == variant
    assert checkpoints[-1]["prompt_version"] == TUNED_PROMPT_VERSION
    invalid_policy = copy.deepcopy(tuned)
    invalid_policy["asv3_workflow_policy"] = "different-policy"
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: invalid_policy)
    selected.reset_mock()
    with pytest.raises(ValueError, match="checkpoint policy"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: tuned)
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
