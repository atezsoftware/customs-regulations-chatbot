"""Only explicit factorial Cropped profiles admit native Flash implicit writes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import llm_adapter
from onyx.asv3.cropped_cost_budget import (
    CroppedCostBudget,
    CroppedResearchBudgetReached,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.model_composition.models import (
    COMPOSITION_ROLES,
    MODEL_COMPOSITION_SERVICE,
    CompositionProfileId,
    ModelBinding,
    ModelComposition,
)
from onyx.asv3.models import RunContext
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.llm.litellm_singleton import litellm
from onyx.llm.models import ChatCompletionMessage, TextContentPart, UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span


def selected(
    name: str = "gemini-3.8-flash",
    *,
    provider: str = "vertex_ai",
    api_base: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    custom_config: dict[str, Any] | None = None,
) -> LitellmLLM:
    return LitellmLLM(
        api_key="test",
        model_provider=provider,
        model_name=name,
        max_input_tokens=1_047_552,
        api_base=api_base,
        model_kwargs=model_kwargs,
        custom_config=custom_config,
    )


@pytest.mark.parametrize("name", ["gemini-3.8-flash", "gemini-3.5-flash-lite"])
def test_native_flash_requires_explicit_factorial_permission(name: str) -> None:
    llm = selected(name)
    prompt = [UserMessage(content="Question")]
    assert not llm.cropped_has_free_implicit_cache_writes(prompt, [])
    assert llm.cropped_has_free_implicit_cache_writes(
        prompt, [], allow_fast_composition_flash=True
    )
    assert not llm.cropped_native_cache_writes_disabled


@pytest.mark.parametrize(
    "provider,name",
    [
        ("vertex_ai", "gemini-3.6-flash"),
        ("openrouter", "gemini-3.8-flash"),
        ("openai", "gemini-3.8-flash"),
    ],
)
def test_permission_does_not_admit_unknown_models_or_provider_routes(
    provider: str, name: str
) -> None:
    assert not selected(name, provider=provider).cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], [], allow_fast_composition_flash=True
    )


@pytest.mark.parametrize(
    "variant",
    ["api_base", "model_kwargs", "nested_base", "custom_config", "prompt", "tools"],
)
def test_factorial_permission_preserves_explicit_cache_and_routing_rejections(
    variant: str,
) -> None:
    llm = selected(
        api_base="https://proxy.example" if variant == "api_base" else None,
        model_kwargs={"cached_content": "cache-id"}
        if variant == "model_kwargs"
        else {"extra_body": {"base_url": "https://proxy.example"}}
        if variant == "nested_base"
        else None,
        custom_config={"cache_control": {"type": "ephemeral"}}
        if variant == "custom_config"
        else None,
    )
    prompt = (
        [
            UserMessage(
                content=[
                    TextContentPart(
                        text="Question", cache_control={"type": "ephemeral"}
                    )
                ]
            )
        ]
        if variant == "prompt"
        else [UserMessage(content="Question")]
    )
    tools = [{"cache_control": {"type": "ephemeral"}}] if variant == "tools" else []
    assert not llm.cropped_has_free_implicit_cache_writes(
        prompt, tools, allow_fast_composition_flash=True
    )


@pytest.mark.parametrize("route", ["global", "environment"])
def test_factorial_permission_preserves_ambient_routing_rejection(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    if route == "global":
        monkeypatch.setattr(litellm, "api_base", "https://proxy.example")
    else:
        monkeypatch.setenv("VERTEXAI_API_BASE", "https://proxy.example")
    assert not selected().cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], [], allow_fast_composition_flash=True
    )


def test_existing_argon_proof_does_not_require_factorial_permission() -> None:
    assert selected("gemini-4-argon").cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], []
    )


@pytest.fixture
def priced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "onyx.asv3.cropped_cost_budget._rates",
        lambda *_args: (0.75e-6, 3.75e-6, 0.075e-6, None),
    )


@pytest.mark.parametrize("name", ["gemini-3.8-flash", "gemini-3.5-flash-lite"])
@pytest.mark.usefixtures("priced")
def test_proof_admits_a_priced_bounded_ticket_without_assuming_zero_token_prices(
    name: str,
) -> None:
    llm = selected(name)
    budget = CroppedCostBudget(RunContext(), "Question")
    with pytest.raises(CroppedResearchBudgetReached, match="unknown prior charges"):
        budget.prepare(name, "vertex_ai", 30_000, 4096, research=True, owner="test")
    ticket = budget.prepare(
        name,
        "vertex_ai",
        30_000,
        4096,
        research=True,
        owner="test",
        cache_writes_disabled=llm.cropped_has_free_implicit_cache_writes(
            [UserMessage(content="Question")], [], allow_fast_composition_flash=True
        ),
    )
    assert ticket.bounded and ticket.reservation_upper_verified
    assert ticket.input_rate > 0 and ticket.output_rate > 0 and ticket.reserved_usd > 0
    assert ticket.cache_write_rate is None


def composition_context(
    llm: LitellmLLM, profile: CompositionProfileId, fast: bool
) -> RunContext:
    binding = ModelBinding(llm, len, 1)
    context = RunContext(
        services={
            "asv3_exp_cropped": True,
            "asv3_fast": fast,
            MODEL_COMPOSITION_SERVICE: ModelComposition(
                profile, 1, dict.fromkeys(COMPOSITION_ROLES, binding)
            ),
            "cropped_token_counter": len,
        }
    )
    context.services["cropped_cost_budget"] = CroppedCostBudget(context, "Question")
    return context


@pytest.mark.parametrize(
    "profile,fast,allowed",
    [
        ("factorial_v1_fss", True, True),
        ("factorial_v1_fss", False, False),
        ("sol_reference_v1", True, False),
        ("flash_sol_v1", True, False),
        ("terra_sol_v1", True, False),
        ("argon_sonnet_v1", True, False),
        ("luna_sol_v1", True, False),
    ],
)
@pytest.mark.parametrize("path", ["native", "search"])
@pytest.mark.usefixtures("priced")
def test_adapter_call_sites_limit_permission_to_factorial_fast_cropped(
    monkeypatch: pytest.MonkeyPatch,
    profile: CompositionProfileId,
    fast: bool,
    allowed: bool,
    path: str,
) -> None:
    class ProviderBoundaryReached(Exception):
        pass

    llm = selected()
    context = composition_context(llm, profile, fast)
    prompt: list[ChatCompletionMessage] = [UserMessage(content="Question")]
    if path == "search":
        scoped = ScopedSearchLLM(llm, context, None)
        if allowed:
            _budget, ticket = scoped._cropped_reservation(prompt, [], None, None, 1024)
            assert ticket is not None and ticket.reservation_upper_verified
        else:
            with pytest.raises(CroppedResearchBudgetReached):
                scoped._cropped_reservation(prompt, [], None, None, 1024)
        return

    span = MagicMock()
    span.span_data = GenerationSpanData(model="gemini-3.8-flash")

    @contextmanager
    def resources(
        *_args: object, **_kwargs: object
    ) -> Iterator[Span[GenerationSpanData]]:
        yield cast(Span[GenerationSpanData], span)

    monkeypatch.setattr(llm_adapter, "_generation_resources", resources)
    invoked = MagicMock(side_effect=ProviderBoundaryReached())
    monkeypatch.setattr(llm, "invoke", invoked)
    model = ResearchModel(llm, context, token_counter=len, lean_native_mode=True)
    with pytest.raises(
        ProviderBoundaryReached if allowed else CroppedResearchBudgetReached
    ):
        model._invoke_once(
            prompt,
            [],
            LLMFlow.ASV3_COORDINATOR,
            max_tokens=1024,
            research=True,
            structured=False,
        )
    assert invoked.call_count == int(allowed)


@pytest.mark.parametrize("name", ["gemini-3.5-flash-lite", "gemini-3.8-flash"])
def test_native_session_pdf_cache_permission_is_limited_to_flash_lite(
    name: str,
) -> None:
    from onyx.llm.models import FileContentPart, FileDetail

    llm = selected(name)
    prompt = [
        UserMessage(
            content=[
                FileContentPart(
                    file=FileDetail(
                        filename="scan.pdf",
                        file_data="data:application/pdf;base64,JVBERg==",
                    )
                )
            ]
        )
    ]
    assert not llm.cropped_has_free_implicit_cache_writes(prompt, [])
    assert llm.cropped_has_free_implicit_cache_writes(
        prompt, [], allow_session_attachment_flash_lite=True
    ) is (name == "gemini-3.5-flash-lite")
    assert not llm.cropped_has_free_implicit_cache_writes(
        [UserMessage(content="question")], [], allow_session_attachment_flash_lite=True
    )
    routed = selected(name, api_base="https://proxy.example")
    assert not routed.cropped_has_free_implicit_cache_writes(
        prompt, [], allow_session_attachment_flash_lite=True
    )
