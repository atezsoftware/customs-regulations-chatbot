"""Argon admission uses its documented context and native thinking contract."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.utils import get_optional_params

from onyx.asv3 import runtime
from onyx.asv3.cropped_cost_budget import (
    CroppedCostBudget,
    CroppedResearchBudgetReached,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.error_handling.exceptions import OnyxError
from onyx.llm import model_capabilities
from onyx.llm.factory import get_llm_token_counter, llm_from_provider
from onyx.llm.models import ReasoningEffort, TextContentPart, UserMessage
from onyx.llm.multi_llm import LitellmLLM
from tests.unit.onyx.asv3.test_runtime import setup_run
from tests.unit.onyx.llm.test_factory import _build_provider_view


@pytest.fixture(autouse=True)
def clear_model_map() -> Iterator[None]:
    model_capabilities.get_model_map.cache_clear()
    yield
    model_capabilities.get_model_map.cache_clear()


@pytest.mark.parametrize("provider", ["vertex_ai", "gemini"])
def test_missing_registry_argon_uses_documented_context(provider: str) -> None:
    with patch.object(litellm, "model_cost", {}):
        model_map = model_capabilities.get_model_map()
    assert (
        model_capabilities.llm_max_input_tokens(model_map, "gemini-4-argon", provider)
        == 1_048_576
    )
    assert (
        model_capabilities.get_llm_max_output_tokens(
            model_map, "gemini-4-argon", provider
        )
        == 32_768
    )
    metadata = model_capabilities.find_model_obj(model_map, provider, "gemini-4-argon")
    assert metadata is not None
    assert metadata["supports_reasoning"] and metadata["supports_function_calling"]
    assert "input_cost_per_token" not in metadata


def test_registry_limits_and_prices_are_preserved_and_long_decode_is_not_requested() -> (
    None
):
    registry = {
        "vertex_ai/gemini-4-argon": {
            "max_input_tokens": 100_000,
            "max_output_tokens": 1_048_576,
            "input_cost_per_token": 0.125,
            "output_cost_per_token": 0.25,
        }
    }
    with patch.object(litellm, "model_cost", registry):
        model_map = model_capabilities.get_model_map()
    metadata = model_map["vertex_ai/gemini-4-argon"]
    assert metadata["max_input_tokens"] == 100_000
    assert metadata["input_cost_per_token"] == 0.125
    assert metadata["output_cost_per_token"] == 0.25
    assert (
        model_capabilities.get_llm_max_output_tokens(
            model_map, "gemini-4-argon", "vertex_ai"
        )
        == 32_768
    )
    assert (
        model_capabilities.get_llm_max_output_tokens(
            {"gemini-4-argon": {"max_output_tokens": 4096}},
            "gemini-4-argon",
            "vertex_ai",
        )
        == 4096
    )
    assert (
        model_capabilities.get_llm_max_output_tokens(
            model_map, "gemini-4-argon", "openrouter"
        )
        == 1_048_576
    )


def test_explicit_admin_context_limit_still_wins() -> None:
    provider = _build_provider_view("vertex_ai", 4096)
    provider.model_configurations[0].name = "gemini-4-argon"
    with patch("onyx.llm.factory.get_llm", return_value=MagicMock()) as factory:
        llm_from_provider("gemini-4-argon", provider)
    assert factory.call_args.kwargs["max_input_tokens"] == 4096


@pytest.mark.parametrize(
    "effort,level",
    [
        (ReasoningEffort.OFF, "LOW"),
        (ReasoningEffort.LOW, "LOW"),
        (ReasoningEffort.AUTO, "MEDIUM"),
        (ReasoningEffort.MEDIUM, "MEDIUM"),
        (ReasoningEffort.HIGH, "HIGH"),
        (ReasoningEffort.XHIGH, "HIGH"),
    ],
)
def test_argon_native_thinking_level_reaches_litellm_without_legacy_budget(
    effort: ReasoningEffort,
    level: str,
) -> None:
    llm = LitellmLLM(
        api_key="test",
        model_provider="vertex_ai",
        model_name="gemini-4-argon",
        max_input_tokens=1_047_552,
    )
    with patch(
        "onyx.llm.litellm_singleton.litellm.completion", return_value=object()
    ) as completion:
        llm._completion(
            prompt=[UserMessage(content="Question")],
            tools=None,
            tool_choice=None,
            stream=False,
            parallel_tool_calls=False,
            reasoning_effort=effort,
        )
    kwargs = completion.call_args.kwargs
    assert "reasoning_effort" not in kwargs and "thinking" not in kwargs
    config = {
        "thinkingLevel": level,
        "includeThoughts": effort is not ReasoningEffort.OFF,
    }
    assert kwargs["thinkingConfig"] == config
    mapped = get_optional_params(
        model="gemini-4-argon", custom_llm_provider="vertex_ai", thinkingConfig=config
    )
    assert mapped["thinkingConfig"] == config


def _argon(**kwargs: Any) -> LitellmLLM:
    return LitellmLLM(
        api_key="test",
        model_provider="vertex_ai",
        model_name="gemini-4-argon",
        max_input_tokens=1_047_552,
        **kwargs,
    )


def test_implicit_cache_contract_admits_known_cost_without_claiming_openai_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _argon()
    proof = llm.cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], []
    )
    assert proof is True and llm.cropped_native_cache_writes_disabled is False
    monkeypatch.setattr(
        "onyx.asv3.cropped_cost_budget._rates", lambda *_args: (2e-6, 1e-5, 1e-7, None)
    )
    budget = CroppedCostBudget(RunContext(), "Question")
    ticket = budget.prepare(
        "gemini-4-argon",
        "vertex_ai",
        30_000,
        4096,
        research=True,
        owner="root",
        cache_writes_disabled=proof,
    )
    assert ticket.bounded and ticket.cache_writes_disabled
    unknown = CroppedCostBudget(RunContext(), "Question")
    with pytest.raises(CroppedResearchBudgetReached, match="unknown prior charges"):
        unknown.prepare(
            "unknown",
            "vertex_ai",
            30_000,
            4096,
            research=True,
            owner="root",
            cache_writes_disabled=False,
        )


@pytest.mark.parametrize(
    "variant", ["api_base", "model", "provider", "model_kwargs", "prompt", "tools"]
)
def test_implicit_contract_does_not_cover_proxies_other_models_or_explicit_cache(
    variant: str,
) -> None:
    llm = _argon(
        api_base="https://proxy.example" if variant == "api_base" else None,
        model_kwargs={"cached_content": "cache-id"}
        if variant == "model_kwargs"
        else None,
    )
    if variant == "model":
        llm = llm.with_model("gemini-3.8-flash", max_input_tokens=1_047_552)
    elif variant == "provider":
        llm = LitellmLLM(
            api_key="test",
            model_provider="openrouter",
            model_name="gemini-4-argon",
            max_input_tokens=1_047_552,
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
    assert llm.cropped_has_free_implicit_cache_writes(prompt, tools) is False


@pytest.mark.parametrize("limit", [30_976, 1_047_552])
def test_initial_native_catalog_requires_argon_context_without_dropping_tools(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    class ProviderBoundaryReached(Exception):
        pass

    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    llm.config = llm.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-4-argon",
            "max_input_tokens": limit,
        }
    )
    kwargs["token_counter"] = get_llm_token_counter(llm)
    prepared: list[tuple[int, int, int]] = []

    def capture(
        self: ResearchModel,
        _view: Any,
        prompt: Any,
        tools: Any,
        output: int,
        **_unused: Any,
    ) -> Any:
        prepared.append((self._input_cost(prompt, tools), output, len(tools)))
        raise ProviderBoundaryReached()

    monkeypatch.setattr(ResearchModel, "_invoke_decision", capture)
    with pytest.raises(OnyxError if limit == 30_976 else ProviderBoundaryReached):
        runtime.run_asv3_loop(
            **kwargs, research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT
        )
    llm.invoke.assert_not_called()
    if limit == 30_976:
        assert prepared == []
        assert (
            checkpoints[-1]["stop_reason"]
            == "The complete question, required originals and native tools exceed the selected model context"
        )
    else:
        assert len(prepared) == 1
        incoming, outgoing, tools = prepared[0]
        assert 27_000 < incoming < 28_000
        assert (outgoing, tools) == (32_768, 37)


@pytest.mark.parametrize(
    "key", ["api_base", "base_url", "custom_llm_provider", "VERTEXAI_API_BASE"]
)
@pytest.mark.parametrize("scope", ["direct", "nested", "custom_config"])
def test_implicit_contract_rejects_passthrough_routing_overrides(
    key: str, scope: str
) -> None:
    kwargs: dict[str, Any] = {}
    if scope == "custom_config":
        kwargs["custom_config"] = {key: "https://proxy.example"}
    else:
        route = {key: "https://proxy.example"}
        kwargs["model_kwargs"] = {"extra_body": route} if scope == "nested" else route
    assert not _argon(**kwargs).cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], []
    )


@pytest.mark.parametrize(
    "provider,variable",
    [("gemini", "GEMINI_API_BASE"), ("vertex_ai", "VERTEXAI_API_BASE")],
)
def test_implicit_contract_rejects_environment_routing(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    variable: str,
) -> None:
    llm = LitellmLLM(
        api_key="test",
        model_provider=provider,
        model_name="gemini-4-argon",
        max_input_tokens=1_047_552,
    )
    monkeypatch.setenv(variable, "https://proxy.example")
    assert not llm.cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], []
    )


def test_implicit_contract_rejects_litellm_global_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "api_base", "https://proxy.example")
    assert not _argon().cropped_has_free_implicit_cache_writes(
        [UserMessage(content="Question")], []
    )
