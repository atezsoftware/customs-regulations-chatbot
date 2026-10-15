"""Secondary search admits only the selected transport's proven cache billing."""

from unittest.mock import patch

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_cost_budget import (
    CroppedCostBudget,
    CroppedResearchBudgetReached,
)
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.configs.constants import MessageType
from onyx.llm.model_response import (
    Choice,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoice,
    Usage,
)
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.secondary_llm_flows.query_expansion import keyword_query_expansion
from onyx.tools.models import ChatMinimalTextMessage


@pytest.fixture(autouse=True)
def native_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_BASE", "VERTEXAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("onyx.llm.litellm_singleton.litellm.api_base", None)
    monkeypatch.setattr(
        "onyx.asv3.cropped_cost_budget._rates",
        lambda *_args: (2e-6, 1e-5, 1e-7, None),
    )


def scoped(selected: LitellmLLM) -> tuple[ScopedSearchLLM, CroppedCostBudget]:
    context = RunContext(
        services={"asv3_exp_cropped": True, "cropped_token_counter": len}
    )
    budget = CroppedCostBudget(context, "The owned search request")
    context.services["cropped_cost_budget"] = budget
    return ScopedSearchLLM(selected, context, None), budget


def argon(
    provider: str = "vertex_ai",
    *,
    api_base: str | None = None,
    model_kwargs: dict[str, str] | None = None,
) -> LitellmLLM:
    return LitellmLLM(
        api_key="test",
        model_provider=provider,
        model_name="gemini-4-argon",
        max_input_tokens=1_047_552,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )


def recorded_calls(budget: CroppedCostBudget) -> list[JsonValue]:
    state = budget.export()["state"]
    assert isinstance(state, dict)
    calls = state["calls"]
    assert isinstance(calls, list)
    return calls


def usage() -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


@pytest.mark.parametrize("provider", ["vertex_ai", "gemini"])
def test_real_keyword_expansion_reaches_selected_provider_and_settles_budget(
    provider: str,
) -> None:
    selected = argon(provider)
    llm, budget = scoped(selected)
    response = ModelResponse(
        id="offline-provider",
        created="0",
        choice=Choice(message=Message(content="transit fire liability\ncustoms duty")),
        usage=usage(),
    )
    with patch.object(selected, "invoke", return_value=response) as provider_call:
        queries = keyword_query_expansion(
            [
                ChatMinimalTextMessage(
                    message="Transit fire?", message_type=MessageType.USER
                )
            ],
            llm,
        )
    assert queries == ["transit fire liability", "customs duty"]
    provider_call.assert_called_once()
    assert provider_call.call_args.kwargs["max_tokens"] == 256
    calls = recorded_calls(budget)
    assert len(calls) == 1 and isinstance(calls[0], dict)
    assert calls[0]["cache_writes_disabled"] is True
    assert calls[0]["known_cost_usd"] == pytest.approx(0.0003)
    assert budget.finish_reason is None


def test_stream_uses_same_actual_argon_contract_for_a_single_message() -> None:
    selected = argon()
    llm, budget = scoped(selected)
    part = ModelResponseStream(
        id="offline-provider", created="0", choice=StreamingChoice(), usage=usage()
    )
    with patch.object(selected, "stream", return_value=iter([part])) as provider_call:
        assert list(llm.stream(UserMessage(content="Query"), max_tokens=256)) == [part]
    provider_call.assert_called_once()
    call = recorded_calls(budget)[0]
    assert isinstance(call, dict) and call["cache_writes_disabled"] is True
    assert call["known_cost_usd"] == pytest.approx(0.0003)


@pytest.mark.parametrize(
    "variant", ["proxy", "routing_kwarg", "explicit_cache", "unknown_model"]
)
def test_unproven_route_or_cache_contract_stops_before_provider(variant: str) -> None:
    selected = argon(
        api_base="https://proxy.example" if variant == "proxy" else None,
        model_kwargs={"api_base": "https://proxy.example"}
        if variant == "routing_kwarg"
        else {"cached_content": "explicit-cache"}
        if variant == "explicit_cache"
        else None,
    )
    if variant == "unknown_model":
        selected = selected.with_model("gemini-3.8-flash", max_input_tokens=1_047_552)
    llm, budget = scoped(selected)
    with patch.object(selected, "invoke") as provider_call:
        with pytest.raises(CroppedResearchBudgetReached, match="unknown prior charges"):
            llm.invoke([UserMessage(content="Query")], max_tokens=256)
    provider_call.assert_not_called()
    assert recorded_calls(budget) == []


def test_existing_openai_native_write_contract_still_controls_reservation() -> None:
    selected = (
        LitellmLLM(
            api_key="test",
            model_provider="openai",
            model_name="gpt-6.1-sol",
            max_input_tokens=1_047_552,
        )
        .with_tool_argument_stream_guard()
        .with_cropped_response_terminal_capture(explicit_cache=True)
    )
    assert selected.cropped_native_cache_writes_disabled
    llm, _budget = scoped(selected)
    _, ticket = llm._cropped_reservation(
        [UserMessage(content="Query")], None, None, None, 256
    )
    assert ticket is not None and ticket.cache_writes_disabled


def test_native_implicit_cache_proof_cannot_admit_unknown_provider_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = argon()
    llm, budget = scoped(selected)

    def unknown_prices(*_args: object) -> tuple[float, float, float, None]:
        raise RunStopped("Selected registry pricing unavailable")

    monkeypatch.setattr("onyx.asv3.cropped_cost_budget._rates", unknown_prices)
    with patch.object(selected, "invoke") as provider_call:
        with pytest.raises(
            CroppedResearchBudgetReached, match="pricing is unavailable"
        ):
            llm.invoke([UserMessage(content="Query")], max_tokens=256)
    provider_call.assert_not_called()
    assert recorded_calls(budget) == []
