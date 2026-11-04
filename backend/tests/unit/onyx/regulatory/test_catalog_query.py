"""Optional query preparation stays bounded, source-neutral, and inexpensive."""

import json
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM, LLMRateLimitError
from onyx.regulatory import catalog_query as query
from onyx.server.manage.llm.models import LLMProviderView

DESCRIPTION = "Şöyle bir şey vardı, ithalat yaparken verdiğimiz teminatın geri alınmasına ilişkin düzenlemeyi bulmak istiyorum."
PREPARED = "İthalat işlemlerinde verilen teminatın iadesi koşulları"


@pytest.mark.parametrize(
    "query_text",
    [
        "gümrük vergisi",
        "Bana Gümrük Kanununu bul",
        "4458 sayılı kanunu getir",
        'Bana "gümrük kıymeti" ile ilgili bölüm bul',
        "Şöyle bir şey vardı 4458 sayılı Kanun kapsamında bir istisnaydı",
    ],
)
def test_explicit_lookup_and_short_keywords_make_no_model_or_database_call(
    query_text: str,
) -> None:
    with patch.object(query, "get_catalog_query_llm") as get_llm:
        result = query.prepare_catalog_search_query(query_text, MagicMock())
    assert result.query == query_text
    assert result.rewritten is False
    get_llm.assert_not_called()


def _response(text: str, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        id="mock",
        created="now",
        choice=Choice(
            finish_reason=finish_reason,
            message=Message(content=text),
        ),
    )


def test_description_uses_one_small_traced_call_with_complete_original_request() -> (
    None
):
    llm = MagicMock()
    llm.config.max_input_tokens = 100_000
    llm.invoke.return_value = _response(PREPARED)
    with (
        patch.object(query, "get_catalog_query_llm", return_value=llm),
        patch.object(query, "llm_generation_span") as span,
        patch.object(query, "record_llm_response"),
    ):
        result = query.prepare_catalog_search_query(DESCRIPTION, MagicMock())
    assert result.query == PREPARED
    assert result.rewritten is True
    llm.invoke.assert_called_once()
    sent = llm.invoke.call_args
    assert json.loads(str(sent.args[0][1].content)) == {"user_request": DESCRIPTION}
    assert sent.kwargs == {
        "max_tokens": 160,
        "timeout_override": 15,
        "reasoning_effort": ReasoningEffort.OFF,
        "use_streaming": False,
        "provider_compatibility_attempts": 1,
    }
    assert span.call_args.kwargs["flow"] is query.LLMFlow.CATALOG_QUERY_PREPARATION


@pytest.mark.parametrize(
    ("original", "output", "finish"),
    [
        (DESCRIPTION, "İthalat iade 4458 sayılı Kanun", "stop"),
        (DESCRIPTION, "İthalat\nTeminat", "stop"),
        (DESCRIPTION, PREPARED, "length"),
        (DESCRIPTION + " AB dışında olsun", "İthalat teminat iadesi", "stop"),
        (DESCRIPTION + " 3 gün bekledik", "İthalat teminat iadesi", "stop"),
    ],
)
def test_invalid_or_incomplete_preparation_falls_back_without_another_call(
    original: str,
    output: str,
    finish: str,
) -> None:
    llm = MagicMock()
    llm.config.max_input_tokens = 100_000
    llm.invoke.return_value = _response(output, finish)
    with (
        patch.object(query, "get_catalog_query_llm", return_value=llm),
        patch.object(query, "llm_generation_span"),
        patch.object(query, "record_llm_response"),
    ):
        result = query.prepare_catalog_search_query(original, MagicMock())
    assert result.query == original
    assert result.rewritten is False
    llm.invoke.assert_called_once()


def test_provider_failure_preserves_original_and_never_retries_or_changes_model() -> (
    None
):
    llm = MagicMock()
    llm.config.max_input_tokens = 100_000
    llm.invoke.side_effect = RuntimeError("private text must not be logged")
    with (
        patch.object(query, "get_catalog_query_llm", return_value=llm) as get_llm,
        patch.object(query, "llm_generation_span"),
        patch.object(query.logger, "warning") as warning,
    ):
        result = query.prepare_catalog_search_query(DESCRIPTION, MagicMock())
    assert result.query == DESCRIPTION
    get_llm.assert_called_once()
    llm.invoke.assert_called_once()
    assert "private text" not in str(warning.call_args)


def test_unconfigured_or_small_context_skips_inference() -> None:
    for llm in (None, MagicMock()):
        if llm is not None:
            llm.config.max_input_tokens = 100
        with patch.object(query, "get_catalog_query_llm", return_value=llm):
            result = query.prepare_catalog_search_query(DESCRIPTION, MagicMock())
        assert result.query == DESCRIPTION
        if llm is not None:
            llm.invoke.assert_not_called()


def _provider(model_names: list[str]) -> MagicMock:
    provider = MagicMock(id=3, provider="openrouter")
    provider.model_configurations = []
    for name in model_names:
        model = MagicMock(is_visible=True, max_input_tokens=1_048_576)
        model.name = name
        provider.model_configurations.append(model)
    return provider


@pytest.mark.parametrize("visible", [True, False])
def test_only_allowlisted_visible_configured_models_are_selected(visible: bool) -> None:
    provider = _provider(
        ["nvidia/nemotron-3-ultra-550b-a55b", "deepseek/deepseek-v4.1-flash:floor"]
    )
    provider.model_configurations[1].is_visible = visible
    configured = LLMProviderView(
        id=3, provider="openrouter", api_key="test-key", model_configurations=[]
    )
    with (
        patch.object(query, "fetch_existing_llm_providers", return_value=[provider]),
        patch.object(query, "can_user_access_llm_provider", return_value=True),
        patch.object(query.LLMProviderView, "from_model", return_value=configured),
    ):
        llm = query.get_catalog_query_llm(MagicMock())
    if visible:
        assert llm is not None
        assert llm.config.model_name == "deepseek/deepseek-v4.1-flash:floor"
        assert llm.config.max_input_tokens == 1_048_576
    else:
        assert llm is None


@pytest.mark.parametrize("status", [200, 429])
def test_preferred_model_http_wire_disables_thinking_and_uses_single_attempt(
    status: int,
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if status == 429:
            return httpx.Response(
                status, json={"error": {"message": "rate limited", "code": 429}}
            )
        return httpx.Response(
            status,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek/deepseek-v4.1-flash:floor",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": PREPARED},
                    }
                ],
            },
        )

    provider = _provider(["openai/gpt-5.6-luna", "deepseek/deepseek-v4.1-flash:floor"])
    configured = LLMProviderView(
        id=3,
        provider="openrouter",
        api_key="test-key",
        model_configurations=[],
        api_base="https://openrouter.ai/api/v1",
    )
    with (
        patch.object(query, "fetch_existing_llm_providers", return_value=[provider]),
        patch.object(query, "can_user_access_llm_provider", return_value=True),
        patch.object(query.LLMProviderView, "from_model", return_value=configured),
        httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
    ):
        llm = query.get_catalog_query_llm(MagicMock())
        assert llm is not None
        assert isinstance(llm, LitellmLLM)
        with pytest.raises(LLMRateLimitError) if status == 429 else nullcontext():
            llm._completion(
                prompt=[UserMessage(content=DESCRIPTION)],
                tools=None,
                tool_choice=None,
                stream=False,
                parallel_tool_calls=False,
                reasoning_effort=ReasoningEffort.OFF,
                max_tokens=160,
                provider_compatibility_attempts=1,
                client=HTTPHandler(client=http_client),
            )
    assert len(requests) == 1
    assert requests[0]["model"] == "deepseek/deepseek-v4.1-flash:floor"
    assert requests[0]["reasoning"] == {"enabled": False}
    assert requests[0]["max_tokens"] == 160
    assert "num_retries" not in requests[0]
    assert "max_retries" not in requests[0]


def test_vague_description_with_year_is_prepared_once_and_retains_year() -> None:
    original = "2023 yılında şöyle bir düzenleme vardı, ithalat teminatının iadesine ilişkin bir şey hatırlıyorum."
    llm = MagicMock()
    llm.config.max_input_tokens = 100_000
    llm.invoke.return_value = _response("2023 yılında ithalat teminatının iadesi")
    with (
        patch.object(query, "get_catalog_query_llm", return_value=llm),
        patch.object(query, "llm_generation_span"),
        patch.object(query, "record_llm_response"),
    ):
        result = query.prepare_catalog_search_query(original, MagicMock())
    assert result.rewritten is True
    assert "2023" in result.query
    llm.invoke.assert_called_once()
