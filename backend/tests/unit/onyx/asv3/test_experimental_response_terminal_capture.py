"""Only the selected Experimental native route preserves provider terminal events."""

from typing import cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm import HTTPHandler
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.native_responses_stream import NativeResponseTerminalError
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.llm.test_asv3_response_terminal_capture import TOOLS, events
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream


@pytest.mark.parametrize(
    "tuned,provider,model,base,tools,expected",
    [
        (True, "openai", "gpt-6.1-sol", None, True, True),
        (True, "openai", "gpt-6.1-sol", "https://api.openai.com/v1", True, True),
        (False, "openai", "gpt-6.1-sol", None, True, False),
        (True, "openai", "gpt-6.1-sol", "https://proxy.example/v1", True, False),
        (True, "openai", "gpt-5-mini", None, True, False),
        (True, "anthropic", "claude-haiku-4-5", None, True, False),
        (True, "openai", "gpt-6.1-sol", None, False, False),
    ],
)
def test_capture_scope_preserves_native128k_kwargs_and_selected_instance(
    tuned: bool,
    provider: str,
    model: str,
    base: str | None,
    tools: bool,
    expected: bool,
) -> None:
    selected = LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name=model,
        api_base=base,
        max_input_tokens=1000000,
        temperature=1,
    )
    context = RunContext(
        services={"asv3_workflow_variant": ASV3_TUNED_VARIANT if tuned else "asv3"}
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    seen: list[LitellmLLM] = []
    content = "Complete source conditions and exceptions [1]. " * 1500
    response = ModelResponse(
        id="provider-completed",
        created="0",
        choice=Choice(finish_reason="stop", message=Message(content=content)),
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=128000,
            total_tokens=128100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    def respond(bound: LitellmLLM, **kwargs: object) -> ModelResponse:
        seen.append(bound)
        assert kwargs["max_tokens"] == 128000
        assert kwargs["tools"] == (TOOLS if tools else None)
        assert bound.config.model_name == selected.config.model_name
        assert bound.config.temperature == selected.config.temperature
        return response

    with patch.object(LitellmLLM, "invoke", autospec=True, side_effect=respond):
        result = adapter._native_decision_response(
            [UserMessage(content="Exact unchanged originals.")],
            cast(list[dict[str, JsonValue]], TOOLS) if tools else [],
            LLMFlow.ASV3_COORDINATOR,
            128000,
            call_llm=selected,
            research=True,
        )
    assert len(seen) == 1 and seen[0]._asv3_response_terminal_capture is expected
    assert selected._asv3_response_terminal_capture is False
    assert selected._tool_argument_stream_guard is False
    assert result is response and result.choice.message.content == content
    assert not adapter.last_response_truncated


def test_failed_actual_native_response_retains_usage_without_recording_action_delivery() -> (
    None
):
    context = RunContext(
        services={
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "evidence": EvidenceLedger(),
        }
    )
    selected = LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=1000000,
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    stream = EventStream(
        events(
            "failed",
            {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112},
            tool=True,
        )
    )
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        stream=stream,
        headers={"content-type": "text/event-stream"},
    )
    with (
        patch("litellm.HTTPHandler", return_value=client),
        patch("onyx.llm.model_capabilities.get_model_map", return_value={}),
        patch("litellm.utils.supports_native_streaming", return_value=False),
        patch("litellm.completion") as ordinary,
    ):
        with pytest.raises(NativeResponseTerminalError) as failed:
            adapter._native_decision_response(
                [UserMessage(content="Preserve source evidence.")],
                cast(list[dict[str, JsonValue]], TOOLS),
                LLMFlow.ASV3_COORDINATOR,
                128000,
                call_llm=selected,
                research=True,
            )
        ordinary.assert_not_called()
    assert failed.value.usage is not None and failed.value.usage.completion_tokens == 12
    assert failed.value.provider_error == {
        "code": "server_error",
        "message": "Generation failed",
    }
    assert adapter.last_call_id is None
    assert context.services["evidence"].export()["deliveries"] == []
    assert stream.closed
