from enum import Enum
from decimal import Decimal

import httpx
import json
import pytest
from pydantic import BaseModel, Field

from fs_explorer_api.llm.base import ChatTurn
from fs_explorer_api.llm.openrouter import (
    OpenRouterLLMClient,
    _provider_compatible_json_schema,
    usage_from_openrouter,
)
from fs_explorer_api.orchestration_models import (
    GlobalPlan,
    SearchAssignmentBatch,
    TaskArtifact,
    WorkerArtifact,
)


class Reply(BaseModel):
    answer: str


class ReplyKind(str, Enum):
    SUCCESS = "success"


class ReferencedReply(BaseModel):
    kind: ReplyKind = Field(description="Result category")


def _ref_nodes(value: object) -> list[dict]:
    if isinstance(value, list):
        return [node for item in value for node in _ref_nodes(item)]
    if not isinstance(value, dict):
        return []
    nodes = [value] if "$ref" in value else []
    return nodes + [node for item in value.values() for node in _ref_nodes(item)]


@pytest.mark.parametrize(
    "schema",
    [GlobalPlan, SearchAssignmentBatch, WorkerArtifact, TaskArtifact],
)
def test_orchestration_schemas_have_no_ref_siblings_after_normalization(
    schema: type[BaseModel],
) -> None:
    normalized = _provider_compatible_json_schema(schema.model_json_schema())

    refs = _ref_nodes(normalized)
    assert refs
    assert all(set(node) == {"$ref"} for node in refs)


def test_schema_normalization_never_discards_validation_keywords() -> None:
    with pytest.raises(ValueError, match=r"beside \$ref.*minLength"):
        _provider_compatible_json_schema(
            {"$ref": "#/$defs/Name", "description": "Name", "minLength": 1}
        )


def test_openrouter_completion_total_is_not_double_counted() -> None:
    usage = usage_from_openrouter(
        {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "cost": "0.00125",
        }
    )

    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (
        100,
        18,
        12,
    )
    assert usage.input_tokens + usage.output_tokens + usage.thinking_tokens == 130
    assert usage.billed_cost_usd == Decimal("0.00125")
    assert usage.cost_source == "provider"


@pytest.mark.asyncio
async def test_structured_request_uses_strict_schema_and_provider_cost() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "cost": "0.00001",
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")], "system", Reply
        )

    assert result.answer == "ok"
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert seen["provider"] == {"require_parameters": True}
    assert seen["messages"][0] == {"role": "system", "content": "system"}
    assert seen["max_tokens"] == 8000
    assert "reasoning" not in seen
    assert usage.generation_id == "gen-1"
    assert usage.billed_cost_usd == Decimal("0.00001")


@pytest.mark.asyncio
async def test_structured_request_removes_annotations_beside_ref() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"kind":"success"}'}}],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            ReferencedReply,
        )

    kind_schema = seen["response_format"]["json_schema"]["schema"]["properties"]["kind"]
    assert kind_schema == {"$ref": "#/$defs/ReplyKind"}
    assert result.kind is ReplyKind.SUCCESS


@pytest.mark.asyncio
async def test_openai_reasoning_model_uses_azure_compatible_token_parameter() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            model="openai/gpt-5.6-sol",
            client=raw_client,
            enable_reasoning_effort=True,
        )
        await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
            thinking_level="medium",
        )

    assert seen["max_completion_tokens"] == 8000
    assert "max_tokens" not in seen
    assert seen["provider"] == {"require_parameters": True}


@pytest.mark.asyncio
async def test_endpoint_eligibility_adapts_parameters_without_changing_model() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if (
            "max_tokens" in payload
            or "max_completion_tokens" in payload
            or "reasoning" in payload
            or "temperature" in payload
        ):
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": (
                            "No endpoints available matching your guardrail "
                            "restrictions and data policy."
                        )
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            model="vendor/model",
            temperature=0.2,
            client=raw_client,
            enable_reasoning_effort=True,
        )
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
            thinking_level="medium",
        )

    assert result.answer == "ok"
    assert len(seen) == 5
    assert all(payload["model"] == "vendor/model" for payload in seen)
    assert all(payload["messages"] == seen[0]["messages"] for payload in seen)
    assert all(
        payload["response_format"] == seen[0]["response_format"] for payload in seen
    )
    assert all(payload["provider"] == {"require_parameters": True} for payload in seen)
    assert "max_tokens" not in seen[-1]
    assert "max_completion_tokens" not in seen[-1]
    assert "reasoning" not in seen[-1]
    assert "temperature" not in seen[-1]


@pytest.mark.asyncio
async def test_generic_provider_400_adapts_same_model_parameters() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Provider returned error",
                        "metadata": {
                            "error_type": "invalid_request",
                            "provider_code": "invalid_argument",
                            "raw": json.dumps(
                                {
                                    "error": {
                                        "message": (
                                            "max_tokens is not supported; use "
                                            "max_completion_tokens"
                                        )
                                    }
                                }
                            ),
                        },
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            model="vendor/model",
            client=raw_client,
        )
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "ok"
    assert len(seen) == 2
    assert seen[0]["max_tokens"] == 8000
    assert seen[1]["max_completion_tokens"] == 8000
    assert all(payload["model"] == "vendor/model" for payload in seen)


@pytest.mark.asyncio
async def test_typed_content_policy_error_is_not_retried_as_compatibility() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {
                        "error_type": "content_policy_violation",
                        "provider_code": "content_filter",
                    },
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        with pytest.raises(RuntimeError, match="Provider returned error"):
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")],
                "system",
                Reply,
            )

    assert calls == 1


@pytest.mark.asyncio
async def test_stream_adapts_token_parameter_before_emitting_text() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": (
                            "No endpoints available matching your guardrail "
                            "restrictions and data policy."
                        )
                    }
                },
            )
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"compatible"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            model="vendor/model",
            client=raw_client,
        )
        chunks = [
            chunk
            async for chunk in client.stream_text(
                [ChatTurn(role="user", text="hello")],
                "system",
            )
        ]

    assert chunks == ["compatible"]
    assert len(seen) == 2
    assert seen[0]["max_tokens"] == 8000
    assert seen[1]["max_completion_tokens"] == 8000
    assert all(payload["model"] == "vendor/model" for payload in seen)


@pytest.mark.asyncio
async def test_truncated_reasoning_response_raises_actionable_error() -> None:
    """A verbose reasoning model (e.g. z-ai/glm-5.2) can burn its whole output
    budget on hidden chain-of-thought and never emit visible content —
    OpenRouter reports this as `finish_reason: "length"` with `content`
    missing. The resulting error should point at the real cause instead of
    the bare "no structured content" message.
    """

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 8000,
                    "completion_tokens_details": {"reasoning_tokens": 8000},
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        with pytest.raises(RuntimeError, match="OPENROUTER_MAX_OUTPUT_TOKENS"):
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")], "system", Reply
            )


@pytest.mark.asyncio
async def test_structured_request_omits_reasoning_for_model_compatibility() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
            thinking_level="high",
        )

    assert "reasoning" not in seen


@pytest.mark.asyncio
async def test_role_enabled_request_sends_normalized_reasoning_effort() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            client=raw_client,
            enable_reasoning_effort=True,
        )
        await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
            thinking_level="high",
        )

    assert seen["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_rejected_request_keeps_bounded_provider_error_detail() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "reasoning is not supported by this model"}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        with pytest.raises(RuntimeError, match="400.*reasoning is not supported"):
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")], "system", Reply
            )


@pytest.mark.asyncio
async def test_routing_policy_error_names_ineligible_model() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": (
                        "No endpoints available matching your guardrail "
                        "restrictions and data policy."
                    )
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(
            api_key="test",
            model="openai/gpt-5.6-sol",
            client=raw_client,
        )
        with pytest.raises(
            RuntimeError,
            match=r"openai/gpt-5\.6-sol.*Privacy and Guardrail policy",
        ):
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")], "system", Reply
            )
