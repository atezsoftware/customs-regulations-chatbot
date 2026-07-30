from enum import Enum
from decimal import Decimal

import httpx
import json
import pytest
from pydantic import BaseModel, Field

from fs_explorer_api.llm.base import ChatTurn
from fs_explorer_api.llm.openrouter import (
    OpenRouterError,
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


@pytest.fixture(autouse=True)
def _use_explicit_output_budget_for_wire_compatibility_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most tests exercise token-parameter adaptation with an opt-in cap."""

    monkeypatch.setenv("OPENROUTER_MAX_OUTPUT_TOKENS", "8000")


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
async def test_structured_request_has_no_application_token_cap_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    monkeypatch.delenv("OPENROUTER_MAX_OUTPUT_TOKENS")

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
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "ok"
    assert "max_tokens" not in seen
    assert "max_completion_tokens" not in seen


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
async def test_truncated_structured_response_is_regenerated_from_original() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "gen-truncated",
                    "choices": [
                        {
                            "message": {
                                "content": '{"answer":"TOP_SECRET_BROKEN_FRAGMENT'
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 6,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "cost": "0.001",
                        "cost_details": {"upstream_inference_cost": "0.0008"},
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "gen-complete",
                "choices": [
                    {
                        "message": {"content": '{"answer":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "cost": "0.002",
                    "cost_details": {"upstream_inference_cost": "0.0016"},
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "ok"
    assert len(seen) == 2
    assert seen[1]["model"] == seen[0]["model"]
    assert seen[1]["provider"] == seen[0]["provider"]
    assert seen[1]["response_format"] == seen[0]["response_format"]
    assert seen[1]["messages"][:-1] == seen[0]["messages"]
    assert "COMPLETE, concise JSON" in seen[1]["messages"][-1]["content"]
    assert "TOP_SECRET_BROKEN_FRAGMENT" not in json.dumps(seen[1])
    assert seen[0]["max_tokens"] == 8000
    assert seen[1]["max_tokens"] == 16000
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.thinking_tokens,
    ) == (22, 7, 2)
    assert usage.generation_id == "gen-complete"
    assert usage.billed_cost_usd == Decimal("0.003")
    assert usage.upstream_cost_usd == Decimal("0.0024")
    assert usage.cost_source == "provider"


@pytest.mark.asyncio
async def test_token_boundary_retries_even_when_json_is_schema_valid() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"answer":"premature"}'},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"answer":"complete"}'},
                        "finish_reason": "stop",
                    }
                ]
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
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "complete"
    assert len(seen) == 2
    assert seen[0]["max_completion_tokens"] == 8000
    assert seen[1]["max_completion_tokens"] == 16000


@pytest.mark.asyncio
async def test_native_max_tokens_finish_reason_triggers_recovery() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"answer":"premature"}'},
                            "finish_reason": "stop",
                            "native_finish_reason": "MAX_TOKENS",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"complete"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "complete"
    assert calls == 2


@pytest.mark.asyncio
async def test_schema_invalid_structured_response_is_regenerated() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        content = "{}" if len(seen) == 1 else '{"answer":"repaired"}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "repaired"
    assert len(seen) == 2
    assert "missing" in seen[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_structured_recovery_has_no_default_attempt_ceiling() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "{}" if calls <= 4 else '{"answer":"eventually valid"}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "eventually valid"
    assert calls == 5


@pytest.mark.asyncio
async def test_transient_transport_recovery_has_no_default_attempt_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(
        "fs_explorer_api.llm.openrouter.asyncio.sleep",
        no_sleep,
    )

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 4:
            return httpx.Response(
                503,
                json={"error": {"message": "temporary upstream overload"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"recovered"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "recovered"
    assert calls == 5


@pytest.mark.asyncio
async def test_malformed_structured_responses_raise_sanitized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    broken_fragment = '{"answer":"TOP_SECRET_BROKEN_FRAGMENT'
    monkeypatch.setenv("OPENROUTER_STRUCTURED_RETRIES", "2")

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": broken_fragment},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "cost": "0.001",
                },
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        with pytest.raises(OpenRouterError) as caught:
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")],
                "system",
                Reply,
            )

    message = str(caught.value)
    assert calls == 3
    assert "complete valid Reply response after 3 attempts" in message
    assert "TOP_SECRET_BROKEN_FRAGMENT" not in message
    assert "input_value" not in message
    assert "errors.pydantic.dev" not in message
    assert caught.value.usage is not None
    assert caught.value.usage.input_tokens == 15
    assert caught.value.usage.output_tokens == 24
    assert caught.value.usage.billed_cost_usd == Decimal("0.003")
    assert caught.value.attempts == 3


@pytest.mark.asyncio
async def test_choice_level_provider_error_is_retried_without_raw_payload() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {},
                            "finish_reason": "error",
                            "error": {
                                "code": "upstream_error",
                                "message": "temporary provider failure",
                                "raw": "TOP_SECRET_PROVIDER_BODY",
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        result, _usage = await client.generate_structured(
            [ChatTurn(role="user", text="hello")],
            "system",
            Reply,
        )

    assert result.answer == "ok"
    assert len(seen) == 2
    assert "temporary provider failure" in seen[1]["messages"][-1]["content"]
    assert "TOP_SECRET_PROVIDER_BODY" not in json.dumps(seen[1])
    assert seen[1]["max_tokens"] == 8000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "finish_reason", "expected"),
    [
        ({"refusal": "cannot comply"}, None, "refused"),
        ({}, "content_filter", "blocked"),
    ],
)
async def test_structured_refusal_or_filter_is_not_retried(
    message: dict,
    finish_reason: str | None,
    expected: str,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        choice: dict = {"message": message}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        return httpx.Response(200, json={"choices": [choice]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    ) as raw_client:
        client = OpenRouterLLMClient(api_key="test", client=raw_client)
        with pytest.raises(OpenRouterError, match=expected):
            await client.generate_structured(
                [ChatTurn(role="user", text="hello")],
                "system",
                Reply,
            )

    assert calls == 1


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
async def test_truncated_reasoning_response_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verbose reasoning model (e.g. z-ai/glm-5.2) can burn its whole output
    budget on hidden chain-of-thought and never emit visible content —
    OpenRouter reports this as `finish_reason: "length"` with `content`
    missing. The resulting error should point at the real cause instead of
    the bare "no structured content" message.
    """

    monkeypatch.setenv("OPENROUTER_STRUCTURED_RETRIES", "0")

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
        with pytest.raises(OpenRouterError, match="output-token boundary"):
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
