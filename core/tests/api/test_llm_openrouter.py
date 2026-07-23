from decimal import Decimal

import httpx
import json
import pytest
from pydantic import BaseModel

from fs_explorer_api.llm.base import ChatTurn
from fs_explorer_api.llm.openrouter import OpenRouterLLMClient, usage_from_openrouter


class Reply(BaseModel):
    answer: str


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
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "cost": "0.00001"},
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
    assert usage.generation_id == "gen-1"
    assert usage.billed_cost_usd == Decimal("0.00001")


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
            [ChatTurn(role="user", text="hello")], "system", Reply, thinking_level="high"
        )

    assert "reasoning" not in seen


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
