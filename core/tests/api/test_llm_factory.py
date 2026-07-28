import pytest
from pydantic import BaseModel

from fs_explorer_api.llm.base import ChatTurn, LLMUsage
from fs_explorer_api.llm.factory import (
    RoleConfiguredLLMClient,
    get_llm_client,
)
from fs_explorer_api.llm.openrouter import OpenRouterLLMClient
from fs_explorer_api.llm.profile import load_llm_profile


class Reply(BaseModel):
    answer: str


class RecordingLLMClient:
    def __init__(self) -> None:
        self.structured_efforts: list[str | None] = []
        self.stream_efforts: list[str | None] = []

    async def generate_structured(
        self,
        history,
        system_prompt,
        schema,
        *,
        thinking_level=None,
    ):
        del history, system_prompt, schema
        self.structured_efforts.append(thinking_level)
        return Reply(answer="ok"), LLMUsage()

    async def stream_text(
        self,
        history,
        system_prompt,
        *,
        thinking_level=None,
    ):
        del history, system_prompt
        self.stream_efforts.append(thinking_level)
        yield "ok"

    def last_stream_usage(self):
        return None


def test_role_profile_has_quality_tiered_defaults() -> None:
    profile = load_llm_profile({})

    assert (
        profile.planner.provider,
        profile.planner.model,
        profile.planner.reasoning_effort,
    ) == ("openrouter", "openai/gpt-5.6-sol", "medium")
    assert (
        profile.task.model,
        profile.task.reasoning_effort,
    ) == ("google/gemini-3.6-flash", "medium")
    assert (
        profile.worker.model,
        profile.worker.reasoning_effort,
    ) == ("google/gemini-3.5-flash-lite", "low")
    assert (
        profile.final.model,
        profile.final.reasoning_effort,
    ) == ("google/gemini-3.6-flash", "high")


def test_role_profile_reads_and_normalizes_role_overrides() -> None:
    profile = load_llm_profile(
        {
            "FS_EXPLORER_WORKER_PROVIDER": " GEMINI ",
            "FS_EXPLORER_WORKER_MODEL": " gemini-worker ",
            "FS_EXPLORER_WORKER_REASONING": " MINIMAL ",
        }
    )

    assert (
        profile.worker.provider,
        profile.worker.model,
        profile.worker.reasoning_effort,
    ) == ("gemini", "gemini-worker", "minimal")


def test_role_profile_rejects_invalid_reasoning_at_startup() -> None:
    with pytest.raises(ValueError, match="FS_EXPLORER_PLANNER_REASONING"):
        load_llm_profile({"FS_EXPLORER_PLANNER_REASONING": "unbounded"})


def test_legacy_factory_path_keeps_global_configuration(monkeypatch) -> None:
    monkeypatch.setenv("FS_EXPLORER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("FS_EXPLORER_LLM_MODEL", "legacy/model")

    client = get_llm_client(api_key="test")

    assert isinstance(client, OpenRouterLLMClient)
    assert client.model == "legacy/model"


def test_role_factory_ignores_legacy_globals_and_captures_profile(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FS_EXPLORER_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("FS_EXPLORER_LLM_MODEL", "legacy/model")

    client = get_llm_client(
        role="planner",
        profile=load_llm_profile({}),
        api_key="test",
    )

    assert isinstance(client, RoleConfiguredLLMClient)
    assert isinstance(client.raw_client, OpenRouterLLMClient)
    assert client.role == "planner"
    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-5.6-sol"
    assert client.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_role_client_applies_default_effort_and_allows_call_override() -> None:
    raw_client = RecordingLLMClient()
    profile = load_llm_profile({})
    client = RoleConfiguredLLMClient(
        raw_client,
        role="planner",
        config=profile.planner,
    )
    history = [ChatTurn(role="user", text="question")]

    await client.generate_structured(history, "system", Reply)
    streamed = [
        chunk
        async for chunk in client.stream_text(
            history,
            "system",
            thinking_level="low",
        )
    ]

    assert raw_client.structured_efforts == ["medium"]
    assert raw_client.stream_efforts == ["low"]
    assert streamed == ["ok"]
