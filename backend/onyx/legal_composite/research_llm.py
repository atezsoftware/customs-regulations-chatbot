from __future__ import annotations

from collections.abc import Iterator

from pydantic import JsonValue

from onyx.asv3.models import RunStopped
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.llm.interfaces import LLM, LLMConfig, LLMUserIdentity
from onyx.llm.model_response import ModelResponse, ModelResponseStream
from onyx.llm.models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions


class BudgetedResearchLLM(LLM):
    """Isolated search helpers share the main workflow's model/time/spend limits."""

    def __init__(self, gateway: BudgetedGateway) -> None:
        self.gateway = gateway

    @property
    def config(self) -> LLMConfig:
        return self.gateway.research_llm.config

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict[str, JsonValue]] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict[str, JsonValue] | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        use_streaming: bool = True,
        provider_compatibility_attempts: int | None = None,
    ) -> ModelResponse:
        del reasoning_effort, use_streaming
        if tools or tool_choice is not None:
            raise RunStopped("Auxiliary search generations cannot execute tools")
        # Search helpers request collected responses; transport streaming is unnecessary.
        return self.gateway.research_invoke(
            prompt,
            structured_response_format=structured_response_format,
            timeout_override=timeout_override,
            max_tokens=max_tokens,
            user_identity=user_identity,
            provider_compatibility_attempts=provider_compatibility_attempts,
        )

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict[str, JsonValue]] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict[str, JsonValue] | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
    ) -> Iterator[ModelResponseStream]:
        del prompt, tools, tool_choice, structured_response_format
        del timeout_override, max_tokens, reasoning_effort, user_identity
        raise RunStopped(
            "Auxiliary search streaming is unavailable within this workflow"
        )
