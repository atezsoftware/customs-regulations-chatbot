"""One selected Gemini model, explicit admissions and full-context transport."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel, JsonValue

from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_review.models import WorkflowPolicy
from onyx.legal_review.transport import model_state
from onyx.llm.interfaces import LLM, LLMConfig, LLMUserIdentity
from onyx.llm.model_response import ModelResponse, ModelResponseStream
from onyx.llm.models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.regulatory.structured_llm import (
    _JSON_ONLY_REMINDER,
    _portable_structured_output_schema,
    generate_structured,
)
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class UsageMeter:
    def __init__(self, policy: WorkflowPolicy) -> None:
        self.policy = policy
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = Lock()

    def check(self) -> None:
        with self._lock:
            if (
                self.input_tokens >= self.policy.max_input_tokens
                or self.output_tokens >= self.policy.max_output_tokens
            ):
                raise RunStopped("Legal review generation token budget exhausted")

    def record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


class MeteredLLM(LLM):
    """Account underlying model usage including secondary search generations."""

    def __init__(self, selected: LLM, context: RunContext, meter: UsageMeter) -> None:
        self.selected = selected.with_stream_cancellation_check(context.check_active)
        self.context = context
        self.meter = meter

    @property
    def config(self) -> LLMConfig:
        return self.selected.config

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
        use_streaming: bool = True,
        provider_compatibility_attempts: int | None = None,
    ) -> ModelResponse:
        del provider_compatibility_attempts
        self.context.check_active()
        self.meter.check()
        result = self.selected.invoke(
            prompt=prompt,
            tools=tools,
            tool_choice=tool_choice,
            structured_response_format=structured_response_format,
            timeout_override=timeout_override,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            user_identity=user_identity,
            use_streaming=use_streaming,
            provider_compatibility_attempts=1,
        )
        if result.usage is not None:
            self.meter.record(
                result.usage.prompt_tokens, result.usage.completion_tokens
            )
        self.context.check_active()
        return result

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
    ) -> Iterator[ModelResponseStream]:
        self.context.check_active()
        self.meter.check()
        usage = None
        try:
            for chunk in self.selected.stream(
                prompt=prompt,
                tools=tools,
                tool_choice=tool_choice,
                structured_response_format=structured_response_format,
                timeout_override=timeout_override,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                user_identity=user_identity,
            ):
                self.context.check_active()
                if chunk.usage is not None:
                    usage = chunk.usage
                yield chunk
        finally:
            if usage is not None:
                self.meter.record(usage.prompt_tokens, usage.completion_tokens)


class GeminiGateway:
    def __init__(
        self,
        *,
        llm: LLM,
        context: RunContext,
        policy: WorkflowPolicy,
        token_counter: Callable[[str], int],
        reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    ) -> None:
        self.llm = llm
        self.context = context
        self.policy = policy
        self.token_counter = token_counter
        self.reasoning_effort = reasoning_effort

    def complete(
        self,
        prompt: str,
        state: dict[str, JsonValue],
        response_model: type[ResponseModel],
        flow: LLMFlow,
        *,
        finalizing: bool = False,
    ) -> ResponseModel:
        if finalizing:
            self.context.check_active()
            deadline = self.context.deadline
            self.context.budget.consume("decisions")
        else:
            self.context.check_research_active()
            deadline = self.context.research_deadline
            self.context.consume_research_decision()
        serialized = json.dumps(model_state(state, flow), ensure_ascii=False)
        validation_schema = response_model.model_json_schema()
        system_prompt = prompt + _JSON_ONLY_REMINDER.format(schema=validation_schema)
        provider_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": _portable_structured_output_schema(validation_schema),
                "strict": False,
            },
        }
        complete_context = (
            system_prompt + serialized + json.dumps(provider_format, ensure_ascii=False)
        )
        if self.token_counter(complete_context) > min(
            self.policy.max_context_tokens, self.llm.config.max_input_tokens
        ):
            raise RunStopped("Complete legal review context exceeds model capacity")
        try:
            return generate_structured(
                self.llm,
                flow=flow,
                system_prompt=prompt,
                user_prompt=serialized,
                response_model=response_model,
                timeout_override=max(
                    1,
                    min(self.policy.max_call_seconds, int(deadline - time.monotonic())),
                ),
                max_tokens=self.policy.max_generation_output_tokens,
                reasoning_effort=self.reasoning_effort,
                max_attempts=1,
                provider_max_attempts=1,
                deadline=deadline,
                use_streaming=False,
            )
        except LLMTimeoutError as error:
            raise TimeoutError("Legal Review Gemini provider timeout") from error
