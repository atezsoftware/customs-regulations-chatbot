"""One selected Gemini model, explicit admissions and full-context transport."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel, JsonValue, ValidationError

from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_review.contracts import ReadingContractError
from onyx.legal_review.models import WorkflowPolicy
from onyx.legal_review.streaming import Deadline, read_stream
from onyx.legal_review.transport import model_state
from onyx.llm.interfaces import LLM, LLMConfig, LLMUserIdentity
from onyx.llm.model_response import ModelResponse, ModelResponseStream
from onyx.llm.models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions
from onyx.llm.multi_llm import LitellmLLM, LLMTimeoutError
from onyx.llm.utils import llm_response_to_string
from onyx.regulatory.structured_llm import (
    _JSON_ONLY_REMINDER,
    StructuredOutputValidationError,
    _portable_structured_output_schema,
    generate_structured,
)
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def root_validation_error(
    content: str, response_model: type[BaseModel], fallback: ValidationError
) -> ValidationError:
    """Keep a valid JSON envelope's errors instead of an incidental nested object."""
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return fallback
    try:
        response_model.model_validate_json(content)
    except ValidationError as primary_error:
        return primary_error
    return fallback


class UsageMeter:
    def __init__(self, policy: WorkflowPolicy) -> None:
        self.policy = policy
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = Lock()

    def check(self) -> None:
        with self._lock:
            if (
                self.policy.max_input_tokens is not None
                and self.input_tokens >= self.policy.max_input_tokens
            ) or (
                self.policy.max_output_tokens is not None
                and self.output_tokens >= self.policy.max_output_tokens
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

    def __init__(
        self,
        selected: LLM,
        context: RunContext,
        meter: UsageMeter,
        reader_deadline: float | None = None,
    ) -> None:
        self.selected = selected.with_stream_cancellation_check(context.check_active)
        self.context = context
        self.meter = meter
        self.reader_deadline = reader_deadline
        self.last_response: ModelResponse | None = None

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
        self.last_response = None
        if self.reader_deadline is not None and isinstance(self.selected, LitellmLLM):
            result = read_stream(
                self.selected,
                deadline=Deadline(
                    self.reader_deadline,
                    self.meter.policy.max_call_seconds,
                    self.context.check_active,
                ),
                prompt=prompt,
                tools=tools,
                tool_choice=tool_choice,
                structured_response_format=structured_response_format,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                user_identity=user_identity,
                record_usage=self.meter.record,
            )
            self.last_response = result
            self.context.check_active()
            return result
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
        self.last_response = result
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
        if finalizing:
            if flow in {LLMFlow.LEGAL_REVIEW_DRAFT, LLMFlow.LEGAL_REVIEW_REPAIR}:
                deadline -= self.policy.publication_reserve_seconds
            else:
                deadline -= self.policy.publication_reserve_seconds + 60
            self.context.services["legal_review_phase_deadline"] = deadline
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Legal Review retained time for final publication review"
                )
        serialized = json.dumps(
            model_state(state, flow), ensure_ascii=False, separators=(",", ":")
        )
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
        capacity = self.llm.config.max_input_tokens
        if self.policy.max_context_tokens is not None:
            capacity = min(capacity, self.policy.max_context_tokens)
        if self.token_counter(complete_context) > capacity:
            raise RunStopped("Complete legal review context exceeds model capacity")
        generation_llm = self.llm
        if isinstance(self.llm, MeteredLLM):
            generation_llm = MeteredLLM(
                self.llm.selected,
                self.context,
                self.llm.meter,
                reader_deadline=deadline,
            )
        try:
            return generate_structured(
                generation_llm,
                flow=flow,
                system_prompt=prompt,
                user_prompt=serialized,
                response_model=response_model,
                timeout_override=max(
                    1,
                    int(min(self.policy.max_call_seconds, deadline - time.monotonic())),
                ),
                max_tokens=self.policy.max_generation_output_tokens,
                reasoning_effort=self.reasoning_effort,
                max_attempts=1,
                provider_max_attempts=1,
                deadline=deadline if math.isfinite(deadline) else None,
                use_streaming=False,
            )
        except LLMTimeoutError as error:
            raise TimeoutError("Legal Review Gemini provider timeout") from error
        except StructuredOutputValidationError as error:
            response = (
                generation_llm.last_response
                if isinstance(generation_llm, MeteredLLM)
                else None
            )
            cause = error.__cause__
            if (
                flow
                in {
                    LLMFlow.LEGAL_REVIEW_READING,
                    LLMFlow.LEGAL_REVIEW_SOURCE_ACCOUNTING,
                }
                and response is not None
                and response.choice.finish_reason == "stop"
                and isinstance(cause, ValidationError)
            ):
                content = llm_response_to_string(response)
                cause = root_validation_error(content, response_model, cause)
                diagnostics = json.dumps(
                    [
                        {
                            "loc": list(row["loc"]),
                            "type": row["type"],
                            "msg": row["msg"],
                        }
                        for row in cause.errors(
                            include_input=False, include_context=False
                        )
                    ],
                    ensure_ascii=False,
                )
                raise ReadingContractError(diagnostics, content) from error
            raise
