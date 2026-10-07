from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from threading import BoundedSemaphore
from typing import TypeVar, cast

from pydantic import BaseModel, JsonValue, ValidationError
from sqlalchemy.orm import Session

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import CallReservation, WorkflowBudget
from onyx.llm.cost import ModelPrice, get_model_price_per_million
from onyx.llm.cost_overrides import get_override
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import (
    ANTHROPIC_REASONING_EFFORT_BUDGET,
    ChatCompletionMessage,
    LanguageModelInput,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.utils import check_number_of_tokens
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_RESEARCH_OUTPUT_TOKENS = 2_048
_FINAL_OUTPUT_TOKENS = 4_096
_PROTOCOL_TOKEN_MARGIN = 256
_TOKEN_ESTIMATE_MARGIN = 1.25
_COMPATIBILITY_ATTEMPTS = 3


def _validate_budgeted_provider_config(settings: Mapping[str, object]) -> None:
    for key, value in settings.items():
        normalized = re.sub(r"[^a-z]", "", key.lower())
        if "retr" in normalized or normalized in {
            "servicetier",
            "processingmode",
            "provider",
            "customllmprovider",
            "route",
            "routing",
            "routingstrategy",
            "priority",
            "batch",
            "flex",
            "models",
            "modelname",
            "modellist",
            "modelgroup",
            "fallbacks",
            "model",
            "deployment",
            "deploymentname",
        }:
            raise RunStopped(
                "Custom provider retry, routing or service-tier settings cannot share this workflow budget"
            )
        nested: object = value
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(value)
            except json.JSONDecodeError as error:
                raise RunStopped(
                    "Provider configuration JSON cannot be verified for budgeting"
                ) from error
        if isinstance(nested, dict):
            _validate_budgeted_provider_config(cast(Mapping[str, object], nested))
        elif isinstance(nested, list):
            for item in nested:
                _validate_budgeted_provider_config({"nested": item})


def _estimated_input_tokens(
    text: str, token_counter: Callable[[str], int] | None = None
) -> int:
    # Provider tokenizers differ; protocol/schema and a safety margin are included.
    generic = check_number_of_tokens(text)
    custom = token_counter(text) if token_counter else generic
    if isinstance(custom, bool) or not isinstance(custom, int) or custom < 0:
        raise ValueError("Invalid token counter result")
    return (
        math.ceil(max(generic, custom) * _TOKEN_ESTIMATE_MARGIN)
        + _PROTOCOL_TOKEN_MARGIN
    )


def _priced_model(
    llm: LLM, db_session: Session | None, max_context_tokens: int
) -> ModelPrice:
    config = llm.config
    price = get_model_price_per_million(
        config.model_name, config.model_provider, db_session
    )
    if price.input_per_mtok is None or price.output_per_mtok is None:
        raise RunStopped("The configured model has no verified token price")
    input_rates = [price.input_per_mtok]
    output_rates = [price.output_per_mtok]
    if (
        db_session is not None
        and get_override(db_session, config.model_name, config.model_provider)
        is not None
    ):
        if any(
            not math.isfinite(rate) or rate < 0 for rate in input_rates + output_rates
        ):
            raise RunStopped("The configured model has an invalid token price")
        return price.model_copy(update={"cache_per_mtok": None})
    try:
        import litellm

        metadata = cast(
            Mapping[str, object],
            litellm.get_model_info(
                model=config.model_name, custom_llm_provider=config.model_provider
            ),
        )
    except Exception:
        metadata = {}
    for key, value in metadata.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        # This workflow requests the default online tier and cannot reach larger contexts.
        if any(tier in key for tier in ("priority", "batch", "flex")):
            continue
        threshold = re.search(r"_above_(\d+)(k)?(?:_tokens|$)", key)
        if threshold:
            boundary = int(threshold.group(1)) * (1_000 if threshold.group(2) else 1)
            if max_context_tokens <= boundary:
                continue
        if key.startswith("input_cost_per_token") or key.startswith(
            "cache_creation_input_token_cost"
        ):
            input_rates.append(float(value) * 1_000_000)
        elif key.startswith("output_cost_per_token"):
            output_rates.append(float(value) * 1_000_000)
    if any(not math.isfinite(rate) or rate < 0 for rate in input_rates + output_rates):
        raise RunStopped("The configured model has an invalid token price")
    return price.model_copy(
        update={
            "input_per_mtok": max(input_rates),
            "output_per_mtok": max(output_rates),
            "cache_per_mtok": None,
        }
    )


class BudgetedGateway:
    """Typed generations with estimated spend and bounded provider timeouts.

    Invoke bypasses transient stream retries. Existing compatibility retries handle
    rejected 400s, assumed nonbillable, and share a conservatively divided timeout.
    Provider tokenizer/pricing drift and transport behavior preclude an invoice SLA.
    """

    def __init__(
        self,
        *,
        selected_llm: LLM,
        research_llm: LLM,
        budget: WorkflowBudget,
        ledger: EvidenceLedger,
        db_session: Session | None = None,
        user_identity: LLMUserIdentity | None = None,
        check_active: Callable[[], None] = lambda: None,
        token_counter: Callable[[str], int] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    ) -> None:
        self.selected_llm = selected_llm
        self.research_llm = research_llm
        self.budget = budget
        self.ledger = ledger
        self.user_identity = user_identity
        self.check_active = check_active
        self.token_counter = token_counter
        self.reasoning_effort = reasoning_effort
        self.last_call_id: str | None = None
        self.last_delivered_citations: set[int] = set()
        self._final_output_tokens = _FINAL_OUTPUT_TOKENS
        if "claude" in selected_llm.config.model_name.lower():
            self._final_output_tokens = max(
                self._final_output_tokens,
                ANTHROPIC_REASONING_EFFORT_BUDGET.get(reasoning_effort, 0) + 1,
            )
        self._prices: dict[tuple[str, str], ModelPrice] = {}
        self._model_slot = BoundedSemaphore(1)
        for llm in (selected_llm, research_llm):
            self.check_active()
            budget.check_active()
            config = llm.config
            _validate_budgeted_provider_config(config.custom_config or {})
            self._prices[(config.model_provider, config.model_name)] = _priced_model(
                llm,
                db_session,
                min(budget.policy.max_context_tokens, config.max_input_tokens),
            )
        selected_price = self._price(selected_llm)
        input_rate = selected_price.input_per_mtok
        output_rate = selected_price.output_per_mtok
        assert input_rate is not None and output_rate is not None
        final_input = min(
            budget.policy.max_context_tokens, selected_llm.config.max_input_tokens
        )
        budget.configure_finalization(
            final_input,
            self._final_output_tokens,
            (final_input * input_rate + self._final_output_tokens * output_rate)
            / 1_000_000,
        )

    def _price(self, llm: LLM) -> ModelPrice:
        config = llm.config
        return self._prices[(config.model_provider, config.model_name)]

    def _fit_messages(
        self, system: str, payload: dict[str, JsonValue], schema: str, llm: LLM
    ) -> tuple[list[ChatCompletionMessage], int, list[dict[str, JsonValue]]]:
        body = cast(dict[str, JsonValue], json.loads(json.dumps(payload)))
        raw_records = body.get("original_evidence", [])
        if not isinstance(raw_records, list) or any(
            not isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("Original evidence must contain serialized records")
        records = cast(list[dict[str, JsonValue]], raw_records)
        required_raw = body.get("required_evidence_numbers", [])
        if not isinstance(required_raw, list) or any(
            isinstance(number, bool) or not isinstance(number, int)
            for number in required_raw
        ):
            raise ValueError("Required evidence identities must be integers")
        required = set(cast(list[int], required_raw))
        recorded = {
            number
            for record in records
            if isinstance(number := record.get("citation"), int)
            and not isinstance(number, bool)
        }
        if required - recorded:
            raise RunStopped(
                "A required original is absent from the generation payload"
            )
        omitted_raw = body.get("omitted_original_ids", [])
        if not isinstance(omitted_raw, list):
            raise ValueError("Omitted evidence identities must be a list")
        omitted = list(omitted_raw)
        cap = min(self.budget.policy.max_context_tokens, llm.config.max_input_tokens)
        while True:
            body["original_evidence"] = cast(list[JsonValue], records)
            body["omitted_original_ids"] = omitted
            messages: list[ChatCompletionMessage] = [
                SystemMessage(
                    content=system
                    + "\nReturn one JSON object matching this schema:\n"
                    + schema
                ),
                UserMessage(content=json.dumps(body, ensure_ascii=False)),
            ]
            protocol = json.dumps(
                {
                    "messages": [
                        message.model_dump(mode="json") for message in messages
                    ],
                    "response_format": {"type": "json_object"},
                },
                ensure_ascii=False,
            )
            input_tokens = _estimated_input_tokens(protocol, self.token_counter)
            if input_tokens <= cap:
                return messages, input_tokens, records
            removable = next(
                (
                    index
                    for index in range(len(records) - 1, -1, -1)
                    if records[index].get("citation") not in required
                ),
                None,
            )
            if removable is None:
                raise RunStopped(
                    "Required originals and protocol exceed the model context budget"
                )
            removed = records.pop(removable)
            number = removed.get("citation")
            if number not in omitted:
                omitted.append(number)

    def _invoke(
        self,
        llm: LLM,
        messages: list[ChatCompletionMessage],
        flow: LLMFlow,
        timeout: int,
        output_tokens: int,
        research: bool,
        response_format: dict[str, JsonValue] | None,
        user_identity: LLMUserIdentity | None,
    ) -> ModelResponse:
        self.check_active()
        self.budget.check_active(not research)
        with llm_generation_span(llm=llm, flow=flow, input_messages=messages) as span:
            response = llm.invoke(
                messages,
                structured_response_format=response_format,
                timeout_override=timeout,
                max_tokens=output_tokens,
                reasoning_effort=ReasoningEffort.LOW
                if research
                else self.reasoning_effort,
                user_identity=user_identity,
                use_streaming=False,
            )
            record_llm_response(span, response)
            return response

    def _generate(
        self,
        llm: LLM,
        messages: list[ChatCompletionMessage],
        flow: LLMFlow,
        input_tokens: int,
        output_tokens: int,
        finalizing: bool,
        response_format: dict[str, JsonValue] | None,
        user_identity: LLMUserIdentity | None,
        invocation_output_tokens: int | None = None,
        requested_seconds: int | None = None,
    ) -> tuple[CallReservation, ModelResponse]:
        while True:
            self.check_active()
            self.budget.check_active(finalizing)
            if self._model_slot.acquire(timeout=0.05):
                break
        try:
            return self._generate_reserved(
                llm,
                messages,
                flow,
                input_tokens,
                output_tokens,
                finalizing,
                response_format,
                user_identity,
                invocation_output_tokens,
                requested_seconds,
            )
        finally:
            self._model_slot.release()

    def _generate_reserved(
        self,
        llm: LLM,
        messages: list[ChatCompletionMessage],
        flow: LLMFlow,
        input_tokens: int,
        output_tokens: int,
        finalizing: bool,
        response_format: dict[str, JsonValue] | None,
        user_identity: LLMUserIdentity | None,
        invocation_output_tokens: int | None,
        requested_seconds: int | None,
    ) -> tuple[CallReservation, ModelResponse]:
        self.check_active()
        self.budget.check_active(finalizing)
        price = self._price(llm)
        input_rate, output_rate = price.input_per_mtok, price.output_per_mtok
        assert input_rate is not None and output_rate is not None
        reservation = self.budget.request(
            input_tokens, output_tokens, input_rate, output_rate, finalizing
        )
        call_seconds = reservation.timeout_seconds
        if requested_seconds is not None:
            call_seconds = min(call_seconds, requested_seconds)
        timeout = math.floor(call_seconds / _COMPATIBILITY_ATTEMPTS)
        if timeout < 1:
            raise RunStopped("Insufficient time remains for a bounded model invocation")
        executor = ThreadPoolExecutor(max_workers=1)
        context = copy_context()
        future = executor.submit(
            context.run,
            self._invoke,
            llm,
            messages,
            flow,
            timeout,
            invocation_output_tokens or output_tokens,
            not finalizing,
            response_format,
            user_identity,
        )
        try:
            response = cast(ModelResponse, future.result(timeout=call_seconds))
        except FutureTimeout as error:
            self.budget.stop(
                "Provider call exceeded its deadline; no further spend authorized"
            )
            raise RunStopped("The bounded model invocation timed out") from error
        except RunStopped:
            raise
        except Exception as error:
            raise RunStopped("The bounded model invocation failed") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if response.usage is not None:
            usage = response.usage
            self.budget.settle(
                reservation, usage.prompt_tokens, usage.completion_tokens
            )
        self.check_active()
        self.budget.check_active(finalizing)
        return reservation, response

    def research_invoke(
        self,
        prompt: LanguageModelInput,
        *,
        structured_response_format: dict[str, JsonValue] | None = None,
        max_tokens: int | None = None,
        timeout_override: int | None = None,
        user_identity: LLMUserIdentity | None = None,
    ) -> ModelResponse:
        """Admit auxiliary search generations through the same research allocation."""
        if max_tokens is not None and (isinstance(max_tokens, bool) or max_tokens < 1):
            raise ValueError("Invalid auxiliary output limit")
        if timeout_override is not None and (
            isinstance(timeout_override, bool)
            or timeout_override < _COMPATIBILITY_ATTEMPTS
        ):
            raise ValueError("Invalid auxiliary timeout")
        messages = prompt if isinstance(prompt, list) else [prompt]
        for message in messages:
            if (
                isinstance(message, UserMessage)
                and isinstance(message.content, list)
                and any(part.type != "text" for part in message.content)
            ):
                raise RunStopped(
                    "Multimodal auxiliary generations have no verified workflow pricing"
                )
        protocol = json.dumps(
            {
                "messages": [message.model_dump(mode="json") for message in messages],
                "response_format": structured_response_format,
            },
            ensure_ascii=False,
        )
        input_tokens = _estimated_input_tokens(protocol, self.token_counter)
        llm = self.research_llm
        if input_tokens > min(
            self.budget.policy.max_context_tokens, llm.config.max_input_tokens
        ):
            raise RunStopped(
                "Auxiliary search generation exceeds the model context budget"
            )
        _, response = self._generate(
            llm,
            messages,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
            input_tokens,
            _RESEARCH_OUTPUT_TOKENS,
            False,
            structured_response_format,
            user_identity or self.user_identity,
            invocation_output_tokens=min(
                max_tokens or _RESEARCH_OUTPUT_TOKENS, _RESEARCH_OUTPUT_TOKENS
            ),
            requested_seconds=timeout_override,
        )
        return response

    def research_proxy(self) -> LLM:
        from onyx.legal_composite.research_llm import BudgetedResearchLLM

        return BudgetedResearchLLM(self)

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel:
        self.last_call_id = None
        self.last_delivered_citations = set()
        self.check_active()
        self.budget.check_active(finalizing)
        research = flow is LLMFlow.LEGAL_COMPOSITE_RESEARCH
        llm = self.research_llm if research else self.selected_llm
        if finalizing == research:
            raise ValueError("Generation flow and phase must agree")
        schema = json.dumps(response_type.model_json_schema(), ensure_ascii=False)
        messages, input_tokens, records = self._fit_messages(
            system, payload, schema, llm
        )
        output_tokens = (
            _RESEARCH_OUTPUT_TOKENS if research else self._final_output_tokens
        )
        reservation, response = self._generate(
            llm,
            messages,
            flow,
            input_tokens,
            output_tokens,
            finalizing,
            {"type": "json_object"},
            self.user_identity,
        )
        self.ledger.record_delivery(reservation.call_id, flow.value, records)
        self.last_call_id = reservation.call_id
        self.last_delivered_citations = self.ledger.completely_delivered(
            reservation.call_id
        )
        self.check_active()
        self.budget.check_active(finalizing)
        content = response.choice.message.content
        if content is None:
            raise RunStopped("The model returned no structured answer")
        try:
            return response_type.model_validate_json(content, strict=True)
        except ValidationError as error:
            raise RunStopped("The model response failed the workflow schema") from error
