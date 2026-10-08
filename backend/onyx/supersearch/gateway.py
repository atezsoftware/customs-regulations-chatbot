"""Selected-model generations retaining complete originals and delivery receipts."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import model_slot
from onyx.asv3.models import RunContext, RunStopped
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import (
    ChatCompletionMessage,
    ReasoningEffort,
    SystemMessage,
    UserMessage,
)
from onyx.llm.utils import check_number_of_tokens
from onyx.regulatory.structured_llm import _portable_structured_output_schema
from onyx.supersearch.payload import METADATA_DECODER, compact_original_evidence_payload
from onyx.supersearch.receipts import (
    RECEIPT_DATA_DECODER,
    compact_receipt_data_payload,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def compact_schema(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: compact_schema(item)
            for key, item in value.items()
            if key not in {"title", "description", "default"}
        }
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    return value


class SelectedModelGateway:
    def __init__(
        self,
        *,
        llm: LLM,
        ledger: EvidenceLedger,
        context: RunContext,
        user_identity: LLMUserIdentity | None = None,
        token_counter: Callable[[str], int] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
    ) -> None:
        self.llm = llm
        self.ledger = ledger
        self.context = context
        self.user_identity = user_identity
        self.token_counter = token_counter
        self.reasoning_effort = reasoning_effort
        self.last_call_id: str | None = None
        self.last_delivered_citations: set[int] = set()

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel:
        self.context.check_active()
        self.last_call_id = None
        self.last_delivered_citations = set()
        schema = compact_schema(response_type.model_json_schema())
        response_format: dict[str, JsonValue] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_type.__name__,
                "schema": _portable_structured_output_schema(schema),
                "strict": False,
            },
        }
        provider_payload, metadata_pooled = compact_original_evidence_payload(payload)
        receipts_pooled = False
        if flow == LLMFlow.SUPERSEARCH_REVIEW:
            provider_payload, receipts_pooled = compact_receipt_data_payload(
                provider_payload
            )
        messages: list[ChatCompletionMessage] = [
            SystemMessage(
                content=system
                + ("\n" + METADATA_DECODER if metadata_pooled else "")
                + ("\n" + RECEIPT_DATA_DECODER if receipts_pooled else "")
                + "\nJSON schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
            UserMessage(
                content=json.dumps(
                    provider_payload, ensure_ascii=False, separators=(",", ":")
                )
            ),
        ]
        protocol = json.dumps(
            {
                "messages": [message.model_dump(mode="json") for message in messages],
                "response_format": response_format,
            },
            ensure_ascii=False,
        )
        # Missing model tokenizers must not turn into silent original clipping.
        input_tokens = max(
            check_number_of_tokens(protocol),
            self.token_counter(protocol) if self.token_counter else 0,
        )
        if input_tokens + 256 > self.llm.config.max_input_tokens:
            raise RunStopped(
                "Complete required PC originals exceed the selected model context; no sources were silently omitted"
            )
        records_raw = payload.get("original_evidence", [])
        if not isinstance(records_raw, list) or any(
            not isinstance(row, dict) for row in records_raw
        ):
            raise ValueError("Original evidence must contain serialized records")
        records = cast(list[dict[str, JsonValue]], records_raw)
        call_id = str(uuid4())
        with model_slot(self.context, research=not finalizing):
            self.context.budget.consume("decisions")
            with llm_generation_span(
                llm=self.llm, flow=flow, input_messages=messages
            ) as span:
                span.span_data.model_config = {
                    **(span.span_data.model_config or {}),
                    "supersearch_run_id": self.context.run_id,
                    "supersearch_call_id": call_id,
                    "supersearch_corpus_only": "true",
                    "supersearch_document_set_id": str(
                        self.context.scope.get("asv3_document_set_id")
                    ),
                }
                response = self.llm.invoke(
                    messages,
                    structured_response_format=response_format,
                    timeout_override=120 if finalizing else None,
                    reasoning_effort=self.reasoning_effort,
                    user_identity=self.user_identity,
                    # Invoke assembles the provider stream before strict validation;
                    # an active long review must not hit a whole-response read timeout.
                    use_streaming=True,
                )
                record_llm_response(span, response)
        self.context.check_active()
        finish_reason = (response.choice.finish_reason or "").lower()
        if response.choice.message.tool_calls or finish_reason == "tool_calls":
            raise RunStopped("Supersearch returned an undeclared provider tool call")
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            raise RunStopped("Supersearch model output was truncated")
        content = response.choice.message.content
        if content is None:
            raise RunStopped("Supersearch model returned no structured result")
        self.ledger.record_delivery(call_id, flow.value, records)
        self.last_call_id = call_id
        self.last_delivered_citations = self.ledger.completely_delivered(call_id)
        try:
            return response_type.model_validate_json(content, strict=True)
        except ValidationError as error:
            raise RunStopped(
                "Supersearch model result failed its structured contract"
            ) from error
