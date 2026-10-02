"""Provider-neutral decisions over a bounded, addressable research record."""

from __future__ import annotations

import copy
import json
import random
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated, Literal

import jsonschema
from pydantic import BaseModel, Field, JsonValue, model_validator

from onyx.asv3.artifacts import ArtifactStore, compact_json
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    ResearchTurn,
    RunContext,
    RunStopped,
)
from onyx.configs.chat_configs import (
    LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S,
    LLM_FIRST_CHUNK_RETRY_JITTER_RATIO,
    LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S,
)
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    ContentPart,
    ImageContentPart,
    ImageUrlDetail,
    ReasoningEffort,
    SystemMessage,
    TextContentPart,
    ToolCall,
    ToolChoiceOptions,
    UserMessage,
)
from onyx.llm.models import (
    FunctionCall as NativeFunctionCall,
)
from onyx.prompts.asv3.research import COORDINATOR_PROMPT, RESEARCHER_PROMPT
from onyx.regulatory.structured_llm import (
    _portable_structured_output_schema,
    _retry_after_seconds,
    is_retryable_provider_error,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response

REQUIRED_NOTIFICATION_PHASES = (
    "started",
    "tools",
    "worker",
    "final",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "resume",
    "native_citation",
)


class LanguageProfile(BaseModel):
    language: str = Field(pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")
    external_requested: bool = Field(default=False, strict=True)
    requires_sources: bool = Field(default=True, strict=True)
    notifications: dict[str, list[str]] = Field(
        json_schema_extra={
            "type": "object",
            "properties": {
                phase: {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 2,
                    "maxItems": 2,
                }
                for phase in REQUIRED_NOTIFICATION_PHASES
            },
            "required": list(REQUIRED_NOTIFICATION_PHASES),
            "additionalProperties": False,
        }
    )

    @model_validator(mode="after")
    def validate_notifications(self) -> LanguageProfile:
        missing = set(REQUIRED_NOTIFICATION_PHASES) - self.notifications.keys()
        extra = self.notifications.keys() - set(REQUIRED_NOTIFICATION_PHASES)
        if missing or extra:
            raise ValueError(
                f"Notification phases missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        for phase, pair in self.notifications.items():
            if len(pair) != 2 or any(not text.strip() for text in pair):
                raise ValueError(
                    f"Notification {phase} requires exactly two nonblank strings"
                )
        return self


class VerificationResult(BaseModel):
    status: Literal["supported", "contradicted", "incomplete", "uncertain"]
    explanation: Annotated[str, Field(min_length=1)]
    required_conditions: list[str]
    missing_conditions: list[str]
    evidence_numbers: list[Annotated[int, Field(strict=True, ge=1)]]
    question_results: list["QuestionVerification"] = Field(default_factory=list)
    safe_to_publish: bool = Field(default=False, strict=True)
    unsupported_claims: list[str] = Field(default_factory=list)


class QuestionVerification(BaseModel):
    question_id: str
    status: Literal["supported", "contradicted", "incomplete", "uncertain"]
    evidence_numbers: list[Annotated[int, Field(strict=True, ge=1)]]
    missing_conditions: list[str]


def structured_model(flow: LLMFlow) -> type[BaseModel] | None:
    if flow == LLMFlow.ASV3_LANGUAGE:
        return LanguageProfile
    if flow == LLMFlow.ASV3_VERIFICATION:
        return VerificationResult
    return None


def strict_json_decoder() -> json.JSONDecoder:
    def object_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = item
        return result

    def invalid_constant(_value: str) -> JsonValue:
        raise ValueError("Non-finite JSON number")

    return json.JSONDecoder(
        object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


def provider_retry_delay(error: Exception, attempt: int) -> float:
    delay = _retry_after_seconds(error)
    if delay is not None:
        return delay
    scheduled = min(
        LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S,
        LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S * (2**attempt),
    )
    jitter = scheduled * LLM_FIRST_CHUNK_RETRY_JITTER_RATIO
    return random.uniform(max(0, scheduled - jitter), scheduled + jitter)


def parse_json_object(text: str) -> dict[str, JsonValue]:
    value = text.strip()
    if value.startswith("```"):
        if "\n" not in value or not value.endswith("```"):
            raise ValueError("Malformed JSON code fence")
        value = value.split("\n", 1)[1].rsplit("```", 1)[0]
    data = strict_json_decoder().decode(value)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def normalize_structured_response(text: str, response_model: type[BaseModel]) -> str:
    decoder = strict_json_decoder()
    candidates: list[BaseModel] = []
    validation_error: ValueError | None = None
    offset = 0
    if text.strip().startswith("["):
        raise ValueError("Expected a JSON object, not an array")
    while (start := text.find("{", offset)) >= 0:
        # Never salvage an inner object from malformed, duplicate, or nonfinite JSON.
        value, end = decoder.raw_decode(text, start)
        offset = end
        if isinstance(value, dict):
            try:
                candidates.append(response_model.model_validate(value))
            except ValueError as error:
                validation_error = error
                continue
    if not candidates and validation_error is not None:
        raise validation_error
    if len(candidates) != 1:
        raise ValueError(
            "Structured response requires exactly one schema-valid JSON object"
        )
    return candidates[0].model_dump_json()


@contextmanager
def model_slot(context: RunContext, *, research: bool = False) -> Iterator[None]:
    check = context.check_research_active if research else context.check_active
    while not context.budget.model_slots.acquire(timeout=0.05):
        check()
    try:
        check()
        yield
    finally:
        context.budget.model_slots.release()


class ResearchModel:
    def __init__(
        self,
        llm: LLM,
        context: RunContext,
        *,
        user_identity: LLMUserIdentity | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        history: str = "",
        updates: Callable[[], list[str]] | None = None,
        pending_tasks: Callable[[], list[dict[str, JsonValue]]] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.llm = llm
        self.context = context
        self.user_identity = user_identity
        self.reasoning_effort = reasoning_effort
        self.history = history
        self.updates = updates or (lambda: [])
        self.pending_tasks = pending_tasks or (lambda: [])
        self.token_counter = token_counter
        self.last_call_id: str | None = None

    def _tokens(self, text: str) -> int:
        if self.token_counter is None:
            return len(text.encode("utf-8"))
        try:
            counted = self.token_counter(text)
        except (NotImplementedError, LookupError):
            return len(text.encode("utf-8"))
        if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
            raise ValueError(
                "The selected model token counter returned an invalid count"
            )
        return counted

    def _input_cost(
        self, prompt: list[ChatCompletionMessage], tools: list[dict[str, JsonValue]]
    ) -> int:
        total = 128 + len(prompt) * 16 + len(tools) * 16
        for message in prompt:
            if isinstance(message, AssistantMessage) and message.tool_calls:
                total += self._tokens(
                    json.dumps(
                        [call.model_dump(mode="json") for call in message.tool_calls]
                    )
                )
            content = message.content
            if isinstance(content, str):
                total += self._tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, TextContentPart):
                        total += self._tokens(part.text)
                    elif isinstance(part, ImageContentPart):
                        total += max(4096, len(part.image_url.url))
                    else:
                        total += len(part.model_dump_json().encode("utf-8"))
        if tools:
            total += self._tokens(json.dumps(tools, ensure_ascii=False))
        return total

    def _limits(self, max_tokens: int) -> tuple[int, int]:
        limit = self.llm.config.max_input_tokens
        output = max(1, min(max_tokens, limit // 4))
        return limit - output, output

    def _compact_payload(
        self, payload: dict[str, JsonValue], stage: int
    ) -> dict[str, JsonValue]:
        compacted = dict(payload)
        compacted["context_omissions"] = {
            "stage": stage,
            "notice": "Context excerpts are shortened, not absent. Reopen original evidence by global number; use read_research_state for tool receipts and list_researchers for task state.",
        }
        evidence = compacted.get("evidence")
        if isinstance(evidence, list):
            items = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                excerpt = dict(item)
                text = str(excerpt.get("text", ""))
                excerpt["text"] = (
                    text[: max(0, 1000 // (stage + 1))] if stage < 3 else ""
                )
                excerpt["truncated"] = bool(text) or bool(excerpt.get("truncated"))
                items.append(excerpt)
            if stage >= 5:
                retained = items[: max(1, 12 // (stage - 3))]
                compacted["evidence_omitted"] = {
                    "count": len(items) - len(retained),
                    "global_number_range": [
                        items[0].get("citation"),
                        items[-1].get("citation"),
                    ]
                    if items
                    else [],
                    "reopen": "read_evidence",
                }
                items = retained
            compacted["evidence"] = items
        receipts = compacted.get("receipts")
        if isinstance(receipts, list):
            retained = receipts[-max(0, 8 - stage) :] if stage < 8 else []
            compacted["receipts_omitted"] = len(receipts) - len(retained)
            sanitized = []
            for item in retained:
                if not isinstance(item, dict):
                    continue
                receipt = dict(item)
                outcome = receipt.get("outcome")
                if isinstance(outcome, dict):
                    outcome = dict(outcome)
                    outcome["data"] = {
                        "context_omitted": True,
                        "reopen": "read_research_state",
                    }
                    outcome["summary"] = str(outcome.get("summary", ""))[:300]
                    receipt["outcome"] = outcome
                receipt["call"] = compact_json(receipt.get("call", {}), max_chars=1200)
                sanitized.append(receipt)
            compacted["receipts"] = sanitized
        tasks = compacted.get("research_tasks")
        if isinstance(tasks, list):
            compacted["research_tasks"] = [
                {
                    key: item[key]
                    for key in ("task_id", "status", "task", "parent_task_id")
                    if key in item
                }
                for item in tasks
                if isinstance(item, dict)
            ]
            compacted["task_details_omitted"] = True
        return compacted

    def _fit(
        self,
        instruction: str,
        data: str,
        tools: list[dict[str, JsonValue]],
        *,
        max_tokens: int,
        research: bool = False,
        repair: str | None = None,
        turns: list[ResearchTurn] | None = None,
    ) -> tuple[list[ChatCompletionMessage], list[dict[str, JsonValue]], int]:
        ceiling, output = self._limits(max_tokens)
        selected = tools
        try:
            payload = parse_json_object(data)
        except ValueError:
            payload = None

        def prompt(text: str) -> list[ChatCompletionMessage]:
            result: list[ChatCompletionMessage] = [SystemMessage(content=instruction)]
            for turn in turns or []:
                result.extend([turn.assistant, *turn.results])
            result.append(UserMessage(content=text))
            if repair:
                result.append(UserMessage(content=repair))
            return result

        initial = prompt(data)
        # Remove complete old call/result pairs; never leave orphan provider calls.
        turns = list(turns or [])
        while turns and self._input_cost(initial, selected) > ceiling:
            turns.pop(0)
            initial = prompt(data)
        if self._input_cost(initial, selected) <= ceiling:
            return initial, selected, output
        if payload is None:
            raise RunStopped(
                "The complete question and instructions exceed the selected model input limit"
            )
        if research and selected:
            needed = {
                "discover_tools",
                "read_evidence",
                "read_research_state",
                "wait_researcher",
                "report_progress",
            }
            receipts = payload.get("receipts")
            if isinstance(receipts, list):
                for receipt in receipts[-4:]:
                    call = receipt.get("call") if isinstance(receipt, dict) else None
                    if isinstance(call, dict):
                        needed.add(str(call.get("name", "")))
                        args = call.get("arguments")
                        if (
                            call.get("name") == "discover_tools"
                            and isinstance(args, dict)
                            and isinstance(args.get("names"), list)
                        ):
                            needed.update(str(name) for name in args["names"])
            selected = [
                tool
                for tool in tools
                if isinstance(function := tool.get("function"), dict)
                and function.get("name") in needed
            ]
            if not any(
                isinstance(function := tool.get("function"), dict)
                and function.get("name") == "discover_tools"
                for tool in selected
            ):
                selected = tools
            else:
                omitted: list[JsonValue] = []
                for tool in tools:
                    definition = tool.get("function")
                    if tool not in selected and isinstance(definition, dict):
                        omitted.append(definition.get("name"))
                payload["capability_context"] = {
                    "definitions_omitted": omitted,
                    "reopen": "discover_tools",
                }
        # Verification/final synthesis preserves all cited operative text. Only
        # uncited supplemental evidence may be removed to fit the selected model.
        evidence = payload.get("evidence")
        if not research and isinstance(evidence, str):
            try:
                records = json.loads(evidence)
            except ValueError:
                records = None
            if isinstance(records, list):
                cited = {
                    int(n)
                    for key in ("claim", "draft")
                    for n in re.findall(r"\[(\d+)\]", str(payload.get(key, "")))
                }
                if cited:
                    payload["evidence"] = json.dumps(
                        [
                            item
                            for item in records
                            if isinstance(item, dict) and item.get("citation") in cited
                        ],
                        ensure_ascii=False,
                    )
                    payload["supplemental_evidence_omitted"] = True
                fitted = prompt(json.dumps(payload, ensure_ascii=False))
                if self._input_cost(fitted, selected) <= ceiling:
                    return fitted, selected, output
                raise RunStopped(
                    "Complete cited evidence and scenario exceed the selected model input limit; reopenable originals were preserved"
                )
        for stage in range(9) if research else range(1):
            current = self._compact_payload(payload, stage) if research else payload
            fitted = prompt(json.dumps(current, ensure_ascii=False))
            if self._input_cost(fitted, selected) <= ceiling:
                return fitted, selected, output
        raise RunStopped(
            "Irreducible scenario and capability context exceed the selected model input limit"
        )

    def _invoke(
        self,
        prompt: list[ChatCompletionMessage],
        tools: list[dict[str, JsonValue]],
        flow: LLMFlow,
        *,
        max_tokens: int,
        research: bool,
    ) -> ModelResponse:
        check = (
            self.context.check_research_active
            if research
            else self.context.check_active
        )
        deadline = self.context.research_deadline if research else self.context.deadline
        for attempt in range(3):
            check()
            try:
                return self._invoke_once(
                    prompt, tools, flow, max_tokens=max_tokens, research=research
                )
            except Exception as error:
                if attempt == 2 or not is_retryable_provider_error(error):
                    raise
                check()
                delay = provider_retry_delay(error, attempt)
                if delay >= deadline - time.monotonic():
                    raise RunStopped(
                        "Provider retry would exceed the remaining run deadline"
                    ) from error
                if research:
                    self.context.budget.consume_research_decision()
                else:
                    self.context.budget.consume("decisions")
                retry_at = time.monotonic() + delay
                while time.monotonic() < retry_at:
                    check()
                    time.sleep(min(0.05, max(0, retry_at - time.monotonic())))
        raise AssertionError("Provider retry loop did not terminate")

    def _invoke_once(
        self,
        prompt: list[ChatCompletionMessage],
        tools: list[dict[str, JsonValue]],
        flow: LLMFlow,
        *,
        max_tokens: int,
        research: bool,
    ) -> ModelResponse:
        response_model = structured_model(flow)
        from onyx.asv3.evidence import EvidenceLedger

        ledger = self.context.services.get("evidence")
        records: list[dict[str, JsonValue]] = []
        if isinstance(ledger, EvidenceLedger):
            for message in prompt:
                content = message.content
                texts = (
                    [content]
                    if isinstance(content, str)
                    else [
                        part.text
                        for part in content or []
                        if isinstance(part, TextContentPart)
                    ]
                )
                for text in texts:
                    try:
                        payload = parse_json_object(text)
                    except ValueError:
                        continue
                    evidence = payload.get("evidence", [])
                    if isinstance(evidence, str):
                        try:
                            evidence = json.loads(evidence)
                        except ValueError:
                            continue
                    if isinstance(evidence, list):
                        records.extend(
                            item for item in evidence if isinstance(item, dict)
                        )
                    outcome = payload.get("outcome")
                    data = outcome.get("data") if isinstance(outcome, dict) else None
                    if isinstance(data, dict):
                        records.append(data)
        with (
            model_slot(self.context, research=research),
            llm_generation_span(self.llm, flow, prompt, tools or None) as span,
        ):
            response = self.llm.invoke(
                prompt=prompt,
                # Provider normalization must not rewrite canonical validation schemas.
                tools=copy.deepcopy(tools) if tools else None,
                tool_choice=ToolChoiceOptions.AUTO if tools else ToolChoiceOptions.NONE,
                structured_response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": copy.deepcopy(
                            _portable_structured_output_schema(
                                response_model.model_json_schema()
                            )
                        ),
                        "strict": False,
                    },
                }
                if response_model is not None
                else None,
                max_tokens=max_tokens,
                timeout_override=max(
                    1,
                    min(
                        120,
                        int(
                            (
                                self.context.research_deadline
                                if research
                                else self.context.deadline
                            )
                            - time.monotonic()
                        ),
                    ),
                ),
                reasoning_effort=self.reasoning_effort,
                user_identity=self.user_identity,
            )
            record_llm_response(span, response)
            self.last_call_id = span.span_id
            if isinstance(ledger, EvidenceLedger):
                ledger.record_delivery(span.span_id, flow.value, records)
        self.context.check_active()
        return response

    def invoke_text(
        self,
        instruction: str,
        data: str,
        flow: LLMFlow,
        *,
        max_tokens: int = 6000,
        consume_budget: bool = True,
    ) -> str:
        self.context.check_active()
        response_model = structured_model(flow)
        if response_model is not None:
            instruction += (
                "\nReturn a JSON object conforming to this complete schema:\n"
                + json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            )
        if consume_budget:
            self.context.budget.consume("decisions")
        prompt, tools, output = self._fit(instruction, data, [], max_tokens=max_tokens)
        response = self._invoke(
            prompt, tools, flow, max_tokens=output, research=not consume_budget
        )

        def valid(result: ModelResponse) -> str:
            text = result.choice.message.content or ""
            if not text.strip():
                raise ValueError("ASv3 model returned an empty response")
            if response_model is not None:
                return normalize_structured_response(text, response_model)
            return text

        try:
            return valid(response)
        except ValueError as error:
            if consume_budget:
                self.context.budget.consume("decisions")
            else:
                self.context.budget.consume_research_decision()
            correction = json.dumps(
                {
                    "format_repair": str(error)[:300],
                    "previous_output": (response.choice.message.content or "")[:1000],
                    "instruction": "Repair the output format once. Preserve the original request and sources; do not invent facts or evidence.",
                },
                ensure_ascii=False,
            )
            prompt, tools, output = self._fit(
                instruction, data, [], max_tokens=max_tokens, repair=correction
            )
            return valid(
                self._invoke(
                    prompt, tools, flow, max_tokens=output, research=not consume_budget
                )
            )

    @staticmethod
    def _decision(
        response: ModelResponse, tools: list[dict[str, JsonValue]]
    ) -> Decision:
        definitions = {
            function["name"]: function.get("parameters", {})
            for tool in tools
            if isinstance(function := tool.get("function"), dict)
            and isinstance(function.get("name"), str)
        }
        calls = []
        for call in response.choice.message.tool_calls or []:
            if (
                not call.id
                or not call.function.name
                or not call.function.arguments
                or call.function.name not in definitions
            ):
                raise ValueError("Malformed or unexposed ASv3 tool call")
            args = parse_json_object(call.function.arguments)
            try:
                jsonschema.Draft202012Validator(
                    definitions[call.function.name]
                ).validate(args)
            except jsonschema.ValidationError as error:
                raise ValueError(
                    "Tool arguments violate the exposed schema at "
                    + "/".join(str(p) for p in error.absolute_path)
                ) from error
            calls.append(
                CapabilityCall(name=call.function.name, arguments=args, call_id=call.id)
            )
        if len(calls) > 32 or len({call.call_id for call in calls}) != len(calls):
            raise ValueError("Invalid tool-call count or duplicate identities")
        answer = response.choice.message.content
        if not calls and not (answer or "").strip():
            raise ValueError("ASv3 model produced neither actions nor an answer")
        return Decision(
            calls=calls,
            answer=answer,
            assistant_message=AssistantMessage(
                content=answer,
                tool_calls=[
                    ToolCall(
                        id=call.id,
                        function=NativeFunctionCall.model_validate(
                            call.function.model_dump()
                        ),
                    )
                    for call in response.choice.message.tool_calls or []
                ]
                or None,
            ),
        )

    def decide(self, view: HarnessView) -> Decision:
        self.context.check_research_active()
        payload = view.model_dump(mode="json")
        payload.pop("tools", None)
        payload.pop("turns", None)
        payload.update(
            language=self.context.language,
            conversation=self.history,
            updates=self.updates(),
            research_tasks=self.pending_tasks(),
        )
        instruction = RESEARCHER_PROMPT if self.context.depth else COORDINATOR_PROMPT
        artifacts = [
            artifact
            for receipt in view.receipts[-3:]
            for artifact in receipt.outcome.artifacts
        ]
        if artifacts:
            payload["image_artifact_context"] = {
                "available_ids": [artifact.artifact_id for artifact in artifacts[:8]],
                "notice": "Only attached image parts are visible. An artifact ID does not mean its image was viewed; reopen the source page when an image is omitted from this context.",
            }
        prompt, tools, output = self._fit(
            instruction,
            json.dumps(payload, ensure_ascii=False),
            view.tools,
            max_tokens=6000,
            research=True,
            turns=view.turns,
        )
        content = prompt[-1].content
        assert isinstance(content, str)
        parts: list[ContentPart] = [TextContentPart(text=content)]
        store = self.context.services.get("artifacts")
        for receipt in reversed(view.receipts[-3:]):
            for artifact in receipt.outcome.artifacts:
                if isinstance(store, ArtifactStore):
                    artifact = store.get(artifact.artifact_id) or artifact
                encoded = artifact.metadata.get("base64")
                if (
                    len(parts) < 3
                    and artifact.media_type.startswith("image/")
                    and isinstance(encoded, str)
                ):
                    image = ImageContentPart(
                        image_url=ImageUrlDetail(
                            url=f"data:{artifact.media_type};base64,{encoded}",
                            detail="high",
                        )
                    )
                    trial = [*prompt[:-1], UserMessage(content=[*parts, image])]
                    ceiling, _ = self._limits(output)
                    if self._input_cost(trial, tools) <= ceiling:
                        parts.append(image)
        prompt[-1] = UserMessage(content=parts)
        flow = (
            LLMFlow.ASV3_RESEARCHER if self.context.depth else LLMFlow.ASV3_COORDINATOR
        )
        response = self._invoke(prompt, tools, flow, max_tokens=output, research=True)
        try:
            decision = self._decision(response, tools)
        except ValueError as error:
            self.context.budget.consume_research_decision()
            payload["format_repair"] = {
                "error": str(error)[:300],
                "previous_output": response.model_dump_json()[:3000],
                "instruction": "Repair the tool-call JSON/schema once using exposed tools and original scenario only. Preserve every requested action and keep already valid calls and their IDs unchanged. Do not invent facts, source IDs or evidence.",
            }
            prompt, tools, output = self._fit(
                instruction,
                json.dumps(payload, ensure_ascii=False),
                tools,
                max_tokens=6000,
                research=True,
                turns=view.turns,
            )
            decision = self._decision(
                self._invoke(prompt, tools, flow, max_tokens=output, research=True),
                tools,
            )
            originals = response.choice.message.tool_calls or []
            if originals and len(originals) != len(decision.calls):
                raise ValueError("Schema repair changed the requested action count")
            repaired = {call.call_id: call for call in decision.calls}
            for original in originals:
                try:
                    valid = self._decision(
                        response.model_copy(
                            update={
                                "choice": response.choice.model_copy(
                                    update={
                                        "message": response.choice.message.model_copy(
                                            update={"tool_calls": [original]}
                                        )
                                    }
                                )
                            }
                        ),
                        tools,
                    ).calls[0]
                except ValueError:
                    continue
                if repaired.get(valid.call_id) != valid:
                    raise ValueError("Schema repair altered an already valid action")
        return decision
