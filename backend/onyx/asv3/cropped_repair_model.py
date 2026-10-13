"""Isolated single-call evidence checks and retained-answer integration for Cropped."""

from __future__ import annotations

import copy
import json
from typing import cast

from pydantic import BaseModel, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import Decision, HarnessView, RunStopped
from onyx.asv3.retained_answer import resolve_retained_answer
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import ChatCompletionMessage
from onyx.prompts.asv3_cropped_repair import CROPPED_REPAIR_INTEGRATION
from onyx.tracing.flows import LLMFlow

_SERVICES = frozenset(
    {
        "asv3_exp_cropped",
        "asv3_workflow_variant",
        "research_profile",
        "lean_native_mode",
        "explicit_research_temperature",
        "task_id",
        "task_outcome_ids",
        "independent_question",
        "evidence",
        "scenario_state",
        "outcome_map",
        "registry",
        "authority_requirements",
        "legal_source_reviews",
        "legal_source_navigation",
        "assistant_instructions",
        "session_research",
        "cropped_user_context",
        "cropped_cost_budget",
        "cropped_task_packet_store",
        "cropped_cost_phase",
        "research_state",
        "cropped_research_controller",
    }
)
_TERMINALS = frozenset({"submit_answer", "submit_partial_answer"})
_RETAINED_TERMINALS = frozenset(
    {"submit_retained_answer", "submit_retained_partial_answer"}
)


class _FreshRepairModel(ResearchModel):
    _generation_started = False
    _integration_instruction = ""

    def _research_instruction(self) -> str:
        instruction = super()._research_instruction()
        if self._integration_instruction:
            instruction += "\n\n" + self._integration_instruction
        return instruction

    def _invoke_once(
        self,
        prompt: list[ChatCompletionMessage],
        tools: list[dict[str, JsonValue]],
        flow: LLMFlow,
        *,
        max_tokens: int,
        research: bool,
        structured: bool = True,
        response_model: type[BaseModel] | None = None,
        call_llm: LLM | None = None,
        native_catalogue: _NativeOriginalCatalogue | None = None,
    ) -> ModelResponse:
        if self._generation_started:
            raise RunStopped("Cropped focused assessment permits one model call")
        self._generation_started = True
        return super()._invoke_once(
            prompt,
            tools,
            flow,
            max_tokens=max_tokens,
            research=research,
            structured=structured,
            response_model=response_model,
            call_llm=call_llm,
            native_catalogue=native_catalogue,
        )


def fresh_model(model: ResearchModel) -> ResearchModel:
    """Share source ownership and execution fences without sharing writer receipts."""
    if (
        model.context.services.get("asv3_exp_cropped") is not True
        or model.context.services.get("asv3_exp_cheap") is True
    ):
        raise ValueError("Focused repair requires the isolated Cropped workflow")
    context = copy.copy(model.context)
    context.services = {
        key: value for key, value in model.context.services.items() if key in _SERVICES
    }
    context.services["cropped_cost_phase"] = "repair"
    context.services["provider_max_attempts"] = 1
    context.services["provider_compatibility_attempts"] = 1
    fresh = _FreshRepairModel(
        model.llm,
        context,
        user_identity=model.user_identity,
        reasoning_effort=model.reasoning_effort,
        token_counter=model.token_counter,
        lean_native_mode=model.lean_native_mode,
    )
    fresh._native_output_capacity = model._native_output_capacity
    return fresh


def _originals(
    model: ResearchModel,
    view: HarnessView,
    records: list[dict[str, JsonValue]],
) -> tuple[list[dict[str, JsonValue]], list[int]]:
    ledger = model.context.services.get("evidence")
    if not isinstance(ledger, EvidenceLedger):
        raise RunStopped("Cropped integration requires its original evidence ledger")
    numbers = set(view.required_evidence_numbers)
    numbers.update(extract_citation_numbers(view.draft_to_repair or ""))
    for row in view.original_evidence:
        number, text, start = (
            row.get("citation"),
            row.get("text"),
            row.get("start_char", 0),
        )
        if (
            type(number) is not int
            or not isinstance(text, str)
            or type(start) is not int
        ):
            raise RunStopped("Cropped integration received an invalid original passage")
        item = ledger.get(number)
        if (
            item is None
            or start < 0
            or item.text[start : start + len(text)] != text
            or any(
                row.get(key) != getattr(item, key)
                for key in ("source_id", "chunk_id", "text_hash")
            )
        ):
            raise RunStopped("Cropped integration original identity or text changed")
        numbers.add(number)
    for record in records:
        status = record.get("status")
        if (
            not isinstance(status, str)
            or status not in {"found", "not_found", "incomplete", "failed"}
            or record.get("absence_proven", False) is not False
        ):
            raise RunStopped("Cropped integration requires completed evidence checks")
        witnessed = record.get("evidence_numbers", [])
        if not isinstance(witnessed, list) or any(
            type(n) is not int or n < 1 for n in witnessed
        ):
            raise RunStopped("Cropped integration received invalid witness citations")
        numbers.update(cast(list[int], witnessed))
    if any(
        type(number) is not int or number < 1 or ledger.get(number) is None
        for number in numbers
    ):
        raise RunStopped("Cropped integration lacks a required recorded original")
    for number in numbers:
        item = ledger.get(number)
        assert item is not None
        canonical = item.metadata.get("canonical_metadata")
        if item.metadata.get("truncated") is True or (
            isinstance(canonical, dict) and canonical.get("truncated") is True
        ):
            raise RunStopped("Cropped integration has an incomplete canonical original")
    required = sorted(numbers)
    full = cast(
        list[dict[str, JsonValue]],
        json.loads(
            ledger.serialize_records(
                required,
                required=required,
                max_chars=None,
                include_witness_spans=any(
                    row.get("witness_spans") for row in view.original_evidence
                ),
            )
        ),
    )
    return full, required


def integrate_repair(
    model: ResearchModel,
    view: HarnessView,
    records: list[dict[str, JsonValue]],
) -> Decision:
    """Make one owned-draft decision with fresh actual delivery of every retained original."""
    gap = view.publication_gap
    if not view.draft_to_repair or not gap or not records:
        raise RunStopped(
            "Cropped integration requires an owned rejected draft and checks"
        )
    originals, required = _originals(model, view, records)
    fresh = fresh_model(model)
    instructions = fresh.context.services.get("assistant_instructions")
    fresh.context.services["assistant_instructions"] = (
        instructions + "\n\n" if isinstance(instructions, str) and instructions else ""
    ) + CROPPED_REPAIR_INTEGRATION
    if (
        fresh.context.depth
        and fresh.context.services.get("independent_question") is not True
    ):
        assert isinstance(fresh, _FreshRepairModel)
        fresh._integration_instruction = CROPPED_REPAIR_INTEGRATION
    prepared = view.model_copy(deep=True)
    prepared.turns = []
    prepared.receipts = []
    prepared.original_evidence = originals
    prepared.required_evidence_numbers = required
    prepared.tools = [
        tool
        for tool in prepared.tools
        if isinstance(function := tool.get("function"), dict)
        and function.get("name") in _TERMINALS
    ]
    prepared.publication_gap = {
        **gap,
        "cropped_evidence_checks": cast(JsonValue, copy.deepcopy(records)),
        "absence_proven": False,
    }
    user_context = fresh.context.services.get("cropped_user_context")
    if isinstance(user_context, str) and user_context:
        prepared.publication_gap["user_supplied_scenario_context"] = user_context
    prompt, tools, output = fresh._fit_native_decision(prepared, repair_observations=[])
    tools = [
        tool
        for tool in tools
        if isinstance(function := tool.get("function"), dict)
        and function.get("name") in _RETAINED_TERMINALS
    ]
    if not tools:
        raise RunStopped("Cropped integration lacks bound retained-answer terminals")
    try:
        decision = fresh._invoke_decision(
            prepared, prompt, tools, output, flow_override=LLMFlow.ASV3_ANSWER_REPAIR
        )
    finally:
        if fresh.last_call_id is not None:
            model.last_call_id = fresh.last_call_id
            model.last_finish_reason = fresh.last_finish_reason
            model.last_response_truncated = fresh.last_response_truncated
            model.context.services["last_model_call_id"] = fresh.last_call_id
    ledger = fresh.context.services.get("evidence")
    if (
        not isinstance(ledger, EvidenceLedger)
        or fresh.last_call_id is None
        or not set(required).issubset(ledger.completely_delivered(fresh.last_call_id))
    ):
        raise RunStopped("Cropped integration did not deliver every required original")
    expanded = resolve_retained_answer(
        decision, fresh.context, prepared.draft_to_repair, request=prepared.request
    )
    if len(expanded.calls) != 1 or any(
        call.name not in _TERMINALS or call.argument_error is not None
        for call in expanded.calls
    ):
        raise RunStopped(
            "Cropped integration requires one valid retained terminal action"
        )
    return expanded
