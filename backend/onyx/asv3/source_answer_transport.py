"""Bind a writer's same-response body to ordinary guarded publication actions."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable

from pydantic import JsonValue

from onyx.llm.model_response import ModelResponse

_BODY_TERMINALS = frozenset({"submit_answer", "submit_partial_answer"})


def source_research_handoff_tools(
    tools: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Clarify the research role without changing executable terminal schemas."""
    result = copy.deepcopy(tools)
    for tool in result:
        function = tool.get("function")
        if (
            not isinstance(function, dict)
            or function.get("name") not in _BODY_TERMINALS
        ):
            continue
        function["description"] = (
            "For a source-supported legal response, hand off complete concise cited findings "
            "to the separate answer writer, on its own; do not write the full final answer "
            "first. Preserve material conditions, factual application, exceptions, favorable "
            "grounds, procedural stages and exact gaps. All original passages and current "
            "metadata also go to that writer, which may request more research. This is not "
            "publication approval. Without source-supported findings, use this action's "
            "ordinary direct-response behavior. Call on its own. All ordinary source, "
            "metadata and publication validation still apply after the writer's decision."
        )
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        answer = properties.get("answer") if isinstance(properties, dict) else None
        if isinstance(answer, dict):
            answer["description"] = (
                "Complete concise source-cited research findings for the writer, including "
                "material qualifications and gaps; no polished final response. For a direct "
                "conversation, facts-only or wholly unsupported response, supply that response."
            )
    return result


def source_answer_wire_tools(
    tools: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """The writer emits its body once, alongside a metadata-only terminal call."""
    result = copy.deepcopy(tools)
    for tool in result:
        function = tool.get("function")
        if (
            not isinstance(function, dict)
            or function.get("name") not in _BODY_TERMINALS
        ):
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties, required = parameters.get("properties"), parameters.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            continue
        body_fields = {"answer", "retained_answer_id", "retained_answer_edits"}
        for field in body_fields:
            properties.pop(field, None)
        parameters["required"] = [
            field for field in required if field not in body_fields
        ]
        function["description"] = (
            str(function.get("description", ""))
            + " Write the complete answer in this same response's assistant text. "
            "This action carries publication metadata only; the host binds that text "
            "as its answer before all ordinary publication checks. Do not duplicate "
            "the body in arguments or use a pointing sentence. For owned draft edits "
            "use the separate retained-answer action. Call on its own."
        )
    return result


def bind_source_answer_body(
    response: ModelResponse,
    parse_arguments: Callable[[str], dict[str, JsonValue]],
) -> ModelResponse:
    """Use the declared body channel, without inferring support or adding metadata."""
    message = response.choice.message
    calls = message.tool_calls or []
    if (
        len(calls) != 1
        or calls[0].function.name not in _BODY_TERMINALS
        or not (message.content or "").strip()
    ):
        return response
    arguments = parse_arguments(calls[0].function.arguments or "")
    if "retained_answer_id" in arguments:
        return response
    arguments["answer"] = message.content
    call = calls[0].model_copy(
        update={
            "function": calls[0].function.model_copy(
                update={"arguments": json.dumps(arguments, ensure_ascii=False)}
            )
        }
    )
    return response.model_copy(
        update={
            "choice": response.choice.model_copy(
                update={"message": message.model_copy(update={"tool_calls": [call]})}
            )
        }
    )
