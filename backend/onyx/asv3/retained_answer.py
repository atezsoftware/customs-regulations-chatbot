"""Invocation-bound draft references preserve text without reusing its approval."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence

from pydantic import JsonValue

from onyx.asv3.models import Decision, ResearchTurn, RunContext
from onyx.asv3.parallel_execution import parallel_execution_enabled

_TERMINAL = frozenset({"submit_answer", "submit_partial_answer"})
_EOL = r"(?:\r\n|\r(?!\n)|(?<!\r)\n)"
_PARAGRAPH_BREAK = re.compile(rf"{_EOL}[ \t]*{_EOL}(?:[ \t]*{_EOL})*")


def _enabled(context: RunContext) -> bool:
    return parallel_execution_enabled(context)


def _units(draft: str, reference: str) -> list[dict[str, JsonValue]]:
    """Partition exact text; each preceding paragraph owns its following separator."""
    ends: list[int] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(draft):
        if draft[start : match.start()].strip():
            ends.append(match.end())
            start = match.end()
    if start < len(draft):
        if not draft[start:].strip() and ends:
            ends[-1] = len(draft)
        else:
            ends.append(len(draft))
    units: list[dict[str, JsonValue]] = []
    start = 0
    for end in ends:
        text = draft[start:end]
        digest = hashlib.sha256(
            json.dumps([reference, start, end, text], ensure_ascii=False).encode()
        ).hexdigest()
        units.append(
            {
                "unit_id": "retained_unit_" + digest,
                "text": text,
                "start_char": start,
                "end_char": end,
            }
        )
        start = end
    return units


def _edited_answer(draft: str, reference: str, edits: JsonValue) -> str | None:
    if not isinstance(edits, list) or not edits:
        return None
    units = _units(draft, reference)
    known = {unit["unit_id"] for unit in units}
    replacements: dict[str, str] = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"unit_id", "replacement"}:
            return None
        identifier, replacement = edit["unit_id"], edit["replacement"]
        if (
            not isinstance(identifier, str)
            or identifier not in known
            or identifier in replacements
            or not isinstance(replacement, str)
            or not replacement.strip()
        ):
            return None
        replacements[identifier] = replacement.strip()
    answer: list[str] = []
    for unit in units:
        identifier, text = unit["unit_id"], unit["text"]
        assert isinstance(identifier, str) and isinstance(text, str)
        if identifier in replacements:
            leading = len(text) - len(text.lstrip())
            trailing = len(text) - len(text.rstrip())
            text = (
                text[:leading]
                + replacements[identifier]
                + (text[len(text) - trailing :] if trailing else "")
            )
        answer.append(text)
    return "".join(answer)


def _reference(
    context: RunContext, draft: str | None, request: str
) -> dict[str, JsonValue] | None:
    if not _enabled(context) or not draft or not draft.strip() or not request.strip():
        return None
    owner = context.services.get("task_id")
    binding = [
        "retained-answer-v1",
        context.run_id,
        owner if isinstance(owner, str) and owner.strip() else "coordinator",
        request,
        context.scope,
        draft,
    ]
    digest = hashlib.sha256(
        json.dumps(
            binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    identifier = "retained_" + digest
    return {
        "retained_answer_id": identifier,
        "answer_hash": hashlib.sha256(draft.encode()).hexdigest(),
        "answer_characters": len(draft),
        "units": _units(draft, identifier),
        "instruction": (
            "For an unchanged draft_to_repair, supply this retained_answer_id instead of "
            "answer and correct the publication metadata. To correct specific draft units, "
            "also supply retained_answer_edits with their exact unit_id and replacement text. "
            "Retain supported detail and citations within each replacement. Other units "
            "and outer whitespace remain unchanged. For a full rewrite supply "
            "answer instead. Current original-delivery and publication checks still apply."
        ),
    }


def bind_retained_answer(
    tools: Sequence[dict[str, JsonValue]],
    context: RunContext,
    draft: str | None,
    *,
    request: str,
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue] | None]:
    """Copy provider schemas; the resolver enforces exactly one body representation."""
    definitions = copy.deepcopy(list(tools))
    reference = _reference(context, draft, request)
    if reference is None:
        return definitions, None
    bound = False
    for definition in definitions:
        function = definition.get("function")
        if not isinstance(function, dict) or function.get("name") not in _TERMINAL:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties, required = parameters.get("properties"), parameters.get("required")
        if (
            not isinstance(properties, dict)
            or "answer" not in properties
            or not isinstance(required, list)
            or "answer" not in required
            or "retained_answer_id" in properties
        ):
            continue
        properties["retained_answer_id"] = {
            "type": "string",
            "enum": [reference["retained_answer_id"]],
            "description": "Reference the exact owned draft; optional retained_answer_edits change only named units. Supply answer or this reference, never both.",
        }
        units = reference["units"]
        assert isinstance(units, list)
        properties["retained_answer_edits"] = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "enum": [
                            unit["unit_id"] for unit in units if isinstance(unit, dict)
                        ],
                    },
                    "replacement": {"type": "string", "minLength": 1},
                },
                "required": ["unit_id", "replacement"],
                "additionalProperties": False,
            },
            "description": "Only with retained_answer_id; replace named unit content while keeping every other unit and separator unchanged. Empty deletions are rejected.",
        }
        parameters["required"] = [field for field in required if field != "answer"]
        bound = True
    return definitions, reference if bound else None


def resolve_retained_answer(
    decision: Decision, context: RunContext, draft: str | None, *, request: str
) -> Decision:
    """Expand parsed host calls while preserving the provider's actual native message."""
    if not _enabled(context):
        return decision
    reference = _reference(context, draft, request)
    calls = []
    for call in decision.calls:
        if call.name not in _TERMINAL or call.argument_error is not None:
            calls.append(call)
            continue
        arguments = call.arguments
        has_body, has_reference = (
            "answer" in arguments,
            "retained_answer_id" in arguments,
        )
        has_edits = "retained_answer_edits" in arguments
        if has_body and not has_reference and not has_edits:
            calls.append(call)
            continue
        if (
            has_reference
            and not has_body
            and reference is not None
            and arguments["retained_answer_id"] == reference["retained_answer_id"]
        ):
            assert isinstance(draft, str)
            identifier = reference["retained_answer_id"]
            assert isinstance(identifier, str)
            answer = (
                _edited_answer(draft, identifier, arguments["retained_answer_edits"])
                if has_edits
                else draft
            )
            if answer is None:
                calls.append(
                    call.model_copy(
                        update={
                            "argument_error": "Use each current owned unit_id once with nonempty replacement text; keep retained_answer_id and omit answer."
                        }
                    )
                )
                continue
            expanded = {
                key: value
                for key, value in arguments.items()
                if key not in {"retained_answer_id", "retained_answer_edits"}
            }
            expanded["answer"] = answer
            calls.append(call.model_copy(update={"arguments": expanded}))
        else:
            calls.append(
                call.model_copy(
                    update={
                        "argument_error": "Supply answer or the current owned retained_answer_id, never both; a stale or foreign reference cannot publish.",
                    }
                )
            )
    return decision.model_copy(update={"calls": calls})


def project_failed_terminal_turns(
    turns: Sequence[ResearchTurn],
    context: RunContext,
    draft: str | None,
    publication_gap: dict[str, JsonValue] | None,
) -> list[ResearchTurn]:
    """Hide obsolete rejected drafts only when the host supplies the current draft/gap."""
    if not _enabled(context) or not draft or not draft.strip() or not publication_gap:
        return list(turns)
    projected: list[ResearchTurn] = []
    for turn in turns:
        calls = turn.assistant.tool_calls or []
        if (
            not calls
            or len(calls) != len(turn.results)
            or bool(turn.assistant.content and turn.assistant.content.strip())
            or [call.id for call in calls]
            != [result.tool_call_id for result in turn.results]
            or len({call.id for call in calls}) != len(calls)
            or any(call.function.name not in _TERMINAL for call in calls)
        ):
            projected.append(turn)
            continue
        rejected = True
        for result in turn.results:
            try:
                payload = json.loads(result.content)
            except (ValueError, TypeError, RecursionError):
                rejected = False
                break
            outcome = payload.get("outcome") if isinstance(payload, dict) else None
            data = outcome.get("data") if isinstance(outcome, dict) else None
            if (
                not isinstance(payload, dict)
                or not isinstance(outcome, dict)
                or outcome.get("status") not in {"invalid", "partial", "denied"}
                or not isinstance(data, dict)
                or not data
                or payload.get("evidence_ids")
                or payload.get("original_evidence")
                or outcome.get("evidence")
                or outcome.get("original_reads")
                or outcome.get("artifacts")
                or any(
                    key in data
                    for key in {
                        "submitted_answer",
                        "assembled_answer",
                        "parallel_answer_receipt",
                        "serial_session_state",
                    }
                )
            ):
                rejected = False
                break
        if not rejected:
            projected.append(turn)
    return projected
