"""Nonpublication worker handoff with reversible references to retained originals."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import cast

from pydantic import JsonValue

from onyx.asv3.cropped_cost_budget import CroppedCostBudget
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    HarnessResult,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    RunStopped,
)

WORKER_HANDOFF_REASON = "cropped_worker_long_tail_handoff"


class CroppedWorkerHandoff(RunStopped):
    def __init__(self) -> None:
        super().__init__(WORKER_HANDOFF_REASON)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _eligible(context: RunContext) -> bool:
    return (
        context.services.get("asv3_exp_cropped") is True
        and context.depth > 0
        and context.services.get("independent_question") is not True
    )


def _controller(context: RunContext) -> CroppedResearchLimit:
    controller = context.services.get("cropped_research_controller")
    if not isinstance(controller, CroppedResearchLimit):
        raise ValueError("Cropped worker requires its shared research controller")
    controller.check_context(context)
    return controller


def should_handoff_worker(context: RunContext) -> bool:
    if not _eligible(context):
        return False
    controller = _controller(context)
    budget = context.services.get("cropped_cost_budget")
    if isinstance(budget, CroppedCostBudget):
        budget.check_context(context)
    return controller.forced_finish or (
        isinstance(budget, CroppedCostBudget) and budget.finish_reason is not None
    )


def _source(binding: dict[str, JsonValue], ledger: EvidenceLedger) -> str:
    number, start, end = (
        binding.get(key) for key in ("citation", "start_char", "end_char")
    )
    if any(type(value) is not int for value in (number, start, end)):
        raise ValueError("Invalid worker source reference range")
    item = ledger.get(cast(int, number))
    if item is None or not 0 <= cast(int, start) < cast(int, end) <= len(item.text):
        raise ValueError("Worker source reference is unavailable")
    if (item.source_id, item.chunk_id, item.text_hash, _hash(item.metadata)) != (
        binding.get("source_id"),
        binding.get("chunk_id"),
        binding.get("text_hash"),
        binding.get("metadata_hash"),
    ):
        raise ValueError("Worker source reference identity changed")
    literal = item.text[cast(int, start) : cast(int, end)]
    if hashlib.sha256(literal.encode()).hexdigest() != binding.get("range_hash"):
        raise ValueError("Worker source reference literal changed")
    return literal


def _binding(
    number: int, ledger: EvidenceLedger, start: int = 0, end: int | None = None
) -> dict[str, JsonValue]:
    item = ledger.get(number)
    if item is None:
        raise ValueError("Worker receipt refers to missing evidence")
    end = len(item.text) if end is None else end
    if not 0 <= start < end <= len(item.text):
        raise ValueError("Worker receipt original range is invalid")
    return {
        "citation": number,
        "source_id": item.source_id,
        "chunk_id": item.chunk_id,
        "text_hash": item.text_hash,
        "metadata_hash": _hash(item.metadata),
        "start_char": start,
        "end_char": end,
        "range_hash": hashlib.sha256(item.text[start:end].encode()).hexdigest(),
    }


def _spelling(text: str, encoding: str) -> str:
    if encoding == "literal":
        return text
    if encoding in {"json_unicode", "json_ascii"}:
        return json.dumps(text, ensure_ascii=encoding == "json_ascii")[1:-1]
    raise ValueError("Unknown worker source reference encoding")


def _pack(
    value: JsonValue,
    bindings: list[dict[str, JsonValue]],
    ledger: EvidenceLedger,
    *,
    parse_json_strings: bool = True,
) -> dict[str, JsonValue]:
    spellings: dict[str, tuple[int, str]] = {}
    for index, binding in enumerate(bindings):
        text = _source(binding, ledger)
        # Short quotations remain inline; they do not dominate handoff transport.
        if len(text) < 64:
            continue
        for encoding in ("literal", "json_unicode", "json_ascii"):
            spellings.setdefault(_spelling(text, encoding), (index, encoding))
    pattern = (
        re.compile(
            "|".join(
                re.escape(text)
                for text in sorted(spellings, key=lambda text: len(text), reverse=True)
            )
        )
        if spellings
        else None
    )
    references: list[JsonValue] = []
    json_references: list[JsonValue] = []

    def visit(node: JsonValue, path: list[JsonValue]) -> JsonValue:
        if isinstance(node, dict):
            return {key: visit(item, [*path, key]) for key, item in node.items()}
        if isinstance(node, list):
            return [visit(item, [*path, index]) for index, item in enumerate(node)]
        if not isinstance(node, str):
            return node
        # Short argument strings do not justify another structured wrapper.
        if (
            parse_json_strings
            and len(node) >= 512
            and node.startswith(("{", "["))
            and node not in spellings
        ):
            try:
                parsed = json.loads(node)
            except ValueError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                for ensure_ascii, compact in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    if (
                        json.dumps(
                            parsed,
                            ensure_ascii=ensure_ascii,
                            separators=(",", ":") if compact else None,
                            allow_nan=False,
                        )
                        == node
                    ):
                        json_references.append(
                            {
                                "path": path,
                                "ensure_ascii": ensure_ascii,
                                "compact": compact,
                                "value": _pack(
                                    parsed, bindings, ledger, parse_json_strings=False
                                ),
                                "sha256": hashlib.sha256(node.encode()).hexdigest(),
                            }
                        )
                        return ""
        if pattern is None:
            return node
        pieces: list[JsonValue] = []
        previous = 0
        for match in pattern.finditer(node):
            if match.start() > previous:
                pieces.append(node[previous : match.start()])
            index, encoding = spellings[match.group()]
            pieces.append({"source": index, "encoding": encoding})
            previous = match.end()
        if not pieces:
            return node
        if previous < len(node):
            pieces.append(node[previous:])
        if len(json.dumps(pieces, ensure_ascii=False)) >= len(
            json.dumps(node, ensure_ascii=False)
        ):
            return node
        references.append(
            {
                "path": path,
                "pieces": pieces,
                "sha256": hashlib.sha256(node.encode()).hexdigest(),
            }
        )
        return ""

    content = visit(value, [])
    packed: dict[str, JsonValue] = {
        "content": content,
        "source_string_references": references,
        "json_string_references": json_references,
        "original_hash": _hash(value),
    }
    packed["packed_hash"] = _hash(packed)
    return packed


def build_worker_handoff(
    *,
    context: RunContext,
    result: HarnessResult,
    ledger: EvidenceLedger,
    turns: list[ResearchTurn],
    research_state: dict[str, JsonValue],
    working_memory: dict[str, JsonValue],
    draft: str | None,
) -> dict[str, JsonValue]:
    if not _eligible(context):
        raise ValueError("Only Cropped nonpublication workers may hand off")
    _controller(context)
    if context.services.get("evidence") is not ledger:
        raise ValueError("Worker handoff ledger differs from this run")
    if (
        result.stop_reason != WORKER_HANDOFF_REASON
        or result.status != OutcomeStatus.PARTIAL
        or result.answer is not None
    ):
        raise ValueError(
            "Worker handoff requires an actual nonpublication partial stop"
        )
    task_id = context.services.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Worker handoff requires its actual task identity")
    bindings: list[dict[str, JsonValue]] = []
    numbers: list[int] = []
    for receipt in result.receipts:
        for number in receipt.evidence_ids:
            if number not in numbers:
                numbers.append(number)
                bindings.append(_binding(number, ledger))
        for read in receipt.outcome.original_reads:
            binding = _binding(read.citation, ledger, read.start_char, read.end_char)
            if binding["text_hash"] != read.text_hash:
                raise ValueError("Worker reopened range has a different original hash")
            if read.citation not in numbers:
                numbers.append(read.citation)
            if binding not in bindings:
                bindings.append(binding)
    payload: dict[str, JsonValue] = {
        "receipts": [receipt.model_dump(mode="json") for receipt in result.receipts],
        "turns": [turn.model_dump(mode="json") for turn in turns],
        "questions": list(result.questions),
        "facts": list(result.facts),
        "research_state": copy.deepcopy(research_state),
        "working_memory": copy.deepcopy(working_memory),
        "publication_gap": result.publication_gap.model_dump(mode="json")
        if result.publication_gap
        else None,
        "unapproved_draft": draft,
        "draft_approved": False,
        "task_need_ids": copy.deepcopy(
            cast(JsonValue, context.services.get("task_need_ids", []))
        ),
        "task_outcome_ids": copy.deepcopy(
            cast(JsonValue, context.services.get("task_outcome_ids", []))
        ),
        "assignment_id": cast(JsonValue, context.services.get("assignment_id")),
        "parent_task_id": cast(JsonValue, context.services.get("parent_task_id")),
    }
    output: dict[str, JsonValue] = {
        "version": 2,
        "kind": WORKER_HANDOFF_REASON,
        "run_id": context.run_id,
        "scope_hash": _hash(context.scope),
        "task_id": task_id,
        "status": OutcomeStatus.PARTIAL.value,
        "stop_reason": result.stop_reason,
        "absence_proven": False,
        "research_complete": False,
        "publication_approved": False,
        "evidence_numbers": numbers,
        "source_bindings": cast(list[JsonValue], bindings),
        "source_bindings_are_model_delivery_receipts": False,
        "instruction": "Read actual handoff.content and source_string_references together. Each reference replaces only its empty-string path with the ordered literal pieces and exact source pieces. A zero-based source index addresses source_bindings; its citation identifies the retained original and exact start_char/end_char range, available through the root original_evidence. literal/json_unicode/json_ascii describes the original string spelling, not a new source. json_string_references retain actual JSON tool/argument strings as their structured value.content plus its source references; ensure_ascii/compact restores the exact original JSON spelling at that empty-string path. Read every structured field as the actual original observation, with no clipping or invented result. Preserve every actual observation, unresolved status and exception. Assistant reasoning, candidate findings and unapproved_draft are unapproved research; these references do not constitute legal approval or a root model-delivery receipt. The coordinator must evaluate the actual original text and use its normal publication guards.",
        "handoff": _pack(payload, bindings, ledger),
    }
    output["handoff_hash"] = _hash(output)
    # Prove exact reconstruction before accepting the handoff for root integration.
    if expand_worker_handoff(output, ledger=ledger, context=context) != payload:
        raise ValueError("Worker handoff did not reconstruct its actual state")
    return output


def _replace_placeholder(
    content: JsonValue, path: list[JsonValue], literal: str
) -> None:
    if not path:
        raise ValueError("Invalid worker handoff reference path")
    node: JsonValue = content
    for key in path[:-1]:
        node = (
            node.get(key)
            if isinstance(node, dict) and isinstance(key, str)
            else node[key]
            if isinstance(node, list) and type(key) is int and 0 <= key < len(node)
            else None
        )
    key = path[-1]
    if isinstance(node, dict) and isinstance(key, str) and node.get(key) == "":
        node[key] = literal
    elif (
        isinstance(node, list)
        and type(key) is int
        and 0 <= key < len(node)
        and node[key] == ""
    ):
        node[key] = literal
    else:
        raise ValueError("Worker handoff reference does not match its placeholder")


def _unpack(
    packed: dict[str, JsonValue], literals: list[str], *, json_table: bool
) -> JsonValue:
    fields = {"content", "source_string_references", "original_hash", "packed_hash"}
    if json_table:
        fields.add("json_string_references")
    if set(packed) != fields:
        raise ValueError("Invalid worker handoff packed grammar")
    if _hash(
        {key: value for key, value in packed.items() if key != "packed_hash"}
    ) != packed.get("packed_hash"):
        raise ValueError("Worker handoff reference integrity changed")
    content, references = (
        copy.deepcopy(packed.get("content")),
        packed.get("source_string_references"),
    )
    if not isinstance(content, (dict, list)) or not isinstance(references, list):
        raise ValueError("Invalid worker handoff payload")
    for reference in references:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "pieces", "sha256"}
            or not isinstance(reference.get("path"), list)
            or not isinstance(reference.get("pieces"), list)
        ):
            raise ValueError("Invalid worker handoff string reference")
        pieces: list[str] = []
        for piece in cast(list[JsonValue], reference["pieces"]):
            if isinstance(piece, str):
                pieces.append(piece)
            elif (
                isinstance(piece, dict)
                and set(piece) == {"source", "encoding"}
                and type(piece["source"]) is int
                and isinstance(piece["encoding"], str)
                and 0 <= piece["source"] < len(literals)
            ):
                pieces.append(_spelling(literals[piece["source"]], piece["encoding"]))
            else:
                raise ValueError("Invalid worker source reference")
        literal = "".join(pieces)
        if hashlib.sha256(literal.encode()).hexdigest() != reference.get("sha256"):
            raise ValueError("Worker handoff string did not reconstruct")
        _replace_placeholder(content, cast(list[JsonValue], reference["path"]), literal)
    json_references = packed.get("json_string_references", [])
    if not isinstance(json_references, list):
        raise ValueError("Invalid worker JSON string references")
    for reference in json_references:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "ensure_ascii", "compact", "value", "sha256"}
            or not isinstance(reference.get("path"), list)
            or type(reference.get("ensure_ascii")) is not bool
            or type(reference.get("compact")) is not bool
            or not isinstance(reference.get("value"), dict)
        ):
            raise ValueError("Invalid worker JSON string reference")
        nested = cast(dict[str, JsonValue], reference["value"])
        if nested.get("json_string_references") != []:
            raise ValueError("Nested worker JSON string references are not permitted")
        actual = _unpack(nested, literals, json_table=True)
        literal = json.dumps(
            actual,
            ensure_ascii=cast(bool, reference["ensure_ascii"]),
            separators=(",", ":") if reference["compact"] else None,
            allow_nan=False,
        )
        if hashlib.sha256(literal.encode()).hexdigest() != reference.get("sha256"):
            raise ValueError("Worker JSON string did not reconstruct")
        _replace_placeholder(content, cast(list[JsonValue], reference["path"]), literal)
    if _hash(content) != packed.get("original_hash"):
        raise ValueError("Worker handoff actual payload hash changed")
    return content


def expand_worker_handoff(
    data: dict[str, JsonValue], *, ledger: EvidenceLedger, context: RunContext
) -> dict[str, JsonValue]:
    _controller(context)
    version = data.get("version")
    if (
        context.services.get("evidence") is not ledger
        or type(version) is not int
        or version not in (1, 2)
        or (data.get("kind"), data.get("run_id"), data.get("scope_hash"))
        != (WORKER_HANDOFF_REASON, context.run_id, _hash(context.scope))
    ):
        raise ValueError("Foreign worker handoff context")
    if _hash(
        {key: value for key, value in data.items() if key != "handoff_hash"}
    ) != data.get("handoff_hash"):
        raise ValueError("Worker handoff integrity changed")
    packed, raw_bindings = data.get("handoff"), data.get("source_bindings")
    if (
        not isinstance(packed, dict)
        or not isinstance(raw_bindings, list)
        or any(not isinstance(binding, dict) for binding in raw_bindings)
    ):
        raise ValueError("Invalid worker handoff structure")
    bindings = cast(list[dict[str, JsonValue]], raw_bindings)
    literals = [_source(binding, ledger) for binding in bindings]
    content = _unpack(packed, literals, json_table=version == 2)
    if not isinstance(content, dict):
        raise ValueError("Invalid worker handoff actual state")
    return content


def contains_worker_handoff(
    data: dict[str, JsonValue], *, ledger: EvidenceLedger, context: RunContext
) -> bool:
    """Validate only actual worker-outcome or wait/list task schema locations."""
    if context.services.get("asv3_exp_cropped") is not True:
        return False

    def inspect(outcome_data: dict[str, JsonValue], task_id: str | None) -> bool:
        if "cropped_worker_handoff" not in outcome_data:
            return False
        handoff = outcome_data["cropped_worker_handoff"]
        if not isinstance(handoff, dict):
            raise ValueError("Invalid worker handoff in actual outcome")
        declared = handoff.get("task_id")
        if (
            not isinstance(declared, str)
            or not declared
            or (task_id is not None and declared != task_id)
        ):
            raise ValueError("Worker handoff task identity differs from its outcome")
        expand_worker_handoff(handoff, ledger=ledger, context=context)
        return True

    owner = context.services.get("task_id") if context.depth else None
    found = inspect(data, owner if isinstance(owner, str) else None)
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            outcome = task.get("outcome")
            nested = outcome.get("data") if isinstance(outcome, dict) else None
            if isinstance(nested, dict) and "cropped_worker_handoff" in nested:
                task_id = task.get("task_id")
                if (
                    not isinstance(task_id, str)
                    or not task_id
                    or outcome.get("status") != OutcomeStatus.PARTIAL.value
                ):
                    raise ValueError(
                        "Worker handoff requires an actual partial task outcome"
                    )
                found = inspect(nested, task_id) or found
    return found
