"""Read-selection navigation after verified handoff, never source approval."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from onyx.asv3.authority import statute_references
from onyx.asv3.cropped_worker_handoff import expand_worker_handoff
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    HarnessView,
    RunContext,
    ToolReceipt,
    model_evidence_metadata,
)
from onyx.asv3.working_memory import WorkingMemory

if TYPE_CHECKING:
    from onyx.asv3.harness import Harness


def _scope_hash(context: RunContext) -> str:
    return hashlib.sha256(
        json.dumps(context.scope, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CroppedPlanningReceiptSnapshot:
    run_id: str
    scope_hash: str
    request: str
    owner_task_id: str
    depth: int
    receipts: tuple[ToolReceipt, ...]
    user_id: str | None = None

    def validate(self, context: RunContext, request: str) -> None:
        if (
            context.services.get("asv3_exp_cropped") is not True
            or context.depth != 0
            or (
                self.run_id,
                self.scope_hash,
                self.request,
                self.owner_task_id,
                self.depth,
                self.user_id,
            )
            != (
                context.run_id,
                _scope_hash(context),
                request,
                str(context.services.get("task_id") or "root"),
                context.depth,
                str(context.services["user_id"])
                if context.services.get("user_id") is not None
                else None,
            )
        ):
            raise ValueError("Foreign Cropped committed planning receipt snapshot")


def snapshot_cropped_planning_receipts(
    harness: Harness,
) -> CroppedPlanningReceiptSnapshot:
    """Capture committed receipts on the root owner's thread, before view clipping."""
    context = harness.context
    snapshot = CroppedPlanningReceiptSnapshot(
        run_id=context.run_id,
        scope_hash=_scope_hash(context),
        request=harness.request,
        owner_task_id=str(context.services.get("task_id") or "root"),
        depth=context.depth,
        receipts=tuple(receipt.model_copy(deep=True) for receipt in harness.receipts),
        user_id=str(context.services["user_id"])
        if context.services.get("user_id") is not None
        else None,
    )
    snapshot.validate(context, harness.request)
    return snapshot


_CARD_FIELDS = frozenset(
    "title document_type article_no paragraph_no clause_label version_unknown version "
    "revision document_date validity_start validity_end "
    "read_as_of_date article_closure_complete".split()
)
_PLANNING_NOTE = (
    "READ PLANNING ONLY. Select a batch of precise missing original reads from actual "
    "user references, worker recommendations, failed operations, candidates, and NAV. "
    "Do not write or approve a legal answer, certify completeness, or infer absence "
    "from not_found/ambiguous/partial. Source cards and reference NAV are navigation, "
    "not proof of governing law or material applicability. Original bodies excluded "
    "here are UNREAD TO THIS PLANNER; only separately supplied exact witness snippets "
    "have been shown. All originals and full research semantics remain retained for "
    "the subsequent writer and unchanged publication checks. Handoff observations "
    "are unapproved; host validation proves identity, not legal correctness. This is "
    "a selective planning view: successful-read telemetry, repeated batch results, "
    "control/wait history and mechanical transport digests remain in host state and "
    "the full writer view. A batch step reference addresses the same actual parent "
    "call_id plus step_id in operations. Source-card common metadata is inherited "
    "by each unit's numbered metadata variant; unit rows are [citation, chunk_id, "
    "metadata_variant]. Cards are limited to actual read, source-reference witness, "
    "and user-named instrument leads. Other cards and rich date annotations remain "
    "in the final writer view. Cards do not certify article closure or current validity."
    " For statute units with an article locator, chunk IDs remain final-writer-only; "
    "use the literal source_id and article for a whole-provision read. Other units "
    "retain actual chunk IDs. A working-navigation receipt reference means its "
    "actual cursor/status data is already in operations, not that the read is complete."
)

_READ_MARKERS = frozenset(
    "source_id source_name article paragraph clause qualifier subunit_verified "
    "scan_truncated evidence_truncated next_position evidence_next_position "
    "absence_proven has_more next_offset parent_known parent_heading sibling_count "
    "offset sibling_selection_complete article_closure_complete oversized_chunk_id "
    "oversized_chunk_instruction seed_chunk_id shared_read_reuse".split()
)

_NAVIGATION_FIELDS = (
    "research_state",
    "research_tasks",
    "worker_states",
    "assistant_recommendations",
    "operations",
    "recorded_batches",
    "working_navigation",
    "source_reference_navigation",
    "source_cards",
)


def expand_cropped_planning_navigation(
    payload: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Decode exact navigation values; this never creates original delivery."""
    if "planning_navigation_codec" not in payload:
        return copy.deepcopy(payload)
    header = payload["planning_navigation_codec"]
    if not isinstance(header, dict) or header.get("version") != 1:
        raise ValueError("Invalid planning navigation codec")
    layouts, strings, fields = (
        header.get("layouts"),
        header.get("strings"),
        header.get("fields"),
    )
    if (
        not isinstance(layouts, list)
        or any(
            not isinstance(row, list)
            or any(not isinstance(key, str) for key in row)
            or len(set(row)) != len(row)
            for row in layouts
        )
        or not isinstance(strings, list)
        or any(not isinstance(value, str) for value in strings)
        or not isinstance(fields, list)
        or any(field not in _NAVIGATION_FIELDS for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("Invalid planning navigation dictionaries")

    def expand(value: JsonValue) -> JsonValue:
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"s"}:
            index = value["s"]
            if type(index) is not int or not 0 <= index < len(strings):
                raise ValueError("Invalid planning navigation string reference")
            return strings[index]
        if set(value) != {"o"}:
            raise ValueError("Invalid planning navigation object reference")
        row = value["o"]
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Invalid planning navigation object")
        index, values = row
        if (
            type(index) is not int
            or not 0 <= index < len(layouts)
            or not isinstance(values, list)
        ):
            raise ValueError("Invalid planning navigation layout reference")
        keys = layouts[index]
        assert isinstance(keys, list)
        if len(keys) != len(values):
            raise ValueError("Planning navigation field count changed")
        return {
            cast(str, key): expand(item) for key, item in zip(keys, values, strict=True)
        }

    result = copy.deepcopy(payload)
    result.pop("planning_navigation_codec")
    for field in fields:
        assert isinstance(field, str)
        if field not in result:
            raise ValueError("Missing planning navigation field")
        result[field] = expand(result[field])
    return result


def _encode_planning_navigation(
    payload: dict[str, JsonValue], token_counter: Callable[[str], int]
) -> str:
    ordinary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    counts: Counter[str] = Counter()

    def count(value: JsonValue) -> None:
        if isinstance(value, dict):
            for item in value.values():
                count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)
        elif isinstance(value, str) and len(value) >= 20:
            counts[value] += 1

    fields = [field for field in _NAVIGATION_FIELDS if field in payload]
    for field in fields:
        count(payload[field])
    strings = [value for value, number in counts.items() if number > 1]
    string_indexes = {value: index for index, value in enumerate(strings)}
    layouts: list[tuple[str, ...]] = []

    def encode(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            layout = tuple(value)
            if layout not in layouts:
                layouts.append(layout)
            return {
                "o": [layouts.index(layout), [encode(item) for item in value.values()]]
            }
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, str) and value in string_indexes:
            return {"s": string_indexes[value]}
        return value

    result = copy.deepcopy(payload)
    for field in fields:
        result[field] = encode(result[field])
    result["planning_navigation_codec"] = {
        "version": 1,
        "fields": cast(list[JsonValue], fields),
        "layouts": [cast(list[JsonValue], list(row)) for row in layouts],
        "strings": cast(list[JsonValue], strings),
        "instruction": (
            "Only the listed planning fields are encoded. {o:[layout_index,values]} "
            "is an object using layouts[layout_index] as ordered field names. {s:index} "
            "inserts the exact literal string from strings[index]. Lists, null, booleans "
            "and numbers retain their types; absent object fields stay absent. Expand "
            "these exact values before choosing reads; they are not new original texts, "
            "delivery receipts, legal approval or proof of absence."
        ),
    }
    if expand_cropped_planning_navigation(result) != payload:
        raise ValueError("Planning navigation encoding changed its logical packet")
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return encoded if token_counter(encoded) < token_counter(ordinary) else ordinary


def _planning_data(
    name: str, status: JsonValue, data: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    if name in {"search_corpus", "query_corpus"}:
        kept = {
            key: value
            for key, value in data.items()
            if key not in {"search_receipt", "original_question"}
        }
    elif status == "found" and name.startswith("read_"):
        kept = {key: value for key, value in data.items() if key in _READ_MARKERS}
        related = data.get("related_source_candidates")
        if isinstance(related, dict) and (
            related.get("candidates") or related.get("has_more") is True
        ):
            kept["related_source_candidates"] = {
                key: value
                for key, value in related.items()
                if key not in {"instruction", "query_variants"}
            }
    else:
        return data
    omitted = sorted(set(data) - set(kept))
    if omitted:
        kept["additional_data_retained_for_final_writer"] = True
    return kept


def serialize_cropped_read_planning_packet(
    packet: dict[str, JsonValue], token_counter: Callable[[str], int] | None = None
) -> str:
    aliases: dict[str, str] = {}
    for operation in _objects(packet.get("operations")):
        call_id = operation.get("call_id")
        if isinstance(call_id, str):
            aliases.setdefault(call_id, f"recorded_call_{len(aliases) + 1}")

    def visit(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {
                key: aliases.get(item, item)
                if key in {"call_id", "parent_call_id", "receipt_id"}
                and isinstance(item, str)
                else visit(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    visible = cast(dict[str, JsonValue], visit(packet))
    owners: list[JsonValue] = []
    operation_fields = [
        "owner",
        "call_id",
        "name",
        "arguments",
        "status",
        "data",
        "summary",
        "parent_call_id",
        "step_id",
        "evidence_citations",
    ]
    rows: list[JsonValue] = []
    for operation in _objects(visible.get("operations")):
        owner = operation.get("owner_task_id")
        if owner not in owners:
            owners.append(owner)
        rows.append(
            [owners.index(owner), *(operation.get(key) for key in operation_fields[1:])]
        )
    visible["operation_fields"] = cast(list[JsonValue], operation_fields)
    visible["operation_owners"] = owners
    visible["operations"] = rows
    excluded = visible.get("excluded_original_bodies")
    if isinstance(excluded, dict) and isinstance(excluded.get("citations"), list):
        excluded["retained_original_count"] = len(
            cast(list[JsonValue], excluded.pop("citations"))
        )
    excluded_cards = visible.get("excluded_source_cards_retained_for_final_writer")
    if isinstance(excluded_cards, list):
        visible["excluded_source_cards_retained_for_final_writer"] = {
            "count": len(excluded_cards),
            "unread_to_planner": True,
        }
    visible["recorded_call_aliases"] = (
        "Local navigation IDs only; exact physical call IDs remain in host state. They do not denote new calls, model delivery or approval."
    )
    if token_counter is not None:
        return _encode_planning_navigation(visible, token_counter)
    return json.dumps(
        visible, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _objects(value: JsonValue) -> list[dict[str, JsonValue]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _target(locator: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: locator[key]
        for key in (
            "instrument_number",
            "source_name",
            "article",
            "paragraph",
            "clause",
            "qualifier",
        )
        if locator.get(key) is not None
    }


def _request_targets(text: str) -> list[dict[str, JsonValue]]:
    return [
        {
            "instrument_number": ref.number,
            "article": ref.article,
            "paragraph": ref.paragraph,
            "clause": ref.clause,
            "qualifier": ref.qualifier,
            "reference_text": ref.reference_text,
        }
        for ref in statute_references(
            text, strict_reference_boundaries=True, syntactic_reference_binding=True
        )
    ]


def build_cropped_read_planning_packet(
    view: HarnessView,
    context: RunContext,
    *,
    research_tasks: list[dict[str, JsonValue]],
    committed_receipt_snapshot: CroppedPlanningReceiptSnapshot | None = None,
) -> dict[str, JsonValue]:
    """Project actual committed state without modifying originals or delivery receipts."""
    ledger = context.services.get("evidence")
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.depth != 0
        or not isinstance(ledger, EvidenceLedger)
    ):
        raise ValueError("Read planning requires the Cropped root's actual ledger")

    records = [(number, ledger.get(number)) for number in ledger.citation_numbers()]
    original_texts: dict[str, list[int]] = {}
    origins: dict[tuple[str, str | None], list[int]] = {}
    for number, item in records:
        if item is not None:
            original_texts.setdefault(item.text, []).append(number)
            origins.setdefault((item.source_id, item.chunk_id), []).append(number)

    def navigation_value(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {key: navigation_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [navigation_value(item) for item in value]
        if not isinstance(value, str):
            return value
        if value == view.request:
            return {"exact_request_reference": True}
        if value in original_texts:
            return {"unread_original_citations": original_texts[value]}
        # A structured result serialized as text carries real status/cursor semantics.
        if value.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            if isinstance(parsed, (dict, list)):
                return {
                    "recorded_json_value": navigation_value(cast(JsonValue, parsed))
                }
        return value

    operations: list[dict[str, JsonValue]] = []
    batches: list[JsonValue] = []
    operation_keys: set[tuple[str, str]] = set()
    committed_values: dict[tuple[str, str], dict[str, JsonValue]] = {}
    recommendations: list[dict[str, JsonValue]] = []
    memories: list[tuple[str, dict[str, JsonValue]]] = []
    tasks: list[JsonValue] = []
    worker_states: list[JsonValue] = []
    seen_workers: dict[str, dict[str, JsonValue]] = {}

    def add_receipt(raw: dict[str, JsonValue], owner: str) -> None:
        call, outcome = raw.get("call"), raw.get("outcome")
        if not isinstance(call, dict) or not isinstance(outcome, dict):
            raise ValueError("Invalid committed planning receipt")
        call_id, name, arguments, data = (
            call.get("call_id"),
            call.get("name"),
            call.get("arguments"),
            outcome.get("data"),
        )
        if (
            not isinstance(call_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
            or not isinstance(data, dict)
        ):
            raise ValueError("Invalid committed planning operation")
        identity = owner, call_id
        if identity in operation_keys:
            if committed_values[identity] != raw:
                raise ValueError("Conflicting committed planning receipt")
            return
        operation_keys.add(identity)
        committed_values[identity] = copy.deepcopy(raw)
        projected = copy.deepcopy(data)
        raw_tasks = _objects(projected.get("tasks"))
        for task in raw_tasks:
            add_task(task)
            task_outcome = task.get("outcome")
            nested = (
                task_outcome.get("data") if isinstance(task_outcome, dict) else None
            )
            if isinstance(nested, dict) and "cropped_worker_handoff" in nested:
                nested["cropped_worker_handoff"] = {
                    "validated_worker_state_reference": task.get("task_id")
                }
        if name in {"wait_researcher", "list_researchers", "spawn_researcher"}:
            return
        projected_arguments = copy.deepcopy(arguments)
        steps = data.get("steps")
        if name == "compose_tool_calls" and isinstance(steps, dict):
            projected_arguments["steps"] = [
                {
                    "id": step.get("id"),
                    "actual_substep_reference": {
                        "call_id": call_id,
                        "step_id": step.get("id"),
                    },
                }
                if isinstance(steps.get(str(step.get("id"))), dict)
                else copy.deepcopy(step)
                for step in _objects(arguments.get("steps"))
            ]
            projected["steps"] = {
                key: {"actual_substep_reference": {"call_id": call_id, "step_id": key}}
                if isinstance(value, dict)
                else copy.deepcopy(value)
                for key, value in steps.items()
            }
        operation: dict[str, JsonValue] = {
            "owner_task_id": owner,
            "call_id": call_id,
            "name": name,
            "arguments": navigation_value(projected_arguments),
            "status": outcome.get("status"),
            "summary": outcome.get("summary"),
            "data": navigation_value(
                _planning_data(name, outcome.get("status"), projected)
            ),
            "evidence_citations": copy.deepcopy(raw.get("evidence_ids", [])),
        }
        if outcome.get("status") == "found" and name.startswith("read_"):
            operation.pop("summary")
        if name == "compose_tool_calls" and isinstance(steps, dict):
            batches.append(
                {
                    "owner_task_id": owner,
                    "call_id": call_id,
                    "name": name,
                    "status": outcome.get("status"),
                    "arguments_except_steps": {
                        key: copy.deepcopy(value)
                        for key, value in arguments.items()
                        if key != "steps"
                    },
                    "step_order": [
                        step.get("id") for step in _objects(arguments.get("steps"))
                    ],
                    "additional_data": navigation_value(
                        {key: value for key, value in data.items() if key != "steps"}
                    ),
                }
            )
        else:
            operations.append(operation)
        # Preserve the batch and also expose exact executed substeps for target admission.
        if name == "compose_tool_calls" and isinstance(steps, dict):
            for step in _objects(arguments.get("steps")):
                result = steps.get(str(step.get("id")))
                if not isinstance(result, dict):
                    continue
                operations.append(
                    {
                        "owner_task_id": owner,
                        "call_id": call_id,
                        "parent_call_id": call_id,
                        "step_id": step.get("id"),
                        "name": step.get("tool"),
                        "arguments": copy.deepcopy(step.get("arguments", {})),
                        "status": result.get("status"),
                        "summary": result.get("summary"),
                        "data": navigation_value(
                            _planning_data(
                                str(step.get("tool")),
                                result.get("status"),
                                cast(dict[str, JsonValue], result.get("data", {})),
                            )
                            if isinstance(result.get("data", {}), dict)
                            else result.get("data")
                        ),
                    }
                )

    def add_turns(turns: list[dict[str, JsonValue]], owner: str) -> None:
        for turn in turns:
            assistant = turn.get("assistant")
            if not isinstance(assistant, dict):
                raise ValueError("Invalid recorded planning assistant turn")
            content = assistant.get("content")
            if content:
                recommendations.append(
                    {
                        "owner_task_id": owner,
                        "assistant_content": copy.deepcopy(content),
                    }
                )
            for call in _objects(assistant.get("tool_calls")):
                if (owner, str(call.get("id"))) not in operation_keys:
                    recommendations.append(
                        {
                            "owner_task_id": owner,
                            "unexecuted_assistant_tool_call": copy.deepcopy(call),
                        }
                    )
            # Results not backed by a committed receipt must not silently disappear.
            for result in _objects(turn.get("results")):
                if (owner, str(result.get("tool_call_id"))) not in operation_keys:
                    recommendations.append(
                        {
                            "owner_task_id": owner,
                            "unmatched_recorded_result": navigation_value(result),
                        }
                    )

    def add_task(task: dict[str, JsonValue]) -> None:
        outcome = task.get("outcome")
        data = outcome.get("data") if isinstance(outcome, dict) else None
        handoff = data.get("cropped_worker_handoff") if isinstance(data, dict) else None
        if handoff is None:
            return
        owner = task.get("task_id")
        if (
            not isinstance(owner, str)
            or not isinstance(outcome, dict)
            or not isinstance(handoff, dict)
            or handoff.get("task_id") != owner
            or outcome.get("status") != "partial"
        ):
            raise ValueError("Invalid partial task handoff identity")
        state = expand_worker_handoff(handoff, ledger=ledger, context=context)
        if owner in seen_workers:
            if seen_workers[owner] != state:
                raise ValueError("Conflicting worker handoff states")
            return
        seen_workers[owner] = state
        task_copy = copy.deepcopy(task)
        copied_outcome = cast(dict[str, JsonValue], task_copy["outcome"])
        copied_data = cast(dict[str, JsonValue], copied_outcome["data"])
        copied_data["cropped_worker_handoff"] = {
            "validated_worker_state_reference": owner
        }
        tasks.append(navigation_value(task_copy))
        worker_states.append(
            navigation_value(
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"receipts", "turns", "working_memory"}
                }
            )
        )
        for receipt in _objects(state.get("receipts")):
            add_receipt(receipt, owner)
        add_turns(_objects(state.get("turns")), owner)
        memory = state.get("working_memory")
        if isinstance(memory, dict):
            memories.append((owner, memory))

    root_owner = str(context.services.get("task_id") or "root")
    root_receipts = view.receipts
    if committed_receipt_snapshot is not None:
        committed_receipt_snapshot.validate(context, view.request)
        root_receipts = list(committed_receipt_snapshot.receipts)
        complete = {receipt.call.call_id: receipt for receipt in root_receipts}
        for visible in view.receipts:
            actual = complete.get(visible.call.call_id)
            if (
                actual is None
                or actual.call != visible.call
                or actual.outcome.status != visible.outcome.status
                or actual.evidence_ids != visible.evidence_ids
                or not actual.outcome.summary.startswith(visible.outcome.summary)
            ):
                raise ValueError("Clipped planning receipt differs from committed call")
    for task in research_tasks:
        add_task(task)
        if (
            not isinstance(task.get("outcome"), dict)
            or not isinstance(
                cast(dict[str, JsonValue], task["outcome"]).get("data"), dict
            )
            or "cropped_worker_handoff"
            not in cast(
                dict[str, JsonValue],
                cast(dict[str, JsonValue], task["outcome"])["data"],
            )
        ):
            tasks.append(navigation_value(copy.deepcopy(task)))
    for receipt in root_receipts:
        add_receipt(
            cast(dict[str, JsonValue], receipt.model_dump(mode="json")), root_owner
        )
    add_turns(
        [
            cast(dict[str, JsonValue], turn.model_dump(mode="json"))
            for turn in view.turns
        ],
        root_owner,
    )
    root_memory = context.services.get("working_memory")
    if isinstance(root_memory, WorkingMemory):
        memories.append((root_owner, root_memory.export()))

    user_refs = _request_targets(view.request)
    recommendation_text = json.dumps(recommendations, ensure_ascii=False)
    relevant_targets = {
        _key(_target(ref))
        for ref in [*user_refs, *_request_targets(recommendation_text)]
    }
    references: dict[str, dict[str, JsonValue]] = {}
    working_navigation: list[JsonValue] = []
    for owner, memory in memories:
        locators: list[JsonValue] = []
        for locator in _objects(memory.get("locators")):
            if locator.get("kind") != "reference":
                if (
                    locator.get("kind") in {"cursor", "source"}
                    and (owner, str(locator.get("receipt_id"))) in operation_keys
                ):
                    continue
                if locator.get("kind") not in {"evidence", "chunk"} and (
                    locator.get("kind") != "source"
                    or locator.get("status")
                    in {"not_found", "ambiguous", "partial", "error"}
                ):
                    locators.append(copy.deepcopy(locator))
                continue
            target = _target(locator)
            if not target:
                locators.append(copy.deepcopy(locator))
                continue
            key = _key(target)
            entry = references.setdefault(
                key, {"target": target, "witness_citations": [], "exact_witnesses": []}
            )
            source, chunk = (
                locator.get("origin_source_id"),
                locator.get("origin_chunk_id"),
            )
            numbers = origins.get(
                (str(source), chunk if isinstance(chunk, str) else None), []
            )
            citations = cast(list[JsonValue], entry["witness_citations"])
            citations.extend(number for number in numbers if number not in citations)
            literal = locator.get("reference_text")
            if (
                key in relevant_targets
                and isinstance(literal, str)
                and len(literal) <= 800
            ):
                for number in numbers:
                    item = ledger.get(number)
                    if item is not None and literal in item.text:
                        quote: JsonValue = {"citation": number, "source_quote": literal}
                        quotes = cast(list[JsonValue], entry["exact_witnesses"])
                        if quote not in quotes:
                            quotes.append(quote)
        working_navigation.append(
            {
                "owner_task_id": owner,
                **{
                    key: copy.deepcopy(value)
                    for key, value in memory.items()
                    if key
                    not in {
                        "locators",
                        "strategies",
                        "scope_hash",
                        "version",
                        "revision",
                        "unchanged_streak",
                    }
                },
                "locators": locators,
            }
        )

    card_numbers: set[int] = set()
    for receipt in root_receipts:
        if receipt.call.name.startswith("read_") and (
            receipt.outcome.data.get("has_more") is True
            or receipt.outcome.data.get("scan_truncated") is True
            or receipt.outcome.data.get("evidence_truncated") is True
        ):
            card_numbers.update(receipt.evidence_ids)
    user_instruments = {str(ref["instrument_number"]) for ref in user_refs}
    recommendation_instruments = {
        str(ref["instrument_number"]) for ref in _request_targets(recommendation_text)
    }
    for reference in references.values():
        target = cast(dict[str, JsonValue], reference["target"])
        if (
            target.get("instrument_number")
            in user_instruments | recommendation_instruments
        ):
            card_numbers.update(cast(list[int], reference["witness_citations"]))
    for number, item in records:
        if item is not None:
            title = str(model_evidence_metadata(item.metadata).get("title", ""))
            if any(
                re.search(rf"(?<!\d){re.escape(instrument)}(?!\d)", title)
                for instrument in user_instruments
            ):
                card_numbers.add(number)

    source_units: dict[str, list[tuple[int, str | None, dict[str, JsonValue]]]] = {}
    for number, item in records:
        if item is None or number not in card_numbers:
            continue
        metadata = model_evidence_metadata(item.metadata)
        identity: dict[str, JsonValue] = {
            key: value
            for key, value in metadata.items()
            if key in _CARD_FIELDS and value is not None
        }
        source_units.setdefault(item.source_id, []).append(
            (number, item.chunk_id, identity)
        )
    source_cards: list[JsonValue] = []
    for source, units in source_units.items():
        common = {
            key: value
            for key, value in units[0][2].items()
            if all(
                metadata.get(key) == value and key in metadata
                for _, _, metadata in units
            )
        }
        variants: list[JsonValue] = []
        rows: list[JsonValue] = []
        for number, chunk, metadata in units:
            delta: JsonValue = {
                key: value for key, value in metadata.items() if key not in common
            }
            if delta not in variants:
                variants.append(delta)
            statute = (
                metadata.get("document_type") in {"kanun", "law", "statute"}
                and metadata.get("article_no") is not None
            )
            rows.append(
                [
                    number,
                    {"retained_for_final_writer": True} if statute else chunk,
                    variants.index(delta),
                ]
            )
        source_cards.append(
            {
                "source_id": source,
                "common_metadata": common,
                "metadata_variants": variants,
                "units": rows,
            }
        )

    unresolved_reads = any(
        operation.get("name")
        in {
            "read_named_provision",
            "read_provision",
            "read_chunk",
            "read_chunk_context",
            "resolve_source",
        }
        and operation.get("status") in {"not_found", "ambiguous", "partial", "error"}
        for operation in operations
    )
    return {
        "stage": "cropped_precise_read_planning",
        "instruction": _PLANNING_NOTE,
        "planning_targets_available": bool(user_refs or references or unresolved_reads),
        "request": view.request,
        "questions": navigation_value(view.questions),
        "facts": copy.deepcopy(view.facts),
        "research_state": navigation_value(view.research_state),
        "research_tasks": tasks,
        "worker_states": worker_states,
        "assistant_recommendations": recommendations,
        "operations": operations,
        "recorded_batches": batches,
        "working_navigation": working_navigation,
        "working_cursor_markers_already_in_actual_operations": True,
        "user_reference_navigation": cast(list[JsonValue], user_refs),
        "source_reference_navigation": cast(list[JsonValue], list(references.values())),
        "source_cards": source_cards,
        "excluded_source_cards_retained_for_final_writer": [
            number for number, _ in records if number not in card_numbers
        ],
        "excluded_original_bodies": {
            "citations": [number for number, _ in records],
            "unread_to_planner": True,
        },
        "legal_approval": False,
        "corpus_absence_proven": False,
    }


def _key(value: dict[str, JsonValue]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
