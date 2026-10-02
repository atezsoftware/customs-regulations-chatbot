from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Iterable

import jsonschema
from pydantic import JsonValue

from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)

if TYPE_CHECKING:
    from onyx.asv3.evidence import EvidenceLedger


class CapabilityRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._serial = threading.RLock()
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate capability: {spec.name}")
        jsonschema.Draft202012Validator.check_schema(spec.parameters)
        self._specs[spec.name] = spec

    def definitions(self, context: RunContext) -> list[dict[str, JsonValue]]:
        return [
            spec.definition()
            for spec in self._specs.values()
            if not (context.corpus_only and spec.external)
        ]

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def dispatch(self, call: CapabilityCall, context: RunContext) -> ToolOutcome:
        try:
            context.check_active()
            spec = self._specs.get(call.name)
            if spec is None:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID, summary="Unknown capability"
                )
            if spec.external and context.corpus_only:
                return ToolOutcome(
                    status=OutcomeStatus.DENIED,
                    summary="External access is disabled for this run",
                )
            try:
                jsonschema.Draft202012Validator(spec.parameters).validate(
                    call.arguments
                )
            except jsonschema.ValidationError as error:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Arguments do not match the capability schema",
                    data={"path": [str(part) for part in error.absolute_path]},
                )
            context.budget.consume("tools")
            acquired = False
            if not spec.orchestrates:
                while not acquired:
                    context.check_active()
                    acquired = context.budget.tool_slots.acquire(timeout=0.05)
            try:
                if spec.parallel_safe:
                    outcome = spec.handler(call.arguments, context)
                else:
                    with self._serial:
                        context.check_active()
                        outcome = spec.handler(call.arguments, context)
            finally:
                if acquired:
                    context.budget.tool_slots.release()
            context.check_active()
            return outcome
        except RunStopped as error:
            return ToolOutcome(
                status=OutcomeStatus.CANCELLED
                if context.is_cancelled()
                else OutcomeStatus.TRUNCATED,
                summary=str(error),
            )
        except Exception:
            # Provider and access details remain in server traces, not model-visible text.
            import logging

            logging.getLogger(__name__).exception(
                "ASv3 capability %s failed", call.name
            )
            return ToolOutcome(
                status=OutcomeStatus.ERROR,
                summary="Capability execution failed; try a different supported method",
            )


def build_core_specs(
    registry: CapabilityRegistry,
    evidence: "EvidenceLedger",
    state_provider: "Callable[[], dict[str, JsonValue]]",
) -> list[ToolSpec]:
    def discover(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        names = args.get("names")
        definitions = registry.definitions(context)
        if isinstance(names, list):
            matching: list[dict[str, JsonValue]] = []
            for definition in definitions:
                function = definition.get("function")
                if isinstance(function, dict) and function.get("name") in names:
                    matching.append(definition)
            definitions = matching
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Authorized capability definitions",
            data={"tools": definitions},
        )

    def read(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        raw_number = args["citation"]
        assert isinstance(raw_number, int)
        item = evidence.get(raw_number)
        if item is None:
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND, summary="Evidence is not recorded"
            )
        start = args.get("start_char", 0)
        count = args.get("num_chars", 16000)
        assert isinstance(start, int) and isinstance(count, int)
        excerpt = item.text[start : start + count]
        return ToolOutcome(
            status=OutcomeStatus.TRUNCATED
            if start + count < len(item.text)
            else OutcomeStatus.FOUND,
            summary="Original recorded evidence",
            data={
                "citation": raw_number,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text_hash": item.text_hash,
                "text": excerpt,
                "start_char": start,
                "total_chars": len(item.text),
                "question_ids": item.question_ids,
            },
        )

    def inspect(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        number = args["citation"]
        assert isinstance(number, int)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Observed evidence path",
            data=evidence.inspect(number),
        )

    def state(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        from onyx.asv3.working_memory import WorkingMemory

        snapshot = state_provider()
        result = {
            key: value
            for key, value in snapshot.items()
            if key
            in {
                "run_id",
                "questions",
                "facts",
                "budget",
                "pending_calls",
                "pending_call_count",
            }
        }
        memory = context.services.get("working_memory")
        locator_ids = args.get("locator_ids")
        if isinstance(memory, WorkingMemory):
            result["working_locators"] = memory.view(
                offset=int(str(args.get("locator_offset", 0))),
                locator_ids=[str(value) for value in locator_ids]
                if isinstance(locator_ids, list)
                else None,
            )
        receipts = snapshot.get("receipts")
        if isinstance(receipts, list):
            references: list[JsonValue] = []
            requested = args.get("receipt_ids")
            if isinstance(requested, list):
                receipts = [
                    receipt
                    for receipt in receipts
                    if isinstance(receipt, dict)
                    and isinstance(call := receipt.get("call"), dict)
                    and call.get("call_id") in requested
                ]
            else:
                offset = args.get("receipt_offset")
                if isinstance(offset, int):
                    result["receipt_count"] = len(receipts)
                    result["next_receipt_offset"] = min(len(receipts), offset + 20)
                    receipts = receipts[offset : offset + 20]
                else:
                    receipts = receipts[-20:]
            for receipt in receipts:
                if not isinstance(receipt, dict):
                    continue
                call, outcome = receipt.get("call"), receipt.get("outcome")
                if not isinstance(call, dict) or not isinstance(outcome, dict):
                    continue
                references.append(
                    {
                        "name": call.get("name"),
                        "receipt_id": call.get("call_id"),
                        "arguments": call.get("arguments"),
                        "status": outcome.get("status"),
                        "summary": str(outcome.get("summary", ""))[:300],
                        "evidence_numbers": receipt.get("evidence_ids", []),
                        "data": outcome.get("data", {}),
                    }
                )
            from onyx.asv3.artifacts import compact_json

            result["receipts"] = compact_json(references, max_chars=9000)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Research state and shared budget",
            data=result,
        )

    integer: dict[str, JsonValue] = {"type": "integer", "minimum": 1}
    return [
        ToolSpec(
            name="discover_tools",
            description="Inspect authorized capability schemas. Optional names narrow the catalog.",
            parameters={
                "type": "object",
                "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                "additionalProperties": False,
            },
            handler=discover,
        ),
        ToolSpec(
            name="read_evidence",
            description="Read immutable original evidence by global citation number; use offsets for complete text.",
            parameters={
                "type": "object",
                "properties": {
                    "citation": integer,
                    "start_char": {"type": "integer", "minimum": 0},
                    "num_chars": {"type": "integer", "minimum": 1, "maximum": 16000},
                },
                "required": ["citation"],
                "additionalProperties": False,
            },
            handler=read,
        ),
        ToolSpec(
            name="inspect_evidence_path",
            description="Inspect observed recording and final inclusion of a cited item. Unknown stages are not invented.",
            parameters={
                "type": "object",
                "properties": {"citation": integer},
                "required": ["citation"],
                "additionalProperties": False,
            },
            handler=inspect,
        ),
        ToolSpec(
            name="read_research_state",
            description="Reopen exact retained locator or receipt IDs, or page locators. Inspect current questions/facts/budget without replaying audit history. Locator leads are not legal evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "locator_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 100,
                    },
                    "locator_offset": {"type": "integer", "minimum": 0},
                    "receipt_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                    },
                    "receipt_offset": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            handler=state,
        ),
    ]
