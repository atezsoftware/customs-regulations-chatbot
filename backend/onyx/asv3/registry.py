from __future__ import annotations

import copy
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Iterable

import jsonschema
from pydantic import JsonValue

from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    RelatedSourceReview,
    RelatedSourceReviewValidationError,
    related_source_reviews_enabled,
    serial_session_diagnostics_enabled,
)
from onyx.asv3.models import (
    CapabilityCall,
    OriginalEvidenceRead,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.parallel_execution import capability_slot, parallel_execution_enabled
from onyx.asv3.research_state import ResearchState
from onyx.asv3.scenario import question_determinations
from onyx.asv3.shared_reads import SharedReads
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.tracing.answer_graph import graph_step

if TYPE_CHECKING:
    from onyx.asv3.evidence import EvidenceLedger


def _inline_metadata_schema(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    references = schema.get("$defs", {})
    assert isinstance(references, dict)

    def inline(value: JsonValue) -> JsonValue:
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        result: dict[str, JsonValue] = {}
        reference = value.get("$ref")
        if isinstance(reference, str):
            if not reference.startswith("#/$defs/"):
                raise ValueError("Outcome schema references must be local")
            resolved = inline(references[reference.removeprefix("#/$defs/")])
            assert isinstance(resolved, dict)
            result.update(resolved)
        result.update(
            {
                key: inline(item)
                for key, item in value.items()
                if key not in {"$ref", "$defs", "title", "default"}
            }
        )
        return result

    result = inline(schema)
    assert isinstance(result, dict)
    return result


def _outcome_metadata_properties() -> dict[str, JsonValue]:
    properties = _inline_metadata_schema(OutcomeUpdate.model_json_schema())[
        "properties"
    ]
    assert isinstance(properties, dict)
    return {
        "_outcomes": properties["outcomes"],
        "_coverage": {
            "type": "object",
            "properties": {
                "conditions": properties["conditions"],
                "resolutions": properties["resolutions"],
            },
            "additionalProperties": False,
        },
    }


def native_research_bindings(
    context: RunContext,
) -> dict[str, JsonValue] | None:
    state = context.services.get("research_state")
    if not (
        context.services.get("lean_native_mode")
        and context.services.get("research_profile") == "experimental"
        and isinstance(state, ResearchState)
        and (
            serial_session_diagnostics_enabled(context)
            or (
                context.services.get("experimental_parallel") is True
                and context.services.get("independent_question") is True
                and context.depth > 0
            )
        )
    ):
        return None
    return {
        "namespace": "local_update_research",
        "question_ids": [f"q{index}" for index in range(len(state.questions))],
        "determinations": [
            {
                "determination_id": item["determination_id"],
                "question_id": item["question_id"],
                "question": item["question"],
            }
            for item in question_determinations(list(state.questions))
        ],
        "notice": (
            "Use these fixed session IDs for update_research.needs; do not use "
            "the parent assignment's question or outcome IDs."
            if serial_session_diagnostics_enabled(context)
            else "Use these local IDs for update_research.needs. OutcomeMap.question_ids "
            "belong to the immutable parent questions, not these local bindings."
        ),
    }


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
        definitions: list[dict[str, JsonValue]] = []
        task_outcome_ids = context.services.get("task_outcome_ids")
        unbound_parallel_child = (
            context.services.get("research_profile") == "experimental"
            and context.services.get("experimental_parallel") is True
            and context.depth > 0
            and (not isinstance(task_outcome_ids, list) or not task_outcome_ids)
        )
        outcome_properties = (
            _outcome_metadata_properties()
            if context.services.get("lean_native_mode")
            and isinstance(context.services.get("outcome_map"), OutcomeMap)
            and not unbound_parallel_child
            else None
        )
        research_bindings = native_research_bindings(context)
        for spec in self._specs.values():
            if context.corpus_only and spec.external:
                continue
            definition = spec.definition()
            function = definition["function"]
            assert isinstance(function, dict)
            if (
                context.services.get("lean_native_mode")
                and context.services.get("research_profile") == "experimental"
                and spec.name == "read_evidence"
            ):
                function["description"] = (
                    "Reopen recorded original evidence by global citation only when the needed "
                    "range is absent or truncated in this decision's original_evidence_ranges. "
                    "Reuse a fully covered original directly for its claims and citations; "
                    "do not reread it solely to reconfirm wording or obtain a citation. "
                    "A missing continuation or surrounding source context still needs its source tool."
                )
            parameters = function["parameters"]
            assert isinstance(parameters, dict)
            if spec.name == "update_research" and research_bindings is not None:
                schema_definitions = parameters["$defs"]
                assert isinstance(schema_definitions, dict)
                need = schema_definitions["ResearchNeed"]
                assert isinstance(need, dict)
                need_properties = need["properties"]
                assert isinstance(need_properties, dict)
                determinations = research_bindings["determinations"]
                assert isinstance(determinations, list)
                for field, identifiers in (
                    ("question_ids", research_bindings["question_ids"]),
                    (
                        "determination_ids",
                        [
                            item["determination_id"]
                            for item in determinations
                            if isinstance(item, dict)
                        ],
                    ),
                ):
                    binding = need_properties[field]
                    assert isinstance(binding, dict)
                    items = binding["items"]
                    assert isinstance(items, dict)
                    items["enum"] = identifiers
            properties = parameters.setdefault("properties", {})
            assert isinstance(properties, dict)
            if context.services.get("lean_native_mode"):
                properties["_language"] = {
                    "type": "string",
                    "pattern": r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$",
                    "description": "Requested answer language (BCP-47); provide alongside a useful action, without a separate profile call.",
                }
                properties["_notifications"] = {
                    "type": "object",
                    "description": "Optional localized short terminal messages for languages other than Turkish/English: phase maps to [title, natural sentence].",
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string"},
                    },
                }
                properties["_external_requested"] = {
                    "type": "boolean",
                    "description": "True only when the user explicitly requested outside/web sources; host authorization still applies.",
                }
                if outcome_properties is not None and (
                    (
                        context.services.get("research_profile") != "experimental"
                        and context.services.get("asv3_workflow_variant")
                        != ASV3_TUNED_VARIANT
                    )
                    or spec.name
                    in {
                        "search_corpus",
                        "resolve_source",
                        "read_provision",
                        "read_named_provision",
                        "update_research",
                        "research_questions",
                        "submit_answer",
                        "submit_partial_answer",
                        "assemble_answers",
                    }
                ):
                    properties.update(copy.deepcopy(outcome_properties))
                if related_source_reviews_enabled(context) and spec.name in {
                    "submit_answer",
                    "submit_partial_answer",
                    "assemble_answers",
                }:
                    properties["_related_source_reviews"] = {
                        "type": "array",
                        "items": _inline_metadata_schema(
                            RelatedSourceReview.model_json_schema()
                        ),
                    }
            if spec.name not in {
                "update_research",
                "inspect_research",
                "record_scenario",
                "report_progress",
            }:
                if spec.research_need_argument == "_need_id":
                    properties["_need_id"] = {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "description": "Existing material research need ID binding this action to its unresolved question and completion test. May be created by update_research in the same decision.",
                    }
                state = context.services.get("research_state")
                if (
                    spec.requires_research_need
                    and isinstance(state, ResearchState)
                    and state.require_need_bindings
                ):
                    required = parameters.setdefault("required", [])
                    assert isinstance(required, list)
                    required.append(spec.research_need_argument)
                    if spec.research_need_argument == "need_ids":
                        binding = properties["need_ids"]
                        assert isinstance(binding, dict)
                        binding["minItems"] = 1
            public_update = (
                spec.exposes_public_update
                if spec.exposes_public_update is not None
                else not spec.orchestrates
            )
            if public_update and spec.name not in {
                "report_progress",
                "record_scenario",
                "discover_tools",
                "read_research_state",
            }:
                properties["_public_update"] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional short title and natural action description in the question language; no tool names or paths.",
                }
            if context.services.get("lean_native_mode"):
                # Native system prompts explain shared metadata once.
                for name in (
                    "_language",
                    "_notifications",
                    "_external_requested",
                    "_need_id",
                    "_public_update",
                ):
                    metadata = properties.get(name)
                    if isinstance(metadata, dict):
                        metadata.pop("description", None)
            definitions.append(definition)
        return definitions

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def research_binding_gap(
        self, call: CapabilityCall, context: RunContext
    ) -> ToolOutcome | None:
        spec = self.get(call.name)
        state = context.services.get("research_state")
        if (
            spec is not None
            and spec.requires_research_need
            and isinstance(state, ResearchState)
        ):
            if spec.research_need_argument == "need_ids":
                needs = call.arguments.get("need_ids")
                if isinstance(needs, list) and needs:
                    for need in needs:
                        gap = state.action_binding_gap(need)
                        if gap is not None:
                            return gap
                    return None
                return state.action_binding_gap(None)
            return state.action_binding_gap(call.arguments.get("_need_id"))
        return None

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
            if call.argument_error is not None:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Invalid arguments; no action was executed",
                    data={
                        "argument_error": call.argument_error,
                        "instruction": "Send a new call with valid JSON values matching the exposed schema. Keep arrays/objects as JSON values, not quoted JSON strings. Do not change the scenario or invent sources.",
                    },
                )
            binding_gap = self.research_binding_gap(call, context)
            if binding_gap is not None:
                return binding_gap
            try:
                arguments = {
                    key: value
                    for key, value in call.arguments.items()
                    if key
                    not in {
                        "_public_update",
                        "_need_id",
                        "_language",
                        "_notifications",
                        "_external_requested",
                        "_outcomes",
                        "_coverage",
                        *(
                            {"_related_source_reviews"}
                            if related_source_reviews_enabled(context)
                            else set()
                        ),
                    }
                }
                jsonschema.Draft202012Validator(spec.parameters).validate(arguments)
            except jsonschema.ValidationError as error:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Arguments do not match the capability schema",
                    data={"path": [str(part) for part in error.absolute_path]},
                )
            metadata_applied = (
                context.services.pop("applied_outcome_metadata_call", None) is call
            )
            metadata_gap = (
                None if metadata_applied else self.outcome_metadata_gap(call, context)
            )
            if metadata_gap is not None:
                return metadata_gap
            raw_reviews = call.arguments.get("_related_source_reviews")
            if raw_reviews is not None:
                reviews = context.services.get("legal_source_reviews")
                from onyx.asv3.evidence import EvidenceLedger

                ledger = context.services.get("evidence")
                call_id = context.services.get("last_model_call_id")
                try:
                    if (
                        not related_source_reviews_enabled(context)
                        or call.name
                        not in {
                            "submit_answer",
                            "submit_partial_answer",
                            "assemble_answers",
                        }
                        or not isinstance(reviews, LegalSourceReviews)
                        or not isinstance(ledger, EvidenceLedger)
                        or not isinstance(call_id, str)
                        or not isinstance(raw_reviews, list)
                    ):
                        raise ValueError(
                            "Related-source assessments require the exposed experimental terminal action"
                        )
                    reviews.apply(
                        raw_reviews,
                        call_id,
                        context,
                        ledger,
                        detailed_errors=(
                            context.services.get("experimental_parallel") is True
                            or serial_session_diagnostics_enabled(context)
                            or context.services.get("asv3_workflow_variant")
                            == ASV3_TUNED_VARIANT
                        ),
                    )
                except ValueError as error:
                    return ToolOutcome(
                        status=OutcomeStatus.INVALID,
                        summary="Correct only the related-source assessment or read its operative original.",
                        data={
                            "detail": str(error),
                            "invalid_related_source_review": True,
                            **(
                                {"related_source_review_error": error.diagnostic}
                                if isinstance(error, RelatedSourceReviewValidationError)
                                else {}
                            ),
                        },
                    )

            def execute(execution_context: RunContext) -> ToolOutcome:
                if spec.consumes_tool_budget:
                    execution_context.budget.consume("tools")
                slot = (
                    nullcontext()
                    if spec.orchestrates
                    else capability_slot(call.name, execution_context)
                )
                with slot:
                    handler_span = (
                        graph_step("asv3.tool_handler", {"tool": call.name})
                        if parallel_execution_enabled(execution_context)
                        else nullcontext()
                    )
                    with handler_span:
                        if spec.parallel_safe:
                            return spec.handler(arguments, execution_context)
                        with self._serial:
                            execution_context.check_active()
                            return spec.handler(arguments, execution_context)

            shared_reads = context.services.get("shared_reads")
            outcome = (
                shared_reads.run(call.name, arguments, context, execute)
                if (
                    context.services.get("research_profile") == "experimental"
                    or context.services.get("asv3_workflow_variant")
                    == ASV3_TUNED_VARIANT
                )
                and isinstance(shared_reads, SharedReads)
                else execute(context)
            )
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

    def outcome_metadata_gap(
        self, call: CapabilityCall, context: RunContext
    ) -> ToolOutcome | None:
        if "_outcomes" not in call.arguments and "_coverage" not in call.arguments:
            return None
        spec = self.get(call.name)
        if (
            spec is None
            or call.argument_error is not None
            or (spec.external and context.corpus_only)
        ):
            return None
        arguments = {
            key: value
            for key, value in call.arguments.items()
            if key
            not in {
                "_public_update",
                "_need_id",
                "_language",
                "_notifications",
                "_external_requested",
                "_outcomes",
                "_coverage",
                *(
                    {"_related_source_reviews"}
                    if related_source_reviews_enabled(context)
                    else set()
                ),
            }
        }
        if not jsonschema.Draft202012Validator(spec.parameters).is_valid(arguments):
            return None
        from onyx.asv3.evidence import EvidenceLedger

        outcomes = context.services.get("outcome_map")
        ledger = context.services.get("evidence")
        try:
            if not isinstance(outcomes, OutcomeMap) or not isinstance(
                ledger, EvidenceLedger
            ):
                raise ValueError("Outcome metadata is unavailable in this context")
            raw = call.arguments.get("_coverage", {})
            if not isinstance(raw, dict):
                raise ValueError("_coverage must be an object")
            payload = dict(raw)
            if "_outcomes" in call.arguments:
                if "outcomes" in payload:
                    raise ValueError("Use only _outcomes to declare outcomes")
                payload["outcomes"] = call.arguments["_outcomes"]
            update = OutcomeUpdate.model_validate(payload)
            subset = context.services.get("task_outcome_ids")
            if context.depth:
                if not isinstance(subset, list) or not subset:
                    raise ValueError(
                        "Researcher outcome metadata needs an assigned subset"
                    )
                allowed = {value for value in subset if isinstance(value, str)}
                if (
                    any(row.outcome_id not in allowed for row in update.outcomes)
                    or any(set(row.outcome_ids) - allowed for row in update.conditions)
                    or any(row.outcome_id not in allowed for row in update.resolutions)
                ):
                    raise ValueError("Researcher may update only assigned outcomes")
            outcomes.update(update, ledger)
        except ValueError as error:
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Invalid outcome metadata; correct only its bindings or source ranges.",
                data={"detail": str(error), "invalid_outcome_metadata": True},
            )
        return None


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
        if not excerpt:
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND,
                summary="Requested range is outside the recorded original",
            )
        return ToolOutcome(
            status=OutcomeStatus.TRUNCATED
            if start + count < len(item.text)
            else OutcomeStatus.FOUND,
            summary="Original recorded evidence",
            original_reads=[
                OriginalEvidenceRead(
                    citation=raw_number,
                    text_hash=item.text_hash,
                    start_char=start,
                    end_char=start + len(excerpt),
                )
            ],
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
            consumes_tool_budget=False,
            requires_research_need=True,
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
