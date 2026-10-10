"""Account for every newly read source in batches before interpreting the findings."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue, create_model, model_validator

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_review.contracts import ReadingContractError
from onyx.legal_review.models import PassageSupport, SourceAction, StrictModel
from onyx.legal_review.passages import resolve_passage
from onyx.prompts.legal_review.prompts import SOURCE_ACCOUNTING_PROMPT
from onyx.tracing.flows import LLMFlow

if TYPE_CHECKING:
    from onyx.legal_review.engine import ModelGateway


class CanonicalRead(SourceAction):
    tool: Literal["read_source_range", "read_provision"]


class SourceAssessment(StrictModel):
    slot: str
    disposition: Literal[
        "supports_existing_finding",
        "material_limitation",
        "needs_operative_read",
        "irrelevant",
    ]
    reason: str
    passage_roles: list[
        Literal[
            "operative_rule",
            "operative_disposition",
            "reasoning",
            "quoted_rule",
            "party_submission",
            "procedural_background",
        ]
    ]
    established_effect: str | None
    supports: list[PassageSupport]
    missing_effect: str | None
    content_status: Literal[
        "operative_effect_read", "operative_effect_missing", "not_needed"
    ]
    requested_read: CanonicalRead | None

    @model_validator(mode="after")
    def actionable_missing_effect(self) -> SourceAssessment:
        if self.content_status == "operative_effect_missing" and not (
            self.missing_effect and self.requested_read
        ):
            raise ValueError(
                "A material missing operative effect needs its concrete canonical read"
            )
        if self.requested_read and self.content_status != "operative_effect_missing":
            raise ValueError(
                "A canonical continuation must identify its missing operative effect"
            )
        if self.content_status == "operative_effect_read":
            if not (self.supports and self.established_effect):
                raise ValueError("A read effect needs its supporting original passage")
            if self.disposition in {
                "material_limitation",
                "needs_operative_read",
            } and not set(self.passage_roles) & {
                "operative_rule",
                "operative_disposition",
                "reasoning",
            }:
                raise ValueError(
                    "A quoted rule, submission or background cannot establish the source's own limiting effect"
                )
        return self


class SourceInventory(StrictModel):
    source_assessments: list[SourceAssessment]


def inventory_response_model(
    slots: dict[str, str], originals: list[dict[str, JsonValue]]
) -> type[SourceInventory]:
    references: list[Any] = []
    for original in originals:
        passages = cast(list[dict[str, JsonValue]], original["passages"])
        references.append(
            create_model(
                f"InventoryPassage{original['citation']}",
                __base__=PassageSupport,
                citation=(Literal.__getitem__((original["citation"],)), Field()),
                span_number=(
                    Literal.__getitem__(tuple(row["span_number"] for row in passages)),
                    Field(),
                ),
            )
        )
    reference_type: Any = (
        references[0] if len(references) == 1 else Union[tuple(references)]
    )
    assessment = create_model(
        "BoundSourceAssessment",
        __base__=SourceAssessment,
        slot=(Literal.__getitem__(tuple(slots)), Field()),
        supports=(GenericAlias(list, reference_type), Field()),
    )
    return create_model(
        "BoundSourceInventory",
        __base__=SourceInventory,
        source_assessments=(GenericAlias(list, assessment), Field()),
    )


class SourceAccountant:
    def __init__(
        self, gateway: ModelGateway, ledger: EvidenceLedger, *, parallelism: int = 4
    ) -> None:
        self.gateway = gateway
        self.ledger = ledger
        self.parallelism = max(1, parallelism)
        self._scope = ""
        self._signatures: dict[str, tuple[tuple[int, str], ...]] = {}
        self._assessments: dict[str, dict[str, JsonValue]] = {}

    def snapshot(self) -> list[dict[str, JsonValue]]:
        return list(self._assessments.values())

    def scan(
        self, state: dict[str, JsonValue], *, finalizing: bool = False
    ) -> list[dict[str, JsonValue]]:
        raw = state.get("original_evidence")
        if not isinstance(raw, list):
            raise ValueError("Source accounting requires canonical originals")
        groups: dict[str, list[dict[str, JsonValue]]] = {}
        for original in raw:
            if not isinstance(original, dict):
                raise ValueError("Source accounting requires original records")
            source_id = original.get("source_id")
            if not isinstance(source_id, str):
                raise ValueError(
                    "Source accounting requires canonical source identities"
                )
            groups.setdefault(source_id, []).append(original)
        signatures = {
            source: tuple(
                (cast(int, row["citation"]), cast(str, row["text_hash"]))
                for row in originals
            )
            for source, originals in groups.items()
        }
        scope = json.dumps(
            {key: state.get(key) for key in ("request", "plan", "review_diagnoses")},
            ensure_ascii=False,
            sort_keys=True,
        )
        changed = [
            source
            for source in groups
            if scope != self._scope
            or signatures[source] != self._signatures.get(source)
        ]
        if not changed:
            return self.snapshot()
        self._scope = scope
        for source in changed:
            self._signatures.pop(source, None)
            self._assessments.pop(source, None)
        # Partition work, never omit it. Small inventories stay in one batch.
        count = min(self.parallelism, max(1, len(changed) // 8))
        batches: list[list[str]] = [[] for _ in range(count)]
        loads = [0] * count
        sizes = {
            source: sum(len(str(row.get("passages"))) for row in groups[source])
            for source in changed
        }
        for source in sorted(changed, key=lambda source: sizes[source], reverse=True):
            index = min(range(count), key=lambda index: loads[index])
            batches[index].append(source)
            loads[index] += sizes[source]

        def read_batch(sources: list[str]) -> list[dict[str, JsonValue]]:
            slots = {f"s{index:04d}": source for index, source in enumerate(sources, 1)}
            packet = {
                key: state[key]
                for key in (
                    "request",
                    "history",
                    "plan",
                    "requirements",
                    "review_diagnoses",
                    "corpus_currency",
                    "tools",
                )
                if key in state
            }
            packet["tools"] = [
                definition
                for definition in cast(list[dict], state.get("tools", []))
                if definition["function"]["name"]
                in {"read_source_range", "read_provision"}
            ]
            packet["source_operations"] = [
                row
                for row in cast(
                    list[dict[str, JsonValue]], state.get("source_operations", [])
                )
                if row.get("tool") != "search_corpus"
                and isinstance(arguments := row.get("arguments"), dict)
                and arguments.get("source_id") in sources
            ]
            packet["original_evidence"] = [
                row for source in sources for row in groups[source]
            ]
            packet["source_inventory"] = [
                {
                    "slot": slot,
                    "source_id": source,
                    "title": cast(
                        dict[str, JsonValue], groups[source][0]["metadata"]
                    ).get("title"),
                    "citations": [row["citation"] for row in groups[source]],
                }
                for slot, source in slots.items()
            ]
            response_model = inventory_response_model(
                slots, [row for source in sources for row in groups[source]]
            )
            try:
                result = self.gateway.complete(
                    SOURCE_ACCOUNTING_PROMPT,
                    packet,
                    response_model,
                    LLMFlow.LEGAL_REVIEW_SOURCE_ACCOUNTING,
                    finalizing=finalizing,
                )
            except ReadingContractError as error:
                packet["contract_correction"] = {
                    "diagnostics": error.diagnostics,
                    "rejected_proposal": error.candidate,
                }
                result = self.gateway.complete(
                    SOURCE_ACCOUNTING_PROMPT
                    + "\nCorrect the rejected proposal's contradictory source interpretation. "
                    "A quotation or procedural request cannot establish the source's own outcome. "
                    "Return the complete source inventory, preserving all unaffected assessments.",
                    packet,
                    response_model,
                    LLMFlow.LEGAL_REVIEW_SOURCE_ACCOUNTING,
                    finalizing=finalizing,
                )
            provided = [row.slot for row in result.source_assessments]
            if len(provided) != len(set(provided)) or set(provided) != set(slots):
                raise ValueError(
                    "Source accounting must preserve every supplied source"
                )
            compiled: list[dict[str, JsonValue]] = []
            for row in result.source_assessments:
                source = slots[row.slot]
                citations = {original["citation"] for original in groups[source]}
                for support in row.supports:
                    if support.citation not in citations:
                        raise ValueError(
                            "Source assessment support belongs to another source"
                        )
                    resolve_passage(support, self.ledger)
                if row.requested_read:
                    arguments = row.requested_read.arguments
                    if arguments.get("source_id") != source:
                        raise ValueError(
                            "Canonical continuation must read its assessed source"
                        )
                    plan = state.get("plan")
                    known_issues = {
                        issue["issue_id"]
                        for issue in cast(dict, plan).get("issues", [])
                    }
                    if set(row.requested_read.issue_ids) - known_issues:
                        raise ValueError(
                            "Canonical continuation refers to an unknown issue"
                        )
                    definitions = {
                        definition["function"]["name"]: definition["function"][
                            "parameters"
                        ]
                        for definition in cast(list[dict], state.get("tools", []))
                    }
                    schema = definitions.get(row.requested_read.tool)
                    if schema is None:
                        raise ValueError("Canonical continuation tool is unavailable")
                    Draft202012Validator(schema).validate(arguments)
                compiled.append(
                    {
                        **row.model_dump(mode="json", exclude={"slot"}),
                        "source_id": source,
                        "citations": [
                            original["citation"] for original in groups[source]
                        ],
                    }
                )
            return compiled

        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(copy_context().run, read_batch, batch)
                for batch in batches
            ]
            for future in as_completed(futures):
                for assessment in cast(list[dict[str, JsonValue]], future.result()):
                    source = cast(str, assessment["source_id"])
                    self._assessments[source] = assessment
                    self._signatures[source] = signatures[source]
        return self.snapshot()
