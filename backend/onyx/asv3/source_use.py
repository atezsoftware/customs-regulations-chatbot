"""Retain draft-blind source requirements and assess their use in current answers."""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.asv3.assertions import (
    AssertionWitness,
    assertion_inventory,
    assertion_witness_valid,
)
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel, StructuredOutputError
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.source_metadata_transport import share_source_metadata
from onyx.asv3.witnesses import original_witness_spans
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_capabilities import get_llm_max_output_tokens, get_model_map
from onyx.prompts.asv3.source_use import SOURCE_USE_INVENTORY_PROMPT, SOURCE_USE_PROMPT
from onyx.tracing.flows import LLMFlow


class SourceUseIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "omitted_condition",
        "inconsistent_application",
        "unsupported_claim",
        "missing_original",
    ]
    answer_unit_ids: list[str]
    witnesses: list[AssertionWitness]
    detail: str = Field(min_length=1)
    applicability: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_operative_witness(self) -> SourceUseIssue:
        if self.kind != "omitted_condition" and not self.answer_unit_ids:
            raise ValueError("An asserted defect needs its actual answer unit")
        if self.kind != "missing_original" and not self.witnesses:
            raise ValueError("A source-use issue needs its delivered original witness")
        return self


class SourceUseRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    witnesses: list[AssertionWitness] = Field(min_length=1)


class SourceUseInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examined_citations: list[Annotated[int, Field(strict=True, ge=1)]]
    requirements: list[SourceUseRequirement]


class SourceUseCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_unit_id: str
    witnesses: list[AssertionWitness] = Field(min_length=1)


class SourceUseResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    status: Literal[
        "covered", "omitted", "misapplied", "not_applicable", "outside_request"
    ]
    answer_unit_ids: list[str]
    explanation: str = Field(
        default="",
        description="For misapplied, describe the changed logic or scope. An omitted requirement already supplies its exact missing detail; do not repeat it.",
    )
    scenario_witness_ids: list[str] = Field(
        default_factory=list,
        description="Select supplied user_fact_spans IDs for factual exclusion or the actual request scope; never recopy user text or select assistant statements.",
    )
    coverage: list[SourceUseCoverage]


def user_fact_spans(
    scenario: str, conversation: list[dict[str, str]]
) -> list[dict[str, JsonValue]]:
    """Address unchanged user text without generating another copy of each fact."""
    texts = {"scenario": scenario}
    texts.update(
        {
            f"conversation-{index}": row["content"]
            for index, row in enumerate(conversation)
            if row.get("role") == "user"
        }
    )
    return [
        {
            "witness_id": f"{key}-{span['witness_id']}",
            "text_ref": key,
            "start_char": span["start_char"],
            "end_char": span["end_char"],
        }
        for key, text in texts.items()
        for span in original_witness_spans(0, text)
    ]


class SourceUseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examined_citations: list[Annotated[int, Field(strict=True, ge=1)]]
    reviewed_answer_unit_ids: list[str]
    resolutions: list[SourceUseResolution]
    issues: list[SourceUseIssue]


def source_use_review_enabled(context: RunContext) -> bool:
    return (
        context.depth == 0
        and context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        and context.services.get("research_profile") == "normal"
    )


def source_requirement_id(requirement: SourceUseRequirement) -> str:
    data = json.dumps(
        requirement.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
    )
    return "sr-" + hashlib.sha256(data.encode()).hexdigest()[:16]


class SourceUseReviewer:
    def __init__(self, model: ResearchModel, ledger: EvidenceLedger) -> None:
        self.model = model
        self.ledger = ledger
        self._inventories: dict[str, tuple[str, SourceUseInventory]] = {}
        self._cache: dict[str, tuple[str, SourceUseReview]] = {}
        self._failures: dict[tuple[LLMFlow, str], tuple[str, str]] = {}
        self._source_reviewers: dict[str, SourceUseReviewer] = {}

    def _source_inventories(
        self, payload: dict[str, JsonValue], numbers: set[int]
    ) -> SourceUseInventory:
        records = cast(list[dict[str, JsonValue]], payload["original_evidence"])
        groups: dict[str, list[dict[str, JsonValue]]] = {}
        for row in records:
            groups.setdefault(str(row["source_id"]), []).append(row)
        if len(groups) == 1:
            return self._complete_inventory(payload, numbers)
        shared = cast(dict[str, JsonValue], payload["original_source_metadata"])
        jobs: list[tuple[SourceUseReviewer, dict[str, JsonValue], set[int]]] = []
        for source, rows in groups.items():
            reviewer = self._source_reviewers.get(source)
            if reviewer is None:
                child = self.model.context.child()
                reviewer = SourceUseReviewer(
                    ResearchModel(
                        self.model.llm,
                        child,
                        user_identity=self.model.user_identity,
                        reasoning_effort=self.model.reasoning_effort,
                        token_counter=self.model.token_counter,
                        research_llm=self.model.research_llm,
                    ),
                    self.ledger,
                )
                self._source_reviewers[source] = reviewer
            owned = {int(cast(int, row["citation"])) for row in rows}
            links = [
                row
                for row in cast(list[dict[str, JsonValue]], payload["source_links"])
                if row["source_id"] == source
            ]
            anchors = {
                int(cast(int, n))
                for link in links
                for n in cast(list[JsonValue], link["anchor_evidence_numbers"])
            }
            linked_rows = [row for row in records if row["citation"] in anchors]
            originals = {
                int(cast(int, row["citation"])): row for row in [*rows, *linked_rows]
            }
            selected = set(originals)
            sources = {str(row["source_id"]) for row in originals.values()}
            scoped = {
                **payload,
                "original_evidence": cast(list[JsonValue], list(originals.values())),
                "original_source_metadata": {
                    identity: shared[identity]
                    for identity in sorted(sources)
                    if identity in shared
                },
                "source_groups": {
                    identity: [
                        n
                        for n, row in originals.items()
                        if row["source_id"] == identity
                    ]
                    for identity in sorted(sources)
                },
                "inventory_source_id": source,
                "required_owned_evidence_numbers": sorted(owned),
                "source_links": cast(list[JsonValue], links),
                "required_evidence_numbers": sorted(selected),
            }
            jobs.append((reviewer, scoped, selected))
        inventories: list[SourceUseInventory] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    contextvars.copy_context().run,
                    reviewer._complete_inventory,
                    scoped,
                    selected,
                )
                for reviewer, scoped, selected in jobs
            ]
            for future in futures:
                inventories.append(cast(SourceUseInventory, future.result()))
        examined = {n for item in inventories for n in item.examined_citations}
        if examined != numbers:
            raise ValueError("Source inventories must cover every exact original")
        return SourceUseInventory(
            examined_citations=sorted(examined),
            requirements=[r for item in inventories for r in item.requirements],
        )

    def _source_links(self, numbers: set[int]) -> list[dict[str, JsonValue]]:
        reviews = self.model.context.services.get("legal_source_reviews")
        if not isinstance(reviews, LegalSourceReviews):
            return []
        view = reviews.view(self.model.context, self.ledger, numbers)
        links: list[dict[str, JsonValue]] = []
        for raw in cast(list[JsonValue], view["reviews"]):
            if not isinstance(raw, dict):
                continue
            anchors = raw.get("anchor_evidence_numbers")
            selected = (
                sorted(n for n in anchors if type(n) is int and n in numbers)
                if isinstance(anchors, list)
                else []
            )
            if selected:
                links.append(
                    {
                        key: raw[key]
                        for key in (
                            "lead_id",
                            "anchor_source_id",
                            "article_no",
                            "qualifier",
                            "source_id",
                        )
                    }
                    | {"anchor_evidence_numbers": selected}
                )
        return links

    def _receipt_matches(self, call_id: str, flow: LLMFlow, numbers: set[int]) -> bool:
        return (
            self.ledger.delivery_flow(call_id) == flow.value
            and self.ledger.completely_delivered(call_id) == numbers
        )

    def _invoke_assessment(
        self,
        instruction: str,
        data: str,
        flow: LLMFlow,
        response_model: type[BaseModel],
        validate: Callable[[str], None],
        numbers: set[int],
    ) -> str:
        key = (flow, hashlib.sha256(data.encode()).hexdigest())
        failure = self._failures.get(key)
        if failure and self._receipt_matches(failure[0], flow, numbers):
            raise StructuredOutputError(failure[1])
        self.model.context.consume_research_decision()
        try:
            return self.model.invoke_text(
                instruction,
                data,
                flow,
                max_tokens=get_llm_max_output_tokens(
                    get_model_map(),
                    self.model.llm.config.model_name,
                    self.model.llm.config.model_provider,
                ),
                consume_budget=False,
                response_model_override=response_model,
                response_validator=validate,
            )
        except StructuredOutputError as error:
            call_id = self.model.last_call_id
            if call_id and self._receipt_matches(call_id, flow, numbers):
                self.ledger.pin_delivery(call_id)
                self._failures[key] = (call_id, str(error))
            raise

    def _complete_inventory(
        self, payload: dict[str, JsonValue], numbers: set[int]
    ) -> SourceUseInventory:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        identity = hashlib.sha256(data.encode()).hexdigest()
        cached = self._inventories.get(identity)
        if cached and self._receipt_matches(
            cached[0], LLMFlow.ASV3_SOURCE_INVENTORY, numbers
        ):
            return cached[1]
        records = cast(list[dict[str, JsonValue]], payload["original_evidence"])
        originals = {
            int(cast(int, row["citation"])): str(row["text"]) for row in records
        }

        def validate(text: str) -> None:
            result = SourceUseInventory.model_validate_json(text)
            if (
                not self.model.last_call_id
                or not self._receipt_matches(
                    self.model.last_call_id, LLMFlow.ASV3_SOURCE_INVENTORY, numbers
                )
                or set(result.examined_citations) != numbers
                or len(result.examined_citations) != len(numbers)
            ):
                raise ValueError(
                    "Inventory must examine every exact completely delivered original"
                )
            identities = [source_requirement_id(item) for item in result.requirements]
            if len(identities) != len(set(identities)):
                raise ValueError(
                    "Group actual duplicate requirements; do not repeat identical records"
                )
            for index, item in enumerate(result.requirements):
                if any(
                    not assertion_witness_valid(witness, originals)
                    for witness in item.witnesses
                ):
                    raise ValueError(
                        f"Requirement {index} needs its actual delivered operative witness"
                    )
                owned = payload.get("required_owned_evidence_numbers")
                if isinstance(owned, list) and not any(
                    witness.citation in owned for witness in item.witnesses
                ):
                    raise ValueError(
                        f"Requirement {index} needs the target source's own operative witness; linked context is not a separate inventory target"
                    )

        text = self._invoke_assessment(
            SOURCE_USE_INVENTORY_PROMPT,
            data,
            LLMFlow.ASV3_SOURCE_INVENTORY,
            SourceUseInventory,
            validate,
            numbers,
        )
        result = SourceUseInventory.model_validate_json(text)
        call_id = self.model.last_call_id
        if call_id is None:
            raise ValueError("Source requirement inventory has no delivery receipt")
        self.ledger.pin_delivery(call_id)
        self._inventories[identity] = (call_id, result)
        return result

    def publication_gap(
        self,
        answer: str,
        scenario: str,
        coordinator_call_id: str | None,
        *,
        conversation: list[dict[str, str]] | None = None,
    ) -> ToolOutcome | None:
        context = self.model.context
        if not source_use_review_enabled(context) or not extract_citation_numbers(
            answer
        ):
            return None
        numbers = self.ledger.completely_delivered(coordinator_call_id or "")
        numbers &= self.ledger.citation_mapping().keys()
        if not numbers:
            return None
        records = cast(
            list[dict[str, JsonValue]],
            json.loads(
                self.ledger.serialize_records(
                    sorted(numbers),
                    required=sorted(numbers),
                    max_chars=None,
                    include_witness_spans=True,
                )
            ),
        )
        records, shared = share_source_metadata(records)
        records.sort(
            key=lambda row: (str(row["source_id"]), int(cast(int, row["citation"])))
        )
        source_groups: dict[str, list[int]] = {}
        for row in records:
            source_groups.setdefault(str(row["source_id"]), []).append(
                int(cast(int, row["citation"]))
            )
        user_conversation: list[dict[str, str]] = []
        supplied_texts = {scenario}
        for row in conversation or []:
            if row.get("role") == "user" and row["content"] not in supplied_texts:
                user_conversation.append(row)
                supplied_texts.add(row["content"])
        blind_payload = cast(
            dict[str, JsonValue],
            {
                "language": context.language,
                "scenario": scenario,
                "conversation": user_conversation,
                "user_fact_spans": user_fact_spans(scenario, user_conversation),
                "original_evidence": records,
                "original_source_metadata": shared,
                "source_groups": source_groups,
                "source_links": self._source_links(numbers),
                "required_evidence_numbers": sorted(numbers),
            },
        )
        previous_call = context.services.get("last_model_call_id")
        try:
            inventory = self._source_inventories(blind_payload, numbers)
            return self._assess_answer(answer, blind_payload, inventory, numbers)
        except StructuredOutputError:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Source-use assessment is incomplete; retain the candidate and originals.",
                data={"source_use_review_unavailable": True},
            )
        finally:
            if previous_call is None:
                context.services.pop("last_model_call_id", None)
            else:
                context.services["last_model_call_id"] = previous_call

    def _assess_answer(
        self,
        answer: str,
        blind_payload: dict[str, JsonValue],
        inventory: SourceUseInventory,
        numbers: set[int],
    ) -> ToolOutcome | None:
        units = assertion_inventory(answer)
        requirements = {
            source_requirement_id(item): item for item in inventory.requirements
        }
        retained = [
            {"requirement_id": identity, **item.model_dump(mode="json")}
            for identity, item in requirements.items()
        ]
        application_candidates: list[dict[str, JsonValue]] = []
        for link in cast(list[dict[str, JsonValue]], blind_payload["source_links"]):
            anchors = cast(list[JsonValue], link["anchor_evidence_numbers"])
            affected = [
                unit["unit_id"]
                for unit in units
                if any(n in anchors for n in unit["evidence_numbers"])
            ]
            if not affected:
                continue
            for requirement_id, requirement in requirements.items():
                if any(
                    (item := self.ledger.get(witness.citation)) is not None
                    and item.source_id == link["source_id"]
                    for witness in requirement.witnesses
                ):
                    application_candidates.append(
                        {
                            "requirement_id": requirement_id,
                            "answer_unit_ids": affected,
                        }
                    )
        # Stable original/fact prefix permits provider cache reuse after draft-only edits.
        payload = {
            **blind_payload,
            "retained_requirements": retained,
            "related_application_candidates": application_candidates,
            "answer_units": units,
        }
        data = json.dumps(payload, ensure_ascii=False)
        identity = hashlib.sha256(data.encode()).hexdigest()
        cached = self._cache.get(identity)
        if cached and self._receipt_matches(
            cached[0], LLMFlow.ASV3_SOURCE_USE_REVIEW, numbers
        ):
            review = cached[1]
        else:
            originals = {
                int(cast(int, row["citation"])): str(row["text"])
                for row in cast(
                    list[dict[str, JsonValue]], blind_payload["original_evidence"]
                )
            }
            units_by_id = {unit["unit_id"]: unit for unit in units}
            unit_ids = set(units_by_id)
            fact_ids = {
                str(row["witness_id"])
                for row in cast(
                    list[dict[str, JsonValue]], blind_payload["user_fact_spans"]
                )
            }

            def validate(text: str) -> None:
                result = SourceUseReview.model_validate_json(text)
                if (
                    not self.model.last_call_id
                    or not self._receipt_matches(
                        self.model.last_call_id, LLMFlow.ASV3_SOURCE_USE_REVIEW, numbers
                    )
                    or set(result.examined_citations) != numbers
                    or len(result.examined_citations) != len(numbers)
                    or set(result.reviewed_answer_unit_ids) != unit_ids
                    or len(result.reviewed_answer_unit_ids) != len(unit_ids)
                ):
                    raise ValueError(
                        "Review every exact delivered original and current answer unit"
                    )
                resolved = [row.requirement_id for row in result.resolutions]
                if set(resolved) != requirements.keys() or len(resolved) != len(
                    requirements
                ):
                    raise ValueError(
                        "Resolve every exact retained requirement once; do not replace or drop IDs"
                    )
                for row in result.resolutions:
                    if set(row.answer_unit_ids) - unit_ids or len(
                        row.answer_unit_ids
                    ) != len(set(row.answer_unit_ids)):
                        raise ValueError(
                            f"Resolution {row.requirement_id} needs exact current answer-unit IDs"
                        )
                    if row.status == "not_applicable":
                        if (
                            not row.scenario_witness_ids
                            or set(row.scenario_witness_ids) - fact_ids
                            or len(row.scenario_witness_ids)
                            != len(set(row.scenario_witness_ids))
                        ):
                            raise ValueError(
                                f"Resolution {row.requirement_id} needs exact supplied user-fact witnesses establishing exclusion"
                            )
                        continue
                    if row.status == "outside_request":
                        if (
                            row.answer_unit_ids
                            or row.coverage
                            or not row.explanation.strip()
                            or not row.scenario_witness_ids
                            or set(row.scenario_witness_ids) - fact_ids
                            or len(row.scenario_witness_ids)
                            != len(set(row.scenario_witness_ids))
                        ):
                            raise ValueError(
                                f"Resolution {row.requirement_id} needs its actual request-scope witness and explanation; an asserted application cannot be outside the request"
                            )
                        continue
                    if not row.answer_unit_ids and row.status != "omitted":
                        raise ValueError(
                            f"Resolution {row.requirement_id} needs affected answer-unit IDs"
                        )
                    if row.status == "covered":
                        covered_ids = [b.answer_unit_id for b in row.coverage]
                        if set(covered_ids) != set(row.answer_unit_ids) or len(
                            covered_ids
                        ) != len(row.answer_unit_ids):
                            raise ValueError(
                                f"Resolution {row.requirement_id} needs one operative coverage binding per affected unit"
                            )
                        if any(
                            not assertion_witness_valid(witness, originals)
                            for binding in row.coverage
                            for witness in binding.witnesses
                        ):
                            raise ValueError(
                                "Coverage needs actual delivered witnesses"
                            )
                    elif row.status == "misapplied" and not row.explanation.strip():
                        raise ValueError(
                            f"Resolution {row.requirement_id} needs its exact actionable defect"
                        )
                for issue in result.issues:
                    if (
                        set(issue.answer_unit_ids) - unit_ids
                        or len(issue.answer_unit_ids) != len(set(issue.answer_unit_ids))
                        or any(
                            not assertion_witness_valid(witness, originals)
                            for witness in issue.witnesses
                        )
                    ):
                        raise ValueError(
                            "Issues need current answer units and actual delivered witnesses"
                        )

            text = self._invoke_assessment(
                SOURCE_USE_PROMPT,
                data,
                LLMFlow.ASV3_SOURCE_USE_REVIEW,
                SourceUseReview,
                validate,
                numbers,
            )
            review = SourceUseReview.model_validate_json(text)
            call_id = self.model.last_call_id
            if call_id is None:
                raise ValueError("Source-use review has no delivery receipt")
            self.ledger.pin_delivery(call_id)
            self._cache[identity] = (call_id, review)
        issues = [*review.issues]
        for resolution in review.resolutions:
            if resolution.status == "covered":
                units_by_id = {unit["unit_id"]: unit for unit in units}
                for binding in resolution.coverage:
                    missing = {w.citation for w in binding.witnesses} - set(
                        units_by_id[binding.answer_unit_id]["evidence_numbers"]
                    )
                    if missing:
                        requirement = requirements[resolution.requirement_id]
                        issues.append(
                            SourceUseIssue(
                                kind="unsupported_claim",
                                answer_unit_ids=[binding.answer_unit_id],
                                witnesses=binding.witnesses,
                                detail=requirement.detail
                                + " Operative inline support is absent: "
                                + ", ".join(f"[{n}]" for n in sorted(missing)),
                                applicability=requirement.applicability,
                            )
                        )
                continue
            if resolution.status not in {"omitted", "misapplied"}:
                continue
            requirement = requirements[resolution.requirement_id]
            issues.append(
                SourceUseIssue(
                    kind="omitted_condition"
                    if resolution.status == "omitted"
                    else "inconsistent_application",
                    answer_unit_ids=resolution.answer_unit_ids,
                    witnesses=requirement.witnesses,
                    detail=" ".join(
                        part
                        for part in (requirement.detail, resolution.explanation.strip())
                        if part
                    ),
                    applicability=requirement.applicability,
                )
            )
        if not issues:
            return None
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Repair the witnessed source-use defects in their affected answer blocks; preserve supported detail and research only genuinely missing originals.",
            data={
                "source_use_gaps": [issue.model_dump(mode="json") for issue in issues],
                "retained_source_requirements": cast(list[JsonValue], retained),
                "affected_answer_units": [
                    cast(dict[str, JsonValue], dict(unit))
                    for unit in units
                    if any(unit["unit_id"] in issue.answer_unit_ids for issue in issues)
                ],
            },
        )


def combine_source_publication_gaps(gaps: list[ToolOutcome]) -> ToolOutcome | None:
    """Return all computed source defects without losing acquisition-routing fields."""
    if not gaps:
        return None
    if len(gaps) == 1:
        return gaps[0]
    data: dict[str, JsonValue] = {}
    instructions: list[JsonValue] = []
    for gap in gaps:
        instructions.append(
            {"summary": gap.summary, "instruction": gap.data.get("instruction")}
        )
        for key, value in gap.data.items():
            if key == "instruction":
                continue
            if key in data and data[key] != value:
                raise ValueError("Conflicting publication check fields")
            data[key] = value
    data["publication_check_instructions"] = instructions
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Address every reported source defect together in the retained candidate; reuse delivered originals and preserve supported detail.",
        data=data,
    )
