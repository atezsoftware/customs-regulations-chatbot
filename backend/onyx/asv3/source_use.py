"""Review complete delivered originals without replaying the native conversation."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.asv3.assertions import (
    AssertionWitness,
    assertion_inventory,
    assertion_witness_valid,
)
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel, StructuredOutputError
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.source_metadata_transport import share_source_metadata
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_capabilities import get_llm_max_output_tokens, get_model_map
from onyx.prompts.asv3.source_use import SOURCE_USE_PROMPT
from onyx.tracing.flows import LLMFlow


class SourceUseIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "omitted_condition",
        "inconsistent_application",
        "unsupported_claim",
        "missing_original",
    ]
    answer_unit_ids: list[str] = Field(min_length=1)
    witnesses: list[AssertionWitness]
    detail: str = Field(min_length=1)
    applicability: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_operative_witness(self) -> SourceUseIssue:
        if self.kind != "missing_original" and not self.witnesses:
            raise ValueError("A source-use issue needs its delivered original witness")
        return self


class SourceUseRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    witnesses: list[AssertionWitness] = Field(min_length=1)
    answer_unit_ids: list[str] = Field(min_length=1)
    status: Literal["covered", "omitted", "misapplied"]


class SourceUseExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str = Field(min_length=1)
    witnesses: list[AssertionWitness] = Field(min_length=1)


class SourceUseSourceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    requirements: list[SourceUseRequirement]
    exclusion: SourceUseExclusion | None

    @model_validator(mode="after")
    def require_source_assessment(self) -> SourceUseSourceAssessment:
        if bool(self.requirements) == (self.exclusion is not None):
            raise ValueError(
                "Extract material requirements or witness the source exclusion"
            )
        return self


class SourceUseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examined_citations: list[Annotated[int, Field(strict=True, ge=1)]]
    reviewed_answer_unit_ids: list[str]
    source_assessments: list[SourceUseSourceAssessment]
    issues: list[SourceUseIssue]


def source_use_review_enabled(context: RunContext) -> bool:
    return (
        context.depth == 0
        and context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        and context.services.get("research_profile") == "normal"
    )


class SourceUseReviewer:
    def __init__(self, model: ResearchModel, ledger: EvidenceLedger) -> None:
        self.model = model
        self.ledger = ledger
        self._cache: dict[str, tuple[str, SourceUseReview]] = {}

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
            return None  # The structural delivery guard reports this defect.
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
        units = assertion_inventory(answer)
        payload = {
            "language": context.language,
            "scenario": scenario,
            "conversation": conversation or [],
            "answer_units": units,
            "original_evidence": records,
            "original_source_metadata": shared,
            "source_groups": source_groups,
            "required_evidence_numbers": sorted(numbers),
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        identity = hashlib.sha256(data.encode()).hexdigest()
        cached = self._cache.get(identity)
        if cached and (
            self.ledger.delivery_flow(cached[0]) == LLMFlow.ASV3_SOURCE_USE_REVIEW.value
            and self.ledger.completely_delivered(cached[0]) == numbers
        ):
            review = cached[1]
        else:
            unit_ids = {unit["unit_id"] for unit in units}
            originals = {
                int(cast(int, row["citation"])): str(row["text"]) for row in records
            }
            units_by_id = {unit["unit_id"]: unit for unit in units}

            def validate(text: str) -> None:
                result = SourceUseReview.model_validate_json(text)
                if (
                    not self.model.last_call_id
                    or self.ledger.delivery_flow(self.model.last_call_id)
                    != LLMFlow.ASV3_SOURCE_USE_REVIEW.value
                    or self.ledger.completely_delivered(self.model.last_call_id)
                    != numbers
                    or set(result.examined_citations) != numbers
                    or len(result.examined_citations) != len(numbers)
                    or set(result.reviewed_answer_unit_ids) != unit_ids
                    or len(result.reviewed_answer_unit_ids) != len(unit_ids)
                    or {item.source_id for item in result.source_assessments}
                    != source_groups.keys()
                    or len(result.source_assessments) != len(source_groups)
                ):
                    raise ValueError(
                        "Review every exact delivered original and current answer unit"
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
                for assessment in result.source_assessments:
                    source_numbers = set(source_groups[assessment.source_id])
                    if assessment.exclusion is not None and any(
                        source_numbers.intersection(unit["evidence_numbers"])
                        for unit in units
                        if not unit["presentation_only"]
                    ):
                        raise ValueError(
                            "A source used in legal answer blocks needs its operative requirements, not an exclusion"
                        )
                    entries: list[SourceUseRequirement | SourceUseExclusion] = [
                        *assessment.requirements
                    ]
                    if assessment.exclusion is not None:
                        entries.append(assessment.exclusion)
                    for entry in entries:
                        if any(
                            witness.citation not in source_numbers
                            or not assertion_witness_valid(witness, originals)
                            for witness in entry.witnesses
                        ):
                            raise ValueError(
                                "Source requirements need that source's actual original witnesses"
                            )
                    for requirement in assessment.requirements:
                        if set(requirement.answer_unit_ids) - unit_ids:
                            raise ValueError(
                                "Bind source requirements to current answer units"
                            )
                        if requirement.status == "covered":
                            witness_numbers = {
                                witness.citation for witness in requirement.witnesses
                            }
                            if any(
                                witness_numbers
                                - set(units_by_id[identity]["evidence_numbers"])
                                for identity in requirement.answer_unit_ids
                            ):
                                raise ValueError(
                                    "A covered requirement needs its operative witness inline in the bound units"
                                )

            previous_call = context.services.get("last_model_call_id")
            try:
                context.consume_research_decision()
                text = self.model.invoke_text(
                    SOURCE_USE_PROMPT,
                    data,
                    LLMFlow.ASV3_SOURCE_USE_REVIEW,
                    max_tokens=get_llm_max_output_tokens(
                        get_model_map(),
                        self.model.llm.config.model_name,
                        self.model.llm.config.model_provider,
                    ),
                    consume_budget=False,
                    response_model_override=SourceUseReview,
                    response_validator=validate,
                )
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
            review = SourceUseReview.model_validate_json(text)
            call_id = self.model.last_call_id
            if call_id is None:
                raise ValueError("Source-use review has no delivery receipt")
            self.ledger.pin_delivery(call_id)
            self._cache[identity] = (call_id, review)
        issues = [*review.issues]
        for assessment in review.source_assessments:
            issues.extend(
                SourceUseIssue(
                    kind="omitted_condition"
                    if requirement.status == "omitted"
                    else "inconsistent_application",
                    answer_unit_ids=requirement.answer_unit_ids,
                    witnesses=requirement.witnesses,
                    detail=requirement.detail,
                    applicability=requirement.applicability,
                )
                for requirement in assessment.requirements
                if requirement.status != "covered"
            )
        if not issues:
            return None
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Repair the witnessed source-use defects in their affected answer blocks; preserve supported detail and research only genuinely missing originals.",
            data={
                "source_use_gaps": [issue.model_dump(mode="json") for issue in issues],
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
