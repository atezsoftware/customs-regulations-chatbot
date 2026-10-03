"""Check material original conditions independently of affirmative claim review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import cast

from pydantic import JsonValue

from onyx.asv3.assertions import (
    assertion_inventory,
    assertion_support_defect,
    assertion_witness_valid,
)
from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    ResearchModel,
    SourceConditionAuditResult,
    StructuredOutputError,
    VerificationResult,
)
from onyx.asv3.quotations import normalized
from onyx.asv3.scenario import question_determinations
from onyx.prompts.asv3.research import SOURCE_CONDITION_PROMPT
from onyx.tracing.flows import LLMFlow


def answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()


def condition_review_defects(
    audit: SourceConditionAuditResult,
    answer: str,
    scenario: str,
    questions: list[str],
    ledger: EvidenceLedger,
    call_id: str | None,
    *,
    expected_citations: set[int] | None = None,
) -> list[str]:
    """Validate identity and actual delivery; semantic materiality remains model-assessed."""
    defects: list[str] = []
    examined = set(audit.examined_citations)
    delivered: set[int] = ledger.completely_delivered(call_id) if call_id else set()
    if (
        not call_id
        or ledger.delivery_flow(call_id) != LLMFlow.ASV3_CONDITION_REVIEW.value
        or len(examined) != len(audit.examined_citations)
        or examined != delivered
        or (expected_citations is not None and examined != expected_citations)
    ):
        defects.append(
            "Examined citations must equal the complete originals actually delivered to this independent condition assessment."
        )
    citable = ledger.citation_mapping()
    if examined - citable.keys():
        defects.append("Condition assessment references a non-citable original.")
    determinations = {
        item["determination_id"] for item in question_determinations(questions)
    }
    units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
    for index, condition in enumerate(audit.conditions):
        original = ledger.get(condition.witness.citation)
        if (
            original is None
            or condition.witness.citation not in examined
            or not assertion_witness_valid(
                condition.witness, {condition.witness.citation: original.text}
            )
        ):
            defects.append(
                f"Condition {index} must select a valid delivered original witness."
            )
        if set(condition.determination_ids) - determinations or len(
            set(condition.determination_ids)
        ) != len(condition.determination_ids):
            defects.append(
                f"Condition {index} uses an invalid requested determination identity."
            )
        if set(condition.answer_unit_ids) - units.keys() or len(
            set(condition.answer_unit_ids)
        ) != len(condition.answer_unit_ids):
            defects.append(
                f"Condition {index} uses an invalid current answer-unit identity."
            )
        if condition.disposition == "covered" and not condition.answer_unit_ids:
            defects.append(
                f"Condition {index} needs the exact answer units communicating it."
            )
        if condition.disposition == "not_applicable" and (
            not condition.scenario_quotes
            or any(
                not quote.strip() or normalized(quote) not in normalized(scenario)
                for quote in condition.scenario_quotes
            )
        ):
            defects.append(
                f"Condition {index} needs literal scenario facts establishing its exclusion."
            )
    return defects


def condition_omissions(
    audit: SourceConditionAuditResult, answer: str
) -> list[MaterialSourceOmission]:
    units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
    omissions: list[MaterialSourceOmission] = []
    for condition in audit.conditions:
        carried = any(
            condition.witness.citation in units[key]["evidence_numbers"]
            for key in condition.answer_unit_ids
            if key in units
        )
        if condition.disposition == "omitted" or (
            condition.disposition == "covered" and not carried
        ):
            omissions.append(
                MaterialSourceOmission(
                    **condition.model_dump(
                        include={
                            "witness",
                            "determination_ids",
                            "detail",
                            "applicability",
                        }
                    )
                )
            )
    return omissions


def disclosed_condition_gaps(
    audit: SourceConditionAuditResult, review: VerificationResult, answer: str
) -> set[int]:
    """Bind a partial notice to its exact condition, answer blocks and unresolved outcomes."""
    units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
    checks = {item.unit_id: item for item in review.assertion_results}
    determinations = {
        item.determination_id: (question, item)
        for question in review.question_results
        for item in question.determinations
    }
    if len(checks) != len(review.assertion_results) or len(determinations) != sum(
        len(question.determinations) for question in review.question_results
    ):
        return set()
    disclosed: set[int] = set()
    for index, condition in enumerate(audit.conditions):
        if condition.disposition != "uncertain" or not condition.answer_unit_ids:
            continue
        selected = set(condition.answer_unit_ids)
        if any(
            (unit := units.get(key)) is None
            or (check := checks.get(key)) is None
            or check.basis != "evidence_gap"
            or assertion_support_defect(unit, check, {}, "", allow_explicit_gaps=True)
            is not None
            or not any(value.strip() for value in check.missing_conditions)
            for key in selected
        ):
            continue
        if any(
            (bound := determinations.get(key)) is None
            or bound[0].status not in {"incomplete", "uncertain"}
            or not any(value.strip() for value in bound[0].missing_conditions)
            or bound[1].status not in {"incomplete", "uncertain"}
            or not any(value.strip() for value in bound[1].missing_conditions)
            or not selected.issubset(bound[1].answer_unit_ids)
            for key in condition.determination_ids
        ):
            continue
        disclosed.add(index)
    return disclosed


def condition_payload(
    answer: str, scenario: str, questions: list[str], evidence: str, language: str
) -> tuple[dict[str, JsonValue], set[int]]:
    records = cast(list[dict[str, JsonValue]], json.loads(evidence))
    originals = [
        record
        for record in records
        if record.get("citable") is True and record.get("truncated") is False
    ]
    numbers = {int(cast(int, record["citation"])) for record in originals}
    # The bounded original projection already excludes labels, receipts and file inventories.
    payload: dict[str, JsonValue] = {
        "language": language,
        "scenario": scenario,
        "determinations": [
            cast(dict[str, JsonValue], dict(item))
            for item in question_determinations(questions)
        ],
        "answer_units": [
            cast(dict[str, JsonValue], dict(unit))
            for unit in assertion_inventory(answer)
        ],
        "original_evidence": cast(JsonValue, originals),
        "required_evidence_numbers": cast(JsonValue, sorted(numbers)),
    }
    return payload, numbers


def complete_condition_review(
    review: VerificationResult,
    model: ResearchModel,
    ledger: EvidenceLedger,
    *,
    answer: str,
    scenario: str,
    questions: list[str],
    evidence: str,
    language: str,
    consume_budget: bool,
    condition_memory: SourceConditionMemory | None = None,
    allow_explicit_gaps: bool = False,
) -> VerificationResult:
    # Provider output cannot supply the host's independent approval receipt.
    result = review.model_copy(
        update={
            "condition_review": None,
            "condition_review_call_id": None,
            "condition_review_answer_hash": None,
        }
    )
    if (
        result.format_error
        or not result.safe_to_publish
        or result.omitted_material_source_details
    ):
        return result
    payload, numbers = condition_payload(
        answer, scenario, questions, evidence, language
    )
    payload["allow_explicit_gaps"] = allow_explicit_gaps
    if condition_memory is not None:
        payload["retained_conditions"] = cast(
            JsonValue, condition_memory.required_conditions()
        )
        missing = set(condition_memory.citations()) - numbers
        if missing:
            result.safe_to_publish = False
            result.status = "incomplete"
            result.missing_conditions = [
                *result.missing_conditions,
                "Retained condition originals were not completely supplied: "
                + str(sorted(missing)),
            ]
            return result
    if not numbers:
        return result
    main_call_id = model.last_call_id

    def validate_response(text: str) -> None:
        audit = SourceConditionAuditResult.model_validate_json(text)
        retention_defects: list[str] = []
        if condition_memory is not None:
            audit, retention_defects = condition_memory.materialize(audit, ledger)
        defects = condition_review_defects(
            audit,
            answer,
            scenario,
            questions,
            ledger,
            model.last_call_id,
            expected_citations=numbers,
        )
        if defects or retention_defects:
            raise ValueError(" ".join([*defects, *retention_defects]))

    try:
        if not consume_budget:
            model.context.consume_research_decision()
        text = model.invoke_text(
            SOURCE_CONDITION_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            LLMFlow.ASV3_CONDITION_REVIEW,
            max_tokens=6000,
            consume_budget=consume_budget,
            response_model_override=SourceConditionAuditResult,
            response_validator=validate_response,
        )
        audit = SourceConditionAuditResult.model_validate_json(text)
        effective_audit = audit
        if condition_memory is not None:
            effective_audit, defects = condition_memory.materialize(audit, ledger)
            if defects:
                raise StructuredOutputError(" ".join(defects))
            condition_memory.remember(audit.conditions, ledger)
        result.condition_review = audit
        result.condition_review_call_id = model.last_call_id
        result.condition_review_answer_hash = answer_hash(answer)
        omissions = condition_omissions(effective_audit, answer)
        unresolved = [
            item.detail + " " + item.applicability
            for item in effective_audit.conditions
            if item.disposition == "uncertain"
        ]
        disclosed: set[int] = (
            disclosed_condition_gaps(effective_audit, result, answer)
            if allow_explicit_gaps
            else set()
        )
        undisclosed_uncertainty = any(
            item.disposition == "uncertain" and index not in disclosed
            for index, item in enumerate(effective_audit.conditions)
        )
        if unresolved:
            result.missing_conditions = list(
                dict.fromkeys([*result.missing_conditions, *unresolved])
            )
        if omissions:
            result.omitted_material_source_details = _distinct_omissions(
                [*result.omitted_material_source_details, *omissions]
            )
        if omissions or undisclosed_uncertainty:
            result.safe_to_publish = False
        if omissions or unresolved:
            result.status = "incomplete"
    except StructuredOutputError as error:
        result.safe_to_publish = False
        result.status = "uncertain"
        result.format_error = (
            "Independent source-condition assessment: " + str(error)[:440]
        )
    finally:
        # Main assertion delivery and independent condition delivery have different receipts.
        model.last_call_id = main_call_id
    return result


def _distinct_omissions(
    items: Sequence[MaterialSourceOmission],
) -> list[MaterialSourceOmission]:
    seen: set[str] = set()
    result: list[MaterialSourceOmission] = []
    for item in items:
        identity = item.model_dump_json()
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result
