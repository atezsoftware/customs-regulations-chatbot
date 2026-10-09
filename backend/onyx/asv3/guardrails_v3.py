"""Frozen packet and typed, fail-open JEV review for Guardrails v3."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from math import isfinite
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from onyx.asv3.candidate_audit import CandidateAudit, CandidateAuditRecord
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.outcome_map import OutcomeMap
from onyx.configs import app_configs
from onyx.prompts.asv3.guardrails_v3 import GUARDRAILS_V3_JEV_INSTRUCTION
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import record_llm_span_output, traced_llm_call

_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_JEV_MODEL = "jev-latest"
_PACKET_VERSION = 1
_CHECKLIST_VERSION = "method-1-method-2-v1"
_MAX_RESPONSE_BYTES = 256_000
_DEFAULT_MAX_PACKET_CHARS = 180_000
_M2_DIMENSIONS = ("d1", "d2", "d3", "d4", "d5", "d6")


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class AnswerUnit(_StrictFrozenModel):
    section_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1)


class ReviewEvidence(_StrictFrozenModel):
    citation: int = Field(ge=1)
    source_id: str = Field(min_length=1, max_length=512)
    chunk_id: str | None = Field(default=None, max_length=512)
    text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    deliveries: list[JsonValue] = Field(default_factory=list)


class FrozenReviewPacket(_StrictFrozenModel):
    version: Literal[1] = _PACKET_VERSION
    checklist_version: Literal["method-1-method-2-v1"] = _CHECKLIST_VERSION
    question: str = Field(min_length=1)
    candidate_answer: str = Field(min_length=1)
    answer_units: list[AnswerUnit] = Field(min_length=1)
    outcome_view: dict[str, JsonValue]
    evidence: list[ReviewEvidence]
    candidate_audit: list[CandidateAuditRecord] = Field(default_factory=list)
    delivery_receipts: list[JsonValue] = Field(default_factory=list)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_answer_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    draft_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    packet_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class FrozenReviewPacketBuild(_StrictFrozenModel):
    packet: FrozenReviewPacket | None = None
    failure_reason: str | None = None


class NormalizedFinding(_StrictFrozenModel):
    finding_id: str = Field(min_length=1, max_length=160)
    outcome_ids: list[str] = Field(default_factory=list, max_length=32)
    dimensions: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=120)
    review_question_ids: list[str] = Field(min_length=1, max_length=32)
    affected_answer_unit_ids: list[str] = Field(default_factory=list, max_length=128)
    draft_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    repair_attempt: Literal[0] = 0


class CombinedJevReview(_StrictFrozenModel):
    actual_model: str = Field(min_length=1, max_length=160)
    answers: dict[str, JsonValue]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GuardrailsV3ReviewOutcome(_StrictFrozenModel):
    review_completed: bool = False
    repair_requested: bool = False
    findings: list[NormalizedFinding] = Field(default_factory=list)
    actual_model: str | None = None
    raw_answers: dict[str, JsonValue] = Field(default_factory=dict)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    review_seconds: float = Field(default=0.0, ge=0)
    packet_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    checklist_version: str = _CHECKLIST_VERSION
    failure_reason: str | None = None


def build_frozen_review_packet(
    *,
    question: str,
    candidate_answer: str,
    outcome_map: OutcomeMap | None,
    ledger: EvidenceLedger,
    candidate_audit: CandidateAudit | None,
    max_state_chars: int = _DEFAULT_MAX_PACKET_CHARS,
    requires_research: bool = True,
) -> FrozenReviewPacketBuild:
    """Build a complete review state or fail open before any provider call."""
    if not question.strip() or not candidate_answer.strip() or max_state_chars < 1:
        return FrozenReviewPacketBuild(failure_reason="review_packet_invalid")
    if outcome_map is None:
        return FrozenReviewPacketBuild(
            failure_reason=(
                "outcome_map_missing" if requires_research else "review_exempt"
            )
        )

    outcome_view = outcome_map.view()
    required = set(extract_citation_numbers(candidate_answer))
    required.update(outcome_map.preferred_citations())
    selected = _allocate_evidence(ledger, outcome_view, required)
    evidence = _packet_evidence(ledger, selected)
    if len(evidence) != len(selected):
        return FrozenReviewPacketBuild(
            failure_reason="review_required_evidence_missing"
        )
    candidate_records = _audit_records(candidate_audit, outcome_view)
    deliveries = _delivery_receipts(ledger, selected)
    source_hash = _digest([item.model_dump(mode="json") for item in evidence])
    draft_hash = _digest(candidate_answer)
    packet_without_hash = {
        "version": _PACKET_VERSION,
        "checklist_version": _CHECKLIST_VERSION,
        "question": question,
        "candidate_answer": candidate_answer,
        "answer_units": [
            unit.model_dump(mode="json") for unit in _answer_units(candidate_answer)
        ],
        "outcome_view": outcome_view,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "candidate_audit": [item.model_dump(mode="json") for item in candidate_records],
        "delivery_receipts": deliveries,
        "source_hash": source_hash,
        "candidate_answer_hash": draft_hash,
        "draft_hash": draft_hash,
    }
    if _json_chars(packet_without_hash) > max_state_chars:
        return FrozenReviewPacketBuild(failure_reason="review_packet_too_large")
    packet_hash = _digest(packet_without_hash)
    return FrozenReviewPacketBuild(
        packet=FrozenReviewPacket.model_validate(
            {**packet_without_hash, "packet_hash": packet_hash}
        )
    )


def review_frozen_packet(
    packet: FrozenReviewPacket,
    *,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout_seconds: float | None = None,
) -> GuardrailsV3ReviewOutcome:
    """Run one strict, independent combined JEV review; provider failures stay open."""
    started = time.monotonic()
    credential = api_key if api_key is not None else app_configs.TYPESAFE_API_KEY
    if not credential or not credential.strip():
        return _failed_outcome(packet, started, "jev_credential_unavailable")
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else app_configs.ASV3_GUARDRAILS_V3_JEV_TIMEOUT_SECONDS
    )
    if not isfinite(timeout) or timeout <= 0:
        return _failed_outcome(packet, started, "jev_invalid_timeout")
    questions = _review_questions(packet)
    payload: dict[str, JsonValue] = {
        "model": _JEV_MODEL,
        "state": _review_state(packet),
        "questions": questions,
    }
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > _MAX_RESPONSE_BYTES:
            raise ValueError("JEV review packet exceeds provider bound")
        with traced_llm_call(
            flow=LLMFlow.ASV3_GUARDRAILS_V3_REVIEW,
            model=_JEV_MODEL,
            provider="typesafe",
            extra_config={"endpoint": _JEV_ENDPOINT, "transport_attempts": "1"},
            input_messages=[{"role": "user", "content": encoded.decode("utf-8")}],
        ) as span:
            span.span_data.request_params = {
                "model": _JEV_MODEL,
                "endpoint": _JEV_ENDPOINT,
                "review_question_ids": list(questions),
                "transport_attempts": 1,
            }
            review = _send_jev_review(
                encoded=encoded,
                credential=credential,
                timeout_seconds=timeout,
                transport=transport,
                questions=questions,
            )
            record_llm_span_output(
                span,
                [{"role": "assistant", "content": json.dumps(review.answers)}],
                usage={
                    "input_tokens": review.input_tokens,
                    "output_tokens": review.output_tokens,
                },
            )
    except httpx.TimeoutException:
        return _failed_outcome(packet, started, "jev_review_timeout")
    except httpx.HTTPError:
        return _failed_outcome(packet, started, "jev_review_http_error")
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        return _failed_outcome(packet, started, "jev_invalid_review_response")
    except Exception:
        return _failed_outcome(packet, started, "jev_review_unavailable")

    findings = normalize_review_findings(packet, review.answers)
    repair_probability = review.answers.get("repair_required")
    return GuardrailsV3ReviewOutcome(
        review_completed=True,
        repair_requested=(
            isinstance(repair_probability, float) and repair_probability >= 0.5
        )
        or bool(findings),
        findings=findings,
        actual_model=review.actual_model,
        raw_answers=review.answers,
        input_tokens=review.input_tokens,
        output_tokens=review.output_tokens,
        review_seconds=time.monotonic() - started,
        packet_hash=packet.packet_hash,
    )


def normalize_review_findings(
    packet: FrozenReviewPacket, answers: dict[str, JsonValue]
) -> list[NormalizedFinding]:
    """Turn typed review output into application-owned, deduplicated defects."""
    findings: dict[tuple[tuple[str, ...], str], NormalizedFinding] = {}
    outcome_ids = _outcome_ids(packet.outcome_view)
    for outcome_id in outcome_ids:
        checks = {
            "evidence": {"missing", "incomplete", "conflicting"},
            "coverage": {"omitted", "partial"},
            "application": {"missing_fact", "misapplied"},
            "conditions": {"lost"},
        }
        for check, material_values in checks.items():
            question_id = f"m1.{outcome_id}.{check}"
            value = answers.get(question_id)
            if value not in material_values:
                continue
            key = ((outcome_id,), f"m1_{check}")
            findings[key] = _finding(
                packet, [outcome_id], [f"m1_{check}"], str(value), [question_id]
            )
    for dimension in _M2_DIMENSIONS:
        applicability_id = f"m2.{dimension}.applicability"
        treatment_id = f"m2.{dimension}.treatment"
        if answers.get(applicability_id) != "relevant":
            continue
        if answers.get(treatment_id) not in {"missing", "partial"}:
            continue
        linked = outcome_ids if dimension in {"d3", "d4", "d5", "d6"} else []
        key = (tuple(linked), dimension)
        findings[key] = _finding(
            packet,
            linked,
            [dimension],
            str(answers[treatment_id]),
            [applicability_id, treatment_id],
        )
    return list(findings.values())


def _finding(
    packet: FrozenReviewPacket,
    outcome_ids: list[str],
    dimensions: list[str],
    reason: str,
    question_ids: list[str],
) -> NormalizedFinding:
    identity = "-".join([*outcome_ids, *dimensions, reason]) or "global"
    return NormalizedFinding(
        finding_id=f"finding-{_digest(identity)[:16]}",
        outcome_ids=outcome_ids,
        dimensions=dimensions,
        reason=reason,
        review_question_ids=question_ids,
        affected_answer_unit_ids=[unit.section_id for unit in packet.answer_units],
        draft_hash=packet.draft_hash,
    )


def _allocate_evidence(
    ledger: EvidenceLedger, outcome_view: dict[str, JsonValue], required: set[int]
) -> list[int]:
    selected = [number for number in ledger.citation_numbers() if number in required]
    outcomes = outcome_view.get("outcomes")
    if not isinstance(outcomes, list):
        return selected
    groups: list[list[int]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        question_ids = outcome.get("question_ids")
        if not isinstance(question_ids, list):
            continue
        groups.append(
            [
                number
                for number in ledger.citation_numbers()
                if (item := ledger.get(number)) is not None
                and any(
                    question_id in item.question_ids for question_id in question_ids
                )
            ]
        )
    while any(groups):
        for group in groups:
            if group:
                number = group.pop(0)
                if number not in selected:
                    selected.append(number)
    return selected


def _packet_evidence(
    ledger: EvidenceLedger, numbers: Iterable[int]
) -> list[ReviewEvidence]:
    evidence: list[ReviewEvidence] = []
    for number in numbers:
        item = ledger.get(number)
        if item is None:
            continue
        inspection = ledger.inspect(number)
        deliveries = inspection.get("deliveries")
        evidence.append(
            ReviewEvidence(
                citation=number,
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                text_hash=item.text_hash,
                text=item.text,
                metadata=item.metadata,
                deliveries=deliveries if isinstance(deliveries, list) else [],
            )
        )
    return evidence


def _delivery_receipts(
    ledger: EvidenceLedger, numbers: Iterable[int]
) -> list[JsonValue]:
    return [ledger.inspect(number) for number in numbers]


def _audit_records(
    candidate_audit: CandidateAudit | None, outcome_view: dict[str, JsonValue]
) -> list[CandidateAuditRecord]:
    if candidate_audit is None:
        return []
    unresolved_raw = outcome_view.get("unassessed_outcome_ids", [])
    unresolved = unresolved_raw if isinstance(unresolved_raw, list) else []
    resolutions = outcome_view.get("resolutions", [])
    if isinstance(resolutions, list):
        unresolved = [
            row.get("outcome_id")
            for row in resolutions
            if isinstance(row, dict) and row.get("status") == "unresolved"
        ]
    unresolved_ids = {item for item in unresolved if isinstance(item, str)}
    return [
        record
        for record in candidate_audit.records()
        if unresolved_ids.intersection(record.outcome_ids)
    ]


def _answer_units(answer: str) -> list[AnswerUnit]:
    sections = [
        section.strip() for section in re.split(r"\n\s*\n", answer) if section.strip()
    ]
    return [
        AnswerUnit(section_id=f"s{index + 1}", text=text)
        for index, text in enumerate(sections)
    ]


def _review_state(packet: FrozenReviewPacket) -> dict[str, JsonValue]:
    return {
        "packet_version": packet.version,
        "checklist_version": packet.checklist_version,
        "packet_hash": packet.packet_hash,
        "question": packet.question,
        "candidate_answer": packet.candidate_answer,
        "answer_units": [unit.model_dump(mode="json") for unit in packet.answer_units],
        "outcome_view": packet.outcome_view,
        "evidence": [item.model_dump(mode="json") for item in packet.evidence],
        "candidate_audit": [
            item.model_dump(mode="json") for item in packet.candidate_audit
        ],
        "delivery_receipts": packet.delivery_receipts,
    }


def _review_questions(packet: FrozenReviewPacket) -> dict[str, dict[str, JsonValue]]:
    questions: dict[str, dict[str, JsonValue]] = {
        "repair_required": {
            "type": "noul",
            "instructions": GUARDRAILS_V3_JEV_INSTRUCTION
            + " Is a material repair required before publication?",
        }
    }
    for outcome_id in _outcome_ids(packet.outcome_view):
        questions.update(
            {
                f"m1.{outcome_id}.inventory": _noul_question(
                    "Is this requested outcome missing from the reviewed inventory?"
                ),
                f"m1.{outcome_id}.evidence": _choice_question(
                    ["adequate", "missing", "incomplete", "conflicting", "uncertain"],
                    "Classify the evidence adequacy for this requested outcome.",
                ),
                f"m1.{outcome_id}.coverage": _choice_question(
                    ["addressed", "omitted", "partial", "uncertain"],
                    "Classify candidate answer coverage for this requested outcome.",
                ),
                f"m1.{outcome_id}.application": _choice_question(
                    ["correct", "missing_fact", "misapplied", "uncertain"],
                    "Classify fact application for this requested outcome.",
                ),
                f"m1.{outcome_id}.conditions": _choice_question(
                    ["preserved", "lost", "uncertain"],
                    "Classify preservation of material conditions and exceptions.",
                ),
            }
        )
    for dimension in _M2_DIMENSIONS:
        questions[f"m2.{dimension}.applicability"] = _choice_question(
            ["relevant", "not_applicable", "uncertain"],
            f"Is Method 2 dimension {dimension.upper()} relevant to this request?",
        )
        questions[f"m2.{dimension}.treatment"] = _choice_question(
            ["adequate", "missing", "partial", "uncertain"],
            f"Assuming Method 2 dimension {dimension.upper()} is relevant, classify its treatment.",
        )
    return questions


def _noul_question(instructions: str) -> dict[str, JsonValue]:
    return {
        "type": "noul",
        "instructions": GUARDRAILS_V3_JEV_INSTRUCTION + " " + instructions,
    }


def _choice_question(choices: list[str], instructions: str) -> dict[str, JsonValue]:
    return {
        "type": "choice",
        "choices": choices,
        "instructions": GUARDRAILS_V3_JEV_INSTRUCTION + " " + instructions,
    }


def _send_jev_review(
    *,
    encoded: bytes,
    credential: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
    questions: dict[str, dict[str, JsonValue]],
) -> CombinedJevReview:
    with httpx.Client(
        transport=transport or httpx.HTTPTransport(retries=0),
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        with client.stream(
            "POST",
            _JEV_ENDPOINT,
            headers={
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
            },
            content=encoded,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise ValueError("JEV response exceeds its bounded size")
                body.extend(chunk)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("JEV response must be an object")
    model = payload.get("model")
    answers = payload.get("answers")
    usage = payload.get("usage")
    if (
        not isinstance(model, str)
        or not re.fullmatch(
            r"jev-(?:latest|[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]+(?:\.[0-9]+)*(?:-[0-9]{8})?)",
            model,
        )
        or not isinstance(answers, dict)
        or set(answers) != set(questions)
        or not isinstance(usage, dict)
    ):
        raise ValueError("JEV response does not match the review protocol")
    normalized: dict[str, JsonValue] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError("JEV answer type does not match the requested question")
        if question["type"] == "noul":
            value = answer.get("noul")
            if type(value) not in {int, float} or not 0 <= float(value) <= 1:
                raise ValueError("JEV probability is invalid")
            normalized[question_id] = float(value)
        else:
            value = answer.get("choice")
            allowed = question.get("choices")
            if (
                not isinstance(value, str)
                or not isinstance(allowed, list)
                or value not in allowed
            ):
                raise ValueError("JEV choice label is invalid")
            normalized[question_id] = value
    input_tokens, output_tokens = (
        usage.get("input_tokens"),
        usage.get("output_tokens", 0),
    )
    if type(input_tokens) is not int or type(output_tokens) is not int:
        raise ValueError("JEV usage is invalid")
    return CombinedJevReview(
        actual_model=model,
        answers=normalized,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _failed_outcome(
    packet: FrozenReviewPacket, started: float, failure_reason: str
) -> GuardrailsV3ReviewOutcome:
    return GuardrailsV3ReviewOutcome(
        review_seconds=time.monotonic() - started,
        packet_hash=packet.packet_hash,
        failure_reason=failure_reason,
    )


def _outcome_ids(outcome_view: dict[str, JsonValue]) -> list[str]:
    outcomes = outcome_view.get("outcomes", [])
    if not isinstance(outcomes, list):
        return []
    result: list[str] = []
    for row in outcomes:
        if not isinstance(row, dict):
            continue
        outcome_id = row.get("outcome_id")
        if isinstance(outcome_id, str):
            result.append(outcome_id)
    return result


def _digest(value: object) -> str:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
