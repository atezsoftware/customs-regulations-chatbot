"""Pre-synthesis evidence coverage and claim-to-source matrix."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from onyx.configs.chat_configs import REGULATORY_REVIEW_TIMEOUT_S
from onyx.llm.interfaces import LLM
from onyx.llm.models import ReasoningEffort
from onyx.prompts.regulatory_evidence_matrix import (
    REGULATORY_EVIDENCE_MATRIX_SYSTEM_PROMPT,
)
from onyx.regulatory.candidate_answer_review import CandidateAnswerEvidenceChunk
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_REQUEST_CHARS = 24_000
_MAX_COVERAGE_CONTRACT_CHARS = 24_000
_MAX_EVIDENCE_CHUNKS = 320
_MAX_EVIDENCE_CHARS = 260_000
_MAX_EVIDENCE_CHARS_PER_CHUNK = 3_000
_MAX_MATRIX_ROWS = 64
_MAX_ROW_TARGET_CHARS = 360
_MAX_PROPOSITION_CHARS = 600
_MAX_MISSING_ASPECTS = 5
_MAX_MISSING_ASPECT_CHARS = 220
_MAX_DOCUMENT_NUMBERS = 12
_MAX_RECOVERY_QUERY_CHARS = 420
_MAX_NAVIGATION_LEADS = 128
_MAX_NAVIGATION_VALUE_CHARS = 420
_MAX_NAVIGATION_TARGETS = 6
_MATRIX_MAX_TOKENS = 24_000


class EvidenceCoverageStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONFLICTING = "conflicting"


class RegulatoryEvidenceMatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(min_length=1, max_length=_MAX_ROW_TARGET_CHARS)
    target_ids: list[str] = Field(default_factory=list, max_length=24)
    status: EvidenceCoverageStatus
    supported_proposition: str = Field(default="", max_length=_MAX_PROPOSITION_CHARS)
    document_numbers: list[int] = Field(
        default_factory=list, max_length=_MAX_DOCUMENT_NUMBERS
    )
    missing_aspects: list[str] = Field(
        default_factory=list, max_length=_MAX_MISSING_ASPECTS
    )
    recovery_query: str | None = Field(
        default=None, max_length=_MAX_RECOVERY_QUERY_CHARS
    )

    @field_validator("target")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("matrix target must not be blank")
        return value

    @field_validator("supported_proposition")
    @classmethod
    def normalize_proposition(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("document_numbers")
    @classmethod
    def normalize_document_numbers(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(number for number in value if number > 0))[
            :_MAX_DOCUMENT_NUMBERS
        ]

    @field_validator("target_ids")
    @classmethod
    def normalize_target_ids(cls, value: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                identifier.strip().upper()
                for identifier in value
                if re.fullmatch(r"T[1-9][0-9]*", identifier.strip().upper())
            )
        )[:24]

    @field_validator("missing_aspects")
    @classmethod
    def normalize_missing_aspects(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for aspect in value:
            bounded = " ".join(aspect.split())[:_MAX_MISSING_ASPECT_CHARS].rstrip()
            identity = bounded.casefold()
            if not bounded or identity in seen:
                continue
            seen.add(identity)
            normalized.append(bounded)
        return normalized[:_MAX_MISSING_ASPECTS]

    @field_validator("recovery_query")
    @classmethod
    def normalize_recovery_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_status_support(self) -> "RegulatoryEvidenceMatrixRow":
        if (
            self.status
            in {
                EvidenceCoverageStatus.SUPPORTED,
                EvidenceCoverageStatus.CONFLICTING,
            }
            and not self.document_numbers
        ):
            raise ValueError("supported or conflicting rows require exact documents")
        if self.status is EvidenceCoverageStatus.SUPPORTED and self.recovery_query:
            raise ValueError("supported rows cannot request recovery")
        return self


class RegulatoryEvidenceMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[RegulatoryEvidenceMatrixRow] = Field(
        default_factory=list, max_length=_MAX_MATRIX_ROWS
    )


class RegulatoryNavigationLead(BaseModel):
    """Metadata-only source-outline lead associated with retrieval targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_title: str = Field(min_length=1, max_length=_MAX_NAVIGATION_VALUE_CHARS)
    article_key: str = Field(min_length=1, max_length=_MAX_NAVIGATION_VALUE_CHARS)
    heading_label: str = Field(min_length=1, max_length=_MAX_NAVIGATION_VALUE_CHARS)
    research_targets: list[str] = Field(
        default_factory=list,
        max_length=_MAX_NAVIGATION_TARGETS,
    )

    @field_validator("document_title", "article_key", "heading_label")
    @classmethod
    def normalize_required_value(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("navigation lead value must not be blank")
        return normalized

    @field_validator("research_targets")
    @classmethod
    def normalize_research_targets(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(target.split()) for target in value]
        return list(dict.fromkeys(target for target in normalized if target))[
            :_MAX_NAVIGATION_TARGETS
        ]


def merge_regulatory_evidence_matrices(
    primary: RegulatoryEvidenceMatrix | None,
    independent: RegulatoryEvidenceMatrix | None,
) -> RegulatoryEvidenceMatrix | None:
    """Union independent audits without collapsing distinct propositions."""

    if primary is None:
        return independent
    if independent is None:
        return primary
    merged_rows = list(primary.rows)
    seen = {
        (
            " ".join(row.target.casefold().split()),
            " ".join(row.supported_proposition.casefold().split()),
            row.status,
        )
        for row in merged_rows
    }
    for row in independent.rows:
        identity = (
            " ".join(row.target.casefold().split()),
            " ".join(row.supported_proposition.casefold().split()),
            row.status,
        )
        if identity in seen:
            continue
        seen.add(identity)
        merged_rows.append(row)
        if len(merged_rows) == _MAX_MATRIX_ROWS:
            break
    return RegulatoryEvidenceMatrix(rows=merged_rows)


class _RegulatoryEvidenceMatrixDraftRow(BaseModel):
    """Provider-facing row that tolerates status/document inconsistencies.

    Gemini can correctly identify a missing target while emitting `supported` with
    an empty document list. Parse that useful structure first; the evidence-bound
    normalization below then downgrades it before the strict public model is built.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(min_length=1, max_length=_MAX_ROW_TARGET_CHARS)
    target_ids: list[str] = Field(default_factory=list, max_length=24)
    status: EvidenceCoverageStatus
    supported_proposition: str = Field(default="", max_length=_MAX_PROPOSITION_CHARS)
    document_numbers: list[int] = Field(
        default_factory=list, max_length=_MAX_DOCUMENT_NUMBERS
    )
    missing_aspects: list[str] = Field(
        default_factory=list, max_length=_MAX_MISSING_ASPECTS
    )
    recovery_query: str | None = Field(
        default=None, max_length=_MAX_RECOVERY_QUERY_CHARS
    )


class _RegulatoryEvidenceMatrixDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[_RegulatoryEvidenceMatrixDraftRow] = Field(
        default_factory=list, max_length=_MAX_MATRIX_ROWS
    )


def _bounded(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n\n[... bounded input omitted ...]\n\n"
    retained = max_chars - len(marker)
    leading = retained * 2 // 3
    return value[:leading] + marker + value[-(retained - leading) :]


def _compact_evidence(
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    selected = list(evidence_chunks[:_MAX_EVIDENCE_CHUNKS])
    if not selected:
        return [], []
    per_chunk_chars = min(
        _MAX_EVIDENCE_CHARS_PER_CHUNK,
        max(320, _MAX_EVIDENCE_CHARS // len(selected)),
    )
    target_ids: dict[str, str] = {}
    compact_chunks: list[dict[str, object]] = []
    for chunk in selected:
        if chunk.retrieval_number is None or not chunk.content.strip():
            continue
        chunk_target_ids: list[str] = []
        for target in chunk.research_target.splitlines():
            normalized_target = " ".join(target.split())
            if not normalized_target:
                continue
            target_id = target_ids.get(normalized_target)
            if target_id is None:
                target_id = f"T{len(target_ids) + 1}"
                target_ids[normalized_target] = target_id
            chunk_target_ids.append(target_id)
        compact_chunks.append(
            {
                "document": chunk.retrieval_number,
                "heading": chunk.heading,
                "target_ids": chunk_target_ids,
                "content": _bounded(chunk.content, per_chunk_chars),
                "content_truncated": chunk.content_truncated
                or len(chunk.content) > per_chunk_chars,
            }
        )
    target_registry = [
        {"target_id": target_id, "target": target}
        for target, target_id in target_ids.items()
    ]
    return compact_chunks, target_registry


def _fallback_recovery_query(target: str) -> str:
    prefix = "Specific evidence target:"
    normalized = " ".join(target.split())
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :].lstrip()
    return normalized[:_MAX_RECOVERY_QUERY_CHARS].rstrip(" .;:")


def build_regulatory_evidence_matrix(
    llm: LLM,
    *,
    user_request: str,
    coverage_contract: str | None,
    evidence_chunks: Sequence[CandidateAnswerEvidenceChunk],
    navigation_leads: Sequence[RegulatoryNavigationLead] = (),
    prior_matrix: RegulatoryEvidenceMatrix | None = None,
) -> RegulatoryEvidenceMatrix | None:
    """Map request-derived targets to exact retrieved evidence before synthesis."""

    compact_evidence, target_registry = _compact_evidence(evidence_chunks)
    if not user_request.strip() or not compact_evidence:
        return None
    payload = json.dumps(
        {
            "user_request": _bounded(user_request.strip(), _MAX_REQUEST_CHARS),
            "coverage_contract": (
                _bounded(coverage_contract.strip(), _MAX_COVERAGE_CONTRACT_CHARS)
                if coverage_contract and coverage_contract.strip()
                else None
            ),
            "research_targets": target_registry,
            "navigation_leads": [
                lead.model_dump(mode="json")
                for lead in navigation_leads[:_MAX_NAVIGATION_LEADS]
            ],
            "prior_open_rows": (
                [
                    row.model_dump(mode="json")
                    for row in prior_matrix.rows
                    if row.status is not EvidenceCoverageStatus.SUPPORTED
                ]
                if prior_matrix is not None
                else None
            ),
            "evidence_chunks": compact_evidence,
        },
        ensure_ascii=False,
    )
    available_documents = {
        document
        for chunk in compact_evidence
        for document in [chunk.get("document")]
        if isinstance(document, int)
    }
    available_target_ids = {target["target_id"] for target in target_registry}
    try:
        draft = generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_EVIDENCE_MATRIX,
            system_prompt=REGULATORY_EVIDENCE_MATRIX_SYSTEM_PROMPT,
            user_prompt=payload,
            response_model=_RegulatoryEvidenceMatrixDraft,
            timeout_override=REGULATORY_REVIEW_TIMEOUT_S,
            max_tokens=_MATRIX_MAX_TOKENS,
            reasoning_effort=ReasoningEffort.HIGH,
        )
    except Exception:
        logger.exception("Regulatory evidence matrix generation failed")
        return prior_matrix

    normalized_rows: list[RegulatoryEvidenceMatrixRow] = []
    seen_rows: set[tuple[str, str, EvidenceCoverageStatus]] = set()
    for row in draft.rows:
        target_identity = " ".join(row.target.casefold().split())
        proposition_identity = " ".join(row.supported_proposition.casefold().split())
        row_identity = (target_identity, proposition_identity, row.status)
        if row_identity in seen_rows:
            continue
        seen_rows.add(row_identity)
        valid_documents = [
            number for number in row.document_numbers if number in available_documents
        ]
        valid_target_ids = [
            target_id
            for target_id in row.target_ids
            if target_id in available_target_ids
        ]
        status = row.status
        if status is not EvidenceCoverageStatus.MISSING and not valid_documents:
            status = EvidenceCoverageStatus.MISSING
        normalized_rows.append(
            RegulatoryEvidenceMatrixRow(
                target=row.target,
                status=status,
                supported_proposition=row.supported_proposition,
                target_ids=valid_target_ids,
                document_numbers=valid_documents,
                missing_aspects=row.missing_aspects,
                recovery_query=(
                    None
                    if status is EvidenceCoverageStatus.SUPPORTED
                    else row.recovery_query or _fallback_recovery_query(row.target)
                ),
            )
        )

    if prior_matrix is not None:
        prior_target_ids = {
            target_id for row in prior_matrix.rows for target_id in row.target_ids
        }
        prior_documents_by_target_id: dict[str, set[int]] = {}
        for prior_row in prior_matrix.rows:
            for target_id in prior_row.target_ids:
                prior_documents_by_target_id.setdefault(target_id, set()).update(
                    prior_row.document_numbers
                )
        updates_by_target = {
            " ".join(row.target.casefold().split()): row for row in normalized_rows
        }
        updates_by_target_id = {
            target_id: row for row in normalized_rows for target_id in row.target_ids
        }
        merged_rows: list[RegulatoryEvidenceMatrixRow] = []
        consumed_update_ids: set[int] = set()
        for prior_row in prior_matrix.rows:
            replacement: RegulatoryEvidenceMatrixRow | None = None
            if prior_row.status is not EvidenceCoverageStatus.SUPPORTED:
                replacement = next(
                    (
                        updates_by_target_id[target_id]
                        for target_id in prior_row.target_ids
                        if target_id in updates_by_target_id
                    ),
                    None,
                ) or updates_by_target.get(
                    " ".join(prior_row.target.casefold().split())
                )
            if replacement is not None:
                consumed_update_ids.add(id(replacement))
            merged_rows.append(replacement or prior_row)
        for row in normalized_rows:
            if id(row) in consumed_update_ids or len(merged_rows) >= _MAX_MATRIX_ROWS:
                continue
            # A refresh may discover a genuinely new atomic effect, but Gemini can
            # also restate every already-supported row. Keep a same-target append
            # only when it brings exact evidence that was not available to any
            # prior row for that target. Rows tied to a newly retrieved target ID
            # remain eligible even when their wording resembles an older row.
            row_prior_target_ids = set(row.target_ids) & prior_target_ids
            if row_prior_target_ids and set(row.target_ids) <= prior_target_ids:
                prior_documents = {
                    document
                    for target_id in row_prior_target_ids
                    for document in prior_documents_by_target_id.get(target_id, set())
                }
                if not set(row.document_numbers) - prior_documents:
                    continue
            merged_rows.append(row)
        return RegulatoryEvidenceMatrix(rows=merged_rows)

    return RegulatoryEvidenceMatrix(rows=normalized_rows)


def evidence_matrix_recovery_queries(
    matrix: RegulatoryEvidenceMatrix | None,
    *,
    limit: int,
) -> list[str]:
    """Return deduplicated focused searches for material open matrix rows."""

    if matrix is None or limit <= 0:
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for row in matrix.rows:
        if row.status is EvidenceCoverageStatus.SUPPORTED or not row.recovery_query:
            continue
        identity = " ".join(row.recovery_query.casefold().split())
        if identity in seen:
            continue
        seen.add(identity)
        queries.append(row.recovery_query)
        if len(queries) == limit:
            break
    return queries


def format_regulatory_evidence_matrix(
    matrix: RegulatoryEvidenceMatrix | None,
) -> str | None:
    """Render the matrix as advisory synthesis data without replacing evidence."""

    if matrix is None or not matrix.rows:
        return None
    payload = {
        "usage_note": (
            "AI-generated evidence analysis, not legal evidence. Every proposition "
            "must still be verified against and cited to the exact numbered evidence. "
            "Do not publish a supported row without stating its material result and "
            "inline citation. For an open row, use newly retrieved exact evidence or "
            "state the precise controlling-source gap."
        ),
        "rows": [row.model_dump(mode="json") for row in matrix.rows],
    }
    return "# Claim-source evidence matrix\n" + json.dumps(payload, ensure_ascii=False)
