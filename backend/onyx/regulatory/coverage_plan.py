"""Bounded request-derived coverage planning for regulatory chat."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onyx.configs.chat_configs import REGULATORY_REVIEW_TIMEOUT_S
from onyx.llm.interfaces import LLM
from onyx.llm.models import ReasoningEffort
from onyx.prompts.regulatory_coverage_plan import (
    REGULATORY_COVERAGE_GAP_AUDIT_SYSTEM_PROMPT,
    REGULATORY_COVERAGE_PLAN_SYSTEM_PROMPT,
    REGULATORY_REQUEST_INVENTORY_SYSTEM_PROMPT,
)
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_REQUEST_CHARS = 24_000
_MAX_COVERAGE_ITEMS = 20
_MAX_REQUEST_SEGMENTS = 12
_MAX_REQUEST_SEGMENT_CHARS = 900
_MAX_ITEM_CHARS = 600
_MAX_BRANCHES_PER_ITEM = 6
_MAX_BRANCH_CHARS = 240
_MAX_EVIDENCE_DIMENSIONS_PER_ITEM = 6
_MAX_EVIDENCE_DIMENSION_CHARS = 300
_MAX_RETRIEVAL_QUERIES_PER_ITEM = _MAX_EVIDENCE_DIMENSIONS_PER_ITEM
_MAX_RETRIEVAL_QUERY_CHARS = 240
_MAX_COMPLETION_TEST_CHARS = 420
_MAX_SOURCE_ANCHORS_PER_ITEM = 12
_MAX_SOURCE_ANCHOR_CHARS = 180
_MAX_REQUEST_OBLIGATIONS = 20
_MAX_REQUEST_CONTEXT_ATOMS = 12
_MAX_REQUEST_CONTEXT_ATOM_CHARS = 240
_COVERAGE_PLAN_MAX_TOKENS = 12_000
_COVERAGE_PLAN_MAX_ATTEMPTS = 2
_REQUEST_INVENTORY_MAX_TOKENS = 12_000
_COVERAGE_GAP_AUDIT_MAX_TOKENS = 12_000
_COVERAGE_GAP_AUDIT_MAX_ADDITIONS = 4


def _strip_required(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("coverage text must not be blank")
    return stripped


class RegulatoryCoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_question: str = Field(min_length=1, max_length=_MAX_ITEM_CHARS)
    material_factual_branches: list[str] = Field(
        default_factory=list, max_length=_MAX_BRANCHES_PER_ITEM
    )
    evidence_dimensions: list[str] = Field(
        default_factory=list, max_length=_MAX_EVIDENCE_DIMENSIONS_PER_ITEM
    )
    retrieval_queries: list[str] = Field(
        default_factory=list, max_length=_MAX_RETRIEVAL_QUERIES_PER_ITEM
    )
    source_anchors: list[str] = Field(
        default_factory=list, max_length=_MAX_SOURCE_ANCHORS_PER_ITEM
    )
    request_segment_ids: list[str] = Field(
        default_factory=list, max_length=_MAX_REQUEST_SEGMENTS
    )
    request_obligation_ids: list[str] = Field(
        default_factory=list, max_length=_MAX_REQUEST_OBLIGATIONS
    )
    request_anchors: list[str] = Field(default_factory=list, max_length=3)
    request_anchor_groups: list[list[str]] = Field(
        default_factory=list, max_length=_MAX_REQUEST_OBLIGATIONS
    )
    completion_test: str = Field(min_length=1, max_length=_MAX_COMPLETION_TEST_CHARS)

    @field_validator("research_question", "completion_test")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("material_factual_branches")
    @classmethod
    def normalize_branches(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for branch in value:
            stripped = branch.strip()[:_MAX_BRANCH_CHARS].rstrip()
            identity = " ".join(stripped.casefold().split())
            if not stripped or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized

    @field_validator("evidence_dimensions")
    @classmethod
    def normalize_evidence_dimensions(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for dimension in value:
            stripped = dimension.strip()[:_MAX_EVIDENCE_DIMENSION_CHARS].rstrip()
            identity = " ".join(stripped.casefold().split())
            if not stripped or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized

    @field_validator("retrieval_queries")
    @classmethod
    def normalize_retrieval_queries(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for query in value:
            stripped = " ".join(query.split())[:_MAX_RETRIEVAL_QUERY_CHARS].rstrip()
            identity = stripped.casefold()
            if not stripped or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized

    @field_validator("source_anchors", mode="before")
    @classmethod
    def normalize_source_anchors(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for anchor in value:
            stripped = anchor.strip()[:_MAX_SOURCE_ANCHOR_CHARS].rstrip()
            identity = " ".join(stripped.casefold().split())
            if not stripped or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized[:_MAX_SOURCE_ANCHORS_PER_ITEM]

    @field_validator("request_segment_ids")
    @classmethod
    def normalize_request_segment_ids(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for segment_id in value:
            identifier = segment_id.strip().upper()
            if not re.fullmatch(r"R[1-9][0-9]*", identifier) or identifier in seen:
                continue
            seen.add(identifier)
            normalized.append(identifier)
        return normalized

    @field_validator("request_obligation_ids")
    @classmethod
    def normalize_request_obligation_ids(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for obligation_id in value:
            identifier = obligation_id.strip().upper()
            if not re.fullmatch(r"O[1-9][0-9]*", identifier) or identifier in seen:
                continue
            seen.add(identifier)
            normalized.append(identifier)
        return normalized

    @field_validator("request_anchors")
    @classmethod
    def normalize_request_anchors(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for anchor in value:
            stripped = " ".join(anchor.split())[:180].rstrip()
            identity = stripped.casefold()
            if not stripped or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized

    @field_validator("request_anchor_groups")
    @classmethod
    def normalize_request_anchor_groups(cls, value: list[list[str]]) -> list[list[str]]:
        normalized: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for group in value:
            bounded_group = cls.normalize_request_anchors(group)
            identity = tuple(anchor.casefold() for anchor in bounded_group)
            if not bounded_group or identity in seen:
                continue
            seen.add(identity)
            normalized.append(bounded_group)
        return normalized


class RegulatoryCoveragePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage_items: list[RegulatoryCoverageItem] = Field(
        default_factory=list, max_length=_MAX_COVERAGE_ITEMS
    )
    request_context_atoms: list[str] = Field(
        default_factory=list, max_length=_MAX_REQUEST_CONTEXT_ATOMS
    )

    @field_validator("request_context_atoms")
    @classmethod
    def normalize_request_context_atoms(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for atom in value:
            stripped = " ".join(atom.split())[:_MAX_REQUEST_CONTEXT_ATOM_CHARS]
            stripped = stripped.strip(" \t\r\n,;:.-")
            identity = stripped.casefold()
            if len(stripped.split()) < 3 or identity in seen:
                continue
            seen.add(identity)
            normalized.append(stripped)
        return normalized


class RegulatoryCoverageGapAudit(BaseModel):
    """Provider-facing delta contract for the independent gap audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage_items: list[RegulatoryCoverageItem] = Field(
        default_factory=list,
        max_length=_COVERAGE_GAP_AUDIT_MAX_ADDITIONS,
    )


class RegulatoryRequestObligation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_id: str
    request_grounded_text: str = Field(min_length=1, max_length=_MAX_ITEM_CHARS)
    verbatim_request_anchors: list[str] = Field(min_length=1, max_length=3)
    source_anchors: list[str] = Field(
        default_factory=list, max_length=_MAX_SOURCE_ANCHORS_PER_ITEM
    )
    request_segment_ids: list[str] = Field(
        default_factory=list, max_length=_MAX_REQUEST_SEGMENTS
    )

    @field_validator("obligation_id")
    @classmethod
    def normalize_obligation_id(cls, value: str) -> str:
        identifier = value.strip().upper()
        if not re.fullmatch(r"O[1-9][0-9]*", identifier):
            raise ValueError("request obligation ID must use the O<number> format")
        return identifier

    @field_validator("request_grounded_text")
    @classmethod
    def strip_grounded_text(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("verbatim_request_anchors")
    @classmethod
    def normalize_verbatim_request_anchors(cls, value: list[str]) -> list[str]:
        return RegulatoryCoverageItem.normalize_request_anchors(value)

    @field_validator("source_anchors", mode="before")
    @classmethod
    def normalize_source_anchors(cls, value: list[str]) -> list[str]:
        return RegulatoryCoverageItem.normalize_source_anchors(value)

    @field_validator("request_segment_ids")
    @classmethod
    def normalize_request_segment_ids(cls, value: list[str]) -> list[str]:
        return RegulatoryCoverageItem.normalize_request_segment_ids(value)


class RegulatoryRequestInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligations: list[RegulatoryRequestObligation] = Field(
        default_factory=list, max_length=_MAX_REQUEST_OBLIGATIONS
    )


class _RequestSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    text: str


_NUMBERED_REQUEST_MARKER = re.compile(r"(?m)(?:^|\n)\s*(?:\d{1,2}[.)]|[-*\u2022])\s+")
_QUESTION_SENTENCE = re.compile(r"(?:^|(?<=[.!;:\n]))\s*([^?\n]{8,}\?)")


def _bounded_preserving_ends(value: str, max_chars: int) -> str:
    """Bound request-derived text without always discarding trailing identifiers."""

    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    omission_marker = " ... "
    retained_chars = max_chars - len(omission_marker)
    leading_chars = retained_chars * 2 // 3
    trailing_chars = retained_chars - leading_chars
    return (
        normalized[:leading_chars].rstrip()
        + omission_marker
        + normalized[-trailing_chars:].lstrip()
    )


def _fallback_coverage_item(
    request_text: str,
    *,
    request_segment_id: str | None = None,
    request_obligation_id: str | None = None,
    source_anchors: list[str] | None = None,
    request_anchors: list[str] | None = None,
) -> RegulatoryCoverageItem:
    """Create a source-neutral bounded retrieval item from request text only."""

    research_question = _bounded_preserving_ends(request_text, _MAX_ITEM_CHARS)
    evidence_dimension = _bounded_preserving_ends(
        request_text, _MAX_EVIDENCE_DIMENSION_CHARS
    )
    return RegulatoryCoverageItem(
        research_question=research_question,
        evidence_dimensions=[evidence_dimension],
        retrieval_queries=[evidence_dimension],
        request_segment_ids=(
            [request_segment_id] if request_segment_id is not None else []
        ),
        request_obligation_ids=(
            [request_obligation_id] if request_obligation_id is not None else []
        ),
        source_anchors=source_anchors or [],
        request_anchors=request_anchors or [],
        completion_test=(
            "Find exact controlling text that resolves this request span, or "
            "identify its precise controlling-source gap."
        ),
    )


def _build_bounded_fallback_plan(
    request_segments: list[_RequestSegment],
) -> RegulatoryCoveragePlan | None:
    """Bound planner failure for syntax-explicit multi-part requests."""

    if request_segments:
        return RegulatoryCoveragePlan(
            coverage_items=[
                _fallback_coverage_item(
                    segment.text,
                    request_segment_id=segment.segment_id,
                )
                for segment in request_segments
            ]
        )
    return None


def _extract_explicit_request_segments(user_request: str) -> list[_RequestSegment]:
    """Extract only syntax-explicit deliverables without interpreting their domain."""

    bounded_request = user_request[:_MAX_REQUEST_CHARS]
    candidates: list[tuple[int, str]] = []
    numbered_markers = list(_NUMBERED_REQUEST_MARKER.finditer(bounded_request))
    for marker_index, marker in enumerate(numbered_markers):
        end = (
            numbered_markers[marker_index + 1].start()
            if marker_index + 1 < len(numbered_markers)
            else len(bounded_request)
        )
        candidates.append((marker.start(), bounded_request[marker.end() : end]))

    for question_match in _QUESTION_SENTENCE.finditer(bounded_request):
        candidates.append((question_match.start(1), question_match.group(1)))

    normalized_segments: list[tuple[int, str]] = []
    seen: set[str] = set()
    for position, candidate in sorted(candidates, key=lambda item: item[0]):
        text = " ".join(candidate.split())[:_MAX_REQUEST_SEGMENT_CHARS].rstrip()
        text = re.sub(r"^(?:\d{1,2}[.)]|[-*\u2022])\s+", "", text)
        identity = text.casefold().rstrip(" ?.;:")
        if len(identity) < 8 or identity in seen:
            continue
        # A numbered item that contains the same question sentence is the richer
        # representation; do not create a second obligation for its substring.
        if any(identity in prior_identity for prior_identity in seen):
            continue
        seen.add(identity)
        normalized_segments.append((position, text))

    normalized_segments.sort(key=lambda item: item[0])
    return [
        _RequestSegment(segment_id=f"R{index}", text=text)
        for index, (_, text) in enumerate(
            normalized_segments[:_MAX_REQUEST_SEGMENTS], start=1
        )
    ]


def _extract_request_context_atoms(user_request: str) -> list[str]:
    """Preserve syntax-level scenario facts that precede explicit questions.

    This deliberately performs no legal interpretation. A colon-introduced
    list is split into its written clauses because embeddings for one clause
    are less likely to be diluted by unrelated facts in the same sentence.
    """

    bounded_request = user_request[:_MAX_REQUEST_CHARS]
    boundary_candidates = [
        match.start() for match in _NUMBERED_REQUEST_MARKER.finditer(bounded_request)
    ]
    boundary_candidates.extend(
        match.start(1) for match in _QUESTION_SENTENCE.finditer(bounded_request)
    )
    if not boundary_candidates:
        return []
    preface = bounded_request[: min(boundary_candidates)].strip()
    if not preface:
        return []

    raw_atoms: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", preface):
        sentence = sentence.strip(" \t\r\n,;:.-")
        if not sentence:
            continue
        if ":" not in sentence:
            raw_atoms.append(sentence)
            continue
        _lead, listed_text = sentence.split(":", 1)
        raw_atoms.extend(re.split(r"\s*[,;]\s*", listed_text))

    return RegulatoryCoveragePlan.normalize_request_context_atoms(raw_atoms)[
        :_MAX_REQUEST_CONTEXT_ATOMS
    ]


def _ensure_explicit_request_coverage(
    plan: RegulatoryCoveragePlan,
    request_segments: list[_RequestSegment],
) -> RegulatoryCoveragePlan:
    """Append request-text fallbacks only for explicit segments the model omitted."""

    if not request_segments:
        return plan
    valid_ids = {segment.segment_id for segment in request_segments}
    normalized_items: list[RegulatoryCoverageItem] = []
    covered_ids: set[str] = set()
    for item in plan.coverage_items:
        retained_ids = [
            segment_id
            for segment_id in item.request_segment_ids
            if segment_id in valid_ids
        ]
        covered_ids.update(retained_ids)
        normalized_items.append(
            item.model_copy(update={"request_segment_ids": retained_ids})
        )

    for segment in request_segments:
        if (
            segment.segment_id in covered_ids
            or len(normalized_items) >= _MAX_COVERAGE_ITEMS
        ):
            continue
        normalized_items.append(
            _fallback_coverage_item(
                segment.text,
                request_segment_id=segment.segment_id,
            )
        )

    return RegulatoryCoveragePlan(coverage_items=normalized_items)


def _ensure_request_obligation_coverage(
    plan: RegulatoryCoveragePlan,
    inventory: RegulatoryRequestInventory,
) -> RegulatoryCoveragePlan:
    """Append request-only rows for inventory obligations the plan omitted."""

    if not inventory.obligations:
        return plan
    valid_ids = {obligation.obligation_id for obligation in inventory.obligations}
    normalized_items: list[RegulatoryCoverageItem] = []
    covered_ids: set[str] = set()
    for item in plan.coverage_items:
        retained_ids = [
            obligation_id
            for obligation_id in item.request_obligation_ids
            if obligation_id in valid_ids
        ]
        covered_ids.update(retained_ids)
        normalized_items.append(
            item.model_copy(update={"request_obligation_ids": retained_ids})
        )

    for obligation in inventory.obligations:
        if (
            obligation.obligation_id in covered_ids
            or len(normalized_items) >= _MAX_COVERAGE_ITEMS
        ):
            continue
        normalized_items.append(
            _fallback_coverage_item(
                obligation.request_grounded_text,
                request_segment_id=(
                    obligation.request_segment_ids[0]
                    if obligation.request_segment_ids
                    else None
                ),
                request_obligation_id=obligation.obligation_id,
                source_anchors=obligation.source_anchors,
                request_anchors=obligation.verbatim_request_anchors,
            )
        )
    return RegulatoryCoveragePlan(coverage_items=normalized_items)


def _normalize_for_verbatim_check(value: str) -> str:
    return " ".join(value.casefold().split())


def _sanitize_request_inventory(
    inventory: RegulatoryRequestInventory,
    user_request: str,
) -> RegulatoryRequestInventory:
    """Keep only obligations carrying genuine contiguous request phrases."""

    normalized_request = _normalize_for_verbatim_check(user_request)
    obligations: list[RegulatoryRequestObligation] = []
    seen_ids: set[str] = set()
    for obligation in inventory.obligations:
        if obligation.obligation_id in seen_ids:
            continue
        verified_anchors = [
            anchor
            for anchor in obligation.verbatim_request_anchors
            if _normalize_for_verbatim_check(anchor) in normalized_request
            and len(anchor) <= 96
            and len(anchor.split()) <= 12
        ]
        if not verified_anchors:
            continue
        seen_ids.add(obligation.obligation_id)
        obligations.append(
            obligation.model_copy(update={"verbatim_request_anchors": verified_anchors})
        )
    return RegulatoryRequestInventory(obligations=obligations)


def _attach_verified_request_anchors(
    plan: RegulatoryCoveragePlan,
    inventory: RegulatoryRequestInventory,
) -> RegulatoryCoveragePlan:
    if not inventory.obligations:
        return plan
    anchors_by_obligation_id = {
        obligation.obligation_id: obligation.verbatim_request_anchors
        for obligation in inventory.obligations
    }
    items: list[RegulatoryCoverageItem] = []
    for item in plan.coverage_items:
        anchors: list[str] = []
        anchor_groups: list[list[str]] = []
        for obligation_id in item.request_obligation_ids:
            obligation_anchors = anchors_by_obligation_id.get(obligation_id, [])
            if obligation_anchors:
                anchor_groups.append(obligation_anchors)
                anchors.extend(obligation_anchors)
        items.append(
            item.model_copy(
                update={
                    "request_anchors": RegulatoryCoverageItem.normalize_request_anchors(
                        anchors
                    ),
                    "request_anchor_groups": (
                        RegulatoryCoverageItem.normalize_request_anchor_groups(
                            anchor_groups
                        )
                    ),
                }
            )
        )
    return RegulatoryCoveragePlan(coverage_items=items)


def _normalized_item_identity(item: RegulatoryCoverageItem) -> str:
    return " ".join(item.research_question.casefold().split())


def _dimension_query_pairs(
    item: RegulatoryCoverageItem,
) -> list[tuple[str, str]]:
    if not item.evidence_dimensions:
        return []
    queries = (
        item.retrieval_queries
        if len(item.retrieval_queries) == len(item.evidence_dimensions)
        else item.evidence_dimensions
    )
    return list(zip(item.evidence_dimensions, queries, strict=True))


def _merge_matching_coverage_items(
    existing: RegulatoryCoverageItem,
    audit: RegulatoryCoverageItem,
) -> RegulatoryCoverageItem:
    """Preserve atomic audit rows even when their umbrella question matches."""

    dimension_pairs = _dimension_query_pairs(existing)
    dimension_identities = {
        " ".join(dimension.casefold().split()) for dimension, _ in dimension_pairs
    }
    for dimension, query in _dimension_query_pairs(audit):
        dimension_identity = " ".join(dimension.casefold().split())
        if (
            dimension_identity in dimension_identities
            or len(dimension_pairs) >= _MAX_EVIDENCE_DIMENSIONS_PER_ITEM
        ):
            continue
        dimension_identities.add(dimension_identity)
        dimension_pairs.append((dimension, query))

    payload = existing.model_dump()
    payload.update(
        {
            "material_factual_branches": RegulatoryCoverageItem.normalize_branches(
                [
                    *existing.material_factual_branches,
                    *audit.material_factual_branches,
                ]
            )[:_MAX_BRANCHES_PER_ITEM],
            "evidence_dimensions": [dimension for dimension, _ in dimension_pairs],
            "retrieval_queries": [query for _, query in dimension_pairs],
            "source_anchors": RegulatoryCoverageItem.normalize_source_anchors(
                [*existing.source_anchors, *audit.source_anchors]
            )[:_MAX_SOURCE_ANCHORS_PER_ITEM],
            "request_segment_ids": (
                RegulatoryCoverageItem.normalize_request_segment_ids(
                    [*existing.request_segment_ids, *audit.request_segment_ids]
                )[:_MAX_REQUEST_SEGMENTS]
            ),
            "request_obligation_ids": (
                RegulatoryCoverageItem.normalize_request_obligation_ids(
                    [
                        *existing.request_obligation_ids,
                        *audit.request_obligation_ids,
                    ]
                )[:_MAX_REQUEST_OBLIGATIONS]
            ),
        }
    )
    return RegulatoryCoverageItem.model_validate(payload)


def _merge_coverage_gap_audit(
    plan: RegulatoryCoveragePlan,
    audit: RegulatoryCoverageGapAudit,
) -> RegulatoryCoveragePlan:
    """Merge a bounded set of genuinely distinct atomic audit findings."""

    merged_items = list(plan.coverage_items)
    seen_identities = {_normalized_item_identity(item) for item in merged_items}
    added_count = 0
    for item in audit.coverage_items:
        identity = _normalized_item_identity(item)
        if identity in seen_identities:
            existing_index = next(
                index
                for index, existing_item in enumerate(merged_items)
                if _normalized_item_identity(existing_item) == identity
            )
            merged_item = _merge_matching_coverage_items(
                merged_items[existing_index], item
            )
            if merged_item != merged_items[existing_index]:
                merged_items[existing_index] = merged_item
                added_count += 1
                if added_count >= _COVERAGE_GAP_AUDIT_MAX_ADDITIONS:
                    break
            continue
        if (
            len(merged_items) >= _MAX_COVERAGE_ITEMS
            or added_count >= _COVERAGE_GAP_AUDIT_MAX_ADDITIONS
        ):
            break
        merged_items.append(item)
        seen_identities.add(identity)
        added_count += 1
    if not added_count:
        return plan
    return RegulatoryCoveragePlan(coverage_items=merged_items)


def build_regulatory_coverage_plan(
    llm: LLM,
    *,
    user_request: str,
) -> RegulatoryCoveragePlan | None:
    """Create a request-only plan with a bounded request-derived fallback."""

    stripped_request = user_request.strip()
    if not stripped_request:
        return None
    bounded_request = stripped_request[:_MAX_REQUEST_CHARS]
    request_segments = _extract_explicit_request_segments(bounded_request)
    request_context_atoms = _extract_request_context_atoms(bounded_request)
    inventory = RegulatoryRequestInventory()
    logger.info(
        "Regulatory coverage planning started model=%s explicit_segments=%d",
        llm.config.model_name,
        len(request_segments),
    )
    try:
        try:
            inventory = generate_structured(
                llm,
                flow=LLMFlow.REGULATORY_REQUEST_INVENTORY,
                system_prompt=REGULATORY_REQUEST_INVENTORY_SYSTEM_PROMPT,
                user_prompt=json.dumps(
                    {
                        "user_request": bounded_request,
                        "request_outline": [
                            segment.model_dump() for segment in request_segments
                        ],
                    },
                    ensure_ascii=False,
                ),
                response_model=RegulatoryRequestInventory,
                timeout_override=REGULATORY_REVIEW_TIMEOUT_S,
                max_tokens=_REQUEST_INVENTORY_MAX_TOKENS,
                reasoning_effort=ReasoningEffort.HIGH,
                # Retry only when the provider response cannot satisfy the
                # schema. Valid responses retain their single-call cost.
                max_attempts=2,
            )
            inventory = _sanitize_request_inventory(inventory, bounded_request)
        except Exception:
            logger.exception(
                "Regulatory request inventory failed; continuing with syntax outline"
            )
        plan = generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_COVERAGE_PLAN,
            system_prompt=REGULATORY_COVERAGE_PLAN_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "user_request": bounded_request,
                    "request_outline": [
                        segment.model_dump() for segment in request_segments
                    ],
                    "request_inventory": inventory.model_dump(),
                    "request_truncated": len(stripped_request) > len(bounded_request),
                },
                ensure_ascii=False,
            ),
            response_model=RegulatoryCoveragePlan,
            timeout_override=REGULATORY_REVIEW_TIMEOUT_S,
            max_tokens=_COVERAGE_PLAN_MAX_TOKENS,
            reasoning_effort=ReasoningEffort.HIGH,
            max_attempts=_COVERAGE_PLAN_MAX_ATTEMPTS,
        )
        try:
            audit = generate_structured(
                llm,
                flow=LLMFlow.REGULATORY_COVERAGE_GAP_AUDIT,
                system_prompt=REGULATORY_COVERAGE_GAP_AUDIT_SYSTEM_PROMPT,
                user_prompt=json.dumps(
                    {
                        "user_request": bounded_request,
                        "request_outline": [
                            segment.model_dump() for segment in request_segments
                        ],
                        "request_inventory": inventory.model_dump(),
                        "draft_plan": plan.model_dump(),
                    },
                    ensure_ascii=False,
                ),
                response_model=RegulatoryCoverageGapAudit,
                timeout_override=REGULATORY_REVIEW_TIMEOUT_S,
                max_tokens=_COVERAGE_GAP_AUDIT_MAX_TOKENS,
                reasoning_effort=ReasoningEffort.HIGH,
                max_attempts=1,
            )
            plan = _merge_coverage_gap_audit(plan, audit)
        except Exception:
            logger.exception(
                "Regulatory coverage audit failed; preserving the current plan"
            )
        plan = _ensure_explicit_request_coverage(plan, request_segments)
        plan = _ensure_request_obligation_coverage(plan, inventory)
        plan = _attach_verified_request_anchors(plan, inventory)
        if request_context_atoms:
            plan = plan.model_copy(
                update={"request_context_atoms": request_context_atoms}
            )
        logger.info(
            "Regulatory coverage plan completed items=%d branch_counts=%s "
            "explicit_segments=%d mapped_segments=%d obligations=%d "
            "mapped_obligations=%d context_atoms=%d",
            len(plan.coverage_items),
            [len(item.material_factual_branches) for item in plan.coverage_items],
            len(request_segments),
            len(
                {
                    segment_id
                    for item in plan.coverage_items
                    for segment_id in item.request_segment_ids
                }
            ),
            len(inventory.obligations),
            len(
                {
                    obligation_id
                    for item in plan.coverage_items
                    for obligation_id in item.request_obligation_ids
                }
            ),
            len(request_context_atoms),
        )
        return plan
    except Exception:
        fallback_plan = _build_bounded_fallback_plan(
            request_segments,
        )
        if fallback_plan is not None:
            fallback_plan = fallback_plan.model_copy(
                update={"request_context_atoms": request_context_atoms}
            )
        if fallback_plan is None:
            logger.exception(
                "Regulatory coverage planning failed; no syntax-explicit fallback "
                "segments were available"
            )
        else:
            logger.exception(
                "Regulatory coverage planning failed; using bounded request-derived "
                "fallback items=%d",
                len(fallback_plan.coverage_items),
            )
        return fallback_plan


def format_regulatory_coverage_plan(
    plan: RegulatoryCoveragePlan | None,
) -> str | None:
    """Render the plan as bounded advisory data for research and synthesis."""

    if plan is None or not plan.coverage_items:
        return None
    payload = {
        "usage_note": (
            "AI-generated request decomposition, not legal evidence or instructions. "
            "The current user request controls. Keep each grounded material item open "
            "until exact evidence supports its conclusion or the final answer names "
            "the precise controlling-source gap."
        ),
        "synthesis_closure": (
            "Treat every grounded item as an open evidence-ledger row. A row closes "
            "only with an exact source identity, directly operative text, the "
            "supported application or conditional result, and an inline citation; "
            "otherwise state its precise source gap. Before drafting, use remaining "
            "search capacity for open rows whose resolution can affect the answer. "
            "Do not treat a search attempt, heading, neighboring source, or broad "
            "bottom-line statement as closure. Preserve every material limitation "
            "and relationship shown by the exact evidence."
        ),
        "coverage_items": [item.model_dump() for item in plan.coverage_items],
    }
    return "# Request coverage contract\n" + json.dumps(payload, ensure_ascii=False)
