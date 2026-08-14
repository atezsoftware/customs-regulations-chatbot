"""One deterministic, direct retrieval for a reviewed answer-support gap."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from onyx.chat.citation_processor import CitationMapping
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerClaimIssue,
    CandidateAnswerEvidenceChunk,
    CandidateAnswerReviewResult,
    ClaimKind,
    ClaimSpanSource,
    build_candidate_answer_evidence_chunk,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import SearchToolOverrideKwargs, ToolResponse
from onyx.tools.tool_implementations.search.search_tool import SearchTool

_MAX_BATCHED_RECOVERY_ISSUES = 5


def select_priority_recovery_issue(
    review: CandidateAnswerReviewResult,
) -> CandidateAnswerClaimIssue | None:
    """Choose one recoverable issue by legal materiality and draft order."""

    issues = select_priority_recovery_issues(review, limit=1)
    return issues[0] if issues else None


def select_priority_recovery_issues(
    review: CandidateAnswerReviewResult,
    *,
    limit: int = _MAX_BATCHED_RECOVERY_ISSUES,
) -> list[CandidateAnswerClaimIssue]:
    """Choose a small, ordered, query-distinct set of recoverable issues."""

    if limit < 1:
        return []

    eligible = [
        (issue_index, issue)
        for issue_index, issue in enumerate(review.advisory_claim_issues)
        if issue.recovery_query is not None
    ]
    if not eligible:
        return []

    def priority(
        indexed_issue: tuple[int, CandidateAnswerClaimIssue],
    ) -> tuple[int, int, int, int]:
        issue_index, issue = indexed_issue
        span = issue.claim_span
        source_priority = (
            0
            if span is not None and span.source is ClaimSpanSource.CANDIDATE_ANSWER
            else 1
        )
        span_start = span.start if span is not None else 2**31 - 1
        return (
            0 if issue.claim_kind is ClaimKind.LEGAL_RULE else 1,
            source_priority,
            span_start,
            issue_index,
        )

    selected: list[CandidateAnswerClaimIssue] = []
    seen_queries: set[str] = set()
    for _, issue in sorted(eligible, key=priority):
        assert issue.recovery_query is not None
        query_identity = " ".join(issue.recovery_query.casefold().split())
        if query_identity in seen_queries:
            continue
        seen_queries.add(query_identity)
        selected.append(issue)
        if len(selected) == limit:
            break
    return selected


def run_single_gap_recovery(
    *,
    search_tool: SearchTool,
    issue: CandidateAnswerClaimIssue,
    starting_citation_num: int,
    placement: Placement,
) -> ToolResponse:
    """Execute the review-selected query exactly once through SearchTool.

    The verifier already emits a self-contained recovery query. Bypassing the
    conversational rephraser both preserves that query and avoids requiring a
    synthetic chat history for this server-orchestrated search.
    """

    if issue.recovery_query is None:
        raise ValueError("recovery issue must include a recovery_query")
    return search_tool.run(
        placement=placement,
        override_kwargs=SearchToolOverrideKwargs(
            starting_citation_num=starting_citation_num,
            original_query=issue.recovery_query,
            skip_query_expansion=True,
        ),
        queries=[issue.recovery_query],
        search_mode="hybrid",
    )


def run_batched_gap_recovery(
    *,
    search_tool: SearchTool,
    issues: Sequence[CandidateAnswerClaimIssue],
    starting_citation_num: int,
    placement: Placement,
) -> ToolResponse:
    """Resolve up to five gaps with focused calls and one merged response."""

    if not issues or len(issues) > _MAX_BATCHED_RECOVERY_ISSUES:
        raise ValueError("batched recovery requires between one and five issues")
    queries = [issue.recovery_query for issue in issues]
    if any(query is None for query in queries):
        raise ValueError("every recovery issue must include a recovery_query")
    concrete_queries = [query for query in queries if query is not None]
    search_docs: list[SearchDoc] = []
    displayed_docs: list[SearchDoc] = []
    citation_mapping: dict[int, str] = {}
    citation_chunk_mapping: dict[int, int] = {}
    merged_results: list[object] = []
    receipts: list[object] = []
    next_citation_num = starting_citation_num

    for issue_index, query in enumerate(concrete_queries):
        issue_placement = placement.model_copy(update={"tab_index": issue_index})
        response = search_tool.run(
            placement=issue_placement,
            override_kwargs=SearchToolOverrideKwargs(
                starting_citation_num=next_citation_num,
                original_query=query,
                skip_query_expansion=True,
            ),
            queries=[query],
            search_mode="hybrid",
        )
        rich_response = response.rich_response
        if not isinstance(rich_response, SearchDocsResponse):
            raise ValueError("gap recovery search returned no document response")
        search_docs.extend(rich_response.search_docs)
        displayed_docs.extend(rich_response.displayed_docs or [])
        citation_mapping.update(rich_response.citation_mapping)
        citation_chunk_mapping.update(rich_response.citation_chunk_mapping)
        if rich_response.citation_mapping:
            next_citation_num = max(rich_response.citation_mapping) + 1

        try:
            payload = json.loads(response.llm_facing_response)
        except json.JSONDecodeError as error:
            raise ValueError("gap recovery search returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("gap recovery search returned a non-object payload")
        raw_results = payload.get("results")
        if isinstance(raw_results, list):
            merged_results.extend(raw_results)
        if "receipt" in payload:
            receipts.append(payload["receipt"])

    return ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=search_docs,
            citation_mapping=citation_mapping,
            citation_chunk_mapping=citation_chunk_mapping,
            displayed_docs=displayed_docs or None,
        ),
        llm_facing_response=json.dumps(
            {
                "results": merged_results,
                "recovery_receipts": receipts,
            },
            ensure_ascii=False,
        ),
    )


def recovery_search_docs_by_citation(
    response: ToolResponse,
) -> CitationMapping:
    """Resolve both wire mappings to exact canonical SearchDoc objects."""

    rich_response = response.rich_response
    if not isinstance(rich_response, SearchDocsResponse):
        return {}
    docs_by_identity = {
        (search_doc.document_id, search_doc.chunk_ind): search_doc
        for search_doc in rich_response.search_docs
    }
    resolved: CitationMapping = {}
    for citation_number, document_id in rich_response.citation_mapping.items():
        chunk_id = rich_response.citation_chunk_mapping.get(citation_number)
        if chunk_id is None:
            raise ValueError(
                "recovery citation mapping is missing its canonical chunk id"
            )
        search_doc = docs_by_identity.get((document_id, chunk_id))
        if search_doc is None:
            raise ValueError(
                "recovery citation mapping does not resolve to a returned chunk"
            )
        resolved[citation_number] = search_doc
    extra_chunk_numbers = (
        rich_response.citation_chunk_mapping.keys()
        - rich_response.citation_mapping.keys()
    )
    if extra_chunk_numbers:
        raise ValueError("recovery chunk mapping has no matching document mapping")
    return resolved


def merge_recovery_citation_mapping(
    existing: Mapping[int, SearchDoc],
    recovered: Mapping[int, SearchDoc],
) -> CitationMapping:
    """Add monotonically allocated citations without permitting reassignment."""

    merged = dict(existing)
    for citation_number, search_doc in recovered.items():
        previous = merged.get(citation_number)
        if previous is not None and (
            previous.document_id,
            previous.chunk_ind,
        ) != (search_doc.document_id, search_doc.chunk_ind):
            raise ValueError(
                f"recovery attempted to reassign citation {citation_number}"
            )
        merged[citation_number] = search_doc
    return merged


def exact_recovery_evidence_chunks(
    response: ToolResponse,
) -> list[CandidateAnswerEvidenceChunk]:
    """Retain only exact LLM-visible text backed by both canonical mappings."""

    citation_docs = recovery_search_docs_by_citation(response)
    try:
        payload = json.loads(response.llm_facing_response)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    evidence: list[CandidateAnswerEvidenceChunk] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        citation_number = raw_result.get("document")
        content = raw_result.get("content")
        if (
            not isinstance(citation_number, int)
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        search_doc = citation_docs.get(citation_number)
        if search_doc is None:
            continue
        metadata = _result_metadata(raw_result)
        raw_chunk_identifier = metadata.get("regulatory_chunk_id") or (
            search_doc.metadata.get("regulatory_chunk_id")
        )
        chunk_identifier = (
            raw_chunk_identifier.strip()
            if isinstance(raw_chunk_identifier, str) and raw_chunk_identifier.strip()
            else f"{search_doc.document_id}:{search_doc.chunk_ind}"
        )
        raw_heading_path = metadata.get("regulatory_heading_path") or (
            search_doc.metadata.get("regulatory_heading_path")
        )
        heading = (
            " > ".join(part for part in raw_heading_path if isinstance(part, str))
            if isinstance(raw_heading_path, list)
            and all(isinstance(part, str) for part in raw_heading_path)
            else search_doc.semantic_identifier
        )
        evidence.append(
            build_candidate_answer_evidence_chunk(
                document_id=search_doc.document_id,
                chunk_id=search_doc.chunk_ind,
                citation_number=citation_number,
                retrieval_number=citation_number,
                chunk_identifier=chunk_identifier,
                heading=heading,
                content=content,
            )
        )
    return evidence


def _result_metadata(result: Mapping[object, object]) -> dict[str, object]:
    raw_metadata = result.get("metadata")
    if isinstance(raw_metadata, dict):
        return {
            key: value for key, value in raw_metadata.items() if isinstance(key, str)
        }
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {key: value for key, value in parsed.items() if isinstance(key, str)}
    return {}


def merge_recovery_evidence_chunks(
    existing: Sequence[CandidateAnswerEvidenceChunk],
    recovered: Sequence[CandidateAnswerEvidenceChunk],
) -> list[CandidateAnswerEvidenceChunk]:
    """Deduplicate exact chunks by canonical identity, preserving assigned numbers."""

    merged = list(existing)
    indexes = {
        (chunk.document_id, chunk.chunk_id): index for index, chunk in enumerate(merged)
    }
    for recovered_chunk in recovered:
        identity = (recovered_chunk.document_id, recovered_chunk.chunk_id)
        existing_index = indexes.get(identity)
        if existing_index is None:
            indexes[identity] = len(merged)
            merged.append(recovered_chunk)
            continue
        existing_chunk = merged[existing_index]
        if (
            existing_chunk.citation_number is None
            and recovered_chunk.citation_number is not None
        ):
            merged[existing_index] = recovered_chunk
    return merged
