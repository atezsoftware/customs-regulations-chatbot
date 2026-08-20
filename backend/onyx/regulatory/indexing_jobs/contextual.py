from __future__ import annotations

import datetime
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from onyx.configs.model_configs import GEN_AI_INPUT_TOKEN_SAFETY_MARGIN
from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
)
from onyx.db.regulatory_indexing_jobs import (
    persist_regulatory_indexing_item_context,
    persist_regulatory_indexing_item_failure,
)
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.llm.constants import LlmProviderNames
from onyx.llm.model_capabilities import get_max_input_tokens
from onyx.natural_language_processing.utils import BaseTokenizer, tokenizer_trim_middle
from onyx.prompts.contextual_retrieval import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
)
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    fit_context_fields_to_embedding_budget,
    visible_regulatory_snapshot_for_target,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    VertexBatchResult,
    vertex_jsonl_line_size,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

_MAX_CACHED_DOCUMENT_CONTEXTS = 4

_VERTEX_BATCH_MAX_OUTPUT_TOKENS = 256


@dataclass(frozen=True, slots=True)
class VertexUTF8ContextualBudgetTokenizer(BaseTokenizer):
    """Conservative local input budget for a frozen Vertex contextual model.

    This is deliberately not an exact Gemini tokenizer. The installed stack has no
    offline Gemini tokenizer, so every UTF-8 byte is treated as a possible token.
    That upper bound may trim more context than necessary, but avoids a provider call
    during preparation. Boundary fragments are discarded during decode so trimming
    never introduces invalid Unicode or replacement characters.
    """

    model_provider: LlmProviderNames
    model_name: str

    def encode(self, string: str) -> list[int]:
        return list(string.encode("utf-8"))

    def tokenize(self, string: str) -> list[str]:
        return [f"{byte:02x}" for byte in string.encode("utf-8")]

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


@lru_cache(maxsize=32)
def _cached_contextual_token_budget_tokenizer(
    model_provider: LlmProviderNames,
    model_name: str,
) -> VertexUTF8ContextualBudgetTokenizer:
    return VertexUTF8ContextualBudgetTokenizer(
        model_provider=model_provider,
        model_name=model_name,
    )


def get_contextual_token_budget_tokenizer(
    *,
    model_provider: LlmProviderNames,
    model_name: str,
) -> VertexUTF8ContextualBudgetTokenizer:
    """Build the local budgeter for the frozen Vertex provider/model contract."""

    if model_provider is not LlmProviderNames.VERTEX_AI:
        raise ValueError("contextual token budgeting requires the Vertex AI provider")
    normalized_model_name = model_name.strip()
    if not normalized_model_name:
        raise ValueError("contextual Vertex model name must not be empty")
    return _cached_contextual_token_budget_tokenizer(
        model_provider,
        normalized_model_name,
    )


class ContextualMappingError(ValueError):
    """Canonical chunks, persisted items, and Vertex results do not agree."""


class ContextAttemptBudgetExhaustedError(RuntimeError):
    """A pending chunk already consumed its durable contextual attempt budget."""


class ContextApplySummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)


def _ordered_rows(
    rows: Sequence[RegulatoryChunk],
) -> list[RegulatoryChunk]:
    ordered = sorted(rows, key=lambda row: (row.position, row.id))
    if len({row.id for row in ordered}) != len(ordered):
        raise ContextualMappingError("canonical regulatory rows contain duplicate ids")
    return ordered


def _row_block(row: RegulatoryChunk) -> str:
    heading_path = " > ".join(row.heading_path)
    return (
        f"[Canonical position: {row.position}]\n"
        f"Heading path: {heading_path}\n{row.text}"
    )


@dataclass(frozen=True, slots=True)
class _PreparedDocumentContext:
    rows: tuple[RegulatoryChunk, ...]
    text: str
    tokens: tuple[int, ...]
    boundary_anchors: str
    boundary_anchor_tokens: tuple[int, ...]


def _prepare_document_context(
    ordered_rows: Sequence[RegulatoryChunk],
    *,
    tokenizer: BaseTokenizer,
) -> _PreparedDocumentContext:
    document = "\n\n".join(_row_block(candidate) for candidate in ordered_rows)
    first_heading = (
        ordered_rows[0].heading_path[-1] if ordered_rows[0].heading_path else ""
    )
    last_heading = (
        ordered_rows[-1].heading_path[-1] if ordered_rows[-1].heading_path else ""
    )
    boundary_anchors = (
        f"[Document boundary headings: {first_heading} | {last_heading}]\n"
    )
    return _PreparedDocumentContext(
        rows=tuple(ordered_rows),
        text=document,
        tokens=tuple(tokenizer.encode(document)),
        boundary_anchors=boundary_anchors,
        boundary_anchor_tokens=tuple(tokenizer.encode(boundary_anchors)),
    )


def _fit_document_context(
    prepared: _PreparedDocumentContext,
    *,
    token_budget: int,
    tokenizer: BaseTokenizer,
) -> str:
    if len(prepared.tokens) <= token_budget:
        return prepared.text
    remaining_budget = token_budget - len(prepared.boundary_anchor_tokens)
    if remaining_budget <= 0:
        raise ContextualMappingError(
            "contextual model input window cannot preserve document boundaries"
        )
    try:
        representative_document = tokenizer_trim_middle(
            list(prepared.tokens),
            remaining_budget,
            tokenizer,
        )
    except AssertionError as error:
        raise ContextualMappingError(
            "contextual model input window is too small for document context"
        ) from error
    return f"{prepared.boundary_anchors}{representative_document}"


def _contextual_model_name(job: RegulatoryIndexingJob) -> str:
    snapshot = job.config_snapshot
    vertex = snapshot.get("vertex")
    if not isinstance(vertex, dict):
        raise ContextualMappingError("indexing job has no Vertex configuration")
    model_name = cast(dict[str, object], vertex).get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ContextualMappingError("indexing job has no contextual model name")
    return model_name


@dataclass(slots=True)
class ContextualRequestFactory:
    """Reuse sorted rows and tokenized document snapshots across chunk prompts."""

    job: RegulatoryIndexingJob
    rows: Sequence[RegulatoryChunk]
    contextual_tokenizer: BaseTokenizer
    embedding_tokenizer: BaseTokenizer | None = None
    _ordered_rows: tuple[RegulatoryChunk, ...] = field(init=False)
    _row_by_id: dict[str, RegulatoryChunk] = field(init=False)
    _documents: OrderedDict[
        datetime.date, tuple[tuple[RegulatoryChunk, ...], _PreparedDocumentContext]
    ] = field(init=False, default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self._ordered_rows = tuple(_ordered_rows(self.rows))
        self._row_by_id = {row.id: row for row in self._ordered_rows}
        if any(row.user_file_id != self.job.user_file_id for row in self._ordered_rows):
            raise ContextualMappingError(
                "canonical rows do not belong to the indexing job"
            )

    def _canonical_row(self, row: RegulatoryChunk) -> RegulatoryChunk:
        canonical_row = self._row_by_id.get(row.id)
        if canonical_row is None:
            raise ContextualMappingError(
                "contextual row is not part of the canonical file"
            )
        return canonical_row

    def _document(
        self, row: RegulatoryChunk
    ) -> tuple[tuple[RegulatoryChunk, ...], _PreparedDocumentContext]:
        canonical_row = self._canonical_row(row)
        reference_date = context_reference_date(
            canonical_row.validity_start_date,
            canonical_row.validity_end_date,
        )
        cached = self._documents.get(reference_date)
        if cached is not None:
            self._documents.move_to_end(reference_date)
            return cached
        visible_rows = tuple(
            _ordered_rows(
                visible_regulatory_snapshot_for_target(
                    self._ordered_rows, canonical_row
                )
            )
        )
        prepared = _prepare_document_context(
            visible_rows,
            tokenizer=self.contextual_tokenizer,
        )
        cached = (visible_rows, prepared)
        self._documents[reference_date] = cached
        if len(self._documents) > _MAX_CACHED_DOCUMENT_CONTEXTS:
            self._documents.popitem(last=False)
        return cached

    def reserve(self, row: RegulatoryChunk) -> int:
        if self.embedding_tokenizer is None:
            raise ContextualMappingError(
                "contextual reserve requires an embedding tokenizer"
            )
        canonical_row = self._canonical_row(row)
        visible_rows, _prepared = self._document(canonical_row)
        if len(visible_rows) <= 1:
            return 0
        return contextual_reserve_for_embedding_text(
            canonical_row.text,
            tokenizer=self.embedding_tokenizer,
            embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
            requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        )

    def request(self, row: RegulatoryChunk) -> VertexBatchRequest:
        canonical_row = self._canonical_row(row)
        _visible_rows, prepared = self._document(canonical_row)
        return _contextual_request_for_prepared_document(
            self.job,
            canonical_row,
            prepared_document=prepared,
            contextual_tokenizer=self.contextual_tokenizer,
        )


def contextual_reserve_for_row(
    rows: Sequence[RegulatoryChunk],
    row: RegulatoryChunk,
    *,
    embedding_tokenizer: BaseTokenizer,
) -> int:
    ordered_rows = _ordered_rows(rows)
    canonical_row = next(
        (candidate for candidate in ordered_rows if candidate.id == row.id),
        None,
    )
    if canonical_row is None:
        raise ContextualMappingError("contextual row is not part of the canonical file")
    visible_rows = visible_regulatory_snapshot_for_target(
        ordered_rows,
        canonical_row,
    )
    if len(visible_rows) <= 1:
        return 0
    return contextual_reserve_for_embedding_text(
        row.text,
        tokenizer=embedding_tokenizer,
        embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
        requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
    )


def contextual_request_for_row(
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    row: RegulatoryChunk,
    *,
    contextual_tokenizer: BaseTokenizer,
) -> VertexBatchRequest:
    return ContextualRequestFactory(
        job=job,
        rows=rows,
        contextual_tokenizer=contextual_tokenizer,
    ).request(row)


def _contextual_request_for_prepared_document(
    job: RegulatoryIndexingJob,
    canonical_row: RegulatoryChunk,
    *,
    prepared_document: _PreparedDocumentContext,
    contextual_tokenizer: BaseTokenizer,
) -> VertexBatchRequest:
    chunk_block = _row_block(canonical_row)
    prompt_without_document = CONTEXTUAL_RAG_PROMPT1.format(
        document=""
    ) + CONTEXTUAL_RAG_PROMPT2.format(chunk=chunk_block)
    contextual_model_input_limit = get_max_input_tokens(
        model_name=_contextual_model_name(job),
        model_provider=LlmProviderNames.VERTEX_AI,
        output_tokens=_VERTEX_BATCH_MAX_OUTPUT_TOKENS,
    )
    safe_input_limit = int(
        contextual_model_input_limit * (1 - GEN_AI_INPUT_TOKEN_SAFETY_MARGIN)
    )
    document_token_budget = safe_input_limit - len(
        contextual_tokenizer.encode(prompt_without_document)
    )
    if document_token_budget <= 0:
        raise ContextualMappingError(
            "contextual model input window is too small for the chunk prompt"
        )

    document = _fit_document_context(
        prepared_document,
        token_budget=document_token_budget,
        tokenizer=contextual_tokenizer,
    )

    return VertexBatchRequest(
        prompt=(
            CONTEXTUAL_RAG_PROMPT1.format(document=document)
            + CONTEXTUAL_RAG_PROMPT2.format(chunk=chunk_block)
        )
    )


def build_contextual_requests(
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    *,
    embedding_tokenizer: BaseTokenizer,
    contextual_tokenizer: BaseTokenizer,
    max_requests: int | None = None,
    max_jsonl_bytes: int | None = None,
    max_attempts: int | None = None,
) -> list[VertexBatchRequest]:
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be positive")
    if max_jsonl_bytes is not None and max_jsonl_bytes < 1:
        raise ValueError("max_jsonl_bytes must be positive")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    ordered_rows = _ordered_rows(rows)
    row_by_id = {row.id: row for row in ordered_rows}
    requests: list[VertexBatchRequest] = []
    item_by_row_id: dict[str, RegulatoryIndexingItem] = {}
    for item in items:
        if item.job_id != job.id:
            raise ContextualMappingError("contextual item does not belong to the job")
        if item.regulatory_chunk_id in item_by_row_id:
            raise ContextualMappingError("contextual items contain a duplicate chunk")
        if item.regulatory_chunk_id not in row_by_id:
            raise ContextualMappingError("contextual item has no canonical chunk")
        item_by_row_id[item.regulatory_chunk_id] = item

    request_factory = ContextualRequestFactory(
        job=job,
        rows=ordered_rows,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
    )
    used_jsonl_bytes = 0
    for row in ordered_rows:
        item = item_by_row_id.get(row.id)
        if item is None:
            continue
        if item.status != RegulatoryIndexingItemStatus.PENDING.value:
            continue
        context_attempt_count = getattr(item, "context_attempt_count", 0)
        if max_attempts is not None and context_attempt_count >= max_attempts:
            raise ContextAttemptBudgetExhaustedError(
                "contextual item attempt budget is exhausted"
            )
        if max_requests is not None and len(requests) >= max_requests:
            break
        if request_factory.reserve(row) == 0:
            raise ContextualMappingError(
                "context-ineligible item was not persisted as skipped"
            )
        request = request_factory.request(row)
        if request.request_hash != item.request_hash:
            raise ContextualMappingError(
                "contextual request hash does not match the persisted item"
            )
        request_bytes = vertex_jsonl_line_size(request)
        if (
            max_jsonl_bytes is not None
            and used_jsonl_bytes + request_bytes > max_jsonl_bytes
        ):
            if not requests:
                raise ContextualMappingError(
                    "contextual request exceeds the JSONL byte limit"
                )
            break
        requests.append(request)
        used_jsonl_bytes += request_bytes
    if len({request.request_hash for request in requests}) != len(requests):
        raise ContextualMappingError("contextual requests contain a duplicate hash")
    return requests


def contextualized_embedding_text(
    row: RegulatoryChunk,
    item: RegulatoryIndexingItem,
) -> str:
    """Return generated context before legal text, or original text when skipped."""

    if item.regulatory_chunk_id != row.id:
        raise ContextualMappingError(
            "embedding item does not match its canonical chunk"
        )
    if item.status == RegulatoryIndexingItemStatus.SKIPPED.value:
        return row.text
    if item.status not in {
        RegulatoryIndexingItemStatus.CONTEXT_READY.value,
        RegulatoryIndexingItemStatus.EMBEDDED.value,
    }:
        raise ContextualMappingError("item is not ready for embedding")
    if not isinstance(item.context, dict):
        raise ContextualMappingError("context-ready item has no contextual metadata")
    contextual_text = item.context.get("contextual_text")
    if not isinstance(contextual_text, str) or not contextual_text.strip():
        raise ContextualMappingError("context-ready item has no contextual text")
    return f"{contextual_text}{row.text}"


def _persist_result(
    *,
    job: RegulatoryIndexingJob,
    row: RegulatoryChunk,
    item: RegulatoryIndexingItem,
    result: VertexBatchResult,
    embedding_tokenizer: BaseTokenizer,
    db_session: Session,
) -> str:
    if result.error is not None:
        persisted = persist_regulatory_indexing_item_failure(
            db_session,
            item_id=item.id,
            expected_generation=job.lease_generation,
            error_code=result.error.value,
            error_message=result.error.value,
        )
        next_status = RegulatoryIndexingItemStatus.FAILED.value
    else:
        if result.context is None:
            raise ContextualMappingError("Vertex result has no outcome")
        contextual_text, _unused_context = fit_context_fields_to_embedding_budget(
            title_prefix="",
            content=row.text,
            metadata_suffix="",
            doc_summary=result.context,
            chunk_context="",
            tokenizer=embedding_tokenizer,
            embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
        )
        if not contextual_text:
            persisted = persist_regulatory_indexing_item_failure(
                db_session,
                item_id=item.id,
                expected_generation=job.lease_generation,
                error_code="context_too_large",
                error_message="context_too_large",
            )
            next_status = RegulatoryIndexingItemStatus.FAILED.value
        else:
            persisted = persist_regulatory_indexing_item_context(
                db_session,
                item_id=item.id,
                expected_generation=job.lease_generation,
                context={"contextual_text": contextual_text},
            )
            next_status = RegulatoryIndexingItemStatus.CONTEXT_READY.value
    if not persisted:
        raise RuntimeError("regulatory indexing lease was lost while applying context")
    return next_status


def apply_contextual_results(
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    results: Mapping[str, VertexBatchResult],
    embedding_tokenizer: BaseTokenizer,
    db_session: Session,
) -> ContextApplySummary:
    ordered_rows = _ordered_rows(rows)
    if any(row.user_file_id != job.user_file_id for row in ordered_rows):
        raise ContextualMappingError("canonical rows do not belong to the indexing job")
    row_by_id = {row.id: row for row in ordered_rows}
    known_hashes = {item.request_hash for item in items}
    if not set(results).issubset(known_hashes):
        raise ContextualMappingError(
            "Vertex results contain an unexpected request hash"
        )

    resulting_statuses: list[str] = []
    for item in items:
        if item.job_id != job.id:
            raise ContextualMappingError("contextual item does not belong to the job")
        row = row_by_id.get(item.regulatory_chunk_id)
        if row is None:
            raise ContextualMappingError("contextual item has no canonical chunk")
        status = item.status
        result = results.get(item.request_hash)
        if status == RegulatoryIndexingItemStatus.PENDING.value and result is not None:
            if result.request_hash != item.request_hash:
                raise ContextualMappingError(
                    "Vertex result hash does not match the persisted item"
                )
            status = _persist_result(
                job=job,
                row=row,
                item=item,
                result=result,
                embedding_tokenizer=embedding_tokenizer,
                db_session=db_session,
            )
        resulting_statuses.append(status)

    context_ready_count = sum(
        status
        in {
            RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            RegulatoryIndexingItemStatus.EMBEDDED.value,
        }
        for status in resulting_statuses
    )
    return ContextApplySummary(
        context_ready_count=context_ready_count,
        failed_count=resulting_statuses.count(
            RegulatoryIndexingItemStatus.FAILED.value
        ),
        pending_count=resulting_statuses.count(
            RegulatoryIndexingItemStatus.PENDING.value
        ),
        skipped_count=resulting_statuses.count(
            RegulatoryIndexingItemStatus.SKIPPED.value
        ),
    )
