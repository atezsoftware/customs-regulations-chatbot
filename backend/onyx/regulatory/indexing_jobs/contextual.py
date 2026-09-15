from __future__ import annotations

import datetime
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, cast

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
from onyx.regulatory.amendments.annexes.context_dependencies import (
    canonical_dependency_ids,
    context_hash,
    effective_context_rows,
    encoder_model_fingerprint,
    freeze_context_source_snapshot,
    freeze_encoder_inputs,
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import (
    ContextGenerationCall,
    ContextSourceSnapshot,
    FrozenContextProjection,
    PreparedContextView,
)
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    fit_context_fields_to_embedding_budget,
    visible_regulatory_snapshot_for_target,
)
from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    VertexBatchResult,
    vertex_jsonl_line_size,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

if TYPE_CHECKING:
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel

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
    row_count: int
    text: str
    tokens: tuple[int, ...] | bytes
    boundary_anchors: str
    boundary_anchor_tokens: tuple[int, ...] | bytes
    truncated: bool


def _utf8_prefix(value: str, max_bytes: int) -> tuple[str, int, bool]:
    """Take a UTF-8-safe prefix without encoding more than the bounded prefix."""

    if max_bytes <= 0:
        return "", 0, not value
    if len(value) <= max_bytes:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value, len(encoded), True
    encoded_prefix = value[:max_bytes].encode("utf-8")
    prefix = encoded_prefix[:max_bytes].decode("utf-8", errors="ignore")
    return prefix, len(prefix.encode("utf-8")), len(prefix) == len(value)


def _append_bounded_row_block(
    parts: list[str],
    row: RegulatoryChunk,
    *,
    remaining_bytes: int,
) -> tuple[int, bool]:
    used_bytes = 0

    def pieces() -> Iterator[str]:
        yield f"[Canonical position: {row.position}]\nHeading path: "
        for index, heading in enumerate(row.heading_path):
            if index:
                yield " > "
            yield heading
        yield "\n"
        yield row.text

    for piece in pieces():
        prefix, prefix_bytes, complete = _utf8_prefix(
            piece, remaining_bytes - used_bytes
        )
        if prefix:
            parts.append(prefix)
            used_bytes += prefix_bytes
        if not complete:
            return used_bytes, False
    return used_bytes, True


def _prepare_document_context(
    ordered_rows: Sequence[RegulatoryChunk],
    *,
    tokenizer: BaseTokenizer,
    max_utf8_bytes: int,
    max_tokens: int,
) -> _PreparedDocumentContext:
    if not ordered_rows:
        raise ContextualMappingError("canonical regulatory document is empty")
    if max_utf8_bytes <= 0 or max_tokens <= 0:
        raise ContextualMappingError("document context limits must be positive")
    first_heading = (
        ordered_rows[0].heading_path[-1] if ordered_rows[0].heading_path else ""
    )
    last_heading = (
        ordered_rows[-1].heading_path[-1] if ordered_rows[-1].heading_path else ""
    )
    boundary_anchors = (
        f"[Document boundary headings: {first_heading} | {last_heading}]\n"
    )
    if isinstance(tokenizer, VertexUTF8ContextualBudgetTokenizer):
        boundary_anchor_tokens: tuple[int, ...] | bytes = boundary_anchors.encode(
            "utf-8"
        )
    else:
        boundary_anchor_tokens = tuple(tokenizer.encode(boundary_anchors))
    context_ceiling = min(max_utf8_bytes, max_tokens)
    document_ceiling = context_ceiling - len(boundary_anchor_tokens)
    if document_ceiling <= 0:
        raise ContextualMappingError(
            "contextual model input window cannot preserve document boundaries"
        )

    parts: list[str] = []
    used_bytes = 0
    truncated = False
    for index, row in enumerate(ordered_rows):
        if index:
            separator, separator_bytes, complete = _utf8_prefix(
                "\n\n", document_ceiling - used_bytes
            )
            if separator:
                parts.append(separator)
                used_bytes += separator_bytes
            if not complete:
                truncated = True
                break
        row_bytes, complete = _append_bounded_row_block(
            parts,
            row,
            remaining_bytes=document_ceiling - used_bytes,
        )
        used_bytes += row_bytes
        if not complete:
            truncated = True
            break
    document = "".join(parts)
    if isinstance(tokenizer, VertexUTF8ContextualBudgetTokenizer):
        tokens: tuple[int, ...] | bytes = document.encode("utf-8")
    else:
        encoded_tokens = tokenizer.encode(document)
        if len(encoded_tokens) > document_ceiling:
            encoded_tokens = encoded_tokens[:document_ceiling]
            document = tokenizer.decode(encoded_tokens)
            truncated = True
        tokens = tuple(encoded_tokens)
    return _PreparedDocumentContext(
        row_count=len(ordered_rows),
        text=document,
        tokens=tokens,
        boundary_anchors=boundary_anchors,
        boundary_anchor_tokens=boundary_anchor_tokens,
        truncated=truncated,
    )


def _fit_document_context(
    prepared: _PreparedDocumentContext,
    *,
    token_budget: int,
    tokenizer: BaseTokenizer,
) -> str:
    if len(prepared.tokens) <= token_budget and not prepared.truncated:
        return prepared.text
    remaining_budget = token_budget - len(prepared.boundary_anchor_tokens)
    if remaining_budget <= 0:
        raise ContextualMappingError(
            "contextual model input window cannot preserve document boundaries"
        )
    if len(prepared.tokens) <= remaining_budget:
        representative_document = prepared.text
    elif isinstance(prepared.tokens, bytes):
        first_size = remaining_budget // 2
        last_size = remaining_budget - first_size
        representative_document = prepared.tokens[:first_size].decode(
            "utf-8", errors="ignore"
        ) + prepared.tokens[-last_size:].decode("utf-8", errors="ignore")
    else:
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


def _contextual_safe_input_limit(job: RegulatoryIndexingJob) -> int:
    return int(
        get_max_input_tokens(
            model_name=_contextual_model_name(job),
            model_provider=LlmProviderNames.VERTEX_AI,
            output_tokens=_VERTEX_BATCH_MAX_OUTPUT_TOKENS,
        )
        * (1 - GEN_AI_INPUT_TOKEN_SAFETY_MARGIN)
    )


def _document_context_utf8_byte_limit(job: RegulatoryIndexingJob) -> int:
    configured = job.config_snapshot.get("context_jsonl_max_bytes", 8 * 1024 * 1024)
    if (
        not isinstance(configured, int)
        or isinstance(configured, bool)
        or configured <= 0
    ):
        raise ContextualMappingError("indexing job has no valid context byte limit")
    return configured


@dataclass(slots=True)
class ContextualRequestFactory:
    """Reuse sorted rows and tokenized document snapshots across chunk prompts."""

    job: RegulatoryIndexingJob
    rows: Sequence[RegulatoryChunk]
    contextual_tokenizer: BaseTokenizer
    embedding_tokenizer: BaseTokenizer | None = None
    reference_date_override: datetime.date | None = None
    _source_snapshots: dict[
        tuple[datetime.date, tuple[str, ...]], ContextSourceSnapshot
    ] = field(init=False, default_factory=dict)
    _ordered_rows: tuple[RegulatoryChunk, ...] = field(init=False)
    _row_by_id: dict[str, RegulatoryChunk] = field(init=False)
    _documents: OrderedDict[datetime.date, _PreparedDocumentContext] = field(
        init=False, default_factory=OrderedDict
    )

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

    def _document(self, row: RegulatoryChunk) -> _PreparedDocumentContext:
        canonical_row = self._canonical_row(row)
        reference_date = self.reference_date_override or context_reference_date(
            canonical_row.validity_start_date,
            canonical_row.validity_end_date,
        )
        cached = self._documents.get(reference_date)
        if cached is not None:
            self._documents.move_to_end(reference_date)
            return cached
        visible_rows = _ordered_rows(
            visible_regulatory_snapshot_for_target(
                self._ordered_rows,
                canonical_row,
                reference_date=self.reference_date_override,
            )
        )
        prepared = _prepare_document_context(
            visible_rows,
            tokenizer=self.contextual_tokenizer,
            max_utf8_bytes=_document_context_utf8_byte_limit(self.job),
            max_tokens=_contextual_safe_input_limit(self.job),
        )
        self._documents[reference_date] = prepared
        if len(self._documents) > _MAX_CACHED_DOCUMENT_CONTEXTS:
            self._documents.popitem(last=False)
        return prepared

    def source_snapshot(self, row: RegulatoryChunk) -> ContextSourceSnapshot:
        canonical = self._canonical_row(row)
        visible = visible_regulatory_snapshot_for_target(
            self._ordered_rows, canonical, reference_date=self.reference_date_override
        )
        reference = self.reference_date_override or context_reference_date(
            canonical.validity_start_date, canonical.validity_end_date
        )
        key = (reference, tuple(source.id for source in visible))
        if key in self._source_snapshots:
            return self._source_snapshots[key]
        snapshot = freeze_context_source_snapshot(
            rows=self._ordered_rows,
            target=canonical,
            reference_date=self.reference_date_override,
            generation_path="durable",
            row_text=_row_block,
        )
        self._source_snapshots[key] = snapshot
        return snapshot

    def request_provenance(
        self, row: RegulatoryChunk, request: VertexBatchRequest
    ) -> dict[str, object]:
        prepared = self._document(row)
        return {
            "generation_path": "durable",
            "request_hash": request.request_hash,
            "prompt": request.prompt,
            "source_snapshot_sha256": self.source_snapshot(row).sha256,
            "model_name": _contextual_model_name(self.job),
            "model_provider": LlmProviderNames.VERTEX_AI.value,
            "temperature": 0,
            "max_output_tokens": _VERTEX_BATCH_MAX_OUTPUT_TOKENS,
            "tokenizer": f"{type(self.contextual_tokenizer).__module__}.{type(self.contextual_tokenizer).__qualname__}",
            "safe_input_limit": _contextual_safe_input_limit(self.job),
            "utf8_byte_limit": _document_context_utf8_byte_limit(self.job),
            "bounded_document": prepared.text,
            "boundary_anchors": prepared.boundary_anchors,
            "pre_truncated": prepared.truncated,
        }

    def reserve(self, row: RegulatoryChunk) -> int:
        if self.embedding_tokenizer is None:
            raise ContextualMappingError(
                "contextual reserve requires an embedding tokenizer"
            )
        canonical_row = self._canonical_row(row)
        prepared = self._document(canonical_row)
        if prepared.row_count <= 1:
            return 0
        return contextual_reserve_for_embedding_text(
            canonical_row.text,
            tokenizer=self.embedding_tokenizer,
            embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
            requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        )

    def request(self, row: RegulatoryChunk) -> VertexBatchRequest:
        canonical_row = self._canonical_row(row)
        prepared = self._document(canonical_row)
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

    def request_for_document(document: str) -> VertexBatchRequest:
        return VertexBatchRequest(
            prompt=(
                CONTEXTUAL_RAG_PROMPT1.format(document=document)
                + CONTEXTUAL_RAG_PROMPT2.format(chunk=chunk_block)
            )
        )

    jsonl_byte_limit = _document_context_utf8_byte_limit(job)
    minimum_request = request_for_document("")
    if vertex_jsonl_line_size(minimum_request) > jsonl_byte_limit:
        raise ContextualMappingError(
            "contextual request template and canonical chunk exceed the JSONL byte limit"
        )

    safe_input_limit = _contextual_safe_input_limit(job)
    document_token_budget = safe_input_limit - len(
        contextual_tokenizer.encode(minimum_request.prompt)
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
    request = request_for_document(document)
    if vertex_jsonl_line_size(request) <= jsonl_byte_limit:
        return request

    best_request = minimum_request
    minimum_context_budget = len(prepared_document.boundary_anchor_tokens) + 1
    maximum_context_budget = min(
        document_token_budget,
        len(prepared_document.tokens) + len(prepared_document.boundary_anchor_tokens),
    )
    while minimum_context_budget <= maximum_context_budget:
        candidate_budget = (minimum_context_budget + maximum_context_budget) // 2
        try:
            candidate_document = _fit_document_context(
                prepared_document,
                token_budget=candidate_budget,
                tokenizer=contextual_tokenizer,
            )
        except ContextualMappingError:
            minimum_context_budget = candidate_budget + 1
            continue
        candidate_request = request_for_document(candidate_document)
        if vertex_jsonl_line_size(candidate_request) <= jsonl_byte_limit:
            best_request = candidate_request
            minimum_context_budget = candidate_budget + 1
        else:
            maximum_context_budget = candidate_budget - 1
    return best_request


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
    requests: list[VertexBatchRequest] = []
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
    from onyx.regulatory.indexing_jobs.projection_identity import (
        ordered_projection_items,
        projection_input,
    )

    try:
        mapped = ordered_projection_items(
            job_id=job.id,
            user_file_id=job.user_file_id,
            rows=rows,
            items=items,
            require_complete=False,
        )
    except ValueError as exc:
        raise ContextualMappingError(str(exc)) from exc
    used_jsonl_bytes = 0
    submitted_hashes: set[str] = set()
    factories: dict[str, ContextualRequestFactory] = {}
    for row, item in mapped:
        if item.status != RegulatoryIndexingItemStatus.PENDING.value:
            continue
        frozen = projection_input(item)
        scope_key = (
            context_hash(
                {
                    "rows": [
                        value.model_dump(mode="json") for value in frozen.context_rows
                    ],
                    "date": frozen.reference_date,
                }
            )
            if frozen
            else "legacy"
        )
        if scope_key not in factories:
            factories[scope_key] = ContextualRequestFactory(
                job=job,
                rows=canonical_snapshot_rows(frozen.context_rows)
                if frozen
                else ordered_rows,
                embedding_tokenizer=embedding_tokenizer,
                contextual_tokenizer=contextual_tokenizer,
                reference_date_override=frozen.reference_date if frozen else None,
            )
        request_factory = factories[scope_key]
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
        if request.request_hash in submitted_hashes:
            continue
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
        submitted_hashes.add(request.request_hash)
        used_jsonl_bytes += request_bytes
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
                context={
                    "contextual_text": contextual_text,
                    "raw_contextual_text": result.context,
                },
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
    known_hashes = {item.request_hash for item in items}
    if not set(results).issubset(known_hashes):
        raise ContextualMappingError(
            "Vertex results contain an unexpected request hash"
        )

    from onyx.regulatory.indexing_jobs.projection_identity import (
        ordered_projection_items,
    )

    resulting_statuses: list[str] = []
    try:
        mapped = ordered_projection_items(
            job_id=job.id,
            user_file_id=job.user_file_id,
            rows=rows,
            items=items,
            require_complete=False,
        )
    except ValueError as exc:
        raise ContextualMappingError(str(exc)) from exc
    for row, item in mapped:
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


def prepare_durable_context_view(
    *,
    job: RegulatoryIndexingJob,
    rows: Sequence[RegulatoryChunk],
    embedding_tokenizer: BaseTokenizer,
    contextual_tokenizer: BaseTokenizer,
    generate: Callable[[VertexBatchRequest], str],
    embedding_model: EmbeddingModel | None = None,
    cached: PreparedContextView | None = None,
    as_of_date: datetime.date | None = None,
    changed_ids: list[str] | None = None,
    resolve_call: Callable[
        [ContextGenerationCall, Callable[[], str]], ContextGenerationCall
    ]
    | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_workers: int = 1,
) -> PreparedContextView:
    """Freeze the existing durable transport's exact inputs and configuration.

    Synchronous jobs require the actual encoder built from the writer's frozen
    SearchSettings. Its tokenizer controls fitting and its encode preprocessing
    controls submitted text. Batch jobs use the job's frozen raw-input contract.
    """
    if any(row.user_file_id != job.user_file_id for row in rows):
        raise ContextualMappingError("canonical rows do not belong to the indexing job")
    if changed_ids:
        rows = rebuild_context_aggregates(list(rows), changed_ids=changed_ids)
    ordered = effective_context_rows(list(rows), as_of_date)
    if not ordered:
        return PreparedContextView()
    snapshot = job.config_snapshot
    required = ("embedding_provider", "embedding_model_name", "effective_dimension")
    if any(snapshot.get(key) is None for key in required):
        raise ContextualMappingError("embedding_configuration_unavailable")
    dimension = snapshot["effective_dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ContextualMappingError("embedding_dimension_unavailable")
    batch_payload = snapshot.get("openrouter_batch")
    batch_config = (
        OpenRouterBatchConfig.model_validate(batch_payload)
        if batch_payload is not None
        else None
    )
    if batch_config is not None:
        if (
            batch_config.model_name != snapshot["embedding_model_name"]
            or batch_config.effective_dimension != dimension
        ):
            raise ContextualMappingError("embedding_configuration_mismatch")
        from onyx.regulatory.indexing_jobs.embedding_receipts import (
            batch_embedding_configuration,
        )

        config = batch_embedding_configuration(
            provider=str(snapshot["embedding_provider"]),
            config=batch_config,
        )
    else:
        if embedding_model is None:
            raise ContextualMappingError("embedding_configuration_unavailable")
        if (
            (
                embedding_model.provider_type.value
                if embedding_model.provider_type
                else None
            )
            != snapshot["embedding_provider"]
            or embedding_model.model_name != snapshot["embedding_model_name"]
            or embedding_model.reduced_dimension != dimension
        ):
            raise ContextualMappingError("embedding_configuration_mismatch")
        embedding_tokenizer = embedding_model.tokenizer
        config = encoder_model_fingerprint(
            embedding_model,
            model_dim=dimension,
            formatter="durable-context-before-text-v1",
        )
        config["transport"] = "synchronous_encoder"
    factory = ContextualRequestFactory(
        job=job,
        rows=rows,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
        reference_date_override=as_of_date,
    )
    context_config: dict[str, str | int | float | bool | None] = {
        "model_provider": LlmProviderNames.VERTEX_AI.value,
        "model_name": _contextual_model_name(job),
        "temperature": 0,
        "max_output_tokens": _VERTEX_BATCH_MAX_OUTPUT_TOKENS,
        "max_input_tokens": _contextual_safe_input_limit(job),
        "utf8_byte_limit": _document_context_utf8_byte_limit(job),
    }
    config_hash = context_hash(context_config)
    cached_calls = (
        {call.request_sha256: call for call in cached.calls} if cached else {}
    )
    calls: dict[str, ContextGenerationCall] = {}
    snapshots: dict[str, ContextSourceSnapshot] = {}
    projections: list[FrozenContextProjection] = []
    if not 1 <= max_workers <= 4:
        raise ValueError("context preparation concurrency must be between one and four")

    def resolve(request: VertexBatchRequest) -> ContextGenerationCall:
        request_hash = context_hash([request.request_hash, config_hash])
        template = ContextGenerationCall(
            request_sha256=request_hash,
            stage="durable_chunk",
            prompt_json=request.model_dump_json(),
            config_sha256=config_hash,
            output="",
            source_text=request.prompt,
            token_budget=_contextual_safe_input_limit(job),
            tokenizer=f"{type(contextual_tokenizer).__module__}.{type(contextual_tokenizer).__qualname__}",
            generation_path="durable",
        )
        prior = cached_calls.get(request_hash)
        if prior is not None:
            if prior.model_copy(update={"output": ""}) != template:
                raise ContextualMappingError("cached context request proof changed")
            return prior
        return (
            resolve_call(template, lambda: generate(request))
            if resolve_call
            else template.model_copy(update={"output": generate(request)})
        )

    def resolve_in_context(
        context: Context, request: VertexBatchRequest
    ) -> ContextGenerationCall:
        return context.run(resolve, request)

    def resolved_rows() -> Iterator[
        tuple[RegulatoryChunk, ContextGenerationCall | None]
    ]:
        # Only a bounded window of prompts and provider calls exists at a time.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for offset in range(0, len(ordered), max_workers):
                window = ordered[offset : offset + max_workers]
                pending = [
                    executor.submit(
                        resolve_in_context, copy_context(), factory.request(row)
                    )
                    if factory.reserve(row)
                    else None
                    for row in window
                ]
                for row, future in zip(window, pending, strict=True):
                    yield row, future.result() if future is not None else None

    for completed, (row, call) in enumerate(resolved_rows(), 1):
        source_snapshot = factory.source_snapshot(row)
        snapshot_hash = source_snapshot.sha256
        snapshots[snapshot_hash] = source_snapshot
        item = RegulatoryIndexingItem(
            regulatory_chunk_id=row.id,
            status=RegulatoryIndexingItemStatus.SKIPPED.value,
        )
        request_hashes: list[str] = []
        output = ""
        if call is not None:
            if not call.output.strip():
                raise ContextualMappingError("context_generation_incomplete")
            output, _ = fit_context_fields_to_embedding_budget(
                title_prefix="",
                content=row.text,
                metadata_suffix="",
                doc_summary=call.output,
                chunk_context="",
                tokenizer=embedding_tokenizer,
                embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
            )
            if not output:
                raise ContextualMappingError("context_generation_incomplete")
            calls[call.request_sha256] = call
            request_hashes.append(call.request_sha256)
            item.status = RegulatoryIndexingItemStatus.CONTEXT_READY.value
            context_metadata: dict[str, object] = {"contextual_text": output}
            item.context = context_metadata
        if progress is not None:
            progress(completed, len(ordered))
        text = contextualized_embedding_text(row, item)
        texts = [text]
        if batch_config is None:
            assert embedding_model is not None
            texts, _ = freeze_encoder_inputs(
                texts,
                embedding_model,
                model_dim=dimension,
                formatter="durable-context-before-text-v1",
            )
        projections.append(
            FrozenContextProjection(
                canonical_chunk_id=row.id,
                canonical_dependency_ids=canonical_dependency_ids(row),
                source_snapshot_sha256=snapshot_hash,
                generation_path="durable",
                request_hashes=request_hashes,
                embedding_input_sha256=context_hash(texts),
                embedding_config_sha256=context_hash(config),
                embedding_texts=texts,
                canonical_text_sha256=context_hash(row.text),
                metadata_sha256=context_hash(
                    [
                        row.chunk_metadata,
                        row.heading_path,
                        row.position,
                        row.validity_start_date,
                        row.validity_end_date,
                    ]
                ),
                doc_summary=output,
                contextual_config=context_config,
                embedding_config=config,
                validity_start=as_of_date or row.validity_start_date,
                validity_end=row.validity_end_date,
            )
        )
    return PreparedContextView(
        projections=projections,
        snapshots=list(snapshots.values()),
        calls=list(calls.values()),
    )
