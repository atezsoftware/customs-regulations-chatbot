"""Project PostgreSQL regulatory chunks into every active search index.

The relational ``regulatory_chunk`` rows are the source of truth. Projection
never re-parses the uploaded file, and it keeps superseded rows so temporal
queries can retrieve the version that was valid on the requested date.
"""

import datetime
import hashlib

from chonkie import SentenceChunker
from sqlalchemy.orm import Session

from onyx.configs.app_configs import (
    BLURB_SIZE,
    USE_CHUNK_SUMMARY,
    USE_DOCUMENT_SUMMARY,
)
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.chunking import extract_blurb
from onyx.indexing.contextual_settings import (
    effective_contextual_rag_enabled,
    require_contextual_rag_llm,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.models import DocAwareChunk
from onyx.llm.constants import LlmProviderNames
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.utils import BaseTokenizer, get_tokenizer
from onyx.regulatory.amendments.annexes.context_dependencies import (
    ContextGenerationRecorder,
    canonical_dependency_ids,
    context_hash,
    contextual_model_fingerprint,
    effective_context_rows,
    freeze_context_source_snapshot,
    freeze_embedding_inputs,
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import (
    ContextSourceSnapshot,
    FrozenContextProjection,
    PreparedContextView,
)
from onyx.regulatory.chunk_evidence import chunk_evidence
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    fit_context_fields_to_embedding_budget,
    visible_regulatory_snapshot_for_target,
)
from onyx.regulatory.heading_path import normalize_regulatory_heading_path
from onyx.regulatory.indexing_jobs.contextual import (
    get_contextual_token_budget_tokenizer,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

logger = setup_logger()


def _get_contextual_tokenizer(llm: LLM) -> BaseTokenizer:
    if llm.config.model_provider == LlmProviderNames.VERTEX_AI:
        return get_contextual_token_budget_tokenizer(
            model_provider=LlmProviderNames.VERTEX_AI,
            model_name=llm.config.model_name,
        )
    return get_tokenizer(
        model_name=llm.config.model_name,
        provider_type=llm.config.model_provider,
    )


def _build_document_shell(
    user_file: UserFile,
    *,
    document_id: str | None = None,
    text: str = "",
) -> Document:
    """Build document metadata without reading the uploaded source file."""

    return Document(
        id=document_id or str(user_file.id),
        source=DocumentSource.USER_FILE,
        semantic_identifier=user_file.name,
        sections=[TextSection(text=text, link=None)],
        metadata={},
    )


def _rows_to_doc_aware_chunks(
    document: Document,
    rows: list[RegulatoryChunk],
    blurb_splitter: SentenceChunker,
) -> list[DocAwareChunk]:
    return [
        DocAwareChunk(
            source_document=document,
            chunk_id=row.projection_ordinal,
            blurb=extract_blurb(row.text, blurb_splitter),
            content=row.text,
            source_links=chunk_evidence(row.chunk_metadata).source_links,
            image_file_id=chunk_evidence(row.chunk_metadata).image_file_id,
            section_continuation=False,
            title_prefix="",
            metadata_suffix_semantic="",
            metadata_suffix_keyword="",
            mini_chunk_texts=None,
            large_chunk_id=None,
            doc_summary="",
            chunk_context="",
            contextual_rag_reserved_tokens=0,
            regulatory_chunk_id=row.id,
            heading_path=normalize_regulatory_heading_path(
                row.heading_path,
                article_no=(
                    str(row.chunk_metadata["article_no"])
                    if row.chunk_metadata.get("article_no") is not None
                    else None
                ),
                chunk_type=row.chunk_type,
                paragraph_no=(
                    str(row.chunk_metadata["paragraph_no"])
                    if row.chunk_metadata.get("paragraph_no") is not None
                    else None
                ),
                clause_label=(
                    str(row.chunk_metadata["clause_label"])
                    if row.chunk_metadata.get("clause_label") is not None
                    else None
                ),
            ),
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
        )
        for row in rows
    ]


def _rows_in_structural_order(rows: list[RegulatoryChunk]) -> list[RegulatoryChunk]:
    """Order legal structure independently from immutable index identity."""

    return sorted(rows, key=lambda row: (row.position, row.id))


def _row_context_text(row: RegulatoryChunk) -> str:
    heading_path = normalize_regulatory_heading_path(
        row.heading_path,
        article_no=(
            str(row.chunk_metadata["article_no"])
            if row.chunk_metadata.get("article_no") is not None
            else None
        ),
        chunk_type=row.chunk_type,
        paragraph_no=(
            str(row.chunk_metadata["paragraph_no"])
            if row.chunk_metadata.get("paragraph_no") is not None
            else None
        ),
        clause_label=(
            str(row.chunk_metadata["clause_label"])
            if row.chunk_metadata.get("clause_label") is not None
            else None
        ),
    )
    heading = " > ".join(heading_path)
    return f"{heading}\n{row.text}" if heading else row.text


def _contextualize_chunks(
    *,
    chunks: list[DocAwareChunk],
    rows: list[RegulatoryChunk],
    user_file: UserFile,
    embedder: DefaultIndexingEmbedder,
    search_settings: SearchSettings,
    context_rows: list[RegulatoryChunk] | None = None,
    context_llm: LLM | None = None,
    recorder: ContextGenerationRecorder | None = None,
    context_date: datetime.date | None = None,
) -> None:
    """Add temporally isolated context without crowding out legal text."""

    llm = context_llm or require_contextual_rag_llm(search_settings)
    assert llm is not None, "contextualization called while disabled"

    canonical_document = chunks[0].source_document
    today = datetime.date.today()
    snapshot_documents: dict[tuple[str, ...], Document] = {}
    contextual_chunks: list[DocAwareChunk] = []
    skipped_for_budget = 0

    snapshot_rows = context_rows if context_rows is not None else rows
    for chunk, row in zip(chunks, rows, strict=True):
        reference_date = context_date or context_reference_date(
            row.validity_start_date,
            row.validity_end_date,
            today=today,
        )
        visible_rows = visible_regulatory_snapshot_for_target(
            snapshot_rows,
            row,
            today=today,
            reference_date=context_date,
        )
        if len(visible_rows) <= 1:
            continue

        embedding_text = (
            f"{chunk.title_prefix}{chunk.content}{chunk.metadata_suffix_semantic}"
        )
        reserve = contextual_reserve_for_embedding_text(
            embedding_text,
            tokenizer=embedder.embedding_model.tokenizer,
            embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
            requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        )
        if reserve == 0:
            skipped_for_budget += 1
            continue

        chunk.contextual_rag_reserved_tokens = reserve
        snapshot_key = tuple(candidate.id for candidate in visible_rows)
        context_document = snapshot_documents.get(snapshot_key)
        if context_document is None:
            snapshot_digest = hashlib.sha256(
                "|".join(snapshot_key).encode()
            ).hexdigest()[:12]
            context_document = _build_document_shell(
                user_file,
                document_id=(
                    f"{user_file.id}::regulatory-context::"
                    f"{reference_date.isoformat()}::{snapshot_digest}"
                ),
                text="\n\n".join(_row_context_text(item) for item in visible_rows),
            )
            snapshot_documents[snapshot_key] = context_document
        chunk.source_document = context_document
        contextual_chunks.append(chunk)

    if not contextual_chunks:
        if skipped_for_budget:
            logger.warning(
                "Skipped contextual enrichment for %d oversized chunks in user_file=%s",
                skipped_for_budget,
                user_file.id,
            )
        return

    llm_tokenizer = _get_contextual_tokenizer(llm)
    try:
        # Lazy import avoids loading the full indexing pipeline for projections
        # that do not use contextual retrieval.
        from onyx.indexing.indexing_pipeline import add_contextual_summaries

        add_contextual_summaries(
            chunks=contextual_chunks,
            llm=llm,
            tokenizer=llm_tokenizer,
            chunk_token_limit=DOC_EMBEDDING_CONTEXT_SIZE * 2,
            raise_on_failure=True,
            **({"recorder": recorder} if recorder is not None else {}),
        )
        for chunk in contextual_chunks:
            chunk.doc_summary, chunk.chunk_context = (
                fit_context_fields_to_embedding_budget(
                    title_prefix=chunk.title_prefix,
                    content=chunk.content,
                    metadata_suffix=chunk.metadata_suffix_semantic,
                    doc_summary=chunk.doc_summary,
                    chunk_context=chunk.chunk_context,
                    tokenizer=embedder.embedding_model.tokenizer,
                    embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
                )
            )

        if USE_CHUNK_SUMMARY:
            incomplete_chunks = [
                chunk for chunk in contextual_chunks if not chunk.chunk_context.strip()
            ]
        elif USE_DOCUMENT_SUMMARY:
            incomplete_chunks = [
                chunk for chunk in contextual_chunks if not chunk.doc_summary.strip()
            ]
        else:
            incomplete_chunks = []
        if incomplete_chunks:
            raise RuntimeError(
                "Regulatory contextual projection is incomplete for "
                f"user_file={user_file.id}: {len(incomplete_chunks)}/"
                f"{len(contextual_chunks)} eligible chunks lack generated context"
            )
    finally:
        # Temporary snapshot ids are only for contextual grouping. Elasticsearch
        # identity remains the canonical user-file id in every index.
        for chunk in chunks:
            chunk.source_document = canonical_document

    if skipped_for_budget:
        logger.warning(
            "Skipped contextual enrichment for %d oversized chunks in user_file=%s",
            skipped_for_budget,
            user_file.id,
        )


def project_user_file_to_index(
    db_session: Session,
    user_file: UserFile,
    tenant_id: str,
    *,
    include_chunked: bool = False,
    include_failed: bool = False,
    current_search_settings_id: int | None = None,
) -> int:
    """Embed and replace one file from its canonical regulatory rows.

    `include_chunked` admits files that have chunks but were never indexed, which
    is what an explicit index request needs.

    `current_search_settings_id` selects only the exact validated PRESENT row.
    This strict mode rejects a concurrent promotion and marks an existing FUTURE
    copy for reconciliation instead of invoking its embedding provider.

    `include_failed` admits a file left FAILED by a terminal legacy indexing job.
    INDEXING remains excluded because another durable job may still own it.
    """

    from onyx.regulatory.writer_publication import republish_user_file

    user_file_id = user_file.id
    if db_session.new or db_session.dirty or db_session.deleted:
        raise ValueError(
            "canonical mutation must be staged under publication ownership"
        )
    db_session.rollback()
    count = republish_user_file(
        user_file_id,
        tenant_id,
        include_chunked=include_chunked,
        include_failed=include_failed,
        current_search_settings_id=current_search_settings_id,
    )
    db_session.refresh(user_file)
    return count


def prepare_normal_context_view(
    *,
    rows: list[RegulatoryChunk],
    user_file: UserFile,
    search_settings: SearchSettings,
    embedder: DefaultIndexingEmbedder,
    llm: LLM | None,
    cached: PreparedContextView | None = None,
    as_of_date: datetime.date | None = None,
    changed_ids: list[str] | None = None,
) -> PreparedContextView:
    """Prepare every potential consumer using the same normal projection path.

    No provider vectors or index writes occur here. Generated context and exact
    embedding inputs are frozen for the approval/publication manifest.
    """
    if not rows:
        return PreparedContextView()
    if any(row.user_file_id != user_file.id for row in rows):
        raise ValueError("context file scope mismatch")
    if changed_ids:
        rows = rebuild_context_aggregates(rows, changed_ids=changed_ids)
    all_rows = _rows_in_structural_order(rows)
    ordered = effective_context_rows(all_rows, as_of_date)
    if not ordered:
        return PreparedContextView()
    recorder = ContextGenerationRecorder(cached_calls=cached.calls if cached else [])
    splitter = SentenceChunker(
        tokenizer_or_token_counter=lambda text: len(
            embedder.embedding_model.tokenizer.encode(text)
        ),
        chunk_size=BLURB_SIZE,
        chunk_overlap=0,
        return_type="texts",
    )
    chunks = _rows_to_doc_aware_chunks(
        _build_document_shell(user_file), ordered, splitter
    )
    if effective_contextual_rag_enabled(search_settings):
        if llm is None:
            raise ValueError("contextual_configuration_unavailable")
        _contextualize_chunks(
            chunks=chunks,
            rows=ordered,
            user_file=user_file,
            embedder=embedder,
            search_settings=search_settings,
            context_llm=llm,
            recorder=recorder,
            context_rows=all_rows,
            context_date=as_of_date,
        )
    return _freeze_normal_context_view(
        rows=ordered,
        context_rows=all_rows,
        chunks=chunks,
        user_file=user_file,
        search_settings=search_settings,
        embedder=embedder,
        llm=llm,
        recorder=recorder,
        as_of_date=as_of_date,
    )


def _freeze_normal_context_view(
    *,
    rows: list[RegulatoryChunk],
    context_rows: list[RegulatoryChunk],
    chunks: list[DocAwareChunk],
    user_file: UserFile,
    search_settings: SearchSettings,
    embedder: DefaultIndexingEmbedder,
    llm: LLM | None,
    recorder: ContextGenerationRecorder,
    as_of_date: datetime.date | None = None,
) -> PreparedContextView:
    if any(row.user_file_id != user_file.id for row in rows):
        raise ValueError("context file scope mismatch")
    ordered, all_rows = rows, context_rows
    snapshots: dict[str, ContextSourceSnapshot] = {}
    projections: list[FrozenContextProjection] = []
    for row, chunk in zip(ordered, chunks, strict=True):
        snapshot = freeze_context_source_snapshot(
            rows=all_rows,
            target=row,
            reference_date=as_of_date,
            generation_path="normal",
            row_text=_row_context_text,
        )
        snapshot_hash = snapshot.sha256
        snapshots[snapshot_hash] = snapshot
        embedding_texts, embedding_config = freeze_embedding_inputs(
            chunk, embedder.embedding_model, model_dim=search_settings.model_dim
        )
        projections.append(
            FrozenContextProjection(
                canonical_chunk_id=row.id,
                canonical_dependency_ids=canonical_dependency_ids(row),
                source_snapshot_sha256=snapshot_hash,
                generation_path="normal",
                request_hashes=recorder.consumer_requests.get(row.id, []),
                embedding_input_sha256=context_hash(embedding_texts),
                embedding_config_sha256=context_hash(embedding_config),
                embedding_texts=embedding_texts,
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
                doc_summary=chunk.doc_summary,
                chunk_context=chunk.chunk_context,
                title=chunk.source_document.get_title_for_document_index(),
                mini_chunk_texts=chunk.mini_chunk_texts or [],
                contextual_config=contextual_model_fingerprint(llm) if llm else {},
                embedding_config=embedding_config,
                validity_start=as_of_date or row.validity_start_date,
                validity_end=row.validity_end_date,
            )
        )
    return PreparedContextView(
        projections=projections,
        snapshots=list(snapshots.values()),
        calls=list(recorder.calls.values()),
    )
