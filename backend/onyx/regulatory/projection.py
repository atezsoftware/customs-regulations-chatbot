"""Project PostgreSQL regulatory chunks into every active search index.

The relational ``regulatory_chunk`` rows are the source of truth. Projection
never re-parses the uploaded file, and it keeps superseded rows so temporal
queries can retrieve the version that was valid on the requested date.
"""

import datetime
import hashlib
from uuid import UUID

from chonkie import SentenceChunker
from sqlalchemy.orm import Session

from onyx.access.models import default_public_access
from onyx.configs.app_configs import (
    BLURB_SIZE,
    ENABLE_CONTEXTUAL_RAG,
    USE_CHUNK_SUMMARY,
    USE_DOCUMENT_SUMMARY,
)
from onyx.configs.constants import DEFAULT_BOOST, DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.db.search_settings import get_active_search_settings_list
from onyx.db.user_file import (
    fetch_persona_ids_for_user_files,
    fetch_user_project_ids_for_user_files,
    lock_completed_user_file_for_projection,
)
from onyx.document_index.factory import get_all_document_indices
from onyx.document_index.interfaces_new import IndexingMetadata
from onyx.httpx.httpx_pool import HttpxPool
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.chunking import extract_blurb
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.models import DocAwareChunk, DocMetadataAwareIndexChunk, IndexChunk
from onyx.llm.factory import get_contextual_rag_llm_for_search_settings
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    fit_context_fields_to_embedding_budget,
    validity_window_contains,
)
from onyx.regulatory.heading_path import normalize_regulatory_heading_path
from onyx.utils.logger import setup_logger
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

logger = setup_logger()


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
    # ``position`` is a logical slot shared by an original and its amendment.
    # Enumerating the complete, stable row order prevents those versions from
    # overwriting one another in Elasticsearch.
    return [
        DocAwareChunk(
            source_document=document,
            chunk_id=chunk_id,
            blurb=extract_blurb(row.text, blurb_splitter),
            content=row.text,
            source_links={0: ""},
            image_file_id=None,
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
        for chunk_id, row in enumerate(rows)
    ]


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
) -> None:
    """Add temporally isolated context without crowding out legal text."""

    llm = get_contextual_rag_llm_for_search_settings(search_settings)
    if llm is None:
        raise ValueError(
            "Contextual retrieval is enabled but no contextual model is configured"
        )

    canonical_document = chunks[0].source_document
    today = datetime.date.today()
    snapshot_documents: dict[tuple[str, ...], Document] = {}
    contextual_chunks: list[DocAwareChunk] = []
    skipped_for_budget = 0

    for chunk, row in zip(chunks, rows):
        reference_date = context_reference_date(
            row.validity_start_date,
            row.validity_end_date,
            today=today,
        )
        valid_candidates = [
            candidate
            for candidate in rows
            if validity_window_contains(
                candidate.validity_start_date,
                candidate.validity_end_date,
                reference_date,
            )
        ]
        candidates_by_position: dict[int, list[RegulatoryChunk]] = {}
        for candidate in valid_candidates:
            candidates_by_position.setdefault(candidate.position, []).append(candidate)
        visible_rows: list[RegulatoryChunk] = []
        all_positions = set(candidates_by_position) | {row.position}
        for position in sorted(all_positions):
            candidates = candidates_by_position.get(position, [])
            if position == row.position:
                visible_rows.append(row)
            elif len(candidates) == 1:
                visible_rows.append(candidates[0])
            # An overlapping validity interval is ambiguous. Omitting that
            # position is safer than leaking contradictory versions into the
            # target chunk's generated context.
        visible_rows.sort(key=lambda candidate: (candidate.position, candidate.id))
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

    llm_tokenizer = get_tokenizer(
        model_name=llm.config.model_name,
        provider_type=llm.config.model_provider,
    )
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


def _enrich_index_chunks(
    *,
    index_chunks: list[IndexChunk],
    user_file_id: str,
    project_ids: dict[str, list[int]],
    persona_ids: dict[str, list[int]],
    tenant_id: str,
) -> list[DocMetadataAwareIndexChunk]:
    return [
        DocMetadataAwareIndexChunk.from_index_chunk(
            index_chunk=chunk,
            access=default_public_access,
            document_sets=set(),
            user_project=project_ids.get(user_file_id, []),
            personas=persona_ids.get(user_file_id, []),
            boost=DEFAULT_BOOST,
            tenant_id=tenant_id,
            aggregated_chunk_boost_factor=1.0,
        )
        for chunk in index_chunks
    ]


def _project_rows_to_search_settings(
    *,
    user_file: UserFile,
    rows: list[RegulatoryChunk],
    search_settings: SearchSettings,
    tenant_id: str,
    project_ids: dict[str, list[int]],
    persona_ids: dict[str, list[int]],
    indexing_metadata: IndexingMetadata,
) -> int:
    """Project immutable PostgreSQL rows into exactly one search setting."""

    user_file_id = str(user_file.id)
    canonical_document = _build_document_shell(user_file)
    embedder = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=search_settings
    )

    def token_counter(text: str) -> int:
        return len(embedder.embedding_model.tokenizer.encode(text))

    blurb_splitter = SentenceChunker(
        tokenizer_or_token_counter=token_counter,
        chunk_size=BLURB_SIZE,
        chunk_overlap=0,
        return_type="texts",
    )
    doc_chunks = _rows_to_doc_aware_chunks(
        canonical_document,
        rows,
        blurb_splitter,
    )
    if search_settings.enable_contextual_rag or ENABLE_CONTEXTUAL_RAG:
        _contextualize_chunks(
            chunks=doc_chunks,
            rows=rows,
            user_file=user_file,
            embedder=embedder,
            search_settings=search_settings,
        )

    index_chunks = embedder.embed_chunks(doc_chunks, tenant_id=tenant_id)
    enriched_chunks = _enrich_index_chunks(
        index_chunks=index_chunks,
        user_file_id=user_file_id,
        project_ids=project_ids,
        persona_ids=persona_ids,
        tenant_id=tenant_id,
    )
    document_indices = get_all_document_indices(
        search_settings,
        None,
        httpx_client=HttpxPool.get("vespa"),
    )
    for document_index in document_indices:
        document_index.index(
            chunks=enriched_chunks,
            indexing_metadata=indexing_metadata,
        )
    logger.info(
        "project_user_file_to_index: wrote %d chunks for user_file=%s "
        "search_settings=%s",
        len(enriched_chunks),
        user_file_id,
        search_settings.id,
    )
    return len(enriched_chunks)


def project_user_file_to_index(
    db_session: Session,
    user_file: UserFile,
    tenant_id: str,
) -> int:
    """Embed and replace one file from its rows in PRESENT and FUTURE indices."""

    user_file_id = str(user_file.id)
    locked_user_file = lock_completed_user_file_for_projection(
        db_session, UUID(user_file_id)
    )
    if locked_user_file is None:
        logger.info(
            "project_user_file_to_index: user file is gone or not completed; "
            "skipping user_file=%s",
            user_file_id,
        )
        return 0
    user_file = locked_user_file
    rows = get_chunks_for_file(db_session, UUID(user_file_id))
    if not rows:
        logger.warning(
            "project_user_file_to_index: no chunk rows for user_file=%s", user_file_id
        )
        return 0

    search_settings_list = get_active_search_settings_list(db_session)
    if not any(settings.status.is_current() for settings in search_settings_list):
        raise RuntimeError("No current search settings found")

    project_ids = fetch_user_project_ids_for_user_files([user_file_id], db_session)
    persona_ids = fetch_persona_ids_for_user_files([user_file_id], db_session)

    old_chunk_cnt = user_file.chunk_count or 0
    new_chunk_cnt = len(rows)
    indexing_metadata = IndexingMetadata(
        doc_id_to_chunk_cnt_diff={
            user_file_id: IndexingMetadata.ChunkCounts(
                old_chunk_cnt=max(old_chunk_cnt, new_chunk_cnt),
                new_chunk_cnt=new_chunk_cnt,
            )
        }
    )

    for search_settings in search_settings_list:
        try:
            _project_rows_to_search_settings(
                user_file=user_file,
                rows=rows,
                search_settings=search_settings,
                tenant_id=tenant_id,
                project_ids=project_ids,
                persona_ids=persona_ids,
                indexing_metadata=indexing_metadata,
            )
        except Exception:
            if search_settings.status.is_current():
                raise
            user_file.secondary_reconcile_pending = True
            logger.exception(
                "Deferred FUTURE regulatory projection for user_file=%s "
                "search_settings=%s",
                user_file_id,
                search_settings.id,
            )
            continue
        if search_settings.status.is_future():
            user_file.secondary_reconcile_pending = False

    user_file.chunk_count = len(rows)
    db_session.add(user_file)
    return len(rows)
