from __future__ import annotations

import datetime
import math
from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from onyx.access.access import get_access_for_user_files
from onyx.access.models import DocumentAccess
from onyx.configs.constants import DEFAULT_BOOST, DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import RegulatoryIndexingItemStatus, UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    UserFile,
)
from onyx.db.user_file import (
    fetch_document_set_names_for_user_files,
    fetch_persona_ids_for_user_files,
    fetch_user_project_ids_for_user_files,
)
from onyx.document_index.factory import build_elasticsearch_document_index
from onyx.document_index.interfaces_new import (
    DocumentIndex,
    IndexingMetadata,
    MetadataUpdateRequest,
)
from onyx.indexing.models import (
    ChunkEmbedding,
    DocMetadataAwareIndexChunk,
    IndexChunk,
)
from onyx.regulatory.heading_path import normalize_regulatory_heading_path
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot


class PublishVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    document_id: str = Field(min_length=1)
    canonical_chunk_count: int = Field(gt=0)
    embedded_item_count: int = Field(gt=0)
    vector_dimension: int = Field(gt=0)
    insertion_record_count: int = Field(gt=0)


def _is_valid_vector(vector: object, expected_dimension: int) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == expected_dimension
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in vector
        )
    )


def _ordered_projection(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    expected_dimension: int,
) -> list[tuple[RegulatoryChunk, RegulatoryIndexingItem]]:
    if user_file.id != job.user_file_id:
        raise ValueError("user file does not belong to the regulatory indexing job")
    ordered_rows = sorted(rows, key=lambda row: (row.position, row.id))
    if not ordered_rows:
        raise ValueError("regulatory indexing job has no canonical chunks")
    if any(row.user_file_id != user_file.id for row in ordered_rows):
        raise ValueError("canonical chunk belongs to a different user file")
    row_ids = [row.id for row in ordered_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("canonical chunks contain duplicate ids")

    item_by_row_id: dict[str, RegulatoryIndexingItem] = {}
    for item in items:
        if item.job_id != job.id:
            raise ValueError("indexing item belongs to a different job")
        if item.regulatory_chunk_id in item_by_row_id:
            raise ValueError("indexing items contain duplicate canonical chunks")
        item_by_row_id[item.regulatory_chunk_id] = item
    if set(item_by_row_id) != set(row_ids):
        raise ValueError("embedded items do not exactly cover canonical chunks")

    ordered = [(row, item_by_row_id[row.id]) for row in ordered_rows]
    for _row, item in ordered:
        if item.status != RegulatoryIndexingItemStatus.EMBEDDED.value:
            raise ValueError("every canonical chunk must be embedded before indexing")
        if not _is_valid_vector(item.vector, expected_dimension):
            raise ValueError("embedded item vector is invalid")
    return ordered


def _expected_verification(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> PublishVerification:
    ordered = _ordered_projection(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        expected_dimension=snapshot.effective_dimension,
    )
    return PublishVerification(
        job_id=job.id,
        document_id=str(user_file.id),
        canonical_chunk_count=len(ordered),
        embedded_item_count=len(ordered),
        vector_dimension=snapshot.effective_dimension,
        insertion_record_count=1,
    )


def _normalized_heading_path(row: RegulatoryChunk) -> list[str]:
    metadata = row.chunk_metadata
    return normalize_regulatory_heading_path(
        row.heading_path,
        article_no=(
            str(metadata["article_no"])
            if metadata.get("article_no") is not None
            else None
        ),
        chunk_type=row.chunk_type,
        paragraph_no=(
            str(metadata["paragraph_no"])
            if metadata.get("paragraph_no") is not None
            else None
        ),
        clause_label=(
            str(metadata["clause_label"])
            if metadata.get("clause_label") is not None
            else None
        ),
    )


def _contextual_text(item: RegulatoryIndexingItem) -> str:
    if item.context is None:
        return ""
    if not isinstance(item.context, dict):
        raise ValueError("embedded item context is invalid")
    value = item.context.get("contextual_text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("embedded item contextual text is invalid")
    return value


def _build_hidden_chunks(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    snapshot: RegulatoryIndexingConfigSnapshot,
    tenant_id: str,
    db_session: Session,
) -> list[DocMetadataAwareIndexChunk]:
    ordered = _ordered_projection(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        expected_dimension=snapshot.effective_dimension,
    )
    document_id = str(user_file.id)
    document = Document(
        id=document_id,
        source=DocumentSource.USER_FILE,
        semantic_identifier=user_file.name,
        title="",
        sections=[
            TextSection(
                text="\n\n".join(row.text for row, _item in ordered),
                link=None,
            )
        ],
        metadata={},
        chunk_count=len(ordered),
    )
    access_by_file = get_access_for_user_files([document_id], db_session)
    project_ids = fetch_user_project_ids_for_user_files([document_id], db_session)
    persona_ids = fetch_persona_ids_for_user_files([document_id], db_session)
    document_sets = fetch_document_set_names_for_user_files([document_id], db_session)
    no_access = DocumentAccess.build(
        user_emails=[],
        user_groups=[],
        external_user_emails=[],
        external_user_group_ids=[],
        is_public=False,
    )

    chunks: list[DocMetadataAwareIndexChunk] = []
    for chunk_id, (row, item) in enumerate(ordered):
        vector = [float(value) for value in item.vector or []]
        index_chunk = IndexChunk(
            source_document=document,
            chunk_id=chunk_id,
            blurb=row.text,
            content=row.text,
            source_links={0: ""},
            image_file_id=None,
            section_continuation=False,
            title_prefix="",
            metadata_suffix_semantic="",
            metadata_suffix_keyword="",
            mini_chunk_texts=None,
            large_chunk_id=None,
            doc_summary=_contextual_text(item),
            chunk_context="",
            contextual_rag_reserved_tokens=0,
            regulatory_chunk_id=row.id,
            heading_path=_normalized_heading_path(row),
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
            embeddings=ChunkEmbedding(
                full_embedding=vector,
                mini_chunk_embeddings=[],
            ),
            title_embedding=None,
        )
        chunks.append(
            DocMetadataAwareIndexChunk.from_index_chunk(
                index_chunk=index_chunk,
                access=access_by_file.get(document_id, no_access),
                document_sets=set(document_sets.get(document_id, [])),
                user_project=project_ids.get(document_id, []),
                personas=persona_ids.get(document_id, []),
                boost=DEFAULT_BOOST,
                aggregated_chunk_boost_factor=1.0,
                tenant_id=tenant_id,
                hidden=True,
            )
        )
    return chunks


def _validate_search_settings(
    search_settings: SearchSettings,
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> None:
    if (
        search_settings.id,
        search_settings.provider_type,
        search_settings.model_name,
        search_settings.model_dim,
        search_settings.reduced_dimension,
        search_settings.final_embedding_dim,
        search_settings.index_name,
    ) != (
        snapshot.search_settings_id,
        snapshot.embedding_provider,
        snapshot.embedding_model_name,
        snapshot.model_dimension,
        snapshot.reduced_dimension,
        snapshot.effective_dimension,
        snapshot.index_name,
    ):
        raise ValueError("SearchSettings no longer matches the indexing job snapshot")


def stage_regulatory_job_in_index(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    search_settings: SearchSettings,
    tenant_id: str,
    db_session: Session,
    document_index: DocumentIndex | None = None,
) -> PublishVerification:
    """Write one complete deterministic file generation with hidden chunks."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    if user_file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}:
        raise ValueError("cancelled or deleting user file cannot be staged")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    _validate_search_settings(search_settings, snapshot)
    expected = _expected_verification(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        snapshot=snapshot,
    )
    chunks = _build_hidden_chunks(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        snapshot=snapshot,
        tenant_id=tenant_id,
        db_session=db_session,
    )
    target_index = document_index or build_elasticsearch_document_index(search_settings)
    old_chunk_count = user_file.chunk_count or 0
    indexing_metadata = IndexingMetadata(
        doc_id_to_chunk_cnt_diff={
            expected.document_id: IndexingMetadata.ChunkCounts(
                old_chunk_cnt=max(old_chunk_count, expected.canonical_chunk_count),
                new_chunk_cnt=expected.canonical_chunk_count,
            )
        }
    )
    insertion_records = target_index.index(
        chunks=chunks,
        indexing_metadata=indexing_metadata,
    )
    if len(insertion_records) != 1:
        raise ValueError("regulatory publication requires exactly one insertion record")
    if insertion_records[0].document_id != expected.document_id:
        raise ValueError("regulatory insertion record has an unexpected document id")
    return expected


def publish_regulatory_job(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    verification: PublishVerification | None = None,
    db_session: Session,
    document_index: DocumentIndex | None = None,
    search_settings: SearchSettings | None = None,
) -> None:
    """Publish verified chunks, then complete the user file through a lease fence."""

    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    expected = _expected_verification(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        snapshot=snapshot,
    )
    if user_file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}:
        raise ValueError("cancelled or deleting user file cannot be published")
    if verification is not None and verification != expected:
        raise ValueError("publication verification no longer matches persisted state")

    if document_index is None:
        if search_settings is None:
            raise ValueError(
                "search_settings is required when document_index is not supplied"
            )
        _validate_search_settings(search_settings, snapshot)
        document_index = build_elasticsearch_document_index(search_settings)
    document_index.update(
        [
            MetadataUpdateRequest(
                document_ids=[expected.document_id],
                doc_id_to_chunk_cnt={
                    expected.document_id: expected.canonical_chunk_count
                },
                hidden=False,
            )
        ]
    )
    completed = indexing_job_repository.complete_regulatory_indexing_user_file(
        db_session,
        job_id=job.id,
        expected_generation=job.lease_generation,
        chunk_count=expected.canonical_chunk_count,
        now=datetime.datetime.now(datetime.timezone.utc),
    )
    if not completed:
        raise RuntimeError("regulatory indexing lease was lost while publishing")
