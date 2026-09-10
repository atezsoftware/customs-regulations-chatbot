from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from onyx.access.access import get_access_for_user_files
from onyx.access.models import DocumentAccess
from onyx.configs.constants import DEFAULT_BOOST, DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingStage,
)
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
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationExpectation,
    DocumentChunkVerificationRequest,
    DocumentChunkVerificationResult,
    DocumentIndex,
)
from onyx.indexing.models import (
    ChunkEmbedding,
    DocMetadataAwareIndexChunk,
    IndexChunk,
)
from onyx.regulatory.chunk_evidence import chunk_evidence
from onyx.regulatory.heading_path import normalize_regulatory_heading_path
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
)
from onyx.regulatory.indexing_jobs.projection_identity import projection_ordinal


class PublishVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    document_id: str = Field(min_length=1)
    canonical_chunk_count: int = Field(gt=0)
    embedded_item_count: int = Field(gt=0)
    vector_dimension: int = Field(gt=0)
    insertion_record_count: int = Field(gt=0)


class PublishOutcome(StrEnum):
    COMPLETED = "COMPLETED"


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
    job_id: UUID,
    user_file_id: UUID,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    expected_dimension: int,
) -> list[tuple[RegulatoryChunk, RegulatoryIndexingItem]]:
    from onyx.regulatory.indexing_jobs.projection_identity import (
        ordered_projection_items,
    )

    ordered = ordered_projection_items(
        job_id=job_id, user_file_id=user_file_id, rows=rows, items=items
    )
    for _row, item in ordered:
        if item.status != RegulatoryIndexingItemStatus.EMBEDDED.value:
            raise ValueError("every canonical chunk must be embedded before indexing")
        if not _is_valid_vector(item.vector, expected_dimension):
            raise ValueError("embedded item vector is invalid")
    return ordered


def _expected_verification(
    *,
    job_id: UUID,
    user_file_id: UUID,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> PublishVerification:
    ordered = _ordered_projection(
        job_id=job_id,
        user_file_id=user_file_id,
        rows=rows,
        items=items,
        expected_dimension=snapshot.effective_dimension,
    )
    return PublishVerification(
        job_id=job_id,
        document_id=str(user_file_id),
        canonical_chunk_count=len({row.id for row, _ in ordered}),
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
    raw_checkpoint = item.context.get("context_input")
    checkpoint = (
        cast(dict[str, object], raw_checkpoint)
        if isinstance(raw_checkpoint, dict)
        else None
    )
    if (
        value is None
        and isinstance(checkpoint, dict)
        and (
            checkpoint.get("request_hash") == item.request_hash
            and checkpoint.get("prompt")
            == f"Context skipped for canonical chunk {item.regulatory_chunk_id}"
        )
    ):
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("embedded item contextual text is invalid")
    return value


def _build_hidden_chunks(
    *,
    job_id: UUID,
    user_file_id: UUID,
    user_file_name: str,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    snapshot: RegulatoryIndexingConfigSnapshot,
    tenant_id: str,
    db_session: Session,
) -> list[DocMetadataAwareIndexChunk]:
    ordered = _ordered_projection(
        job_id=job_id,
        user_file_id=user_file_id,
        rows=rows,
        items=items,
        expected_dimension=snapshot.effective_dimension,
    )
    document_id = str(user_file_id)
    document = Document(
        id=document_id,
        source=DocumentSource.USER_FILE,
        semantic_identifier=user_file_name,
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
            chunk_id=projection_ordinal(row, item, chunk_id),
            blurb=row.text,
            content=row.text,
            source_links=chunk_evidence(row.chunk_metadata).source_links,
            image_file_id=chunk_evidence(row.chunk_metadata).image_file_id,
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
            validity_start_date=item.effective_start
            if getattr(item, "projection_id", None) is not None
            else row.validity_start_date,
            validity_end_date=item.effective_end
            if getattr(item, "projection_id", None) is not None
            else row.validity_end_date,
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


def _verification_request(
    *,
    expected: PublishVerification,
    rows: Sequence[RegulatoryChunk],
    hidden: bool,
    items: Sequence[RegulatoryIndexingItem] | None = None,
) -> DocumentChunkVerificationRequest:
    if items is None:
        expected_chunks = tuple(
            DocumentChunkVerificationExpectation(
                chunk_index=ordinal, regulatory_chunk_id=row.id
            )
            for ordinal, row in enumerate(
                sorted(rows, key=lambda row: (row.position, row.id))
            )
        )
    else:
        mapped = _ordered_projection(
            job_id=expected.job_id,
            user_file_id=UUID(expected.document_id),
            rows=rows,
            items=items,
            expected_dimension=expected.vector_dimension,
        )
        expected_chunks = tuple(
            DocumentChunkVerificationExpectation(
                chunk_index=projection_ordinal(row, item, ordinal),
                regulatory_chunk_id=row.id,
            )
            for ordinal, (row, item) in enumerate(mapped)
        )
    return DocumentChunkVerificationRequest(
        document_id=expected.document_id,
        expected_chunks=expected_chunks,
        require_contiguous=items is None,
        expected_hidden=hidden,
        content_vector_dimension=expected.vector_dimension,
    )


def _validate_index_verification(
    result: DocumentChunkVerificationResult,
    expected: PublishVerification,
    *,
    hidden: bool,
) -> None:
    if (
        result.document_id != expected.document_id
        or result.chunk_count != expected.embedded_item_count
        or len(result.document_chunk_ids) != expected.embedded_item_count
        or result.hidden is not hidden
    ):
        raise ValueError("document index verification returned unexpected invariants")


def _validate_call_identity(
    lease: indexing_job_repository.RegulatoryIndexingExternalMutationLease,
    *,
    user_file: UserFile,
    search_settings: SearchSettings | None,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
) -> None:
    """Reject mismatched caller objects while trusting only locked DB state."""

    if user_file.id != lease.user_file_id:
        raise ValueError("user file does not belong to the locked indexing job")
    if search_settings is not None and search_settings.id != lease.search_settings_id:
        raise ValueError("SearchSettings does not belong to the locked indexing job")
    if {row.id for row in rows} != {row.id for row in lease.regulatory_chunks}:
        raise ValueError("caller chunks do not match the locked canonical projection")
    if {item.id for item in items} != {item.id for item in lease.indexing_items}:
        raise ValueError("caller items do not match the locked indexing projection")


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
    """Stage the frozen job with hidden sources under shared actual ES fencing."""
    return _owned_checkpoint(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        search_settings=search_settings,
        tenant_id=tenant_id,
        db_session=db_session,
        document_index=document_index,
        stage=RegulatoryIndexingStage.INDEX_WRITE,
    )


def verify_staged_regulatory_job(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    db_session: Session,
    document_index: DocumentIndex | None = None,
    search_settings: SearchSettings | None = None,
) -> PublishVerification:
    """Reseal and verify the same hidden inventory under the new delivery token."""
    from shared_configs.contextvars import get_current_tenant_id

    return _owned_checkpoint(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        search_settings=search_settings,
        tenant_id=get_current_tenant_id(),
        db_session=db_session,
        document_index=document_index,
        stage=RegulatoryIndexingStage.VERIFY,
    )


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
) -> PublishOutcome:
    """Publish and complete the durable job in the canonical activation transaction."""
    from shared_configs.contextvars import get_current_tenant_id

    if verification is not None and verification != _expected_verification(
        job_id=job.id,
        user_file_id=user_file.id,
        rows=rows,
        items=items,
        snapshot=RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot),
    ):
        raise ValueError("durable publication verification identity mismatch")
    _owned_checkpoint(
        job=job,
        user_file=user_file,
        rows=rows,
        items=items,
        search_settings=search_settings,
        tenant_id=get_current_tenant_id(),
        db_session=db_session,
        document_index=document_index,
        stage=RegulatoryIndexingStage.PUBLISH,
    )
    return PublishOutcome.COMPLETED


def _owned_checkpoint(
    *,
    job: RegulatoryIndexingJob,
    user_file: UserFile,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    search_settings: SearchSettings | None,
    tenant_id: str,
    db_session: Session,
    document_index: DocumentIndex | None,
    stage: RegulatoryIndexingStage,
) -> PublishVerification:
    from onyx.regulatory.indexing_jobs.owned_publication import (
        execute_owned_durable_stage,
    )

    if document_index is not None:
        raise ValueError(
            "durable publication requires the configured fenced Elasticsearch transport"
        )
    if (
        not tenant_id.strip()
        or user_file.id != job.user_file_id
        or (
            search_settings is not None and search_settings.id != job.search_settings_id
        )
    ):
        raise ValueError("durable publication caller scope mismatch")
    job_id, file_id, generation = job.id, user_file.id, job.lease_generation
    row_ids, item_ids = {row.id for row in rows}, {item.id for item in items}
    db_session.rollback()
    runtime = execute_owned_durable_stage(
        job_id=job_id,
        file_id=file_id,
        generation=generation,
        stage=stage,
        tenant_id=tenant_id,
        caller_row_ids=row_ids,
        caller_item_ids=item_ids,
    )
    return _expected_verification(
        job_id=job_id,
        user_file_id=file_id,
        rows=runtime.regulatory_chunks,
        items=runtime.indexing_items,
        snapshot=RegulatoryIndexingConfigSnapshot.model_validate(
            runtime.job.config_snapshot
        ),
    )
