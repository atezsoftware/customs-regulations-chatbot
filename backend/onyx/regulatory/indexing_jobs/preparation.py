from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from hashlib import sha256
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import RegulatoryIndexingStage, UserFileStatus
from onyx.db.models import RegulatoryChunk, RegulatoryIndexingJob, UserFile
from onyx.db.regulatory_chunks import get_chunks_for_file_snapshot
from onyx.db.regulatory_indexing_jobs import (
    claim_regulatory_indexing_job,
    create_or_get_regulatory_indexing_job,
    get_regulatory_indexing_job,
)
from onyx.llm.constants import LlmProviderNames
from onyx.natural_language_processing.utils import BaseTokenizer, get_tokenizer
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.indexing import (
    document_text_for_regulatory_indexing,
)
from onyx.regulatory.indexing_jobs.configuration import (
    resolve_regulatory_indexing_snapshot,
)
from onyx.regulatory.indexing_jobs.contextual import (
    get_contextual_token_budget_tokenizer,
)
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    RegulatoryInputHashVersion,
)


def regulatory_documents_content_hash(
    documents: Sequence[Document],
    input_hash_version: RegulatoryInputHashVersion,
) -> str:
    if input_hash_version is RegulatoryInputHashVersion.LEGACY_V1:
        payload = [
            {
                "semantic_identifier": document.semantic_identifier,
                "title": document.title,
                "text": document_text_for_regulatory_indexing(document),
            }
            for document in documents
        ]
    elif input_hash_version is RegulatoryInputHashVersion.CANONICAL_V2:
        payload = [
            document.model_dump(mode="json", exclude={"doc_updated_at"})
            for document in documents
        ]
    else:
        raise ValueError(
            f"unsupported regulatory input hash version: {input_hash_version}"
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode()).hexdigest()


def regulatory_chunks_content_hash(rows: Sequence[RegulatoryChunk]) -> str:
    """Hash the canonical CHUNKED rows consumed by a durable indexing job."""

    ordered_rows = sorted(rows, key=lambda row: (row.position, row.id))
    if not ordered_rows:
        raise ValueError("regulatory indexing requires canonical chunks")
    user_file_ids = {row.user_file_id for row in ordered_rows}
    if len(user_file_ids) != 1:
        raise ValueError("canonical chunks belong to different user files")
    payload = [
        {
            "id": row.id,
            "position": row.position,
            "text": row.text,
            "chunk_type": row.chunk_type,
            "heading_path": row.heading_path,
            "chunk_metadata": row.chunk_metadata,
            "validity_start_date": (
                row.validity_start_date.isoformat()
                if row.validity_start_date is not None
                else None
            ),
            "validity_end_date": (
                row.validity_end_date.isoformat()
                if row.validity_end_date is not None
                else None
            ),
            "status": row.status,
            "source": row.source,
            "supersedes_chunk_id": row.supersedes_chunk_id,
            "superseded_by_chunk_id": row.superseded_by_chunk_id,
        }
        for row in ordered_rows
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode()).hexdigest()


def resolve_regulatory_documents_input_hash_version(
    documents: Sequence[Document],
    *,
    persisted_content_hash: str,
    declared_version: RegulatoryInputHashVersion,
) -> RegulatoryInputHashVersion:
    """Resolve migration-only hash ambiguity without weakening current identity."""

    if declared_version is not RegulatoryInputHashVersion.LEGACY_OR_CANONICAL:
        if (
            regulatory_documents_content_hash(documents, declared_version)
            != persisted_content_hash
        ):
            raise ValueError(
                "regulatory documents do not match the claimed job revision"
            )
        return declared_version

    matching_versions = tuple(
        version
        for version in (
            RegulatoryInputHashVersion.LEGACY_V1,
            RegulatoryInputHashVersion.CANONICAL_V2,
        )
        if regulatory_documents_content_hash(documents, version)
        == persisted_content_hash
    )
    if len(matching_versions) != 1:
        raise ValueError(
            "regulatory documents do not uniquely resolve the claimed job revision"
        )
    return matching_versions[0]


def _create_owned_regulatory_indexing_job(
    db_session: Session,
    *,
    user_file_id: UUID,
    tenant_id: str,
    snapshot: RegulatoryIndexingConfigSnapshot,
    now: datetime.datetime,
) -> RegulatoryIndexingJob:
    from uuid import uuid4

    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    db_session.rollback()
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        return create_or_get_regulatory_indexing_job(
            db_session,
            user_file_id=user_file_id,
            content_hash=snapshot.input_content_hash,
            search_settings_id=snapshot.search_settings_id,
            prompt_hash=snapshot.prompt_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            config_snapshot=snapshot.model_dump(mode="json"),
            now=now,
            publication_owner=owner,
        )
    finally:
        db_session.rollback()
        authority.release(owner)


def prepare_regulatory_indexing_job(
    user_file_id: UUID,
    documents: Sequence[Document],
    tenant_id: str,
    db_session: Session,
) -> UUID:
    """Create canonical chunks and durable items for one immutable file revision."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    if not documents:
        raise ValueError("regulatory indexing requires at least one document")
    if any(document.id != str(user_file_id) for document in documents):
        raise ValueError("regulatory documents must be stamped with the user file id")

    input_hash_version = RegulatoryInputHashVersion.CANONICAL_V2
    content_hash = regulatory_documents_content_hash(documents, input_hash_version)
    snapshot = resolve_regulatory_indexing_snapshot(
        db_session,
        input_content_hash=content_hash,
        input_hash_version=input_hash_version,
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    job = _create_owned_regulatory_indexing_job(
        db_session,
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        snapshot=snapshot,
        now=now,
    )
    if job.chunk_generation_hash != snapshot.chunk_generation_hash:
        return job.id
    unclaimed_generation = job.lease_generation
    claimed = claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=unclaimed_generation,
        now=now,
    )
    if not claimed:
        return job.id

    return prepare_claimed_regulatory_indexing_job(
        job_id=job.id,
        expected_generation=unclaimed_generation + 1,
        documents=documents,
        tenant_id=tenant_id,
        db_session=db_session,
    )


def prepare_regulatory_indexing_job_from_chunks(
    user_file_id: UUID,
    tenant_id: str,
    db_session: Session,
) -> UUID:
    """Create a durable job that reuses the production CHUNKED source of truth."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    user_file = db_session.get(UserFile, user_file_id)
    if user_file is None:
        raise ValueError("regulatory indexing user file does not exist")
    if user_file.status not in {
        UserFileStatus.CHUNKED,
        UserFileStatus.INDEXING,
        UserFileStatus.COMPLETED,
    }:
        raise ValueError(
            "durable regulatory indexing requires a CHUNKED, INDEXING, or "
            "COMPLETED user file"
        )
    rows = get_chunks_for_file_snapshot(db_session, user_file_id)
    content_hash = regulatory_chunks_content_hash(rows)
    snapshot = resolve_regulatory_indexing_snapshot(
        db_session,
        input_content_hash=content_hash,
        input_hash_version=RegulatoryInputHashVersion.CHUNK_ROWS_V3,
    )
    if user_file.regulatory_chunk_generation_hash is None:
        raise ValueError("CHUNKED user file has no chunk-generation identity")
    if user_file.regulatory_chunk_generation_hash != snapshot.chunk_generation_hash:
        raise ValueError("CHUNKED user file generation does not match indexing config")

    now = datetime.datetime.now(datetime.timezone.utc)
    job = _create_owned_regulatory_indexing_job(
        db_session,
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        snapshot=snapshot,
        now=now,
    )
    if job.content_hash != content_hash:
        raise RuntimeError(
            "another regulatory indexing revision is still active for this file"
        )
    if job.chunk_generation_hash != snapshot.chunk_generation_hash:
        return job.id
    unclaimed_generation = job.lease_generation
    claimed = claim_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=unclaimed_generation,
        now=now,
    )
    if not claimed:
        return job.id
    return prepare_claimed_regulatory_indexing_job_from_chunks(
        job_id=job.id,
        expected_generation=unclaimed_generation + 1,
        tenant_id=tenant_id,
        db_session=db_session,
    )


def prepare_claimed_regulatory_indexing_job_from_chunks(
    *,
    job_id: UUID,
    expected_generation: int,
    tenant_id: str,
    db_session: Session,
) -> UUID:
    """Prepare an already-claimed v3 job without loading or rechunking Markdown."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    job = get_regulatory_indexing_job(db_session, job_id)
    if job is None:
        raise ValueError(f"regulatory indexing job {job_id} does not exist")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    if snapshot.input_hash_version is not RegulatoryInputHashVersion.CHUNK_ROWS_V3:
        raise ValueError("claimed regulatory job is not a chunk-row job")
    if snapshot.input_content_hash != job.content_hash:
        raise ValueError("regulatory job input hash does not match its snapshot")
    if snapshot.chunk_generation_hash != job.chunk_generation_hash:
        raise ValueError("regulatory job generation hash does not match its snapshot")
    user_file = db_session.get(UserFile, job.user_file_id)
    if user_file is None:
        raise ValueError("regulatory indexing user file disappeared")
    if user_file.regulatory_chunk_generation_hash != snapshot.chunk_generation_hash:
        raise ValueError(
            "CHUNKED user file generation identity is absent or mismatched"
        )

    embedding_tokenizer = get_tokenizer(
        snapshot.embedding_model_name,
        snapshot.embedding_provider,
    )
    contextual_tokenizer = get_contextual_token_budget_tokenizer(
        model_provider=LlmProviderNames.VERTEX_AI,
        model_name=snapshot.vertex.model_name,
    )

    return _prepare_claimed_owned_items(
        job=job,
        snapshot=snapshot,
        expected_generation=expected_generation,
        tenant_id=tenant_id,
        db_session=db_session,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
        resolved_input_hash_version=RegulatoryInputHashVersion.CHUNK_ROWS_V3,
    )


def prepare_claimed_regulatory_indexing_job(
    *,
    job_id: UUID,
    expected_generation: int,
    documents: Sequence[Document],
    tenant_id: str,
    db_session: Session,
) -> UUID:
    """Prepare a PREPARING lease already claimed by ordinary or stale recovery."""

    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    if not documents:
        raise ValueError("regulatory indexing requires at least one document")

    job = get_regulatory_indexing_job(db_session, job_id)
    if job is None:
        raise ValueError(f"regulatory indexing job {job_id} does not exist")
    if any(document.id != str(job.user_file_id) for document in documents):
        raise ValueError("regulatory documents must be stamped with the user file id")
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    resolved_input_hash_version = resolve_regulatory_documents_input_hash_version(
        documents,
        persisted_content_hash=job.content_hash,
        declared_version=snapshot.input_hash_version,
    )
    if snapshot.input_content_hash != job.content_hash:
        raise ValueError("regulatory job input hash does not match its snapshot")
    if snapshot.chunk_generation_hash != job.chunk_generation_hash:
        raise ValueError("regulatory job generation hash does not match its snapshot")
    embedding_tokenizer = get_tokenizer(
        snapshot.embedding_model_name,
        snapshot.embedding_provider,
    )
    contextual_tokenizer = get_contextual_token_budget_tokenizer(
        model_provider=LlmProviderNames.VERTEX_AI,
        model_name=snapshot.vertex.model_name,
    )

    return _prepare_claimed_owned_items(
        job=job,
        snapshot=snapshot,
        expected_generation=expected_generation,
        tenant_id=tenant_id,
        db_session=db_session,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
        resolved_input_hash_version=resolved_input_hash_version,
        documents=documents,
    )


def _prepare_claimed_owned_items(
    *,
    job: RegulatoryIndexingJob,
    snapshot: RegulatoryIndexingConfigSnapshot,
    expected_generation: int,
    tenant_id: str,
    db_session: Session,
    embedding_tokenizer: BaseTokenizer,
    contextual_tokenizer: BaseTokenizer,
    resolved_input_hash_version: RegulatoryInputHashVersion,
    documents: Sequence[Document] | None = None,
) -> UUID:
    from datetime import timedelta
    from uuid import uuid4

    from onyx.db.regulatory_publication import PublicationStore
    from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config
    from onyx.regulatory.amendments.annexes.publication_execution import (
        publication_heartbeat,
    )
    from onyx.regulatory.indexing_jobs.projection_preparation import (
        prepare_owned_durable_items,
    )
    from onyx.regulatory.writer_publication import recover_owned_writer_before_next

    file_id = job.user_file_id
    db_session.rollback()
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=2))
    try:
        owner = recover_owned_writer_before_next(owner)
        inputs = load_owned_writer_inputs(owner)
        if not inputs.canonical:
            if not documents:
                raise ValueError(
                    "durable preparation has no canonical chunks or original document"
                )
            from onyx.db.regulatory_writer_publication import (
                persist_owned_initial_chunks,
            )

            persist_owned_initial_chunks(
                owner,
                documents,
                embedding_tokenizer,
                contextual=True,
                generation_hash=snapshot.chunk_generation_hash,
                preparing_job_id=job.id,
                preparing_job_generation=expected_generation,
            )
            inputs = load_owned_writer_inputs(owner)
        if (
            resolved_input_hash_version is RegulatoryInputHashVersion.CHUNK_ROWS_V3
            and regulatory_chunks_content_hash(
                canonical_snapshot_rows(inputs.canonical)
            )
            != job.content_hash
        ):
            raise ValueError("canonical chunks changed after durable job creation")
        with publication_heartbeat(owner):
            prepared_items = prepare_owned_durable_items(
                owner=owner,
                job=job,
                inputs=inputs,
                embedding_tokenizer=embedding_tokenizer,
                contextual_tokenizer=contextual_tokenizer,
            )
        persisted = indexing_job_repository.persist_regulatory_indexing_preparation(
            db_session,
            job_id=job.id,
            expected_generation=expected_generation,
            prepare_items=lambda: prepared_items,
            resolved_input_hash_version=resolved_input_hash_version.value,
            now=datetime.datetime.now(datetime.timezone.utc),
            publication_owner=owner,
        )
        if not persisted:
            raise RuntimeError("regulatory indexing lease was lost during preparation")
        return job.id
    finally:
        db_session.rollback()
        authority.release(owner)
