from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from hashlib import sha256
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import RegulatoryIndexingStage
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.db.regulatory_indexing_jobs import (
    claim_regulatory_indexing_job,
    create_or_get_regulatory_indexing_job,
    get_regulatory_indexing_job,
)
from onyx.llm.constants import LlmProviderNames
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.regulatory.indexing import (
    document_text_for_regulatory_indexing,
    documents_to_regulatory_chunks,
)
from onyx.regulatory.indexing_jobs.configuration import (
    resolve_regulatory_indexing_snapshot,
)
from onyx.regulatory.indexing_jobs.contextual import (
    contextual_request_for_row,
    contextual_reserve_for_row,
    get_contextual_token_budget_tokenizer,
)
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest


def _regulatory_documents_content_hash(documents: Sequence[Document]) -> str:
    payload = [
        {
            "semantic_identifier": document.semantic_identifier,
            "title": document.title,
            "text": document_text_for_regulatory_indexing(document),
        }
        for document in documents
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode()).hexdigest()


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

    snapshot = resolve_regulatory_indexing_snapshot(db_session)
    now = datetime.datetime.now(datetime.timezone.utc)
    job = create_or_get_regulatory_indexing_job(
        db_session,
        user_file_id=user_file_id,
        content_hash=_regulatory_documents_content_hash(documents),
        search_settings_id=snapshot.search_settings_id,
        prompt_hash=snapshot.prompt_hash,
        config_snapshot=snapshot.model_dump(mode="json"),
        now=now,
    )
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
    if _regulatory_documents_content_hash(documents) != job.content_hash:
        raise ValueError("regulatory documents do not match the claimed job revision")

    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(job.config_snapshot)
    embedding_tokenizer = get_tokenizer(
        snapshot.embedding_model_name,
        snapshot.embedding_provider,
    )
    contextual_tokenizer = get_contextual_token_budget_tokenizer(
        model_provider=LlmProviderNames.VERTEX_AI,
        model_name=snapshot.vertex.model_name,
    )

    def prepare_items() -> list[indexing_job_repository.RegulatoryIndexingPreparedItem]:
        documents_to_regulatory_chunks(
            documents=documents,
            db_session=db_session,
            tokenizer=embedding_tokenizer,
            enable_contextual_rag=True,
        )
        rows = get_chunks_for_file(db_session, job.user_file_id)
        if not rows:
            raise ValueError("regulatory indexing produced no canonical chunks")

        prepared_items: list[
            indexing_job_repository.RegulatoryIndexingPreparedItem
        ] = []
        for row in rows:
            contextual_reserve = contextual_reserve_for_row(
                rows,
                row,
                embedding_tokenizer=embedding_tokenizer,
            )
            request = (
                contextual_request_for_row(
                    job,
                    rows,
                    row,
                    contextual_tokenizer=contextual_tokenizer,
                )
                if contextual_reserve > 0
                else VertexBatchRequest(
                    prompt=f"Context skipped for canonical chunk {row.id}"
                )
            )
            prepared_items.append(
                indexing_job_repository.RegulatoryIndexingPreparedItem(
                    regulatory_chunk_id=row.id,
                    request_hash=request.request_hash,
                    skip_context=contextual_reserve == 0,
                )
            )
        return prepared_items

    persisted = indexing_job_repository.persist_regulatory_indexing_preparation(
        db_session,
        job_id=job.id,
        expected_generation=expected_generation,
        prepare_items=prepare_items,
        now=datetime.datetime.now(datetime.timezone.utc),
    )
    if not persisted:
        raise RuntimeError("regulatory indexing lease was lost during preparation")
    return job.id
