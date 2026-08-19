from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from hashlib import sha256
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
)
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.db.regulatory_indexing_jobs import (
    advance_regulatory_indexing_job,
    claim_regulatory_indexing_job,
    create_or_get_regulatory_indexing_item,
    create_or_get_regulatory_indexing_job,
    persist_regulatory_indexing_item_skipped,
)
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
)
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

    claimed_generation = unclaimed_generation + 1
    tokenizer = get_tokenizer(
        snapshot.embedding_model_name,
        snapshot.embedding_provider,
    )
    documents_to_regulatory_chunks(
        documents=documents,
        db_session=db_session,
        tokenizer=tokenizer,
        enable_contextual_rag=True,
    )
    rows = get_chunks_for_file(db_session, user_file_id)
    if not rows:
        raise ValueError("regulatory indexing produced no canonical chunks")

    for row in rows:
        contextual_reserve = contextual_reserve_for_row(rows, row, tokenizer)
        request = (
            contextual_request_for_row(job, rows, row, tokenizer)
            if contextual_reserve > 0
            else VertexBatchRequest(
                prompt=f"Context skipped for canonical chunk {row.id}"
            )
        )
        item = create_or_get_regulatory_indexing_item(
            db_session,
            job_id=job.id,
            regulatory_chunk_id=row.id,
            request_hash=request.request_hash,
            expected_generation=claimed_generation,
        )
        if item is None:
            raise RuntimeError("regulatory indexing lease was lost during preparation")
        if (
            contextual_reserve == 0
            and item.status == RegulatoryIndexingItemStatus.PENDING.value
            and not persist_regulatory_indexing_item_skipped(
                db_session,
                item_id=item.id,
                expected_generation=claimed_generation,
            )
        ):
            raise RuntimeError(
                "regulatory indexing lease was lost while skipping context"
            )

    advanced = advance_regulatory_indexing_job(
        db_session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.PREPARING,
        expected_generation=claimed_generation,
        next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        next_status=RegulatoryIndexingJobStatus.QUEUED,
        now=now,
    )
    if not advanced:
        raise RuntimeError("regulatory indexing lease was lost after preparation")
    return job.id
