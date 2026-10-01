"""Reopen only batch 306's verified unmatched instruction in DEV."""

import hashlib
import os
from uuid import UUID

from sqlalchemy import select

from onyx.background.celery.tasks.regulatory_amendments.tasks import (
    enqueue_amendment_batch,
)
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import AmendmentBatch, AmendmentProposal, RegulatoryChunk
from onyx.db.regulatory_amendments import reset_batch_attention_for_retry
from shared_configs.contextvars import get_current_tenant_id

BATCH_ID = 306
SOURCE_SHA256 = "41fdbdc4679b2e63e22db8ff9f88926cb634e48d8395abf40467b2fbbc4bd36d"
TARGET_FILE_ID = UUID("c7bed4e2-77a2-4609-bb52-3fdffcd15cb1")
TARGET_CHUNK_ID = "rc_cf2191636b0857c03551c430312de392f2820bf5"


def main() -> None:
    from onyx.regulatory.amendments.annexes import config as annex_config

    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("This retry is restricted to the DEV database")
    if annex_config.REGULATORY_ANNEX_ENVIRONMENT != "dev":
        raise RuntimeError("This retry is restricted to DEV publication")
    SqlEngine.init_engine(pool_size=1, max_overflow=0)

    with get_session_with_current_tenant() as session:
        batch = session.scalar(
            select(AmendmentBatch)
            .where(AmendmentBatch.id == BATCH_ID)
            .with_for_update()
        )
        if (
            batch is None
            or batch.status != "analyzed"
            or batch.superseded_by_batch_id is not None
            or hashlib.sha256(batch.raw_text.encode()).hexdigest() != SOURCE_SHA256
            or len(batch.segmented_instructions) != 1
            or batch.processed_instruction_indices != [0]
            or len(batch.unmatched_instructions) != 1
            or str(TARGET_FILE_ID) not in batch.user_file_ids
        ):
            raise RuntimeError("Batch 306 retry preconditions changed")
        instruction_text = batch.segmented_instructions[0]["instruction_text"]
        attention = batch.unmatched_instructions[0]
        if not attention.startswith(instruction_text + "\n\n"):
            raise RuntimeError("Unmatched instruction identity changed")
        if (
            session.scalar(
                select(AmendmentProposal.id).where(
                    AmendmentProposal.batch_id == BATCH_ID
                )
            )
            is not None
        ):
            raise RuntimeError("Batch 306 already has proposals")
        target = session.get(RegulatoryChunk, TARGET_CHUNK_ID)
        if (
            target is None
            or target.user_file_id != TARGET_FILE_ID
            or target.status != "active"
            or "**2) Gemilere satış mağazalarından yapılan satışlar**"
            not in target.text
            or "mükerrer satış" not in target.text
        ):
            raise RuntimeError("Verified target source changed")
        retried = reset_batch_attention_for_retry(session, batch_id=BATCH_ID)
        if retried is None:
            raise RuntimeError("Batch 306 could not reopen its unmatched instruction")

    enqueue_amendment_batch(batch_id=BATCH_ID, tenant_id=get_current_tenant_id())
    print("RETRIED", BATCH_ID, "instruction=0", "source_chunk=" + TARGET_CHUNK_ID)


if __name__ == "__main__":
    main()
