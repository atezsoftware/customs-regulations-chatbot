"""Fixed read-only retained evidence for the a8a1406 Markdown timeout."""

import re
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

RUNTIME = "a8a1406d4d4d359298247949bb0e3190f3069598"
MARKDOWN_FILE = UUID("681a26d9-919f-4d29-a3ec-e24a95175598")


def load_markdown_progress(
    session: "Session", *, expected_scope_key: str
) -> dict[str, object]:
    from sqlalchemy import select

    from onyx.db.models import (
        AmendmentBatch,
        ChatSession,
        DocumentSet,
        KVStore,
        RegulatoryCanonicalRevision,
        RegulatoryFilePublication,
        RegulatoryIndexingJob,
        RegulatoryTemporalProjection,
        UserFile,
    )
    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.file_processing.original_ingestion import OriginalIngestionReceipt

    result: dict[str, object] = {
        "status": "scope_refused",
        "scope_failure": "run_missing",
    }
    saved = session.get(KVStore, "regulatory_annex_acceptance:" + RUNTIME)
    if saved is None:
        return result
    run = CanaryRun.model_validate(saved.value)
    if (
        run.release_sha != RUNTIME
        or run.run_id != UUID("e96ad70f-f49f-45c1-a414-cf02e02f6359")
        or run.user_id != UUID("7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844")
        or run.file_id != UUID("eca03e71-2c07-497b-aa1c-84504cddc12b")
        or run.package_id != UUID("961f595b-b903-4c10-ad1a-bcb7898fbd52")
        or run.document_set_id != 27
        or run.batch_id != 51
        or run.persona_id != 3
        or run.phase != "cleaned"
        or run.evidence.get("cleanup_complete") is not True
        or run.created_at != datetime.fromisoformat("2026-09-12T20:20:10.448214+00:00")
        or run.markdown_file_ids != [MARKDOWN_FILE]
        or [
            intent.artifact_id
            for intent in run.creation_intents
            if intent.kind == "markdown"
            and intent.marker == "ANNEXCANARY" + run.run_id.hex + ".md"
        ]
        != [MARKDOWN_FILE]
    ):
        return {**result, "scope_failure": "run_ownership"}
    scope = session.get(DocumentSet, 27)
    batch = session.get(AmendmentBatch, 51)
    if (
        scope is None
        or scope.is_public
        or scope.user_id != run.user_id
        or scope.name != run.name
        or batch is None
        or batch.created_by != run.user_id
        or batch.document_set_id != 27
        or batch.source_package_id != run.package_id
        or batch.user_file_ids != [str(run.file_id)]
    ):
        return {**result, "scope_failure": "private_scope"}
    owner = session.get(RegulatoryFilePublication, MARKDOWN_FILE)
    file = session.get(UserFile, MARKDOWN_FILE)
    if owner is not None and owner.scope_key != expected_scope_key:
        return {**result, "scope_failure": "publication_scope"}
    if file is not None and (
        file.user_id != run.user_id or [item.id for item in file.document_sets] != [27]
    ):
        return {**result, "scope_failure": "file_ownership"}
    chat_times: dict[str, object] = {}
    for label, chat_id, day in (
        ("old_chat", UUID("0d6e8e7b-0c11-4c53-8669-aeb079c4daa2"), "2026-09-09"),
        ("new_chat", UUID("c94409a8-f6bd-4ef4-aaf5-3af0dc71968a"), "2026-09-10"),
    ):
        if chat_id not in run.chat_ids or not any(
            intent.kind == "chat"
            and intent.artifact_id == chat_id
            and intent.marker == run.name + " / dated " + day
            for intent in run.creation_intents
        ):
            return {**result, "scope_failure": "chat_ownership"}
        chat = session.get(ChatSession, chat_id)
        if chat is not None and (
            chat.user_id != run.user_id
            or chat.persona_id != 3
            or chat.description != run.name + " / dated " + day
        ):
            return {**result, "scope_failure": "chat_ownership"}
        chat_times[label + "_created_at"] = (
            chat.time_created.isoformat() if chat else None
        )
        chat_times[label + "_updated_at"] = (
            chat.time_updated.isoformat() if chat else None
        )
    canonical = list(
        session.scalars(
            select(RegulatoryCanonicalRevision)
            .where(RegulatoryCanonicalRevision.user_file_id == MARKDOWN_FILE)
            .limit(65)
        )
    )
    temporal = list(
        session.scalars(
            select(RegulatoryTemporalProjection)
            .where(RegulatoryTemporalProjection.user_file_id == MARKDOWN_FILE)
            .limit(65)
        )
    )
    jobs = list(
        session.scalars(
            select(RegulatoryIndexingJob)
            .where(RegulatoryIndexingJob.user_file_id == MARKDOWN_FILE)
            .limit(65)
        )
    )
    if any(len(rows) > 64 for rows in (canonical, temporal, jobs)):
        return {**result, "scope_failure": "evidence_limit"}
    raw_receipt = owner.original_ingestion_receipt if owner is not None else None
    receipt = None
    try:
        if raw_receipt is not None:
            candidate = OriginalIngestionReceipt.model_validate(raw_receipt)
            if candidate.file_id and all(
                re.fullmatch("[0-9a-f]{64}", value)
                for value in (
                    candidate.raw_sha256,
                    candidate.documents_sha256,
                    candidate.plaintext_sha256,
                    candidate.canonical_sha256,
                    candidate.generation_hash,
                )
            ):
                receipt = candidate
    except ValueError:
        pass
    result = {
        "status": "read",
        "owner_present": owner is not None,
        "gate_closed": owner.gate_closed if owner is not None else None,
        "writer_manifest_present": owner.writer_manifest is not None
        if owner is not None
        else False,
        "receipt_present": raw_receipt is not None,
        "receipt_valid": receipt is not None,
        "receipt_raw_sha256": receipt.raw_sha256 if receipt else None,
        "receipt_canonical_sha256": receipt.canonical_sha256 if receipt else None,
        "receipt_generation_sha256": receipt.generation_hash if receipt else None,
        "canonical_count": len(canonical),
        "temporal_count": len(temporal),
        "temporal_retired_count": sum(row.retired_at is not None for row in temporal),
        "first_canonical_at": min(row.created_at for row in canonical).isoformat()
        if canonical
        else None,
        "last_canonical_at": max(row.created_at for row in canonical).isoformat()
        if canonical
        else None,
        **chat_times,
        "first_published_at": min(row.published_at for row in temporal).isoformat()
        if temporal
        else None,
        "last_published_at": max(row.published_at for row in temporal).isoformat()
        if temporal
        else None,
        "file_present": file is not None,
        "file_status": file.status.value if file else None,
        "job_count": len(jobs),
        "jobs": [
            {
                "job_status": job.status,
                "job_stage": job.stage,
                "attempt_count": job.attempt_count,
                "next_retry_at": job.next_retry_at.isoformat()
                if job.next_retry_at
                else None,
                "has_error_code": job.error_code is not None,
            }
            for job in jobs
        ],
    }
    return result


def read_markdown_progress() -> dict[str, object]:
    """Configuration owns an earlier engine; never nest its reset with this read."""
    import os

    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from onyx.db.regulatory_annex_dev_cutover import configured_indices
    from onyx.document_index.publication_models import (
        PublicationScope,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY
    from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    result: dict[str, object] = {
        "stage": "markdown_a8",
        "status": "scope_refused",
        "scope_failure": "env",
        "database_read_only": True,
    }
    if not (
        os.environ.get("POSTGRES_DB") == "customs-regulations-dev"
        and os.environ.get("REGULATORY_ANNEX_ENVIRONMENT") == "dev"
        and os.environ.get("PGOPTIONS") == "-c default_transaction_read_only=on"
    ):
        return result
    set_is_ee_based_on_env_variable()
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    configured_indices()
    from onyx.configs import app_configs

    result.pop("scope_failure")
    result.update(
        configuration_verified=True,
        current_batch_indexing=app_configs.REGULATORY_BATCH_INDEXING_ENABLED,
        current_deferred_indexing=app_configs.DEFER_USER_FILE_INDEXING,
        current_vector_disabled=app_configs.DISABLE_VECTOR_DB,
    )
    scope_key = publication_digest(
        PublicationScope(
            tenant_id="public",
            environment="dev",
            database_identity=ANNEX_DATABASE_IDENTITY,
        ).model_dump(mode="json")
    )
    with SqlEngine.scoped_engine(
        pool_size=2,
        max_overflow=0,
        connect_args={
            "options": "-c default_transaction_read_only=on",
            "connect_timeout": 10,
        },
    ):
        with get_session_with_current_tenant() as session:
            result.update(load_markdown_progress(session, expected_scope_key=scope_key))
    return result
