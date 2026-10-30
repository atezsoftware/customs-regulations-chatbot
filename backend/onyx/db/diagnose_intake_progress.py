"""Read-only DEV progress for the four exact reported new-document imports."""

import hashlib
import json
import os
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
from onyx.db.models import (
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    User,
)
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    METADATA_MODEL,
    METADATA_VERSION,
    TARGETS_PER_REQUEST,
    build_metadata_request,
    metadata_input_hash,
)

_BATCH = UUID("8b91e6cf-e755-4f0c-9f27-861f40273ab0")
_MEMBERS = {
    UUID("2814a713-41f9-4292-aa0d-3cf05104e6ce"): UUID(
        "eb5aba9a-5f27-4b50-b977-30f5539f9676"
    ),
    UUID("38993fc3-fb3c-4eaa-aa0f-576e925b820a"): UUID(
        "6a9e0ada-2939-4397-ae41-5b7c8d7da69a"
    ),
    UUID("6b1d6522-a26b-4eba-8d82-288feb57f70a"): UUID(
        "9e421263-6504-4b6b-8e4a-ad36f7d5df43"
    ),
    UUID("ef8dcdf7-8144-44bb-83cc-6312817b13c1"): UUID(
        "17e7eec4-2bf8-4eee-b64a-370c7f9cc96a"
    ),
}
_MAX_CHUNKS = 2000


def inspect_progress(session: Session) -> dict[str, object]:
    batch = session.get_one(RegulatoryDocumentIntake, _BATCH)
    if (
        batch.source_package_id != UUID("9dc3a63a-431e-4198-a922-da463525833d")
        or batch.document_set_id != 15
        or batch.labeling_model_configuration_id != 46
    ):
        raise RuntimeError("Exact intake scope changed")
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == _BATCH)
            .order_by(RegulatoryDocumentIntakeFile.id)
            .limit(5)
        )
    )
    if {member.id: member.user_file_id for member in members} != _MEMBERS:
        raise RuntimeError("Exact intake membership changed")
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    files: list[dict[str, object]] = []
    for member in members:
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == member.user_file_id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
                .limit(_MAX_CHUNKS + 1)
            )
        )
        if len(rows) > _MAX_CHUNKS:
            raise RuntimeError("Intake file exceeds diagnostic chunk bound")
        chunks = metadata_chunks(rows)
        input_hash = metadata_input_hash(chunks)
        canonical_ids = {chunk.chunk_id for chunk in chunks}
        audits = [
            row.chunk_metadata.get(METADATA_AUDIT_KEY)
            for row in rows
            if row.id in canonical_ids
        ]
        receipts = sum(
            isinstance(audit, dict) and audit.get("model") == METADATA_MODEL
            for audit in audits
        )
        complete = (
            bool(chunks)
            and receipts == len(chunks)
            and all(
                isinstance(audit, dict) and audit.get("input_sha256") == input_hash
                for audit in audits
            )
        )
        cache_ids = []
        if not complete:
            prefix = f"metadata-v{METADATA_VERSION}:public:{batch.id}:{member.id}:{binding.fingerprint}"
            for offset in range(0, len(chunks), TARGETS_PER_REQUEST):
                request = build_metadata_request(
                    chunks,
                    [
                        chunk.chunk_id
                        for chunk in chunks[offset : offset + TARGETS_PER_REQUEST]
                    ],
                )
                submission = hashlib.sha256(
                    f"{prefix}:{request.request_hash}".encode()
                ).hexdigest()
                cache_ids.append(
                    f"regulatory-labeling-direct/{binding.fingerprint}/{submission}/{request.request_hash}.json"
                )
        cached = (
            len(
                list(
                    session.scalars(
                        select(FileRecord.file_id)
                        .where(
                            FileRecord.file_id.in_(cache_ids),
                            FileRecord.file_origin == FileOrigin.OTHER,
                            FileRecord.file_type == "application/json",
                        )
                        .limit(len(cache_ids) + 1)
                    )
                )
            )
            if cache_ids
            else None
            if complete
            else 0
        )
        files.append(
            {
                "document_id": str(member.id),
                "file_id": str(member.user_file_id),
                "canonical_count": len(chunks),
                "total_windows": (len(chunks) + TARGETS_PER_REQUEST - 1)
                // TARGETS_PER_REQUEST,
                "metadata_receipt_chunks": receipts,
                "metadata_complete": complete,
                "cached_base_response_count": cached,
                "cache_count_basis": "not_reconstructed_after_metadata_commit"
                if complete
                else "record_existence_only_not_strict_validation",
            }
        )
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": str(batch.id),
        "status": batch.status,
        "stage": batch.stage,
        "error": batch.error,
        "lease_expires_at": batch.lease_expires_at,
        "binding_fingerprint": binding.fingerprint,
        "files": files,
    }


def main() -> None:
    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("Expected explicit DEV database")
    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database="customs-regulations-dev",
        ),
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c search_path=public -c statement_timeout=30000 -c lock_timeout=2000 -c application_name=intake_progress_diagnostic",
            **(
                {"sslmode": os.environ["POSTGRES_SSLMODE"]}
                if os.environ.get("POSTGRES_SSLMODE")
                else {}
            ),
        },
        pool_size=1,
        max_overflow=0,
    )
    try:
        with Session(engine, autoflush=False) as session:
            session.execute(text("SET TRANSACTION READ ONLY"))
            if session.scalar(
                text("SELECT current_database()")
            ) != "customs-regulations-dev" or any(
                session.scalar(text(f"SHOW {setting}")) != "on"
                for setting in (
                    "default_transaction_read_only",
                    "transaction_read_only",
                )
            ):
                raise RuntimeError("DEV read-only connection guard failed")
            print(
                "INTAKE_PROGRESS",
                json.dumps(inspect_progress(session), default=str),
                flush=True,
            )
            session.rollback()
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
