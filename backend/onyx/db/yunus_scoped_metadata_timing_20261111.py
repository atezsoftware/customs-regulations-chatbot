"""Measure fresh scoped metadata on the same DEV sources without applying it."""

import json
import time
from threading import Lock
from unittest.mock import patch
from uuid import UUID

from sqlalchemy import select, text

from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.labeling_configuration import (
    resolve_intake_labeling_provider_binding,
    resolve_labeling_gateway,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    User,
)
from onyx.regulatory.document_intake.metadata import (
    MAX_METADATA_REQUEST_BYTES,
    MAX_METADATA_RESPONSE_BYTES,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.metadata_scopes_job import (
    extract_scoped_metadata_document,
)
from onyx.tracing.flows import LLMFlow
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


def main() -> None:
    SqlEngine.init_engine(pool_size=3, max_overflow=1)
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    identifier = UUID("2f5ae4fb-6756-47c2-8776-d624d134ff8e")
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        assert (
            session.scalar(text("SELECT current_database()"))
            == "customs-regulations-dev"
        )
        batch = session.get_one(RegulatoryDocumentIntake, identifier)
        assert str(batch.created_by) == "9ea33f67-5578-4dc4-bbea-e32b16f3f1dc"
        assert batch.status == "ready" and batch.stage == "review"
        owner = session.get_one(User, batch.created_by)
        binding = resolve_intake_labeling_provider_binding(
            session, batch.labeling_model_configuration_id, user=owner
        )
        gateway = resolve_labeling_gateway(
            session,
            batch.labeling_model_configuration_id,
            user=owner,
            expected_binding=binding,
            direct_flow=LLMFlow.REGULATORY_INTAKE_METADATA,
            direct_max_request_bytes=MAX_METADATA_REQUEST_BYTES,
            direct_max_response_bytes=MAX_METADATA_RESPONSE_BYTES,
            direct_request_timeout_seconds=240,
        )
        members = list(
            session.scalars(
                select(RegulatoryDocumentIntakeFile)
                .where(RegulatoryDocumentIntakeFile.intake_id == identifier)
                .order_by(RegulatoryDocumentIntakeFile.role.desc())
            )
        )
        assert len(members) == 2
        root = next(member for member in members if member.role == "root")
        annex = next(member for member in members if member.role == "annex")
        assert annex.parent_document_id == root.id
        views = []
        for member in members:
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == member.user_file_id)
                    .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
                )
            )
            chunks = metadata_chunks(rows)
            assert len(chunks) == (338 if member.role == "root" else 20)
            views.append((member.role, member.id, chunks))
        session.rollback()

    submit = gateway.submit
    submissions: list[list[str]] = []
    lock = Lock()

    def measured_submit(requests, **kwargs):
        with lock:
            if len(submissions) >= 6:
                raise ValueError(
                    "The fresh two-document metadata measurement exceeded six submissions"
                )
            submissions.append([request.request_hash for request in requests])
        return submit(requests, **kwargs)

    measured = []
    total_started = time.monotonic()
    with patch.object(gateway, "submit", side_effect=measured_submit):
        for role, document_id, chunks in views:
            started = time.monotonic()
            count_before = len(submissions)
            print(
                "SCOPED_DOCUMENT_START",
                json.dumps({"role": role, "targets": len(chunks)}),
                flush=True,
            )
            results = extract_scoped_metadata_document(
                gateway=gateway,
                chunks=chunks,
                identity_prefix=f"yunus-fresh-scoped-timing-20261111-v1:{identifier}:{document_id}:{metadata_input_hash(chunks)}",
                heartbeat=lambda: None,
            )
            assert sum(len(result.chunks) for _hash, result in results) == len(chunks)
            record = {
                "role": role,
                "targets": len(chunks),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "gateway_submissions": len(submissions) - count_before,
                "validated_fields": sum(
                    len(target.fields)
                    for _hash, result in results
                    for target in result.chunks
                ),
            }
            measured.append(record)
            print("SCOPED_DOCUMENT_DONE", json.dumps(record), flush=True)
    print(
        "SCOPED_TIMING_DONE",
        json.dumps(
            {
                "documents": measured,
                "elapsed_seconds": round(time.monotonic() - total_started, 3),
                "gateway_submissions": len(submissions),
                "canonical_metadata_applied": False,
                "indexing_requested": False,
                "approval_requested": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
