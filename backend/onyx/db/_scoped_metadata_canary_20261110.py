"""Read actual DEV source; measure scoped extraction without applying or indexing."""

import importlib.util
import json
import sys
import time
from pathlib import Path
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
from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256
from onyx.tracing.flows import LLMFlow


def main() -> None:
    source_root = Path(sys.argv[1])
    for leaf in ("metadata_structure", "metadata_scopes", "metadata_scopes_job"):
        name = "onyx.regulatory.document_intake." + leaf
        path = source_root / "backend/onyx/regulatory/document_intake" / f"{leaf}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from onyx.regulatory.document_intake.metadata_scopes import (
        build_scoped_metadata_request,
    )
    from onyx.regulatory.document_intake.metadata_scopes_job import (
        extract_scoped_metadata_document,
    )

    assert (
        metadata_recipe_sha256()
        == "56c3504da668fd7299ffd8bd3af7bf438b3224bbf98fc590e3b94e5f0cac45f2"
    )
    SqlEngine.init_engine(pool_size=3, max_overflow=1)
    batch_id = UUID("2f5ae4fb-6756-47c2-8776-d624d134ff8e")
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        assert (
            session.scalar(text("SELECT current_database()"))
            == "customs-regulations-dev"
        )
        batch = session.get_one(RegulatoryDocumentIntake, batch_id)
        assert str(batch.created_by) == "9ea33f67-5578-4dc4-bbea-e32b16f3f1dc"
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
                .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
                .order_by(RegulatoryDocumentIntakeFile.role.desc())
            )
        )
        assert (
            len(members) == 2
            and sum(m.role == "root" for m in members) == 1
            and sum(m.role == "annex" for m in members) == 1
        )
        parent = next(m for m in members if m.role == "root")
        annex = next(m for m in members if m.role == "annex")
        assert annex.parent_document_id == parent.id
        views = []
        for member in members:
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == member.user_file_id)
                    .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
                )
            )
            assert len(rows) == (350 if member.role == "root" else 20)
            views.append((str(member.id), member.role, metadata_chunks(rows)))
        session.rollback()
    from onyx.regulatory.document_intake.metadata import MetadataChunk
    from onyx.regulatory.document_intake.metadata_structure import (
        VerifiedStructuralFact,
    )

    control = json.loads(
        (
            source_root
            / "backend/tests/unit/onyx/regulatory/fixtures/md_annex_navigation/scoped_temporal_controls.json"
        ).read_text()
    )
    views.append(
        (
            "controlled-temporal",
            "controlled",
            [MetadataChunk.model_validate(row) for row in control["chunks"]],
        )
    )
    submissions = []
    submit = gateway.submit

    def measured_submit(requests, **kwargs):
        if len(submissions) >= 6:
            raise ValueError("Read-only canary exhausted its six-submission budget")
        submissions.append([r.request_hash for r in requests])
        return submit(requests, **kwargs)

    report = []
    with patch.object(gateway, "submit", side_effect=measured_submit):
        for document_id, role, chunks in views:
            started = time.monotonic()
            request = build_scoped_metadata_request(chunks)
            print(
                "CANARY_START",
                json.dumps(
                    {
                        "role": role,
                        "targets": len(chunks),
                        "request_bytes": len(request.prompt.encode()),
                        "legacy_requests": (len(chunks) + 7) // 8,
                    }
                ),
                flush=True,
            )
            results = extract_scoped_metadata_document(
                gateway=gateway,
                chunks=chunks,
                identity_prefix=f"scoped-readonly-canary:{batch_id}:{document_id}:{metadata_input_hash(chunks)}",
                heartbeat=lambda: None,
            )
            if role == "controlled":
                starts = {
                    target.chunk_id: fact.value
                    for _hash, result in results
                    for target in result.chunks
                    for fact in target.fields
                    if fact.field == "validity_start_date"
                }
                for target_id, expected in control["expected_start_dates"].items():
                    assert starts.get(target_id) == expected, (
                        target_id,
                        expected,
                        starts.get(target_id),
                    )
            local_facts = sum(
                isinstance(fact, VerifiedStructuralFact)
                for _hash, result in results
                for target in result.chunks
                for fact in target.fields
            )
            report.append(
                {
                    "role": role,
                    "targets": len(chunks),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "requests": len({h for h, _r in results}),
                    "locally_verified_facts": local_facts,
                    "validated_facts": sum(
                        len(t.fields) for _h, r in results for t in r.chunks
                    ),
                    "source_input_sha256": metadata_input_hash(chunks),
                    "expanded_targets": sum(len(r.chunks) for _h, r in results),
                }
            )
            assert report[-1]["expanded_targets"] == len(chunks)
            print("CANARY_DOCUMENT", json.dumps(report[-1]), flush=True)
    print(
        "CANARY_DONE",
        json.dumps(
            {
                "documents": report,
                "gateway_submissions": len(submissions),
                "metadata_applied": False,
                "legal_approval": False,
                "index_or_embedding_requested": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
