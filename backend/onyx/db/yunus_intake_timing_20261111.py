"""One explicit unpublished DEV replay, delivered through the normal source queue."""

import json
from uuid import UUID

from sqlalchemy import select, text

from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
    enqueue_intake_batch,
)
from onyx.db.document_intake_reprepare import reprepare_snapshot, request_reprepare
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReprepare,
    RegulatorySourceAsset,
)
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


def main() -> None:
    SqlEngine.init_engine(pool_size=2, max_overflow=1)
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    identifier = UUID("2f5ae4fb-6756-47c2-8776-d624d134ff8e")
    key = UUID("6fe095c4-54a9-46a3-89f7-805ffb665094")
    with get_session_with_current_tenant() as session:
        assert (
            session.scalar(text("SELECT current_database()"))
            == "customs-regulations-dev"
        )
        batch = session.get_one(
            RegulatoryDocumentIntake, identifier, with_for_update=True
        )
        assert str(batch.created_by) == "9ea33f67-5578-4dc4-bbea-e32b16f3f1dc"
        assert (
            session.get_one(DocumentSet, batch.document_set_id).name == "PC Külliyatı"
        )
        package = session.get_one(AmendmentSourcePackage, batch.source_package_id)
        assert package.environment == "dev"
        members = list(
            session.scalars(
                select(RegulatoryDocumentIntakeFile).where(
                    RegulatoryDocumentIntakeFile.intake_id == identifier
                )
            )
        )
        assert len(members) == 2
        root = next(member for member in members if member.role == "root")
        annex = next(member for member in members if member.role == "annex")
        assert annex.parent_document_id == root.id
        assert {
            session.get_one(RegulatorySourceAsset, member.source_asset_id).sha256
            for member in members
        } == {
            "eeb7becbde4ceb7b06c07d1a1f30683bf2c3287ef228563ff7e12f4f2c2df467",
            "e31fb245f59a57c9e227dbca322c549b26e5f7d61c59e6bb059ab7e33adbc8fa",
        }
        prior = session.scalar(
            select(RegulatoryDocumentIntakeReprepare).where(
                RegulatoryDocumentIntakeReprepare.idempotency_key == key
            )
        )
        if prior is not None:
            assert prior.intake_id == identifier
            if prior.status != "queued":
                print(
                    "TIMING_ALREADY_DELIVERED",
                    json.dumps({"request_id": str(prior.id), "status": prior.status}),
                    flush=True,
                )
                return
            request = prior
        else:
            assert batch.status == "ready" and batch.stage == "review"
            assert (
                batch.background_labeling_enabled
                and batch.background_label_status == "completed"
            )
            view = reprepare_snapshot(session, batch)
            request = request_reprepare(
                session,
                batch,
                expected_sha256=view["expected_sha256"],
                idempotency_key=key,
            )
        request_id = request.id
        created = request.created_at
        session.commit()
    print(
        "TIMING_REQUESTED",
        json.dumps(
            {
                "intake_id": str(identifier),
                "request_id": str(request_id),
                "created_at": str(created),
                "indexing_requested": False,
                "approval_requested": False,
            }
        ),
        flush=True,
    )
    enqueue_intake_batch(
        batch_id=identifier, tenant_id="public", reprepare_request_id=request_id
    )
    print("TIMING_DELIVERED_NORMAL_SOURCE_QUEUE", flush=True)


if __name__ == "__main__":
    main()
