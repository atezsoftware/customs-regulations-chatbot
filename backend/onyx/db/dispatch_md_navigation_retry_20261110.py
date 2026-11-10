"""Retry one failed DEV metadata run, preserving originals, chunks and ownership."""

import hashlib
import json
from uuid import UUID

from sqlalchemy import select, text

from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
    enqueue_intake_batch,
)
from onyx.db import document_intake as intake
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatorySourceAsset,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

assert config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
assert (
    metadata_recipe_sha256()
    == "56c3504da668fd7299ffd8bd3af7bf438b3224bbf98fc590e3b94e5f0cac45f2"
)
batch_id = UUID("70a4b080-00eb-4399-bc89-88024477a04c")
package_id = UUID("28c45534-021a-4b20-8ca7-71848d780b54")
SqlEngine.init_engine(pool_size=2, max_overflow=0)
with get_session_with_current_tenant() as session:
    session.execute(text("SET LOCAL lock_timeout='3s'"))
    session.execute(text("SET LOCAL statement_timeout='30s'"))
    assert (
        session.scalar(text("SELECT current_database()")) == "customs-regulations-dev"
    )
    batch = session.get_one(RegulatoryDocumentIntake, batch_id, with_for_update=True)
    assert batch.source_package_id == package_id
    assert str(batch.created_by) == "9ea33f67-5578-4dc4-bbea-e32b16f3f1dc"
    package = session.get_one(AmendmentSourcePackage, package_id)
    assert package.environment == "dev" and package.status == "ready"
    assert set(
        session.scalars(
            select(RegulatorySourceAsset.sha256).where(
                RegulatorySourceAsset.package_id == package_id
            )
        )
    ) == {
        "eeb7becbde4ceb7b06c07d1a1f30683bf2c3287ef228563ff7e12f4f2c2df467",
        "e31fb245f59a57c9e227dbca322c549b26e5f7d61c59e6bb059ab7e33adbc8fa",
    }
    members = intake.intake_files(session, batch)
    assert len(members) == 2
    parent = next(member for member in members if member.role == "root")
    annex = next(member for member in members if member.role == "annex")
    assert annex.parent_document_id == parent.id
    ids = [member.user_file_id for member in members]
    for query in (
        "SELECT count(*) FROM user_file_projection_repair WHERE user_file_id=ANY(:ids)",
        "SELECT count(*) FROM regulatory_indexing_job WHERE user_file_id=ANY(:ids)",
        "SELECT count(*) FROM regulatory_temporal_projection WHERE user_file_id=ANY(:ids)",
    ):
        assert (
            session.scalar(
                text(query),
                {"ids": ids},
            )
            == 0
        )
    assert (
        session.scalar(
            text(
                "SELECT count(*) FROM regulatory_file_publication WHERE user_file_id=ANY(:ids) AND (epoch<>0 OR writer_manifest IS NOT NULL)"
            ),
            {"ids": ids},
        )
        == 0
    )
    rows = list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id.in_(ids))
            .order_by(RegulatoryChunk.id)
        )
    )
    assert len(rows) == 370
    assert sum(row.user_file_id == parent.user_file_id for row in rows) == 350
    source_digest = hashlib.sha256(
        json.dumps([(row.id, row.text) for row in rows]).encode()
    ).hexdigest()
    if batch.status == "failed":
        assert batch.stage == "metadata" and batch.labeling_run_id is None
        assert (
            batch.error
            == "New-document metadata failed strict validation after bounded repair attempts"
        )
        assert all(member.status == "chunked" for member in members)
        assert batch.lease_token is None and batch.lease_expires_at is None
        assert intake.retry_intake(session, batch) == []
        assert batch.status == "preparing" and batch.stage == "chunking"
        assert batch.metadata_checkpoint is not None
        assert batch.metadata_checkpoint["recipe_sha256"] == metadata_recipe_sha256()
        assert all(member.status == "chunked" for member in members)
        session.commit()
        print("RETRY_COMMITTED", str(batch_id), source_digest, "index_requests=0")
    should_dispatch = (
        batch.status == "preparing"
        and batch.stage == "chunking"
        and batch.lease_token is None
        and batch.lease_expires_at is None
    )
if should_dispatch:
    enqueue_intake_batch(batch_id=batch_id, tenant_id="public")
    print(
        "DISPATCHED metadata retry; existing chunked members are reused", str(batch_id)
    )
else:
    print("No delivery needed; intake already claimed or advanced", str(batch_id))
