"""Send one broker hint for an already committed owner-preserving DEV retry."""

from uuid import UUID

from sqlalchemy import select, text

from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
    enqueue_intake_batch,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatorySourceAsset,
)
from onyx.regulatory.amendments.annexes import config

assert config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
batch_id = UUID("70a4b080-00eb-4399-bc89-88024477a04c")
package_id = UUID("28c45534-021a-4b20-8ca7-71848d780b54")
with get_session_with_current_tenant() as session:
    session.execute(text("SET TRANSACTION READ ONLY"))
    assert (
        session.scalar(text("SELECT current_database()")) == "customs-regulations-dev"
    )
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
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
    should_dispatch = (
        batch.status == "preparing"
        and batch.stage == "chunking"
        and batch.lease_token is None
        and batch.lease_expires_at is None
    )
if should_dispatch:
    enqueue_intake_batch(batch_id=batch_id, tenant_id="public")
    print("DISPATCHED exact pending Markdown retry", batch_id)
else:
    print("No delivery needed; intake already claimed or advanced", batch_id)
