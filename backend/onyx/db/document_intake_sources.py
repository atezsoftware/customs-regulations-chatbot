"""Create immutable, user-scoped Markdown source requests for new intake only."""

import hashlib
import json
from io import BytesIO
from uuid import UUID, uuid5

from sqlalchemy import update
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.amendment_sources import create_source_package, list_source_assets
from onyx.db.document_intake import resolve_intake_target
from onyx.db.models import AmendmentSourcePackage, User
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.server.features.regulatory.models import (
    AmendmentSourceAssetSnapshot,
    AmendmentSourcePackageSnapshot,
)

MARKDOWN_INPUT_KIND = "legislation_markdown_v1"


def create_markdown_source_package(
    session: Session,
    *,
    user: User,
    environment: str,
    idempotency_key: UUID,
    display_name: str,
    content: bytes,
    store: FileStore | None = None,
) -> tuple[AmendmentSourcePackage, bool]:
    target = resolve_intake_target(session, user)
    mime = (
        "application/zip" if display_name.lower().endswith(".zip") else "text/markdown"
    )
    spec = {
        "kind": MARKDOWN_INPUT_KIND,
        "preparation_method": "supplied_markdown",
        "display_name": display_name,
        "mime_type": mime,
        "input_sha256": hashlib.sha256(content).hexdigest(),
    }
    request_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode() + b"\0" + content
    ).hexdigest()
    package, created = create_source_package(
        session,
        document_set_id=target.id,
        environment=environment,
        idempotency_key=f"intake-markdown:{user.id}:{idempotency_key}",
        request_hash=request_hash,
        input_spec=spec,
        created_by=user.id,
    )
    if package.created_by != user.id:
        raise ValueError("Markdown source package belongs to another user")
    if created:
        file_store = store or get_default_file_store()
        identifier = str(uuid5(package.id, "markdown-input"))
        metadata = {
            "source_package_id": str(package.id),
            "sha256": hashlib.sha256(content).hexdigest(),
            "preparation_method": "supplied_markdown",
        }
        if isinstance(file_store, PostgresBackedFileStore):
            package.input_file_id = file_store.save_file(
                BytesIO(content),
                display_name,
                FileOrigin.OTHER,
                mime,
                file_id=identifier,
                file_metadata=metadata,
                db_session=session,
            )
        else:
            package.input_file_id = file_store.save_file(
                BytesIO(content),
                display_name,
                FileOrigin.OTHER,
                mime,
                file_id=identifier,
                file_metadata=metadata,
            )
        session.flush()
    return package, created


def markdown_source_snapshot(
    session: Session, package: AmendmentSourcePackage
) -> AmendmentSourcePackageSnapshot:
    snapshot = AmendmentSourcePackageSnapshot.model_validate(package)
    snapshot.assets = [
        AmendmentSourceAssetSnapshot.model_validate(asset)
        for asset in list_source_assets(session, package.id)
    ]
    return snapshot


def fail_markdown_source_package(
    session: Session,
    *,
    package_id: UUID,
    environment: str,
    lease_token: UUID,
    detail: str,
) -> None:
    session.execute(
        update(AmendmentSourcePackage)
        .where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.environment == environment,
            AmendmentSourcePackage.input_spec["kind"].as_string()
            == MARKDOWN_INPUT_KIND,
            AmendmentSourcePackage.status == "processing",
            AmendmentSourcePackage.lease_token == lease_token,
        )
        .values(
            status="failed",
            issues=[
                {
                    "code": "invalid_markdown_package",
                    "retryable": False,
                    "failure_detail": detail[:2000],
                }
            ],
            lease_token=None,
            lease_expires_at=None,
        )
    )
    session.commit()
