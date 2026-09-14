"""Persistence and scope checks for amendment source evidence."""

import datetime
import hashlib
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, AmendmentSourcePackage, RegulatorySourceAsset
from onyx.regulatory.amendments.annexes.models import AcquisitionResult
from onyx.regulatory.amendments.annexes.source_limits import (
    MAX_SOURCE_PREPARATION_SECONDS,
    SOURCE_PACKAGE_LEASE_MARGIN_SECONDS,
)


def create_source_package(
    db_session: Session,
    *,
    document_set_id: int,
    environment: str,
    idempotency_key: str,
    request_hash: str,
    input_spec: dict[str, Any],
    created_by: UUID | None,
    input_file_id: str | None = None,
) -> tuple[AmendmentSourcePackage, bool]:
    package_id = uuid4()
    created_id = db_session.scalar(
        insert(AmendmentSourcePackage)
        .values(
            id=package_id,
            document_set_id=document_set_id,
            environment=environment,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            input_spec=input_spec,
            input_file_id=input_file_id,
            created_by=created_by,
        )
        .on_conflict_do_nothing(constraint="uq_amendment_source_request")
        .returning(AmendmentSourcePackage.id)
    )
    package = db_session.scalar(
        select(AmendmentSourcePackage).where(
            AmendmentSourcePackage.document_set_id == document_set_id,
            AmendmentSourcePackage.environment == environment,
            AmendmentSourcePackage.idempotency_key == idempotency_key,
        )
    )
    if package is None:
        raise RuntimeError("Source package insert did not resolve")
    if package.request_hash != request_hash:
        raise ValueError("The idempotency key belongs to a different source request")
    return package, created_id is not None


def get_source_package(
    db_session: Session, *, package_id: UUID, document_set_id: int, environment: str
) -> AmendmentSourcePackage | None:
    return db_session.scalar(
        select(AmendmentSourcePackage).where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.document_set_id == document_set_id,
            AmendmentSourcePackage.environment == environment,
        )
    )


def require_ready_source_package(
    db_session: Session, *, package_id: UUID, document_set_id: int, environment: str
) -> AmendmentSourcePackage:
    package = get_source_package(
        db_session,
        package_id=package_id,
        document_set_id=document_set_id,
        environment=environment,
    )
    if package is None:
        raise ValueError(
            "Source package does not belong to this document set and environment"
        )
    if (
        package.status != "ready"
        or package.issues
        or package.asset_count == 0
        or not package.manifest_file_id
        or not package.manifest_sha256
    ):
        raise ValueError(
            "Only a complete, verified ready source package can be analyzed"
        )
    return package


def list_source_assets(
    db_session: Session, package_id: UUID
) -> list[RegulatorySourceAsset]:
    return list(
        db_session.scalars(
            select(RegulatorySourceAsset)
            .where(RegulatorySourceAsset.package_id == package_id)
            .order_by(RegulatorySourceAsset.created_at, RegulatorySourceAsset.id)
        )
    )


def claim_source_package(
    db_session: Session, *, package_id: UUID, environment: str
) -> tuple[AmendmentSourcePackage, UUID] | None:
    token = uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)
    claimed = db_session.scalar(
        update(AmendmentSourcePackage)
        .where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.environment == environment,
            AmendmentSourcePackage.status == "processing",
            or_(
                AmendmentSourcePackage.lease_expires_at.is_(None),
                AmendmentSourcePackage.lease_expires_at < now,
            ),
        )
        .values(
            lease_token=token, lease_expires_at=now + datetime.timedelta(minutes=10)
        )
        .returning(AmendmentSourcePackage)
    )
    db_session.commit()
    return (claimed, token) if claimed is not None else None


def extend_source_package_lease(
    db_session: Session,
    *,
    package_id: UUID,
    environment: str,
    lease_token: UUID,
    lease_seconds: int,
) -> bool:
    if (
        not 0
        < lease_seconds
        <= (MAX_SOURCE_PREPARATION_SECONDS + SOURCE_PACKAGE_LEASE_MARGIN_SECONDS)
    ):
        raise ValueError("Invalid source preparation lease duration")
    now = datetime.datetime.now(datetime.timezone.utc)
    extended = db_session.scalar(
        update(AmendmentSourcePackage)
        .where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.environment == environment,
            AmendmentSourcePackage.status == "processing",
            AmendmentSourcePackage.lease_token == lease_token,
            AmendmentSourcePackage.lease_expires_at > now,
        )
        .values(lease_expires_at=now + datetime.timedelta(seconds=lease_seconds))
        .returning(AmendmentSourcePackage.id)
    )
    db_session.commit()
    return extended is not None


def retry_source_package(
    db_session: Session, *, package_id: UUID, document_set_id: int, environment: str
) -> AmendmentSourcePackage:
    package = db_session.scalar(
        select(AmendmentSourcePackage)
        .where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.document_set_id == document_set_id,
            AmendmentSourcePackage.environment == environment,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if package is None or package.status == "ready":
        raise ValueError("Source package is missing or already complete")
    now = datetime.datetime.now(datetime.timezone.utc)
    if package.lease_expires_at and package.lease_expires_at > now:
        raise ValueError("Source package acquisition is still running")
    package.status = "processing"
    package.lease_token = None
    package.lease_expires_at = None
    db_session.commit()
    return package


def finish_source_package(
    db_session: Session,
    *,
    package_id: UUID,
    environment: str,
    lease_token: UUID,
    result: AcquisitionResult,
    assets: list[RegulatorySourceAsset],
    manifest_file_id: str,
    manifest_sha256: str,
) -> bool:
    package = db_session.scalar(
        select(AmendmentSourcePackage)
        .where(
            AmendmentSourcePackage.id == package_id,
            AmendmentSourcePackage.environment == environment,
            AmendmentSourcePackage.lease_token == lease_token,
            AmendmentSourcePackage.status == "processing",
        )
        .with_for_update()
    )
    if package is None:
        return False
    hashes = {asset.sha256 for asset in result.assets}
    if result.status == "ready" and (
        result.issues
        or not hashes
        or any(
            link.parent_asset_hash not in hashes or link.target_asset_hash not in hashes
            for link in result.links
        )
    ):
        raise ValueError("Incomplete evidence cannot be marked ready")
    if any(
        asset.package_id != package_id or asset.sha256 not in hashes for asset in assets
    ):
        raise ValueError("Source asset scope mismatch")
    existing = {asset.sha256 for asset in list_source_assets(db_session, package_id)}
    for asset in assets:
        if asset.sha256 not in existing:
            db_session.add(asset)
            existing.add(asset.sha256)
    if hashes != existing:
        raise ValueError("Source manifest does not cover all stored assets")
    package.status = result.status
    package.asset_count = len(result.assets)
    package.total_bytes = sum(len(asset.content) for asset in result.assets)
    package.issues = [issue.model_dump(mode="json") for issue in result.issues]
    package.manifest_file_id = manifest_file_id
    package.manifest_sha256 = manifest_sha256
    package.lease_expires_at = None
    db_session.commit()
    return True


def mark_source_package_failed(
    db_session: Session,
    *,
    package_id: UUID,
    environment: str,
    lease_token: UUID | None = None,
    failure: BaseException | None = None,
) -> None:
    statement = update(AmendmentSourcePackage).where(
        AmendmentSourcePackage.id == package_id,
        AmendmentSourcePackage.environment == environment,
        AmendmentSourcePackage.status == "processing",
    )
    statement = (
        statement.where(AmendmentSourcePackage.lease_token == lease_token)
        if lease_token
        else statement.where(AmendmentSourcePackage.lease_token.is_(None))
    )
    if isinstance(failure, TimeoutError):
        code, retryable = "source_preparation_timeout", True
    elif (
        isinstance(failure, ValueError)
        and str(failure) == "image_source_requires_new_preparation"
    ):
        code, retryable = "image_source_requires_new_preparation", False
    else:
        code, retryable = "acquisition_failed", True
    issue: dict[str, Any] = {"code": code, "retryable": retryable}
    if failure is not None:
        from onyx.regulatory.failure_details import safe_failure_detail

        issue["failure_detail"] = safe_failure_detail("source_package", failure)
    db_session.execute(
        statement.values(
            status="failed",
            issues=[issue],
            lease_expires_at=None,
        )
    )
    db_session.commit()


def attach_source_package_to_batch(
    db_session: Session, *, batch: AmendmentBatch, package_id: UUID, environment: str
) -> None:
    require_ready_source_package(
        db_session,
        package_id=package_id,
        document_set_id=batch.document_set_id,
        environment=environment,
    )
    batch.source_package_id = package_id
    batch.source_text_sha256 = hashlib.sha256(batch.raw_text.encode()).hexdigest()
    db_session.flush()


def get_source_asset(
    db_session: Session, *, package_id: UUID, asset_id: UUID
) -> RegulatorySourceAsset | None:
    return db_session.scalar(
        select(RegulatorySourceAsset).where(
            RegulatorySourceAsset.id == asset_id,
            RegulatorySourceAsset.package_id == package_id,
        )
    )
