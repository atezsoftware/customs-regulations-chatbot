"""Retained single-tenant physical operations share the publication clock."""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import (
    IndexAttempt,
    RegulatoryPhysicalIndexOperation,
    RegulatoryTemporalProjection,
    SearchSettings,
)
from onyx.db.regulatory_index_lifecycle import lock_index_publication_barrier
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    PublicationModel,
    PublicationScope,
    publication_digest,
)


class PhysicalIndexOperation(PublicationModel):
    id: UUID
    scope: PublicationScope
    index_name: str
    index_uuid: str
    operation: Literal["delete", "recreate"]
    request: dict[str, JsonValue]
    request_sha256: str
    phase: Literal["prepared", "deleting", "deleted", "creating", "complete"]
    owner_id: UUID
    lease_expires_at: datetime
    result_index_uuid: str | None


def _snapshot(
    row: RegulatoryPhysicalIndexOperation, scope: PublicationScope
) -> PhysicalIndexOperation:
    if row.scope_key != publication_digest(scope.model_dump(mode="json")):
        raise ValueError("physical index operation scope differs")
    return PhysicalIndexOperation.model_validate(
        {
            "scope": scope,
            **{
                name: getattr(row, name)
                for name in PhysicalIndexOperation.model_fields
                if name != "scope"
            },
        }
    )


def pending_physical_operation(
    scope: PublicationScope, index_name: str
) -> PhysicalIndexOperation | None:
    with get_session_with_tenant(tenant_id=scope.tenant_id) as session:
        row = session.scalar(
            select(RegulatoryPhysicalIndexOperation).where(
                RegulatoryPhysicalIndexOperation.index_name == index_name,
                RegulatoryPhysicalIndexOperation.completed_at.is_(None),
            )
        )
        return _snapshot(row, scope) if row else None


def require_physical_index_available(
    session: Session, index_name: str | None = None, *, operation_id: UUID | None = None
) -> None:
    query = select(RegulatoryPhysicalIndexOperation.id).where(
        RegulatoryPhysicalIndexOperation.completed_at.is_(None)
    )
    if index_name is not None:
        query = query.where(RegulatoryPhysicalIndexOperation.index_name == index_name)
    if operation_id is not None:
        query = query.where(RegulatoryPhysicalIndexOperation.id != operation_id)
    if session.scalar(query.limit(1)) is not None:
        raise ValueError(
            "physical index operation must finish before publication or name reuse"
        )


def validate_physical_index_snapshots(
    session: Session, indexes: list[PublicationIndexSnapshot]
) -> None:
    for index in indexes:
        require_physical_index_available(session, index.index_name)
        latest = session.scalar(
            select(RegulatoryPhysicalIndexOperation)
            .where(
                RegulatoryPhysicalIndexOperation.index_name == index.index_name,
                RegulatoryPhysicalIndexOperation.completed_at.is_not(None),
            )
            .order_by(RegulatoryPhysicalIndexOperation.created_at.desc())
            .limit(1)
        )
        if latest is not None and (
            (latest.operation == "delete" and latest.index_uuid == index.index_uuid)
            or (
                latest.operation == "recreate"
                and latest.result_index_uuid != index.index_uuid
            )
        ):
            raise ValueError(
                "physical index UUID changed after publication preparation"
            )


def claim_physical_operation(
    scope: PublicationScope,
    *,
    index_name: str,
    index_uuid: str,
    operation: Literal["delete", "recreate"],
    request: dict[str, JsonValue],
    discard_search_settings_id: int | None = None,
) -> PhysicalIndexOperation:
    now = datetime.now(timezone.utc)
    with get_session_with_tenant(tenant_id=scope.tenant_id) as session:
        if not lock_index_publication_barrier(session):
            raise ValueError("physical operation waits for pending publication")
        row = session.scalar(
            select(RegulatoryPhysicalIndexOperation)
            .where(
                RegulatoryPhysicalIndexOperation.index_name == index_name,
                RegulatoryPhysicalIndexOperation.completed_at.is_(None),
            )
            .with_for_update()
        )
        digest = publication_digest(request)
        if row is not None:
            existing = _snapshot(row, scope)
            if (existing.index_uuid, existing.operation, existing.request_sha256) != (
                index_uuid,
                operation,
                digest,
            ):
                raise ValueError(
                    "physical operation request differs from retained authority"
                )
            if existing.phase == "deleting":
                raise ValueError(
                    "physical deletion submission is indeterminate; reconcile the original request before retry"
                )
            if existing.lease_expires_at > now:
                raise ValueError("physical operation is already owned")
            row.owner_id = uuid4()
            row.lease_expires_at = now + timedelta(minutes=2)
        else:
            retained = session.scalar(
                select(RegulatoryTemporalProjection.id)
                .where(
                    RegulatoryTemporalProjection.payload["index"]["index_name"].astext
                    == index_name,
                )
                .limit(1)
            )
            if retained is not None:
                raise ValueError(
                    "retained publication authority prevents physical replacement"
                )
            row = RegulatoryPhysicalIndexOperation(
                id=uuid4(),
                scope_key=publication_digest(scope.model_dump(mode="json")),
                index_name=index_name,
                index_uuid=index_uuid,
                operation=operation,
                request=request,
                request_sha256=digest,
                phase="prepared",
                owner_id=uuid4(),
                lease_expires_at=now + timedelta(minutes=2),
            )
            session.add(row)
            if discard_search_settings_id is not None:
                setting = session.get(
                    SearchSettings, discard_search_settings_id, with_for_update=True
                )
                if setting is None:
                    raise ValueError(
                        "physical cleanup has no removable setting or retained operation"
                    )
                if setting is not None:
                    if setting.status.is_current() or setting.index_name != index_name:
                        raise ValueError(
                            "physical cleanup cannot discard the active or a different index"
                        )
                    session.execute(
                        delete(IndexAttempt).where(
                            IndexAttempt.search_settings_id == setting.id
                        )
                    )
                    session.delete(setting)
        session.commit()
        return _snapshot(row, scope)


def advance_physical_operation(
    operation: PhysicalIndexOperation,
    *,
    expected_phase: str,
    phase: str,
    result_index_uuid: str | None = None,
    terminal_evidence: dict[str, JsonValue] | None = None,
) -> PhysicalIndexOperation:
    with get_session_with_tenant(tenant_id=operation.scope.tenant_id) as session:
        PublicationStore(operation.scope).lock_clock(session)
        row = session.get(
            RegulatoryPhysicalIndexOperation, operation.id, with_for_update=True
        )
        if (
            row is None
            or row.owner_id != operation.owner_id
            or row.phase != expected_phase
        ):
            raise ValueError("physical operation ownership or phase changed")
        if phase == "deleting" and row.lease_expires_at <= datetime.now(timezone.utc):
            raise ValueError("physical operation lease expired before deletion")
        row.phase = phase
        row.result_index_uuid = result_index_uuid
        if terminal_evidence is not None:
            row.terminal_evidence = terminal_evidence
        if phase == "complete":
            row.completed_at = datetime.now(timezone.utc)
        session.commit()
        return _snapshot(row, operation.scope)


def release_physical_operation(operation: PhysicalIndexOperation) -> None:
    with get_session_with_tenant(tenant_id=operation.scope.tenant_id) as session:
        row = session.get(
            RegulatoryPhysicalIndexOperation, operation.id, with_for_update=True
        )
        if (
            row is not None
            and row.owner_id == operation.owner_id
            and row.phase != "deleting"
        ):
            row.lease_expires_at = datetime.now(timezone.utc)
            session.commit()
