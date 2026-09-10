"""Short index lifecycle transitions share the permanent publication clock."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryFilePublication
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import PublicationScope
from onyx.regulatory.amendments.annexes import config


def port_user_file_ids(document_ids: list[str], tenant_id: str) -> set[str]:
    """Retain classification after physical deletion so old port work cannot revive it."""
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import UserFile

    candidates: dict[str, UUID] = {}
    for identifier in document_ids:
        try:
            candidates[identifier] = UUID(identifier)
        except ValueError:
            continue
    if not candidates:
        return set()
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        known = set(
            session.scalars(
                select(UserFile.id).where(UserFile.id.in_(candidates.values()))
            )
        )
        known.update(
            session.scalars(
                select(RegulatoryFilePublication.user_file_id).where(
                    RegulatoryFilePublication.user_file_id.in_(candidates.values())
                )
            )
        )
    return {identifier for identifier, value in candidates.items() if value in known}


def active_port_setting_id(index_name: str, tenant_id: str) -> int:
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.enums import IndexModelStatus
    from onyx.db.models import SearchSettings

    with get_session_with_tenant(tenant_id=tenant_id) as session:
        identifier = session.scalar(
            select(SearchSettings.id).where(
                SearchSettings.index_name == index_name,
                SearchSettings.status == IndexModelStatus.FUTURE,
            )
        )
        if identifier is None:
            raise ValueError("user-file port target is no longer FUTURE")
        return identifier


def lock_index_publication_clock(session: Session) -> None:
    """Serialize only short publication/index state transitions."""
    translations = (
        session.connection().get_execution_options().get("schema_translate_map") or {}
    )
    authority = PublicationStore(
        PublicationScope(
            tenant_id=translations.get(None, "public"),
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    authority.lock_clock(session)


def lock_index_publication_barrier(session: Session) -> bool:
    """Staged writers must finish before a status transition."""
    lock_index_publication_clock(session)
    return (
        session.scalar(
            select(RegulatoryFilePublication.user_file_id)
            .where(RegulatoryFilePublication.gate_closed.is_(True))
            .limit(1)
        )
        is None
    )


def require_index_publication_barrier(session: Session) -> None:
    from onyx.db.regulatory_physical_indexes import require_physical_index_available

    with session.no_autoflush:
        if not lock_index_publication_barrier(session):
            raise ValueError("index transition waits for pending file publication")
        require_physical_index_available(session)


def unreconciled_user_files(
    session: Session, present_name: str, future_name: str
) -> list["UUID"]:
    """Compare dated canonical authority, independently of the target encoder space."""
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import RegulatoryTemporalProjection, UserFile
    from onyx.document_index.publication_models import publication_digest

    eligible: set[UUID] = {
        UUID(str(value))
        for value in session.scalars(
            select(UserFile.id).where(UserFile.status == UserFileStatus.COMPLETED)
        )
    }
    pending_metadata = set(
        session.scalars(
            select(UserFile.id).where(
                UserFile.status.in_([UserFileStatus.COMPLETED, UserFileStatus.FAILED]),
                (
                    UserFile.needs_project_sync
                    | UserFile.needs_persona_sync
                    | UserFile.needs_document_set_sync
                    | UserFile.secondary_reconcile_pending
                ),
            )
        )
    )
    histories: dict[UUID, dict[str, set[str]]] = {}
    rows = session.scalars(
        select(RegulatoryTemporalProjection)
        .join(UserFile, UserFile.id == RegulatoryTemporalProjection.user_file_id)
        .where(
            RegulatoryTemporalProjection.retired_at.is_(None),
            UserFile.status != UserFileStatus.DELETING,
            RegulatoryTemporalProjection.payload["index"]["index_name"].astext.in_(
                [present_name, future_name]
            ),
        )
    )
    for row in rows:
        name = row.payload["index"]["index_name"]
        if name == present_name:
            eligible.add(UUID(str(row.user_file_id)))
        payload = row.payload
        signature = publication_digest(
            {
                "canonical_id": row.canonical_chunk_id,
                "revision_id": str(row.canonical_revision_id),
                "start": row.effective_start.isoformat()
                if row.effective_start
                else None,
                "end": row.effective_end.isoformat() if row.effective_end else None,
                **{
                    key: payload[key]
                    for key in (
                        "canonical_base_sha256",
                        "representation_text",
                        "representation_metadata",
                        "semantic_position",
                    )
                },
            }
        )
        histories.setdefault(row.user_file_id, {}).setdefault(name, set()).add(
            signature
        )
    return sorted(
        pending_metadata
        | {
            identifier
            for identifier in eligible
            if (
                not histories.get(identifier, {}).get(present_name)
                or histories[identifier].get(present_name)
                != histories[identifier].get(future_name)
            )
        },
        key=lambda identifier: str(identifier),
    )


def schedule_user_file_index_reconciliation(
    session: Session, identifiers: list["UUID"]
) -> None:
    """Called after releasing the clock; the existing sync worker drains these flags."""
    from sqlalchemy import update

    from onyx.db.models import UserFile

    session.execute(
        update(UserFile)
        .where(UserFile.id.in_(identifiers))
        .values(secondary_reconcile_pending=True)
    )
    session.commit()
