"""Read-only corpus inventory for publication baseline preparation."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import exists, select, text

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryFilePublication,
    SearchSettings,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.db.search_settings import get_active_search_settings_list
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest


@dataclass(frozen=True)
class BaselineAuditInputs:
    canonical: list[AnnexCanonicalSnapshot]
    bindings: list[AnnexTemporalProjection]
    status: str
    gate_closed: bool
    pending_manifest: bool
    manifest: WriterPublicationManifest | None


def baseline_file_ids(
    tenant: str, *, after: UUID | None = None, limit: int | None = None
) -> list[UUID]:
    with get_session_with_tenant(tenant_id=tenant) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        query = (
            select(UserFile.id)
            .where(
                UserFile.status == UserFileStatus.COMPLETED,
                exists(
                    select(RegulatoryChunk.id).where(
                        RegulatoryChunk.user_file_id == UserFile.id
                    )
                ),
            )
            .order_by(UserFile.id)
        )
        if after is not None:
            query = query.where(UserFile.id > after)
        if limit is not None:
            query = query.limit(limit)
        return list(session.scalars(query))


def baseline_audit_settings(
    tenant: str, expected_database: str
) -> list[SearchSettings]:
    with get_session_with_tenant(tenant_id=tenant) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        if session.scalar(text("SELECT current_database()")) != expected_database:
            raise ValueError("baseline database identity mismatch")
        settings = get_active_search_settings_list(session)
        for setting in settings:
            _ = setting.cloud_provider
        session.expunge_all()
        return settings


def baseline_audit_inputs(tenant: str, file_id: UUID) -> BaselineAuditInputs:
    with get_session_with_tenant(tenant_id=tenant) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        file = session.get(UserFile, file_id)
        if file is None:
            raise ValueError("baseline file unavailable")
        publication = session.get(RegulatoryFilePublication, file_id)
        manifest = None
        if publication is not None and publication.writer_manifest is not None:
            if (
                publication_digest(publication.writer_manifest)
                != publication.writer_manifest_sha256
            ):
                raise ValueError("baseline recovery manifest changed")
            manifest = WriterPublicationManifest.model_validate(
                publication.writer_manifest
            )
        return BaselineAuditInputs(
            canonical=capture_canonical_scope(session, file_id),
            bindings=load_file_temporal_bindings(session, file_id),
            status=file.status.value,
            gate_closed=bool(publication and publication.gate_closed),
            pending_manifest=bool(publication and publication.writer_manifest),
            manifest=manifest,
        )


def baseline_audit_inputs_batch(
    tenant: str, file_ids: list[UUID]
) -> dict[UUID, BaselineAuditInputs]:
    """Batch scalar/canonical reads; qualified payloads retain their normal verifier."""
    from collections import defaultdict

    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.regulatory.amendments.annexes.publication_representations import _snapshot

    with get_session_with_tenant(tenant_id=tenant) as session:
        session.execute(
            text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        )
        files = {
            row.id: row
            for row in session.scalars(
                select(UserFile).where(UserFile.id.in_(file_ids))
            )
        }
        publications = {
            row.user_file_id: row
            for row in session.scalars(
                select(RegulatoryFilePublication).where(
                    RegulatoryFilePublication.user_file_id.in_(file_ids)
                )
            )
        }
        canonical: dict[UUID, list[AnnexCanonicalSnapshot]] = defaultdict(list)
        for row in session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id.in_(file_ids))
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        ):
            canonical[row.user_file_id].append(_snapshot(row))
        qualified = set(
            session.scalars(
                select(RegulatoryTemporalProjection.user_file_id)
                .where(
                    RegulatoryTemporalProjection.user_file_id.in_(file_ids),
                    RegulatoryTemporalProjection.retired_at.is_(None),
                )
                .distinct()
            )
        )
        result = {}
        for file_id in file_ids:
            publication = publications.get(file_id)
            manifest = None
            if publication is not None and publication.writer_manifest is not None:
                if (
                    publication_digest(publication.writer_manifest)
                    != publication.writer_manifest_sha256
                ):
                    raise ValueError("baseline recovery manifest changed")
                manifest = WriterPublicationManifest.model_validate(
                    publication.writer_manifest
                )
            result[file_id] = BaselineAuditInputs(
                canonical=canonical[file_id],
                bindings=load_file_temporal_bindings(session, file_id)
                if file_id in qualified
                else [],
                status=files[file_id].status.value,
                gate_closed=bool(publication and publication.gate_closed),
                pending_manifest=manifest is not None,
                manifest=manifest,
            )
        return result
