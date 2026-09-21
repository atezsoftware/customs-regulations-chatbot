"""Durable normal writer staging and activation; authority precedes all row locks."""

import json
from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_publication import PublicationStore, archive_canonical_revisions
from onyx.document_index.publication_models import (
    FileOwnership,
    PublicationVerification,
    publication_digest,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest

if TYPE_CHECKING:
    from onyx.connectors.models import Document
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_indexing_jobs import UserFileDeletionCleanupPlan
    from onyx.document_index.interfaces_new import MetadataUpdateRequest
    from onyx.natural_language_processing.utils import BaseTokenizer
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

from dataclasses import dataclass

from onyx.db.models import SearchSettings
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexProjectionAccess,
    AnnexTemporalProjection,
    PreparedContextView,
)


def canonical_scope_digest(session: Session, user_file_id: UUID) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json")
            for row in capture_canonical_scope(session, user_file_id)
        ]
    )


def pending_writer_manifest(owner: FileOwnership) -> WriterPublicationManifest | None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owner.user_file_id)
        assert row is not None
        if row.writer_manifest is None:
            return None
        if publication_digest(row.writer_manifest) != row.writer_manifest_sha256:
            raise ValueError("durable writer manifest changed")
        manifest = WriterPublicationManifest.model_validate(row.writer_manifest)
        if manifest.scope != owner.scope or manifest.user_file_id != owner.user_file_id:
            raise ValueError("durable writer manifest scope mismatch")
        return manifest


def stage_writer_publication(
    owner: FileOwnership,
    manifest: WriterPublicationManifest,
    *,
    durable_generation: int | None = None,
) -> None:
    authority = PublicationStore(owner.scope)
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = authority.lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owner.user_file_id)
        assert row is not None
        payload = manifest.model_dump(mode="json")
        if manifest.kind == "cancellation":
            from onyx.db.regulatory_durable_publication import lock_owned_cancellation

            if manifest.cancellation_job_id is None or durable_generation is None:
                raise ValueError("cancellation publication requires job ownership")
            lock_owned_cancellation(
                session,
                owner,
                job_id=manifest.cancellation_job_id,
                expected_generation=durable_generation,
            )
            if row.writer_manifest is not None and row.writer_manifest != payload:
                previous = WriterPublicationManifest.model_validate(row.writer_manifest)
                if (
                    previous.kind != "durable"
                    or previous.durable_job_id != manifest.cancellation_job_id
                    or publication_digest(row.writer_manifest)
                    != manifest.cancelled_manifest_sha256
                    or row.writer_manifest_sha256 != manifest.cancelled_manifest_sha256
                ):
                    raise ValueError(
                        "cancellation cannot replace another writer's recovery"
                    )
                row.writer_manifest = None
                row.writer_manifest_sha256 = None
                session.flush()
        if row.writer_manifest is not None:
            if (
                row.writer_manifest != payload
                or row.writer_manifest_sha256 != publication_digest(payload)
            ):
                raise ValueError("another durable writer must recover first")
            return
        if manifest.kind not in {"delete", "cancellation"}:
            from onyx.db.enums import UserFileStatus

            file = session.get(UserFile, owner.user_file_id, with_for_update=True)
            if file is None or file.status in {
                UserFileStatus.CANCELED,
                UserFileStatus.DELETING,
            }:
                raise ValueError(
                    "new publication cannot restart a cancelled or deleting file"
                )
        if reservations.gate_closed and not (
            manifest.kind == "cancellation" and manifest.cancelled_manifest_sha256
        ):
            raise ValueError("closed publication requires its existing recovery")
        if manifest.scope != owner.scope or manifest.user_file_id != owner.user_file_id:
            raise ValueError("writer ownership scope mismatch")
        if (
            canonical_scope_digest(session, owner.user_file_id)
            != manifest.canonical_before_sha256
        ):
            raise ValueError("writer canonical baseline changed")
        if any(
            binding.projection.ordinal not in reservations.ordinals
            for binding in manifest.bindings
        ):
            raise ValueError("writer projection was not reserved under ownership")
        if manifest.kind == "durable":
            _validate_durable_writer(
                session, owner, manifest, durable_generation, "INDEX_WRITE"
            )
        archive_canonical_revisions(session, owner)
        row.writer_manifest = payload
        row.writer_manifest_sha256 = publication_digest(payload)
        authority.record_event(session, owner)
        from onyx.db.regulatory_physical_indexes import (
            validate_physical_index_snapshots,
        )

        validate_physical_index_snapshots(session, manifest.indexes)
        if (
            manifest.index_state_sha256 is not None
            and manifest.index_state_sha256
            != writer_index_state_digest(
                session, [index.search_settings_id for index in manifest.indexes]
            )
        ):
            raise ValueError("writer index settings changed before staging")
        session.commit()


def _apply_canonical(session: Session, manifest: WriterPublicationManifest) -> None:
    if manifest.canonical_after is not None:
        _apply_canonical_rows(session, manifest.user_file_id, manifest.canonical_after)


def _apply_canonical_rows(
    session: Session, user_file_id: UUID, after: list[AnnexCanonicalSnapshot]
) -> None:
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows

    existing = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == user_file_id)
            .with_for_update()
        )
    }
    if not set(existing).issubset({row.id for row in after}):
        raise ValueError("writer cannot discard canonical history")
    for desired in canonical_snapshot_rows(after):
        current = existing.get(desired.id)
        if current is None:
            session.add(desired)
            continue
        if current.projection_ordinal != desired.projection_ordinal:
            raise ValueError("writer changed stable canonical ordinal")
        for name in (
            "text",
            "position",
            "chunk_type",
            "heading_path",
            "chunk_metadata",
            "validity_start_date",
            "validity_end_date",
            "status",
            "source",
            "supersedes_chunk_id",
            "superseded_by_chunk_id",
        ):
            setattr(current, name, getattr(desired, name))
    session.flush()


def _validate_history_coverage(
    active: list[RegulatoryTemporalProjection],
    manifest: WriterPublicationManifest,
    before: list[AnnexCanonicalSnapshot],
) -> None:
    if manifest.kind == "delete":
        return
    after = {row.id: row for row in manifest.canonical_after or []}
    corrected_windows = (
        {
            row.id
            for row in before
            if row.id in after
            and (
                row.validity_start_date != after[row.id].validity_start_date
                or row.validity_end_date != after[row.id].validity_end_date
            )
        }
        if manifest.kind == "correction"
        else set()
    )
    coverage: dict[tuple[str, str], list[tuple[date, date]]] = {}
    for binding in manifest.bindings:
        key = (
            binding.index.index_uuid,
            json.loads(binding.projection.source_json)["regulatory_chunk_id"],
        )
        coverage.setdefault(key, []).append(
            (binding.effective_start or date.min, binding.effective_end or date.max)
        )
    for intervals in coverage.values():
        intervals.sort()
    for previous in active:
        start, end = (
            previous.effective_start or date.min,
            previous.effective_end or date.max,
        )
        if manifest.kind == "amendment" and previous.canonical_chunk_id in after:
            end = min(
                end, after[previous.canonical_chunk_id].validity_end_date or date.max
            )
        if previous.canonical_chunk_id in after and (
            manifest.kind == "validity"
            or previous.canonical_chunk_id in corrected_windows
        ):
            legal = after[previous.canonical_chunk_id]
            start = max(start, legal.validity_start_date or date.min)
            end = min(end, legal.validity_end_date or date.max)
            if start >= end:
                continue
        intervals = coverage.get((previous.index_uuid, previous.canonical_chunk_id), [])
        covered = start
        for lower, upper in intervals:
            if lower <= covered:
                covered = max(covered, upper)
        if covered < end:
            raise ValueError("writer would discard qualified legal history")


def finalize_writer_publication(
    owner: FileOwnership,
    manifest: WriterPublicationManifest,
    proofs: list[PublicationVerification],
    *,
    durable_generation: int | None = None,
) -> None:
    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_context_projections import (
        activate_temporal_projection,
        persist_context_evidence,
    )

    authority = PublicationStore(owner.scope)
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = authority.lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owner.user_file_id)
        assert row is not None
        if row.writer_manifest != manifest.model_dump(
            mode="json"
        ) or row.writer_manifest_sha256 != publication_digest(row.writer_manifest):
            raise ValueError("writer activation manifest changed")
        if (
            canonical_scope_digest(session, owner.user_file_id)
            != manifest.canonical_before_sha256
        ):
            raise ValueError("writer canonical baseline changed")
        if [proof.index for proof in proofs] != manifest.indexes:
            raise ValueError("writer activation requires every physical index proof")
        for proof in proofs:
            expected = tuple(
                sorted(
                    binding.projection.ordinal
                    for binding in manifest.bindings
                    if binding.index.index_uuid == proof.index.index_uuid
                )
            )
            if proof.reservations != reservations or proof.live_ordinals != expected:
                raise ValueError(
                    "writer proof does not cover complete reserved inventory"
                )
        if manifest.kind == "durable":
            _validate_durable_writer(
                session, owner, manifest, durable_generation, "PUBLISH"
            )
        active = list(
            session.scalars(
                select(RegulatoryTemporalProjection)
                .where(
                    RegulatoryTemporalProjection.user_file_id == owner.user_file_id,
                    RegulatoryTemporalProjection.index_uuid.in_(
                        [index.index_uuid for index in manifest.indexes]
                    ),
                    RegulatoryTemporalProjection.retired_at.is_(None),
                )
                .with_for_update()
            )
        )
        active_ids = {binding.id for binding in active}
        if active_ids != set(manifest.previous_binding_ids):
            retired = [
                session.get(RegulatoryTemporalProjection, identifier)
                for identifier in set(manifest.previous_binding_ids) - active_ids
            ]
            if (
                manifest.kind != "delete"
                or not active_ids.issubset(manifest.previous_binding_ids)
                or any(
                    item is None
                    or item.retired_at is None
                    or item.user_file_id != owner.user_file_id
                    for item in retired
                )
            ):
                raise ValueError("writer qualified history baseline changed")
        _validate_history_coverage(
            active, manifest, capture_canonical_scope(session, owner.user_file_id)
        )
        retained = {binding.id: binding for binding in manifest.bindings}
        for previous in active:
            if previous.id in retained:
                if previous.payload != retained[previous.id].model_dump(mode="json"):
                    raise ValueError("writer changed immutable retained binding")
            else:
                previous.retired_at = datetime.now(timezone.utc)
        session.flush()
        if manifest.amendment_proposal_id is not None:
            from onyx.db.models import AmendmentProposal
            from onyx.db.regulatory_amendments import approve_amendment_proposal

            proposal = session.get(AmendmentProposal, manifest.amendment_proposal_id)
            if (
                proposal is None
                or _amendment_review_digest(proposal)
                != manifest.amendment_review_sha256
            ):
                raise ValueError("amendment human review changed during publication")
            approve_amendment_proposal(session, proposal, publication_owner=owner)
            if (
                capture_canonical_scope(session, owner.user_file_id)
                != manifest.canonical_after
            ):
                raise ValueError(
                    "amendment no longer matches frozen canonical transition"
                )
        _apply_canonical(session, manifest)
        for view in manifest.views:
            persist_context_evidence(
                session, user_file_id=owner.user_file_id, view=view
            )
        existing_ids = {binding.id for binding in active}
        pending = [
            binding for binding in manifest.bindings if binding.id not in existing_ids
        ]
        canonical_by_binding = {
            binding.id: json.loads(binding.projection.source_json)[
                "regulatory_chunk_id"
            ]
            for binding in pending
        }
        while pending:
            ready = [
                binding
                for binding in pending
                if not any(
                    other is not binding
                    and other.index.index_uuid == binding.index.index_uuid
                    and canonical_by_binding[other.id] in binding.dependency_ids
                    for other in pending
                )
            ]
            if not ready:
                raise ValueError("writer temporal dependency cycle")
            for binding in ready:
                activate_temporal_projection(
                    session,
                    user_file_id=owner.user_file_id,
                    binding=binding,
                    canonical_revision_id=manifest.canonical_revisions.get(binding.id),
                )
                pending.remove(binding)
        file = session.get(UserFile, owner.user_file_id)
        if file is None:
            raise ValueError("writer live file disappeared")
        if manifest.name_after is not None:
            file.name = manifest.name_after
        if manifest.complete_file and file.status != UserFileStatus.DELETING:
            file.status = UserFileStatus.COMPLETED
        if manifest.chunk_count_after is not None:
            file.chunk_count = manifest.chunk_count_after
        if manifest.secondary_reconcile_pending is not None:
            file.secondary_reconcile_pending = manifest.secondary_reconcile_pending
        if manifest.amendment_proposal_id is not None:
            from onyx.db.regulatory_amendments import (
                finalize_amendment_proposal_projection,
            )

            if not finalize_amendment_proposal_projection(
                session, proposal_id=manifest.amendment_proposal_id, succeeded=True
            ):
                raise ValueError("owned amendment could not be finalized")
        if manifest.kind == "durable":
            from onyx.db.regulatory_indexing_jobs import (
                complete_regulatory_indexing_publication,
            )

            assert (
                manifest.durable_job_id is not None and durable_generation is not None
            )
            if not complete_regulatory_indexing_publication(
                session,
                job_id=manifest.durable_job_id,
                expected_generation=durable_generation,
                chunk_count=manifest.chunk_count_after or 0,
                now=datetime.now(timezone.utc),
                commit=False,
            ):
                raise ValueError("durable publication completion lease changed")
        if manifest.kind == "cancellation":
            from onyx.db.regulatory_durable_publication import lock_owned_cancellation
            from onyx.db.regulatory_indexing_jobs import (
                finalize_regulatory_indexing_cancellation,
            )

            if manifest.cancellation_job_id is None or durable_generation is None:
                raise ValueError("cancellation publication requires job ownership")
            job = lock_owned_cancellation(
                session,
                owner,
                job_id=manifest.cancellation_job_id,
                expected_generation=durable_generation,
            )
            job.cancellation_phase = "FINALIZE"
            session.flush()
            if not finalize_regulatory_indexing_cancellation(
                session,
                job_id=job.id,
                expected_generation=durable_generation,
                now=datetime.now(timezone.utc),
                commit=False,
                preserve_published_history=bool(active),
            ):
                raise ValueError("cancellation publication completion lease changed")
        authority.finalize(session, owner, proofs[0])
        if (
            manifest.index_state_sha256 is not None
            and manifest.index_state_sha256
            != writer_index_state_digest(
                session, [index.search_settings_id for index in manifest.indexes]
            )
        ):
            raise ValueError("writer index settings changed before activation")
        if manifest.kind == "delete":
            file.status = UserFileStatus.DELETING
            authority.record_event(session, owner)
        else:
            row.writer_manifest = None
            row.writer_manifest_sha256 = None
        session.commit()


def owned_metadata_baseline(
    owner: FileOwnership, index_names: list[str]
) -> tuple[str, list["AnnexTemporalProjection"], dict[UUID, UUID], str]:
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if reservations.gate_closed:
            raise ValueError("metadata must wait for the existing publication recovery")
        archive_canonical_revisions(session, owner)
        bindings = [
            binding
            for binding in load_file_temporal_bindings(session, owner.user_file_id)
            if binding.index.index_name in index_names
        ]
        if set(index_names) != {binding.index.index_name for binding in bindings}:
            raise ValueError("metadata index needs its canonical publication first")
        revisions = {}
        for binding in bindings:
            row = session.get(RegulatoryTemporalProjection, binding.id)
            if row is None or row.canonical_revision_id is None:
                raise ValueError("metadata binding has no retained canonical revision")
            revisions[binding.id] = row.canonical_revision_id
        digest = canonical_scope_digest(session, owner.user_file_id)
        index_state = writer_index_state_digest(
            session, [binding.index.search_settings_id for binding in bindings]
        )
        session.commit()
        return digest, bindings, revisions, index_state


def has_file_publication_history(user_file_id: UUID, tenant_id: str) -> bool:
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        return session.get(RegulatoryFilePublication, user_file_id) is not None


@dataclass(frozen=True)
class OwnedMetadataSyncPlan:
    request: "MetadataUpdateRequest"
    index_names: list[str]
    requires_content: bool
    cleanup_failed: bool


def _owned_metadata_sync_request(
    session: Session,
    owner: FileOwnership,
) -> OwnedMetadataSyncPlan | None:
    from onyx.access.access import get_access_for_user_files
    from onyx.db.enums import UserFileStatus
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.db.user_file import (
        fetch_document_set_names_for_user_files,
        fetch_persona_ids_for_user_files,
        fetch_user_project_ids_for_user_files,
    )
    from onyx.document_index.interfaces_new import MetadataUpdateRequest

    file = session.get(UserFile, owner.user_file_id)
    if file is None:
        raise ValueError("owned metadata file disappeared")
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings

    bindings = load_file_temporal_bindings(session, owner.user_file_id)
    eligible = file.status in {UserFileStatus.COMPLETED, UserFileStatus.FAILED}
    if not eligible or not any(
        (
            file.needs_project_sync,
            file.needs_persona_sync,
            file.needs_document_set_sync,
            file.secondary_reconcile_pending,
        )
    ):
        return None
    identifier = str(file.id)
    request = MetadataUpdateRequest(
        document_ids=[identifier],
        doc_id_to_chunk_cnt={identifier: file.chunk_count or -1},
        access=get_access_for_user_files([identifier], session)[identifier],
        document_sets=set(
            fetch_document_set_names_for_user_files([identifier], session).get(
                identifier, []
            )
        ),
        project_ids=set(
            fetch_user_project_ids_for_user_files([identifier], session).get(
                identifier, []
            )
        ),
        persona_ids=set(
            fetch_persona_ids_for_user_files([identifier], session).get(identifier, [])
        ),
    )
    names = [
        settings.index_name for settings in get_active_search_settings_list(session)
    ]
    represented = {binding.index.index_name for binding in bindings}
    return OwnedMetadataSyncPlan(
        request,
        names,
        bool(file.secondary_reconcile_pending or not set(names).issubset(represented)),
        file.status == UserFileStatus.FAILED and not bindings,
    )


def owned_metadata_sync_request(
    owner: FileOwnership,
) -> OwnedMetadataSyncPlan | None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        return _owned_metadata_sync_request(session, owner)


def finish_owned_metadata_sync(
    owner: FileOwnership, request: "MetadataUpdateRequest"
) -> None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        file = session.scalar(
            select(UserFile).where(UserFile.id == owner.user_file_id).with_for_update()
        )
        if file is None:
            raise ValueError("owned metadata file disappeared")
        current = _owned_metadata_sync_request(session, owner)
        if current is None or current.request != request:
            return
        file.needs_project_sync = False
        file.needs_persona_sync = False
        file.needs_document_set_sync = False
        file.secondary_reconcile_pending = False
        file.last_project_sync_at = datetime.now(timezone.utc)
        session.commit()


@dataclass(frozen=True)
class OwnedWriterInputs:
    file: UserFile
    settings: list[SearchSettings]
    access: AnnexProjectionAccess
    canonical: list[AnnexCanonicalSnapshot]
    bindings: list[AnnexTemporalProjection]
    revisions: dict[UUID, UUID]
    canonical_revisions: dict[str, UUID]
    cached: PreparedContextView
    index_state_sha256: str


def load_owned_writer_inputs(owner: FileOwnership) -> OwnedWriterInputs:
    from onyx.access.access import get_access_for_user_files
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
    from onyx.db.regulatory_context_projections import load_context_generation_calls
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.db.user_file import (
        fetch_document_set_names_for_user_files,
        fetch_persona_ids_for_user_files,
        fetch_user_project_ids_for_user_files,
    )

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if reservations.gate_closed:
            raise ValueError(
                "writer must recover the existing closed publication first"
            )
        canonical_revisions = archive_canonical_revisions(session, owner)
        file = session.get(UserFile, owner.user_file_id)
        if file is None:
            raise ValueError("owned writer file disappeared")
        settings = get_active_search_settings_list(session)
        for setting in settings:
            _ = setting.cloud_provider
        bindings = load_file_temporal_bindings(session, owner.user_file_id)
        revisions = {}
        for binding in bindings:
            row = session.get(RegulatoryTemporalProjection, binding.id)
            if row is None or row.canonical_revision_id is None:
                raise ValueError("writer binding canonical revision is missing")
            revisions[binding.id] = row.canonical_revision_id
        identifier = str(file.id)
        result = OwnedWriterInputs(
            file=file,
            settings=settings,
            access=AnnexProjectionAccess(
                access=get_access_for_user_files([identifier], session)[identifier],
                project_ids=fetch_user_project_ids_for_user_files(
                    [identifier], session
                ).get(identifier, []),
                persona_ids=fetch_persona_ids_for_user_files([identifier], session).get(
                    identifier, []
                ),
                document_sets=fetch_document_set_names_for_user_files(
                    [identifier], session
                ).get(identifier, []),
            ),
            canonical=capture_canonical_scope(session, owner.user_file_id),
            bindings=bindings,
            revisions=revisions,
            canonical_revisions=canonical_revisions,
            index_state_sha256=writer_index_state_digest(
                session, [item.id for item in settings]
            ),
            cached=PreparedContextView(
                calls=load_context_generation_calls(
                    session, user_file_id=owner.user_file_id
                )
            ),
        )
        session.commit()
        session.expunge_all()
        return result


def apply_owned_deferred_edit(
    owner: FileOwnership,
    *,
    canonical_before_sha256: str,
    canonical_after: list[AnnexCanonicalSnapshot] | None = None,
    name: str | None = None,
) -> bool:
    """Edit never-published deferred chunks without admitting a search publication."""
    from onyx.db.enums import UserFileStatus

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        publication = session.get_one(RegulatoryFilePublication, owner.user_file_id)
        file = session.get(UserFile, owner.user_file_id, with_for_update=True)
        if file is None:
            raise ValueError("owned file no longer exists")
        if (
            file.status not in {UserFileStatus.CHUNKED, UserFileStatus.INDEXING}
            or reservations.gate_closed
            or publication.epoch != 0
            or publication.writer_manifest is not None
            or session.scalar(
                select(RegulatoryTemporalProjection.id)
                .where(RegulatoryTemporalProjection.user_file_id == owner.user_file_id)
                .limit(1)
            )
            is not None
        ):
            return False
        if (
            canonical_scope_digest(session, owner.user_file_id)
            != canonical_before_sha256
        ):
            raise ValueError("deferred edit canonical baseline changed")
        if canonical_after is not None:
            archive_canonical_revisions(session, owner)
            _apply_canonical_rows(session, owner.user_file_id, canonical_after)
            archive_canonical_revisions(session, owner)
        if name is not None:
            file.name = name
        session.commit()
        return True


def rename_owned_unprojected_file(owner: FileOwnership, name: str) -> None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if reservations.gate_closed or capture_canonical_scope(
            session, owner.user_file_id
        ):
            raise ValueError("rename requires canonical publication")
        file = session.get(UserFile, owner.user_file_id, with_for_update=True)
        if file is None:
            raise ValueError("owned file no longer exists")
        file.name = name
        session.commit()


def persist_owned_initial_chunks(
    owner: FileOwnership,
    documents: "Sequence[Document]",
    tokenizer: "BaseTokenizer",
    *,
    contextual: bool,
    generation_hash: str,
    preparing_job_id: UUID | None = None,
    preparing_job_generation: int | None = None,
) -> int:
    from onyx.db.enums import UserFileStatus
    from onyx.regulatory.indexing import documents_to_regulatory_chunks

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        if preparing_job_id is not None:
            from onyx.db.models import RegulatoryIndexingJob

            PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
            file = session.get(UserFile, owner.user_file_id, with_for_update=True)
            job = session.get(
                RegulatoryIndexingJob, preparing_job_id, with_for_update=True
            )
            if (
                file is None
                or file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}
                or job is None
                or (
                    job.user_file_id != owner.user_file_id
                    or job.status != "RUNNING"
                    or job.stage != "PREPARING"
                    or job.lease_generation != preparing_job_generation
                )
            ):
                raise ValueError(
                    "durable initial preparation lease is no longer current"
                )
        chunks = documents_to_regulatory_chunks(
            documents,
            session,
            tokenizer,
            enable_contextual_rag=contextual,
            publication_owner=owner,
        )
        if not chunks:
            raise ValueError("initial chunking produced no canonical chunks")
        file = session.get(UserFile, owner.user_file_id, with_for_update=True)
        if file is None or file.status == UserFileStatus.DELETING:
            raise ValueError("initial chunking file is gone or deleting")
        file.status = UserFileStatus.CHUNKED
        file.chunk_count = len(chunks)
        file.regulatory_chunk_generation_hash = generation_hash
        from onyx.file_processing.original_ingestion import (
            LoadedUserFileDocuments,
            OriginalIngestionReceipt,
            documents_sha256,
        )

        if isinstance(documents, LoadedUserFileDocuments):
            proof = documents.original_extraction
            if (
                proof.file_id != file.file_id
                or proof.documents_sha256 != documents_sha256(documents)
            ):
                raise ValueError(
                    "original extraction no longer matches canonical input"
                )
            publication = session.get(RegulatoryFilePublication, owner.user_file_id)
            assert publication is not None
            if publication.original_ingestion_receipt is None:
                publication.original_ingestion_receipt = OriginalIngestionReceipt(
                    **proof.model_dump(),
                    canonical_sha256=canonical_scope_digest(
                        session, owner.user_file_id
                    ),
                    generation_hash=generation_hash,
                ).model_dump(mode="json")
        session.commit()
        return len(chunks)


def writer_file_exists(user_file_id: UUID, tenant_id: str) -> bool:
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        return session.get(UserFile, user_file_id) is not None


def begin_owned_deletion(owner: FileOwnership) -> "UserFileDeletionCleanupPlan":
    from onyx.db.regulatory_indexing_jobs import request_user_file_deletion_cleanup

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        return request_user_file_deletion_cleanup(
            session,
            user_file_id=owner.user_file_id,
            now=datetime.now(timezone.utc),
            publication_owner=owner,
        )


def _unindexed_deletion_file(session: Session, owner: FileOwnership) -> UserFile | None:
    from sqlalchemy import or_

    from onyx.db.enums import UserFileStatus
    from onyx.db.models import (
        AnnexChangeSet,
        AnnexPublicationIntent,
        RegulatoryCanonicalRevision,
        RegulatoryIndexingJob,
        RegulatoryPublicationOrdinal,
    )

    reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
    row = session.get_one(RegulatoryFilePublication, owner.user_file_id)
    file = session.get(UserFile, owner.user_file_id, with_for_update=True)
    if (
        file is None
        or file.status != UserFileStatus.DELETING
        or file.chunk_count != 0
        or reservations.gate_closed
        or reservations.ordinals
        or row.epoch != 0
        or row.next_ordinal != 0
        or row.writer_manifest is not None
        or row.original_ingestion_receipt is not None
    ):
        return None
    has_authority = session.scalar(
        select(
            or_(
                select(RegulatoryChunk.id)
                .where(RegulatoryChunk.user_file_id == owner.user_file_id)
                .exists(),
                select(RegulatoryCanonicalRevision.id)
                .where(RegulatoryCanonicalRevision.user_file_id == owner.user_file_id)
                .exists(),
                select(RegulatoryTemporalProjection.id)
                .where(RegulatoryTemporalProjection.user_file_id == owner.user_file_id)
                .exists(),
                select(RegulatoryPublicationOrdinal.ordinal)
                .where(RegulatoryPublicationOrdinal.user_file_id == owner.user_file_id)
                .exists(),
                select(RegulatoryIndexingJob.id)
                .where(RegulatoryIndexingJob.user_file_id == owner.user_file_id)
                .exists(),
                select(AnnexPublicationIntent.id)
                .join(AnnexChangeSet)
                .where(AnnexChangeSet.user_file_id == owner.user_file_id)
                .exists(),
            )
        )
    )
    return None if has_authority else file


def owned_unindexed_deletion_file_id(owner: FileOwnership) -> str | None:
    """Prove the completed no-index upload has no historical or pending ES authority."""
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        file = _unindexed_deletion_file(session, owner)
        return file.file_id if file is not None else None


def owned_deletion_file_id(owner: FileOwnership) -> str:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owner.user_file_id)
        file = session.get(UserFile, owner.user_file_id)
        if (
            not reservations.gate_closed
            or row is None
            or row.writer_manifest is None
            or row.writer_manifest.get("kind") != "delete"
            or file is None
        ):
            raise ValueError("file deletion requires its verified retained manifest")
        return file.file_id


def finish_owned_deletion(
    owner: FileOwnership, *, without_index_authority: bool = False
) -> None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        if without_index_authority:
            file = _unindexed_deletion_file(session, owner)
            if file is None:
                raise ValueError("unindexed deletion lost its positive absence proof")
            session.delete(file)
            session.commit()
            return
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        row = session.get(RegulatoryFilePublication, owner.user_file_id)
        if (
            not reservations.gate_closed
            or row is None
            or row.writer_manifest is None
            or row.writer_manifest.get("kind") != "delete"
        ):
            raise ValueError("deletion lost its retained publication gate")
        active = session.scalar(
            select(RegulatoryTemporalProjection.id)
            .where(
                RegulatoryTemporalProjection.user_file_id == owner.user_file_id,
                RegulatoryTemporalProjection.retired_at.is_(None),
            )
            .limit(1)
        )
        if active is not None:
            raise ValueError("deletion still has active historical representations")
        file = session.get(UserFile, owner.user_file_id, with_for_update=True)
        if file is not None:
            session.delete(file)
        row.writer_manifest = None
        row.writer_manifest_sha256 = None
        session.commit()


def amendment_writer_target(
    proposal_id: int, tenant_id: str
) -> tuple[UUID, list[str]] | None:
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_chunks import make_regulatory_chunk_id

    with get_session_with_tenant(tenant_id=tenant_id) as session:
        proposal = session.get(AmendmentProposal, proposal_id)
        if proposal is None or proposal.status != "approving":
            return None
        changes = list(getattr(proposal, "chunk_changes", None) or [])
        if len(changes) > 1:
            file_ids = {
                UUID(change["new_chunk_draft"]["user_file_id"]) for change in changes
            }
            if len(file_ids) != 1:
                raise ValueError("Atomic amendment spans multiple source files")
            file_id = next(iter(file_ids))
            applied = list(getattr(proposal, "applied_new_chunk_ids", None) or [])
            canonical_ids = applied or [
                make_regulatory_chunk_id(
                    file_id,
                    change["new_chunk_draft"]["position"],
                    change["new_chunk_draft"]["text"],
                    version_key=f"amendment:{proposal.id}:{index}",
                )
                for index, change in enumerate(changes)
            ]
            return file_id, canonical_ids
        draft = proposal.new_chunk_draft
        file_id = UUID(draft["user_file_id"])
        return file_id, [
            proposal.applied_new_chunk_id
            or make_regulatory_chunk_id(
                file_id,
                draft["position"],
                draft["text"],
                version_key=f"amendment:{proposal.id}",
            )
        ]


def _amendment_review_digest(proposal: "AmendmentProposal") -> str:
    return publication_digest(
        {
            "id": proposal.id,
            "new_chunk_draft": proposal.new_chunk_draft,
            "chunk_changes": list(getattr(proposal, "chunk_changes", None) or []),
            "old_chunk_id": proposal.old_chunk_id,
            "old_chunk_snapshot": proposal.old_chunk_snapshot,
            "instruction_text": proposal.instruction_text,
            "instruction_texts": proposal.instruction_texts,
            "decided_by": str(proposal.decided_by) if proposal.decided_by else None,
        }
    )


def preview_owned_amendment(
    owner: FileOwnership, proposal_id: int
) -> tuple[list[AnnexCanonicalSnapshot], str]:
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_amendments import approve_amendment_proposal

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        proposal = session.get(AmendmentProposal, proposal_id)
        if proposal is None:
            raise ValueError("amendment proposal disappeared")
        reviewed = _amendment_review_digest(proposal)
        result = approve_amendment_proposal(session, proposal, publication_owner=owner)
        if any(chunk.user_file_id != owner.user_file_id for chunk in result.new_chunks):
            raise ValueError("amendment escaped publication file scope")
        after = capture_canonical_scope(session, owner.user_file_id)
        session.rollback()
        return after, reviewed


def record_owned_amendment_failure(proposal_id: int, tenant_id: str) -> None:
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_amendments import reset_amendment_proposal_approval

    with get_session_with_tenant(tenant_id=tenant_id) as session:
        proposal = session.get(AmendmentProposal, proposal_id)
        if proposal is None or proposal.status != "approving":
            return
        file_id = UUID(proposal.new_chunk_draft["user_file_id"])
        publication = session.get(RegulatoryFilePublication, file_id)
        if (
            publication is not None
            and publication.writer_manifest is not None
            and publication.writer_manifest.get("amendment_proposal_id") == proposal_id
        ):
            proposal.approval_error = (
                "Indexing interrupted. The frozen publication will resume."
            )
            session.commit()
        else:
            reset_amendment_proposal_approval(session, proposal_id=proposal_id)
            session.commit()


def _validate_durable_writer(
    session: Session,
    owner: FileOwnership,
    manifest: WriterPublicationManifest,
    generation: int | None,
    stage: str,
) -> None:
    from onyx.db.enums import RegulatoryIndexingStage
    from onyx.db.regulatory_durable_publication import (
        durable_publication_input_digest,
        lock_owned_durable_runtime,
    )

    if manifest.durable_job_id is None or generation is None:
        raise ValueError("durable publication requires its current job generation")
    runtime = lock_owned_durable_runtime(
        session,
        owner,
        job_id=manifest.durable_job_id,
        expected_stage=RegulatoryIndexingStage(stage),
        expected_generation=generation,
    )
    if durable_publication_input_digest(runtime) != manifest.durable_input_sha256:
        raise ValueError("durable publication input checkpoint changed")


def writer_index_state_digest(session: Session, target_ids: list[int]) -> str:
    from onyx.db.search_settings import get_active_search_settings_list

    settings = {item.id: item for item in get_active_search_settings_list(session)}
    for identifier in target_ids:
        if identifier not in settings:
            item = session.get(SearchSettings, identifier)
            if item is not None:
                settings[identifier] = item
    return publication_digest(
        [
            {
                "id": item.id,
                "status": item.status.value,
                "name": item.index_name,
                "model": item.model_name,
                "dimension": item.model_dim,
                "reduced_dimension": item.reduced_dimension,
                "provider": item.provider_type.value if item.provider_type else None,
                "normalize": item.normalize,
                "query_prefix": item.query_prefix,
                "passage_prefix": item.passage_prefix,
                "endpoint_sha256": publication_digest(item.api_url),
                "api_version": item.api_version,
                "deployment_name": item.deployment_name,
                "contextual": item.enable_contextual_rag,
                "contextual_model_id": item.contextual_rag_model_configuration_id,
                "reclaim_status": item.reclaim_status.value
                if item.reclaim_status
                else None,
            }
            for item in sorted(settings.values(), key=lambda item: item.id)
        ]
    )


@dataclass(frozen=True)
class WriterIndexInventory:
    settings: list[SearchSettings]
    bindings: list[AnnexTemporalProjection]
    revisions: dict[UUID, UUID]
    index_state_sha256: str


def owned_writer_index_inventory(
    owner: FileOwnership, target_ids: list[int]
) -> WriterIndexInventory:
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
    from onyx.db.search_settings import get_active_search_settings_list

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        archive_canonical_revisions(session, owner)
        settings = get_active_search_settings_list(session)
        for setting in settings:
            _ = setting.cloud_provider
        bindings = load_file_temporal_bindings(session, owner.user_file_id)
        revisions = {}
        for binding in bindings:
            row = session.get(RegulatoryTemporalProjection, binding.id)
            if row is None or row.canonical_revision_id is None:
                raise ValueError("writer inventory lacks retained canonical authority")
            revisions[binding.id] = row.canonical_revision_id
        digest = writer_index_state_digest(session, target_ids)
        for setting in settings:
            _ = setting.cloud_provider
        session.commit()
        session.expunge_all()
        return WriterIndexInventory(settings, bindings, revisions, digest)
