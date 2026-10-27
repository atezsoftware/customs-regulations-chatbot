"""Owned replacement of an unpublished intake's invalid frozen index preparation."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_original_ingestion import (
    has_initial_intake_canonical_state,
)
from onyx.db.document_intake_reprepare import reprepared_original_receipt
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryAnnexElementChunk,
    RegulatoryChunk,
    RegulatoryContextGeneration,
    RegulatoryContextProjection,
    RegulatoryContextSnapshot,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakePublicationRecovery,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatoryPublicationOrdinal,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_publication import PublicationStore, archive_canonical_revisions
from onyx.db.regulatory_writer_publication import (
    OwnedWriterInputs,
    canonical_scope_digest,
    writer_index_state_digest,
)
from onyx.document_index.publication_models import (
    FileOwnership,
    publication_streaming_digest,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.amendments.annexes.models import PreparedContextView
from onyx.regulatory.writer_publication_models import WriterPublicationManifest


def pending_failed_intake_publication(session: Session, file_id: UUID) -> bool:
    row = session.get(RegulatoryFilePublication, file_id, populate_existing=True)
    return bool(row and (row.writer_manifest is not None or row.gate_closed))


def _no_activated_history(session: Session, file_id: UUID) -> bool:
    for model in (
        RegulatoryTemporalProjection,
        RegulatoryContextSnapshot,
        RegulatoryContextGeneration,
    ):
        if session.scalar(select(model).where(model.user_file_id == file_id).limit(1)):
            return False
    if session.scalar(
        select(RegulatoryContextProjection)
        .join(
            RegulatoryChunk,
            RegulatoryChunk.id == RegulatoryContextProjection.canonical_chunk_id,
        )
        .where(RegulatoryChunk.user_file_id == file_id)
        .limit(1)
    ):
        return False
    if session.scalar(
        select(RegulatoryAnnexElementChunk)
        .join(
            RegulatoryChunk, RegulatoryChunk.id == RegulatoryAnnexElementChunk.chunk_id
        )
        .where(RegulatoryChunk.user_file_id == file_id)
        .limit(1)
    ):
        return False
    return (
        session.scalar(
            select(RegulatoryIndexingJob)
            .where(
                RegulatoryIndexingJob.user_file_id == file_id,
                RegulatoryIndexingJob.status.not_in(("FAILED", "CANCELLED")),
            )
            .limit(1)
        )
        is None
    )


def _validate_source(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    file: UserFile,
    publication: RegulatoryFilePublication,
    *,
    allow_activated_history: bool = False,
) -> str:
    asset = session.get(
        RegulatorySourceAsset, member.source_asset_id, populate_existing=True
    )
    package = session.get(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    try:
        original = OriginalIngestionReceipt.model_validate(
            publication.original_ingestion_receipt
        )
        effective = reprepared_original_receipt(session, file.id, original)
        if (
            not batch.review_sha256
            or intake.review_digest(session, batch) != batch.review_sha256
            or member.intake_id != batch.id
            or member.user_file_id != file.id
            or file.user_id != batch.created_by
            or file.file_id != original.file_id
            or asset is None
            or asset.package_id != batch.source_package_id
            or asset.sha256 != original.raw_sha256
            or package is None
            or package.status != "ready"
            or package.issues
            or package.created_by != batch.created_by
            or package.document_set_id != batch.document_set_id
            or effective.file_id != original.file_id
            or effective.raw_sha256 != original.raw_sha256
            or effective.generation_hash != file.regulatory_chunk_generation_hash
            or not has_initial_intake_canonical_state(session, [file.id])
            or not allow_activated_history
            and not _no_activated_history(session, file.id)
        ):
            raise ValueError("The approved original or unpublished history changed")
        return canonical_scope_digest(session, file.id)
    except (ValueError, intake.IntakeConflictError) as error:
        raise intake.IntakeConflictError(
            "The approved source or publication history changed"
        ) from error


def _manifest(publication: RegulatoryFilePublication) -> WriterPublicationManifest:
    if (
        not publication.gate_closed
        or publication.writer_manifest is None
        or publication_streaming_digest(publication.writer_manifest)
        != publication.writer_manifest_sha256
    ):
        raise intake.IntakeConflictError("The closed publication proof changed")
    value = WriterPublicationManifest.model_validate(publication.writer_manifest)
    if (
        value.kind != "reindex"
        or not value.complete_file
        or value.previous_binding_ids
        or value.amendment_proposal_id is not None
        or value.baseline_origin_proposal_id is not None
        or value.durable_job_id is not None
        or value.cancellation_job_id is not None
        or value.name_after is not None
    ):
        raise intake.IntakeConflictError(
            "Only an unpublished initial intake reindex can be rebuilt"
        )
    return value


def request_intake_publication_recovery(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    file: UserFile,
) -> UUID:
    from onyx.db.user_file import claim_user_file_projection_repair

    publication = session.get(
        RegulatoryFilePublication, file.id, populate_existing=True
    )
    repair = session.get(
        UserFileProjectionRepair, file.id, with_for_update=True, populate_existing=True
    )
    if (
        batch.status not in {"failed", "partial"}
        or batch.stage != "indexing"
        or file.status != UserFileStatus.FAILED
        or member.status != "failed"
        or publication is None
        or publication.lease_expires_at > datetime.now(timezone.utc)
        or repair is None
        or not repair.initial_index
        or repair.status != UserFileProjectionRepairStatus.FAILED
        or repair.attempt_id != member.approval_attempt_id
    ):
        raise intake.IntakeConflictError(
            "The failed initial publication is not available for recovery"
        )
    original = _manifest(publication)
    canonical = _validate_source(session, batch, member, file, publication)
    if (
        original.user_file_id != file.id
        or original.canonical_before_sha256 != canonical
        or original.canonical_after is None
        or publication_streaming_digest(
            [item.model_dump(mode="json") for item in original.canonical_after]
        )
        != canonical
    ):
        raise intake.IntakeConflictError(
            "The frozen publication differs from the approved original"
        )
    prior = repair.attempt_id
    file.status = UserFileStatus.CHUNKED
    replacement = claim_user_file_projection_repair(
        session, file.id, initial_index=True
    )
    if replacement is None:
        raise intake.IntakeConflictError(
            "The failed publication cannot currently be replaced"
        )
    session.add(
        RegulatoryDocumentIntakePublicationRecovery(
            id=uuid4(),
            intake_id=batch.id,
            document_id=member.id,
            user_file_id=file.id,
            previous_attempt_id=prior,
            replacement_attempt_id=replacement,
            approved_review_sha256=batch.review_sha256,
            canonical_sha256=canonical,
            original_manifest_sha256=publication.writer_manifest_sha256,
            original_manifest=deepcopy(publication.writer_manifest),
            original_fencing_token=publication.fencing_token,
            status="queued",
        )
    )
    session.flush()
    return replacement


def recovery_for_attempt(
    session: Session, file_id: UUID, attempt_id: UUID
) -> UUID | None:
    identifier = session.scalar(
        select(RegulatoryDocumentIntakePublicationRecovery.id).where(
            RegulatoryDocumentIntakePublicationRecovery.user_file_id == file_id,
            RegulatoryDocumentIntakePublicationRecovery.replacement_attempt_id
            == attempt_id,
        )
    )
    publication = session.get(
        RegulatoryFilePublication, file_id, populate_existing=True
    )
    if (
        identifier is None
        and publication is not None
        and publication.gate_closed
        and publication.writer_manifest is not None
        and session.scalar(
            select(RegulatoryDocumentIntakePublicationRecovery.id)
            .where(
                RegulatoryDocumentIntakePublicationRecovery.user_file_id == file_id,
                RegulatoryDocumentIntakePublicationRecovery.status != "completed",
                or_(
                    RegulatoryDocumentIntakePublicationRecovery.original_manifest_sha256
                    == publication.writer_manifest_sha256,
                    RegulatoryDocumentIntakePublicationRecovery.replacement_manifest_sha256
                    == publication.writer_manifest_sha256,
                ),
            )
            .limit(1)
        )
        is not None
    ):
        raise intake.IntakeConflictError(
            "The recovery delivery changed; retry the reviewed import"
        )
    return identifier


def committed_recovery_delivery(
    session: Session, file_id: UUID, attempt_id: UUID
) -> bool:
    """Only a completed owned recovery may finish an already-started delivery."""
    file = session.get(UserFile, file_id, populate_existing=True)
    repair = session.get(UserFileProjectionRepair, file_id, populate_existing=True)
    return bool(
        file is not None
        and file.status == UserFileStatus.COMPLETED
        and repair is not None
        and repair.initial_index
        and repair.attempt_id == attempt_id
        and repair.status == UserFileProjectionRepairStatus.RUNNING
        and session.scalar(
            select(RegulatoryDocumentIntakePublicationRecovery.id).where(
                RegulatoryDocumentIntakePublicationRecovery.user_file_id == file_id,
                RegulatoryDocumentIntakePublicationRecovery.replacement_attempt_id
                == attempt_id,
                RegulatoryDocumentIntakePublicationRecovery.replacement_manifest.is_not(
                    None
                ),
            )
        )
        is not None
    )


def associate_pending_recovery(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    file: UserFile,
    repair: UserFileProjectionRepair,
) -> None:
    """Bind an expired delivery's rotated token through the same reviewed Retry."""
    previous = session.scalar(
        select(RegulatoryDocumentIntakePublicationRecovery)
        .where(RegulatoryDocumentIntakePublicationRecovery.user_file_id == file.id)
        .order_by(
            RegulatoryDocumentIntakePublicationRecovery.created_at.desc(),
            RegulatoryDocumentIntakePublicationRecovery.id.desc(),
        )
        .limit(1)
    )
    if (
        previous is None
        or previous.status == "completed"
        or previous.replacement_attempt_id == repair.attempt_id
    ):
        return
    publication = session.get(
        RegulatoryFilePublication, file.id, populate_existing=True
    )
    if (
        batch.stage != "indexing"
        or batch.status not in {"failed", "partial"}
        or file.status != UserFileStatus.CHUNKED
        or not repair.initial_index
        or repair.status != UserFileProjectionRepairStatus.PENDING
        or member.approval_attempt_id != previous.replacement_attempt_id
        or previous.intake_id != batch.id
        or previous.document_id != member.id
        or previous.approved_review_sha256 != batch.review_sha256
        or publication is None
        or publication.lease_expires_at > datetime.now(timezone.utc)
    ):
        raise intake.IntakeConflictError(
            "The rotated recovery delivery cannot be associated"
        )
    pending = _manifest(publication)
    canonical = _validate_source(session, batch, member, file, publication)
    expected = (
        previous.replacement_manifest
        if previous.replacement_manifest is not None
        else previous.original_manifest
    )
    if (
        publication.writer_manifest != expected
        or pending.canonical_before_sha256 != canonical
        or previous.canonical_sha256 != canonical
    ):
        raise intake.IntakeConflictError(
            "The retained recovery delivery snapshot changed"
        )
    session.add(
        RegulatoryDocumentIntakePublicationRecovery(
            id=uuid4(),
            intake_id=batch.id,
            document_id=member.id,
            user_file_id=file.id,
            previous_attempt_id=previous.replacement_attempt_id,
            replacement_attempt_id=repair.attempt_id,
            approved_review_sha256=batch.review_sha256,
            canonical_sha256=canonical,
            original_manifest_sha256=publication.writer_manifest_sha256,
            original_manifest=deepcopy(publication.writer_manifest),
            original_fencing_token=publication.fencing_token,
            status="queued",
        )
    )
    previous.status, previous.error = (
        "failed",
        "The expired delivery token was replaced",
    )
    session.flush()


def _owned_recovery(
    session: Session,
    owner: FileOwnership,
    recovery_id: UUID,
    attempt_id: UUID,
) -> tuple[
    RegulatoryDocumentIntakePublicationRecovery, RegulatoryFilePublication, UserFile
]:
    PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
    row = session.get(
        RegulatoryDocumentIntakePublicationRecovery,
        recovery_id,
        with_for_update=True,
        populate_existing=True,
    )
    if (
        row is None
        or row.user_file_id != owner.user_file_id
        or row.replacement_attempt_id != attempt_id
    ):
        raise intake.IntakeConflictError("The publication recovery ownership changed")
    batch = session.get_one(
        RegulatoryDocumentIntake, row.intake_id, populate_existing=True
    )
    member = session.get_one(
        RegulatoryDocumentIntakeFile, row.document_id, populate_existing=True
    )
    file = session.get_one(
        UserFile, row.user_file_id, with_for_update=True, populate_existing=True
    )
    repair = session.get(
        UserFileProjectionRepair,
        row.user_file_id,
        with_for_update=True,
        populate_existing=True,
    )
    publication = session.get_one(
        RegulatoryFilePublication, row.user_file_id, populate_existing=True
    )
    if (
        batch.status not in {"approving", "failed"}
        or batch.stage != "indexing"
        or batch.review_sha256 != row.approved_review_sha256
        or member.approval_attempt_id != attempt_id
        or repair is None
        or repair.attempt_id != attempt_id
        or repair.status != UserFileProjectionRepairStatus.RUNNING
        or not repair.initial_index
        or file.status != UserFileStatus.CHUNKED
        or owner.fencing_token <= row.original_fencing_token
        or publication_streaming_digest(row.original_manifest)
        != row.original_manifest_sha256
        or _validate_source(session, batch, member, file, publication)
        != row.canonical_sha256
    ):
        raise intake.IntakeConflictError("The approved recovery snapshot changed")
    expected = (
        row.replacement_manifest
        if row.replacement_manifest is not None
        else row.original_manifest
    )
    expected_sha = (
        row.replacement_manifest_sha256
        if row.replacement_manifest is not None
        else row.original_manifest_sha256
    )
    if (
        not publication.gate_closed
        or publication.writer_manifest != expected
        or publication.writer_manifest_sha256 != expected_sha
    ):
        raise intake.IntakeConflictError("The retained pending recovery changed")
    return row, publication, file


def load_recovery_inputs(
    owner: FileOwnership, recovery_id: UUID, attempt_id: UUID
) -> OwnedWriterInputs | Literal["completed"] | None:
    from onyx.access.access import get_access_for_user_files
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.db.user_file import (
        fetch_document_set_names_for_user_files,
        fetch_persona_ids_for_user_files,
        fetch_user_project_ids_for_user_files,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexProjectionAccess

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if _complete_committed_recovery(session, owner, recovery_id, attempt_id):
            session.commit()
            return "completed"
        row, _publication, file = _owned_recovery(
            session, owner, recovery_id, attempt_id
        )
        if row.replacement_manifest is not None:
            return None
        settings = get_active_search_settings_list(session)
        for setting in settings:
            _ = setting.cloud_provider
        original = WriterPublicationManifest.model_validate(row.original_manifest)
        current_state = writer_index_state_digest(
            session, [item.search_settings_id for item in original.indexes]
        )
        if (
            original.scope != owner.scope
            or original.index_state_sha256 != current_state
        ):
            raise intake.IntakeConflictError(
                "The original recovery index configuration changed"
            )
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
            canonical=capture_canonical_scope(session, file.id),
            bindings=[],
            revisions={},
            canonical_revisions=archive_canonical_revisions(session, owner),
            cached=PreparedContextView(),
            index_state_sha256=current_state,
        )
        row.status, row.error = "preparing", None
        PublicationStore(owner.scope).renew_in_session(session, owner)
        session.commit()
        session.expunge_all()
        return result


def _complete_committed_recovery(
    session: Session,
    owner: FileOwnership,
    recovery_id: UUID,
    attempt_id: UUID,
) -> bool:
    row = session.get_one(
        RegulatoryDocumentIntakePublicationRecovery,
        recovery_id,
        with_for_update=True,
        populate_existing=True,
    )
    file = session.get_one(UserFile, owner.user_file_id, populate_existing=True)
    if file.status != UserFileStatus.COMPLETED:
        return False
    publication = session.get_one(
        RegulatoryFilePublication, file.id, populate_existing=True
    )
    batch = session.get_one(
        RegulatoryDocumentIntake, row.intake_id, populate_existing=True
    )
    member = session.get_one(
        RegulatoryDocumentIntakeFile, row.document_id, populate_existing=True
    )
    repair = session.get(UserFileProjectionRepair, file.id, populate_existing=True)
    if (
        row.user_file_id != file.id
        or row.replacement_attempt_id != attempt_id
        or row.replacement_manifest is None
        or publication_streaming_digest(row.replacement_manifest)
        != row.replacement_manifest_sha256
        or publication.gate_closed
        or publication.writer_manifest is not None
        or publication.writer_manifest_sha256 is not None
        or publication.epoch <= 0
        or member.approval_attempt_id != attempt_id
        or repair is None
        or repair.attempt_id != attempt_id
        or not repair.initial_index
        or repair.status
        not in {
            UserFileProjectionRepairStatus.RUNNING,
            UserFileProjectionRepairStatus.SUCCEEDED,
        }
        or batch.review_sha256 != row.approved_review_sha256
        or (batch.status, batch.stage)
        not in {
            ("approving", "indexing"),
            ("failed", "indexing"),
            ("completed", "completed"),
        }
        or _validate_source(
            session, batch, member, file, publication, allow_activated_history=True
        )
        != row.canonical_sha256
    ):
        raise intake.IntakeConflictError("The completed recovery proof changed")
    manifest = WriterPublicationManifest.model_validate(row.replacement_manifest)
    active = list(
        session.scalars(
            select(RegulatoryTemporalProjection).where(
                RegulatoryTemporalProjection.user_file_id == file.id,
                RegulatoryTemporalProjection.retired_at.is_(None),
            )
        )
    )
    expected = {
        binding.id: binding.model_dump(mode="json") for binding in manifest.bindings
    }
    if {binding.id for binding in active} != set(expected) or any(
        binding.payload != expected[binding.id] for binding in active
    ):
        raise intake.IntakeConflictError("The committed recovery bindings changed")
    row.status, row.error = "completed", None
    return True


def stage_recovery_replacement(
    owner: FileOwnership,
    recovery_id: UUID,
    attempt_id: UUID,
    manifest: WriterPublicationManifest,
) -> None:
    from onyx.db.regulatory_physical_indexes import validate_physical_index_snapshots

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        row, publication, _file = _owned_recovery(
            session, owner, recovery_id, attempt_id
        )
        if row.replacement_manifest is not None:
            if row.replacement_manifest != manifest.model_dump(mode="json"):
                raise intake.IntakeConflictError(
                    "The prepared replacement is immutable"
                )
            return
        if (
            manifest.scope != owner.scope
            or manifest.user_file_id != owner.user_file_id
            or manifest.kind != "reindex"
            or not manifest.complete_file
            or manifest.canonical_before_sha256 != row.canonical_sha256
            or manifest.previous_binding_ids
            or manifest.amendment_proposal_id is not None
            or manifest.canonical_after is None
            or publication_streaming_digest(
                [item.model_dump(mode="json") for item in manifest.canonical_after]
            )
            != row.canonical_sha256
            or manifest.index_state_sha256
            != writer_index_state_digest(
                session, [item.search_settings_id for item in manifest.indexes]
            )
        ):
            raise intake.IntakeConflictError(
                "The replacement differs from the approved canonical scope"
            )
        allocations = {
            ordinal: key
            for ordinal, key in session.execute(
                select(
                    RegulatoryPublicationOrdinal.ordinal,
                    RegulatoryPublicationOrdinal.allocation_key,
                ).where(RegulatoryPublicationOrdinal.user_file_id == owner.user_file_id)
            ).all()
        }
        if not manifest.bindings or any(
            not allocations.get(binding.projection.ordinal, "").startswith(
                f"intake-recovery:{recovery_id}:"
            )
            for binding in manifest.bindings
        ):
            raise intake.IntakeConflictError(
                "Recovery requires newly reserved projection ordinals"
            )
        validate_physical_index_snapshots(session, manifest.indexes)
        payload = manifest.model_dump(mode="json")
        digest = publication_streaming_digest(payload)
        row.replacement_manifest, row.replacement_manifest_sha256 = payload, digest
        row.status, row.error = "prepared", None
        session.flush([row])
        publication.writer_manifest, publication.writer_manifest_sha256 = (
            payload,
            digest,
        )
        PublicationStore(owner.scope).record_event(session, owner)
        PublicationStore(owner.scope).renew_in_session(session, owner)
        session.commit()


def finish_recovery(
    owner: FileOwnership, recovery_id: UUID, *, error: str | None = None
) -> None:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        row = session.get_one(
            RegulatoryDocumentIntakePublicationRecovery,
            recovery_id,
            with_for_update=True,
            populate_existing=True,
        )
        file = session.get_one(UserFile, owner.user_file_id, populate_existing=True)
        publication = session.get_one(
            RegulatoryFilePublication, owner.user_file_id, populate_existing=True
        )
        if row.user_file_id != file.id:
            raise intake.IntakeConflictError("Recovery completion file changed")
        if _complete_committed_recovery(
            session, owner, recovery_id, row.replacement_attempt_id
        ):
            session.commit()
            return
        if error is None and (
            file.status != UserFileStatus.COMPLETED
            or publication.gate_closed
            or publication.writer_manifest is not None
        ):
            raise intake.IntakeConflictError(
                "Recovery has no completed verified publication"
            )
        row.status, row.error = (
            ("failed", error[:1000]) if error else ("completed", None)
        )
        session.commit()
