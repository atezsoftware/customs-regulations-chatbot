"""Admission and review are bounded to files created by one new-document import."""

import datetime
import hashlib
import json
from html import unescape
from typing import Any
from uuid import UUID, uuid4, uuid5

from mistune.helpers import unescape_char
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from onyx.db.amendment_sources import require_ready_source_package
from onyx.db.document_intake_graph import (
    require_frozen_zip_intake_graph,
    require_frozen_zip_members,
)
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.labeling_configuration import (
    resolve_intake_labeling_provider_binding,
    select_intake_labeling_provider_binding,
)
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.db.regulatory_chunks import get_chunks_for_file_page_snapshot
from onyx.db.user_file import (
    claim_user_file_projection_repair,
    get_user_file_projection_repair,
)
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import INTAKE_DELIVERY_SECONDS
from onyx.regulatory.document_intake.source_graph import IntakeGraph

TARGET_DOCUMENT_SET_NAME = "PC Külliyatı"


class IntakeConflictError(RuntimeError):
    pass


def intake_source_links(
    session: Session, package_id: UUID, user: User, environment: str
) -> list[SourceLink]:
    from onyx.file_store.file_store import get_default_file_store

    target = resolve_intake_target(session, user)
    package = session.get(AmendmentSourcePackage, package_id)
    if (
        package is None
        or package.document_set_id != target.id
        or package.environment != environment
        or package.created_by != user.id
    ):
        raise ValueError("Source package not found")
    if not package.manifest_file_id or not package.manifest_sha256:
        return []
    with get_default_file_store().read_file(package.manifest_file_id) as stream:
        content = stream.read(150 * 1024 * 1024 + 1)
    if (
        len(content) > 150 * 1024 * 1024
        or hashlib.sha256(content).hexdigest() != package.manifest_sha256
    ):
        raise IntakeConflictError("Source manifest integrity changed")
    return [SourceLink.model_validate(value) for value in json.loads(content)["links"]]


def resolve_intake_target(session: Session, user: User) -> DocumentSet:
    targets = list(
        session.scalars(
            select(DocumentSet).where(DocumentSet.name == TARGET_DOCUMENT_SET_NAME)
        )
    )
    if len(targets) != 1:
        raise ValueError(
            "Exactly one PC Külliyatı document set must exist before importing"
        )
    target = get_document_set_by_id_for_user(
        db_session=session, document_set_id=targets[0].id, user=user, get_editable=True
    )
    if target is None:
        raise ValueError("PC Külliyatı is not accessible for editing")
    return target


def create_intake(
    session: Session,
    *,
    source_package_id: UUID,
    idempotency_key: UUID,
    labeling_model_configuration_id: int | None = None,
    user: User,
    environment: str,
) -> tuple[RegulatoryDocumentIntake, bool]:
    target = resolve_intake_target(session, user)
    session.get(DocumentSet, target.id, with_for_update=True)
    package = require_ready_source_package(
        session,
        package_id=source_package_id,
        document_set_id=target.id,
        environment=environment,
    )
    if package.created_by != user.id:
        raise ValueError("Source package belongs to another user")
    existing = session.scalar(
        select(RegulatoryDocumentIntake).where(
            or_(
                RegulatoryDocumentIntake.idempotency_key == idempotency_key,
                RegulatoryDocumentIntake.source_package_id == package.id,
            )
        )
    )
    if existing is not None:
        if (
            existing.created_by != user.id
            or existing.source_package_id != package.id
            or (
                labeling_model_configuration_id is not None
                and existing.labeling_model_configuration_id
                != labeling_model_configuration_id
            )
        ):
            raise IntakeConflictError(
                "This request already belongs to a different intake"
            )
        return existing, False
    if labeling_model_configuration_id is None:
        labeling_model_configuration_id = select_intake_labeling_provider_binding(
            session, user=user
        ).model_configuration_id
    else:
        resolve_intake_labeling_provider_binding(
            session, labeling_model_configuration_id, user=user
        )
    row = RegulatoryDocumentIntake(
        source_package_id=package.id,
        document_set_id=target.id,
        created_by=user.id,
        idempotency_key=idempotency_key,
        labeling_model_configuration_id=labeling_model_configuration_id,
        status="preparing",
        stage="chunking",
    )
    session.add(row)
    session.flush()
    return row, True


def get_intake(
    session: Session, intake_id: UUID, user: User
) -> RegulatoryDocumentIntake:
    target = resolve_intake_target(session, user)
    row = session.scalar(
        select(RegulatoryDocumentIntake).where(
            RegulatoryDocumentIntake.id == intake_id,
            RegulatoryDocumentIntake.document_set_id == target.id,
            RegulatoryDocumentIntake.created_by == user.id,
        )
    )
    if row is None:
        raise ValueError("New document intake not found")
    return row


def list_intakes(session: Session, user: User) -> list[RegulatoryDocumentIntake]:
    target = resolve_intake_target(session, user)
    return list(
        session.scalars(
            select(RegulatoryDocumentIntake)
            .where(
                RegulatoryDocumentIntake.document_set_id == target.id,
                RegulatoryDocumentIntake.created_by == user.id,
            )
            .order_by(RegulatoryDocumentIntake.created_at.desc())
            .limit(100)
        )
    )


def intake_files(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[RegulatoryDocumentIntakeFile]:
    return list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .order_by(
                RegulatoryDocumentIntakeFile.role.desc(),
                RegulatoryDocumentIntakeFile.id,
            )
        )
    )


def intake_preparation_work(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[tuple[UUID, UUID, UUID, str, str]]:
    return [
        (
            member.id,
            member.user_file_id,
            member.source_asset_id,
            member.status,
            session.get_one(UserFile, member.user_file_id).file_id,
        )
        for member in intake_files(session, batch)
    ]


def pending_intake_file_ids(
    session: Session, file_ids: list[UUID], *, include_approving: bool = False
) -> set[UUID]:
    """Ordinary upload indexing must not bypass the Updates admission review."""
    if not file_ids:
        return set()
    from onyx.db.document_intake_review import source_evidence_file_ids

    return source_evidence_file_ids(session, file_ids) | set(
        session.scalars(
            select(RegulatoryDocumentIntakeFile.user_file_id)
            .join(
                RegulatoryDocumentIntake,
                RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
            )
            .where(
                RegulatoryDocumentIntakeFile.user_file_id.in_(file_ids),
                RegulatoryDocumentIntake.status.not_in(
                    ("completed",) if include_approving else ("approving", "completed")
                ),
            )
        )
    )


def _labels(session: Session, batch: RegulatoryDocumentIntake) -> dict[str, list[str]]:
    if batch.labeling_run_id is None:
        return {}
    return {
        item.regulatory_chunk_id: list(item.labels)
        for item in session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == batch.labeling_run_id
            )
        )
    }


def review_digest(session: Session, batch: RegulatoryDocumentIntake) -> str:
    from onyx.db.document_lineage import lineage_review_payload

    members = intake_files(session, batch)
    ids = [item.user_file_id for item in members]
    membership = set(
        session.scalars(
            select(DocumentSet__UserFile.user_file_id).where(
                DocumentSet__UserFile.document_set_id == batch.document_set_id,
                DocumentSet__UserFile.user_file_id.in_(ids),
            )
        )
    )
    from onyx.db.document_intake_review import (
        publication_decision_payload,
        source_evidence_file_ids,
    )

    expected_membership = set(ids)
    if batch.stage in {"indexing", "completed", "lineage"} and batch.review_sha256:
        expected_membership -= source_evidence_file_ids(session, ids)
    if not ids or membership != expected_membership:
        raise IntakeConflictError("The imported document membership changed")
    chunks = list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id.in_(ids))
            .order_by(RegulatoryChunk.id)
        )
    )
    labels = _labels(session, batch)
    payload = {
        "source_package_id": str(batch.source_package_id),
        "files": [
            {
                "id": str(item.user_file_id),
                "asset": str(item.source_asset_id),
                "role": item.role,
                "parent_document_id": str(item.parent_document_id)
                if item.parent_document_id
                else None,
                "relationships": item.relationships,
            }
            for item in members
        ],
        "chunks": [
            {
                "id": row.id,
                "file_id": str(row.user_file_id),
                "text": row.text,
                "chunk_type": row.chunk_type,
                "position": row.position,
                "metadata": row.chunk_metadata,
                "headings": list(row.heading_path),
                "status": row.status,
                "start": str(row.validity_start_date),
                "end": str(row.validity_end_date),
                "labels": labels.get(row.id, []),
            }
            for row in chunks
        ],
    }
    if any("intake_display_name" in row.chunk_metadata for row in chunks):
        payload["display_names"] = [
            {"id": str(file_id), "name": name}
            for file_id, name in session.execute(
                select(UserFile.id, UserFile.name)
                .where(UserFile.id.in_(ids))
                .order_by(UserFile.id)
            )
        ]
    decisions = publication_decision_payload(session, batch)
    if decisions:
        from onyx.db.document_intake_review import validate_publication_review

        validate_publication_review(session, batch, require_current_parent=False)
        payload["publication_review"] = decisions
    lineage = lineage_review_payload(session, batch)
    if lineage:
        payload["document_lineage"] = lineage
    return publication_digest(payload)


def refresh_intake(session: Session, batch: RegulatoryDocumentIntake) -> None:
    from onyx.db.document_intake_review import effective_intake_files

    # Polling must re-read the worker's lease under the same lock as transitions.
    session.refresh(batch, with_for_update=True)
    if (
        batch.status == "preparing"
        and batch.stage
        in {
            "chunking",
            "metadata",
            "reprepare",
        }
        and not (
            batch.stage in {"chunking", "reprepare"}
            and batch.lease_token is None
            and batch.lease_expires_at is None
        )
    ):
        deadline = batch.lease_expires_at or (
            batch.updated_at + datetime.timedelta(seconds=INTAKE_DELIVERY_SECONDS)
        )
        if deadline < datetime.datetime.now(datetime.timezone.utc):
            batch.status, batch.error = (
                "failed",
                "Preparation worker lease expired; retry this import",
            )
            batch.lease_token, batch.lease_expires_at = None, None
    if (
        batch.stage == "labeling"
        and batch.status == "preparing"
        and batch.labeling_run_id
    ):
        run = session.get(RegulatoryLabelingRun, batch.labeling_run_id)
        if run is None:
            batch.status, batch.error = "failed", "The labeling run is unavailable"
        elif run.status == "completed":
            expected = {
                str(item.user_file_id)
                for item in effective_intake_files(session, batch)
            }
            if (
                set(run.file_ids) != expected
                or run.failed_chunks
                or run.stale_chunks
                or run.unresolved_derived_chunks
            ):
                batch.status, batch.error = (
                    "partial",
                    "Labels do not cover the exact imported files",
                )
            else:
                batch.status, batch.stage = "ready", "review"
                batch.error = None
                batch.review_sha256 = review_digest(session, batch)
                for item in effective_intake_files(session, batch):
                    item.status = "ready"
        elif run.status in {"failed", "completed_with_errors", "cancelled"}:
            batch.status, batch.error = (
                "partial",
                run.error or "Labeling has unresolved or failed chunks",
            )
        elif run.status in {"queued", "running"}:
            batch.error = None
    if batch.stage == "indexing" and batch.status in {"approving", "failed"}:
        files = effective_intake_files(session, batch)
        for item in files:
            file = session.get(UserFile, item.user_file_id)
            if file is None or file.status in {
                UserFileStatus.FAILED,
                UserFileStatus.CANCELED,
                UserFileStatus.DELETING,
            }:
                item.status, item.error = (
                    "failed",
                    "The imported file could not be indexed",
                )
            elif file.status == UserFileStatus.COMPLETED:
                item.status, item.error = "completed", None
            else:
                repair = get_user_file_projection_repair(session, item.user_file_id)
                if repair and repair.status == UserFileProjectionRepairStatus.FAILED:
                    item.status, item.error = (
                        "failed",
                        repair.failure_code or "Indexing failed",
                    )
        if files and all(item.status == "completed" for item in files):
            batch.status, batch.stage = "completed", "completed"
        elif any(item.status == "failed" for item in files):
            batch.status, batch.error = (
                "failed",
                "Some new documents could not be indexed; retry is available",
            )
    if batch.status == "completed":
        from onyx.db.document_lineage import activate_intake_lineage

        try:
            activate_intake_lineage(session, batch)
        except ValueError as exc:
            batch.status, batch.stage, batch.error = "blocked", "lineage", str(exc)
    session.flush()


def intake_snapshot(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[str, Any]:
    from onyx.db.document_intake_edit import edit_review_digest
    from onyx.db.document_lineage import intake_lineage_selections
    from onyx.regulatory.document_intake.models import IntakeSourceAsset
    from onyx.regulatory.document_intake.text import attachment_reference_label

    refresh_intake(session, batch)
    from onyx.db.document_intake_review import (
        effective_intake_files,
        effective_parent_document_ids,
        latest_publication_decisions,
        publication_review_digest,
    )

    lineage_selections = intake_lineage_selections(session, batch)
    decisions = latest_publication_decisions(session, batch.id)
    labels = _labels(session, batch)
    members = intake_files(session, batch)
    publication_parents = effective_parent_document_ids(members, decisions)
    failed_counts = intake_labeling_failed_counts(session, batch)
    assets = {
        member.source_asset_id: session.get_one(
            RegulatorySourceAsset, member.source_asset_id
        )
        for member in members
    }
    by_hash = {assets[member.source_asset_id].sha256: member for member in members}
    asset_views = [IntakeSourceAsset.model_validate(asset) for asset in assets.values()]
    documents = []
    for member in members:
        file = session.get(UserFile, member.user_file_id)
        asset = assets[member.source_asset_id]
        chunk_ids = list(
            session.scalars(
                select(RegulatoryChunk.id).where(
                    RegulatoryChunk.user_file_id == member.user_file_id,
                    RegulatoryChunk.status == "active",
                )
            )
        )
        document_links = []
        for raw_link in member.relationships:
            link = SourceLink.model_validate(raw_link)
            target = by_hash.get(link.target_asset_hash or "")
            if link.parent_asset_hash != asset.sha256 or target is None:
                continue
            document_links.append(
                {
                    "href": unescape(
                        unescape_char(link.original_url or link.requested_url or "")
                    ),
                    "label": attachment_reference_label(
                        link, package_assets=asset_views
                    )[:500]
                    or assets[target.source_asset_id].display_name,
                    "target_document_id": str(target.id),
                }
            )
        decision = decisions.get(member.id)
        documents.append(
            {
                "id": str(member.id),
                "asset_id": str(asset.id),
                "display_name": file.name if file else asset.display_name,
                "source_display_name": asset.display_name,
                "mime_type": asset.mime_type,
                "role": member.role,
                "parent_document_id": str(publication_parents.get(member.id))
                if publication_parents.get(member.id)
                else None,
                "source_url": asset.final_url or asset.original_url,
                "status": (
                    "source_evidence"
                    if decision and decision.disposition == "source_evidence"
                    else member.status
                )
                if file
                else "deleted",
                "publication_disposition": decision.disposition
                if decision
                else "legislation",
                "duplicate_of_document_id": str(decision.duplicate_of_document_id)
                if decision and decision.duplicate_of_document_id
                else None,
                "existing_parent": decision.existing_parent_snapshot
                if decision and decision.existing_parent_file_id
                else None,
                "publication_review": {
                    "decision_id": str(decision.id),
                    "reason": decision.reason,
                    "reviewed_at": decision.created_at,
                }
                if decision
                else None,
                "chunk_count": len(chunk_ids),
                "labeling_failed_count": failed_counts.get(member.user_file_id, 0),
                "labels": {
                    "labels": sorted(
                        {
                            label
                            for identifier in chunk_ids
                            for label in labels.get(identifier, [])
                        }
                    )
                },
                "error": member.error
                if file
                else "The imported document was deleted; its frozen source is retained",
                "user_file_id": str(member.user_file_id),
                "document_links": document_links,
                "lineage_review": lineage_selections.get(member.id),
            }
        )
    package = session.get_one(AmendmentSourcePackage, batch.source_package_id)
    return {
        "id": str(batch.id),
        "source_package_id": str(batch.source_package_id),
        "status": batch.status,
        "stage": ("queued_reprepare" if batch.stage == "reprepare" else "queued")
        if batch.status == "preparing"
        and batch.stage in {"chunking", "reprepare"}
        and batch.lease_token is None
        and batch.lease_expires_at is None
        else batch.stage,
        "target_document_set_id": batch.document_set_id,
        "target_document_set_name": TARGET_DOCUMENT_SET_NAME,
        "review_sha256": batch.review_sha256,
        "edit_review_sha256": edit_review_digest(session, batch),
        "publication_review_sha256": publication_review_digest(session, batch),
        "active_document_count": len(effective_intake_files(session, batch)),
        "issues": list(package.issues),
        "error": batch.error,
        "documents": documents,
        "labeling_failed_count": sum(failed_counts.values()),
        "labeling_run_id": str(batch.labeling_run_id)
        if batch.labeling_run_id
        else None,
        "created_at": batch.created_at,
        "updated_at": batch.updated_at,
    }


def intake_chunk_page(
    session: Session,
    batch: RegulatoryDocumentIntake,
    document_id: UUID,
    offset: int,
    limit: int,
) -> tuple[list[tuple[RegulatoryChunk, list[str]]], int]:
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.id == document_id,
            RegulatoryDocumentIntakeFile.intake_id == batch.id,
        )
    )
    if member is None:
        raise ValueError("Intake document not found")
    rows, total = get_chunks_for_file_page_snapshot(
        session, member.user_file_id, offset=offset, limit=limit
    )
    labels = _labels(session, batch)
    return [(row, labels.get(row.id, [])) for row in rows], total


def intake_labeling_failed_counts(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[UUID, int]:
    """Count current owned failures across all pages, including abstained targets."""
    if batch.labeling_run_id is None:
        return {}
    member_files = select(RegulatoryDocumentIntakeFile.user_file_id).where(
        RegulatoryDocumentIntakeFile.intake_id == batch.id
    )
    rows = session.execute(
        select(RegulatoryLabelingItem.user_file_id, func.count())
        .join(
            RegulatoryChunk,
            (RegulatoryChunk.id == RegulatoryLabelingItem.regulatory_chunk_id)
            & (RegulatoryChunk.user_file_id == RegulatoryLabelingItem.user_file_id),
        )
        .where(
            RegulatoryLabelingItem.run_id == batch.labeling_run_id,
            RegulatoryLabelingItem.user_file_id.in_(member_files),
            RegulatoryLabelingItem.status == "failed",
            RegulatoryChunk.status == "active",
        )
        .group_by(RegulatoryLabelingItem.user_file_id)
    )
    return {file_id: count for file_id, count in rows}


def intake_chunk_labeling_diagnostics(
    session: Session,
    batch: RegulatoryDocumentIntake,
    chunk_ids: list[str],
) -> dict[str, dict[str, str | None]]:
    """Expose only the current import's visible labeling outcomes, without mutation."""
    if batch.labeling_run_id is None or not chunk_ids:
        return {}
    member_files = select(RegulatoryDocumentIntakeFile.user_file_id).where(
        RegulatoryDocumentIntakeFile.intake_id == batch.id
    )
    rows = session.execute(
        select(
            RegulatoryLabelingItem.regulatory_chunk_id,
            RegulatoryLabelingItem.status,
            RegulatoryLabelingItem.error,
        ).where(
            RegulatoryLabelingItem.run_id == batch.labeling_run_id,
            RegulatoryLabelingItem.regulatory_chunk_id.in_(chunk_ids),
            RegulatoryLabelingItem.user_file_id.in_(member_files),
        )
    )
    return {
        chunk_id: {"status": status, "error": error} for chunk_id, status, error in rows
    }


def approve_intake(
    session: Session, batch: RegulatoryDocumentIntake, expected_review_sha256: str
) -> list[tuple[UUID, UUID]]:
    session.refresh(batch, with_for_update=True)
    refresh_intake(session, batch)
    if batch.status in {"completed", "approving"}:
        return []
    if (
        batch.status != "ready"
        or batch.review_sha256 != expected_review_sha256
        or review_digest(session, batch) != expected_review_sha256
    ):
        raise IntakeConflictError(
            "The ready document review changed; reload before approving"
        )
    from onyx.db.document_intake_review import (
        detach_source_evidence_memberships,
        effective_intake_files,
        validate_publication_review,
    )
    from onyx.db.document_lineage import freeze_intake_lineage

    validate_publication_review(session, batch)
    freeze_intake_lineage(session, batch)
    deliveries = []
    for member in effective_intake_files(session, batch):
        file = session.get_one(UserFile, member.user_file_id, with_for_update=True)
        if file.status != UserFileStatus.CHUNKED:
            raise IntakeConflictError(
                "Only the reviewed, unpublished imported files can be approved"
            )
        token = claim_user_file_projection_repair(session, file.id, initial_index=True)
        if token is None:
            raise IntakeConflictError(
                "The imported file already has an indexing request"
            )
        member.approval_attempt_id, member.status = token, "indexing"
        deliveries.append((file.id, token))
    detach_source_evidence_memberships(session, batch)
    batch.status, batch.stage, batch.error = "approving", "indexing", None
    session.flush()
    return deliveries


def retry_intake(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[tuple[UUID, UUID]]:
    from onyx.db.document_intake_review import (
        effective_intake_files,
        has_active_publication_decisions,
    )

    session.refresh(batch, with_for_update=True)
    if batch.status == "blocked" and batch.stage == "lineage":
        batch.status, batch.stage, batch.error = "completed", "completed", None
        session.flush()
        refresh_intake(session, batch)
        return []
    if batch.status not in {"failed", "partial"}:
        raise IntakeConflictError("Only failed or incomplete intakes can be retried")
    if batch.stage == "reprepare":
        batch.status, batch.error = "preparing", None
        batch.lease_token, batch.lease_expires_at = None, None
        session.flush()
        return []
    if batch.stage == "indexing":
        if (
            not batch.review_sha256
            or review_digest(session, batch) != batch.review_sha256
        ):
            raise IntakeConflictError("The approved document snapshot changed")
        from onyx.db.document_intake_indexing_retry import (
            log_failed_intake_index_retry,
            restore_failed_intake_index_request,
        )

        retry_audits: list[tuple[dict[str, object], UUID]] = []
        with session.begin_nested():
            deliveries = []
            for member in effective_intake_files(session, batch):
                file = session.get_one(
                    UserFile, member.user_file_id, with_for_update=True
                )
                if file.status == UserFileStatus.COMPLETED:
                    continue
                if file.status == UserFileStatus.FAILED:
                    from onyx.db.document_intake_publication_recovery import (
                        pending_failed_intake_publication,
                        request_intake_publication_recovery,
                    )

                    if pending_failed_intake_publication(session, file.id):
                        token = request_intake_publication_recovery(
                            session, batch, member, file
                        )
                        member.approval_attempt_id, member.status, member.error = (
                            token,
                            "indexing",
                            None,
                        )
                        deliveries.append((file.id, token))
                        continue
                retry_evidence = (
                    restore_failed_intake_index_request(session, batch, member, file)
                    if file.status == UserFileStatus.FAILED
                    else None
                )
                repair = get_user_file_projection_repair(session, member.user_file_id)
                if repair and repair.status == UserFileProjectionRepairStatus.RUNNING:
                    continue
                if repair and repair.status == UserFileProjectionRepairStatus.PENDING:
                    from onyx.db.document_intake_publication_recovery import (
                        associate_pending_recovery,
                    )

                    associate_pending_recovery(session, batch, member, file, repair)
                    token = repair.attempt_id
                else:
                    token = claim_user_file_projection_repair(
                        session, file.id, initial_index=True
                    )
                if token is None:
                    raise IntakeConflictError(
                        "The imported file cannot currently be retried"
                    )
                member.approval_attempt_id, member.status, member.error = (
                    token,
                    "indexing",
                    None,
                )
                deliveries.append((file.id, token))
                if retry_evidence is not None:
                    retry_audits.append((retry_evidence, token))
            batch.status, batch.error = "approving", None
            session.flush()
        for evidence, token in retry_audits:
            log_failed_intake_index_retry(evidence, token)
        return deliveries
    if has_active_publication_decisions(session, batch):
        from onyx.db.document_intake_review import _replace_labels

        _replace_labels(session, batch, set())
        return []
    batch.status, batch.stage, batch.error = "preparing", "chunking", None
    batch.lease_token, batch.lease_expires_at = None, None
    session.flush()
    return []


def claim_intake(session: Session, intake_id: UUID) -> UUID | None:
    batch = session.get(
        RegulatoryDocumentIntake,
        intake_id,
        with_for_update=True,
        populate_existing=True,
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    if (
        batch is None
        or batch.status != "preparing"
        or batch.stage != "chunking"
        or batch.lease_token is not None
        or batch.lease_expires_at is not None
    ):
        return None
    batch.lease_token, batch.lease_expires_at = (
        uuid4(),
        now + datetime.timedelta(minutes=10),
    )
    batch.error = None
    session.flush()
    return batch.lease_token


def admit_intake_assets(
    session: Session,
    batch: RegulatoryDocumentIntake,
    assets: list[RegulatorySourceAsset],
    links: list[SourceLink],
    *,
    root_hash: str,
    owned_file_ids: dict[UUID, str] | None = None,
    intake_graph: IntakeGraph | None = None,
    source_manifest_bytes: bytes | None = None,
) -> list[RegulatoryDocumentIntakeFile]:
    package = session.get_one(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    if (
        package.document_set_id != batch.document_set_id
        or package.created_by != batch.created_by
        or package.status != "ready"
        or package.issues
    ):
        raise IntakeConflictError("The verified source package scope changed")
    hashes = {asset.sha256 for asset in assets}
    if (
        not assets
        or len(hashes) != len(assets)
        or root_hash not in hashes
        or any(asset.package_id != package.id for asset in assets)
    ):
        raise IntakeConflictError("The source assets or root do not match this package")
    stored_ids = set(
        session.scalars(
            select(RegulatorySourceAsset.id).where(
                RegulatorySourceAsset.package_id == package.id
            )
        )
    )
    if (
        package.asset_count != len(assets)
        or stored_ids != {asset.id for asset in assets}
        or any(
            link.parent_asset_hash not in hashes or link.target_asset_hash not in hashes
            for link in links
        )
    ):
        raise IntakeConflictError("The source graph does not cover the frozen package")
    try:
        graph = require_frozen_zip_intake_graph(
            session,
            package,
            assets,
            links,
            intake_graph=intake_graph,
            manifest_bytes=source_manifest_bytes,
        )
    except ValueError as error:
        raise IntakeConflictError(str(error)) from error
    if (
        session.get_one(DocumentSet, batch.document_set_id).name
        != TARGET_DOCUMENT_SET_NAME
    ):
        raise IntakeConflictError("The target PC Külliyatı document set changed")
    existing = intake_files(session, batch)
    if existing:
        if {item.source_asset_id for item in existing} != {
            asset.id for asset in assets
        }:
            raise IntakeConflictError("The frozen source asset membership changed")
        if graph is not None:
            try:
                require_frozen_zip_members(existing, assets, links, graph)
            except ValueError as error:
                raise IntakeConflictError(str(error)) from error
        return existing
    if owned_file_ids is not None and (
        set(owned_file_ids) != {asset.id for asset in assets}
        or len(set(owned_file_ids.values())) != len(assets)
        or set(owned_file_ids.values()).intersection(asset.file_id for asset in assets)
    ):
        raise IntakeConflictError(
            "Every imported original requires separate owned storage"
        )
    roots = set(graph.root_hashes) if graph is not None else {root_hash}
    documents = {
        asset.sha256: RegulatoryDocumentIntakeFile(
            id=uuid4(),
            intake_id=batch.id,
            source_asset_id=asset.id,
            user_file_id=uuid4(),
            role="root" if asset.sha256 in roots else "annex",
            relationships=[
                link.model_dump(mode="json")
                for link in links
                if link.target_asset_hash == asset.sha256
                or link.parent_asset_hash == asset.sha256
            ],
            status="preparing",
        )
        for asset in assets
    }
    for asset in assets:
        member = documents[asset.sha256]
        session.add(
            UserFile(
                id=member.user_file_id,
                user_id=batch.created_by,
                file_id=owned_file_ids[asset.id]
                if owned_file_ids is not None
                else asset.file_id,
                name=asset.display_name,
                file_type=asset.mime_type,
                content_type=asset.mime_type,
                link_url=asset.final_url or asset.original_url,
                status=UserFileStatus.CHUNKED,
            )
        )
    session.flush()
    session.add_all(documents.values())
    session.flush()
    for digest, member in documents.items():
        incoming = next(
            (
                link
                for link in links
                if link.target_asset_hash == digest and link.parent_asset_hash != digest
            ),
            None,
        )
        if graph is not None:
            parent_hash = graph.primary_parent_by_hash.get(digest)
            member.parent_document_id = (
                documents[parent_hash].id if parent_hash is not None else None
            )
        elif incoming is not None and member.role != "root":
            member.parent_document_id = documents[incoming.parent_asset_hash].id
        session.add(
            DocumentSet__UserFile(
                document_set_id=batch.document_set_id, user_file_id=member.user_file_id
            )
        )
    session.flush()
    return list(documents.values())


def load_intake_job(
    session: Session, intake_id: UUID, token: UUID
) -> tuple[
    RegulatoryDocumentIntake, AmendmentSourcePackage, list[RegulatorySourceAsset]
]:
    batch = session.get_one(RegulatoryDocumentIntake, intake_id)
    if batch.lease_token != token or batch.status != "preparing":
        raise IntakeConflictError("The intake preparation lease changed")
    package = session.get_one(AmendmentSourcePackage, batch.source_package_id)
    assets = list(
        session.scalars(
            select(RegulatorySourceAsset).where(
                RegulatorySourceAsset.package_id == package.id
            )
        )
    )
    return batch, package, assets


def mark_intake_file_prepared(
    session: Session, intake_id: UUID, token: UUID, document_id: UUID
) -> None:
    batch, _package, _assets = load_intake_job(session, intake_id, token)
    member = session.get_one(RegulatoryDocumentIntakeFile, document_id)
    if member.intake_id != batch.id:
        raise IntakeConflictError("The document belongs to another intake")
    member.status, member.error = "chunked", None


def start_intake_labeling(
    session: Session, intake_id: UUID, token: UUID
) -> UUID | None:
    batch, _package, _assets = load_intake_job(session, intake_id, token)
    session.refresh(batch, with_for_update=True)
    if batch.lease_token != token or batch.status != "preparing":
        raise IntakeConflictError("The intake preparation lease changed")
    return _start_intake_labeling_for_batch(session, batch)


def resume_intake_labeling(session: Session, intake_id: UUID) -> UUID | None:
    batch = session.get_one(
        RegulatoryDocumentIntake,
        intake_id,
        with_for_update=True,
        populate_existing=True,
    )
    if (
        batch.status != "preparing"
        or batch.stage != "waiting_labeling"
        or batch.lease_token is not None
        or batch.lease_expires_at is not None
    ):
        return None
    return _start_intake_labeling_for_batch(session, batch)


def _start_intake_labeling_for_batch(
    session: Session, batch: RegulatoryDocumentIntake
) -> UUID | None:
    from onyx.db import regulatory_labeling as labeling
    from onyx.db.document_intake_review import effective_intake_files

    files = effective_intake_files(session, batch)
    if not files or any(item.status != "chunked" for item in files):
        raise IntakeConflictError("Every new document must be chunked before labeling")
    existing = (
        session.get(RegulatoryLabelingRun, batch.labeling_run_id)
        if batch.labeling_run_id
        else None
    )
    from onyx.db.document_intake_reprepare import latest_reprepare

    reprepare = latest_reprepare(session, batch.id)
    prior_reprepare_run = (
        session.get(RegulatoryLabelingRun, reprepare.previous_labeling_run_id)
        if reprepare is not None and reprepare.previous_labeling_run_id
        else None
    )
    if (
        existing
        and existing.status in {"queued", "running", "completed"}
        and set(existing.file_ids) == {str(member.user_file_id) for member in files}
    ):
        run = existing
    else:
        session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
        if labeling.get_active_run_id(session, batch.document_set_id) is not None:
            batch.stage, batch.error = "waiting_labeling", None
            batch.lease_token, batch.lease_expires_at = None, None
            session.flush()
            return None
        user = session.get_one(User, batch.created_by)
        binding = resolve_intake_labeling_provider_binding(
            session, batch.labeling_model_configuration_id, user=user
        )
        run, _created = labeling.create_labeling_run(
            session,
            document_set_id=batch.document_set_id,
            taxonomy=existing.taxonomy
            if existing is not None
            else prior_reprepare_run.taxonomy
            if prior_reprepare_run is not None
            else labeling.get_label_settings(session).taxonomy,
            model_configuration_id=batch.labeling_model_configuration_id,
            model=binding.model,
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=batch.created_by,
            idempotency_key=uuid5(
                batch.id,
                str(existing.id)
                if existing
                else "reprepare:" + str(reprepare.id)
                if reprepare is not None
                else "labels",
            ),
            uses_current_labels=existing.uses_current_labels
            if existing is not None
            else prior_reprepare_run.uses_current_labels
            if prior_reprepare_run is not None
            else True,
            target_file_ids=[item.user_file_id for item in files],
        )
        if existing is not None:
            from onyx.db.document_intake_edit import reuse_completed_intake_labels

            reuse_completed_intake_labels(session, previous=existing, replacement=run)
    batch.labeling_run_id, batch.stage = run.id, "labeling"
    batch.lease_token, batch.lease_expires_at = None, None
    for item in files:
        item.status = "labeling"
    session.flush()
    return run.id


def fail_intake(
    session: Session, intake_id: UUID, token: UUID | None, message: str
) -> None:
    batch = session.get(RegulatoryDocumentIntake, intake_id, with_for_update=True)
    if batch is None or (token is not None and batch.lease_token != token):
        return
    batch.status, batch.error = "failed", message[:1000]
    batch.lease_token, batch.lease_expires_at = None, None


def mark_intake_label_dispatch_failed(session: Session, intake_id: UUID) -> None:
    batch = session.get(
        RegulatoryDocumentIntake,
        intake_id,
        with_for_update=True,
        populate_existing=True,
    )
    if batch is not None and batch.status == "preparing" and batch.stage == "labeling":
        # The committed labeling run is the durable intent; its recovery task
        # redelivers a lost broker hint without preparing these files again.
        batch.error = "Labeling delivery is pending; automatic recovery will retry"


def mark_intake_dispatch_failed(
    session: Session, batch: RegulatoryDocumentIntake, message: str
) -> None:
    current = session.get(
        RegulatoryDocumentIntake,
        batch.id,
        with_for_update=True,
        populate_existing=True,
    )
    if (
        current is None
        or current.status != "preparing"
        or current.stage not in {"chunking", "reprepare"}
        or current.lease_token is not None
    ):
        return
    current.error = message[:1000]
    session.flush()


def mark_intake_index_dispatch_failed(
    session: Session, batch: RegulatoryDocumentIntake, message: str
) -> None:
    from onyx.db.enums import UserFileProjectionRepairStatus
    from onyx.db.models import UserFileProjectionRepair

    current = session.get(
        RegulatoryDocumentIntake,
        batch.id,
        with_for_update=True,
        populate_existing=True,
    )
    if current is None or current.status != "approving":
        return
    for member in intake_files(session, current):
        request = session.get(UserFileProjectionRepair, member.user_file_id)
        if (
            request is not None
            and request.attempt_id == member.approval_attempt_id
            and request.initial_index
            and request.status == UserFileProjectionRepairStatus.PENDING
        ):
            current.status, current.error = "failed", message[:1000]
            session.flush()
            return
