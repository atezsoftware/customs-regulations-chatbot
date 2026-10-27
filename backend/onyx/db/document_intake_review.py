"""Explicit, retained decisions about publication scope and attachment identity."""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, aliased

from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
    RegulatoryDocumentLineageReview,
    RegulatoryFilePublication,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.document_index.publication_models import publication_digest

PublicationDisposition = Literal["legislation", "source_evidence"]


def latest_publication_decisions(
    session: Session, batch_id: UUID
) -> dict[UUID, RegulatoryDocumentIntakeReviewDecision]:
    rows = session.scalars(
        select(RegulatoryDocumentIntakeReviewDecision)
        .where(RegulatoryDocumentIntakeReviewDecision.intake_id == batch_id)
        .order_by(RegulatoryDocumentIntakeReviewDecision.sequence)
        .execution_options(populate_existing=True)
    )
    return {row.document_id: row for row in rows}


def effective_intake_files(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[RegulatoryDocumentIntakeFile]:
    from onyx.db.document_intake import intake_files

    decisions = latest_publication_decisions(session, batch.id)
    return [
        member
        for member in intake_files(session, batch)
        if member.id not in decisions
        or decisions[member.id].disposition != "source_evidence"
    ]


def source_evidence_file_ids(session: Session, file_ids: list[UUID]) -> set[UUID]:
    if not file_ids:
        return set()
    newest = (
        select(
            RegulatoryDocumentIntakeReviewDecision.document_id,
            func.max(RegulatoryDocumentIntakeReviewDecision.sequence).label("sequence"),
        )
        .group_by(RegulatoryDocumentIntakeReviewDecision.document_id)
        .subquery()
    )
    return set(
        session.scalars(
            select(RegulatoryDocumentIntakeFile.user_file_id)
            .join(
                RegulatoryDocumentIntakeReviewDecision,
                RegulatoryDocumentIntakeReviewDecision.document_id
                == RegulatoryDocumentIntakeFile.id,
            )
            .join(
                newest,
                (
                    newest.c.document_id
                    == RegulatoryDocumentIntakeReviewDecision.document_id
                )
                & (
                    newest.c.sequence == RegulatoryDocumentIntakeReviewDecision.sequence
                ),
            )
            .where(
                RegulatoryDocumentIntakeFile.user_file_id.in_(file_ids),
                RegulatoryDocumentIntakeReviewDecision.disposition == "source_evidence",
            )
        )
    )


def require_publication_file(session: Session, file_id: UUID) -> None:
    if source_evidence_file_ids(session, [file_id]):
        raise ValueError("Retained source evidence cannot be published as legislation")


def require_publication_file_for_tenant(file_id: UUID, tenant_id: str) -> None:
    from onyx.db.engine.sql_engine import get_session_with_tenant

    with get_session_with_tenant(tenant_id=tenant_id) as session:
        require_publication_file(session, file_id)


def _blob_sha256(pointer: str) -> str:
    from onyx.file_store.file_store import get_default_file_store

    digest = hashlib.sha256()
    try:
        with get_default_file_store().read_file(pointer) as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
    except (OSError, ValueError) as exc:
        from onyx.db.document_intake import IntakeConflictError

        raise IntakeConflictError("The retained source bytes are unavailable") from exc
    return digest.hexdigest()


def _source_snapshot(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
) -> dict[str, Any]:
    from onyx.db.document_intake import IntakeConflictError

    asset = session.get(RegulatorySourceAsset, member.source_asset_id)
    package = session.get(AmendmentSourcePackage, batch.source_package_id)
    file = session.get(UserFile, member.user_file_id)
    if (
        asset is None
        or package is None
        or file is None
        or package.status != "ready"
        or package.issues
        or package.created_by != batch.created_by
        or package.document_set_id != batch.document_set_id
        or asset.package_id != package.id
        or file.user_id != batch.created_by
        or not package.manifest_file_id
        or not package.manifest_sha256
        or not asset.file_id
        or not asset.text_file_id
        or not asset.text_sha256
        or not file.file_id
        or _blob_sha256(package.manifest_file_id) != package.manifest_sha256
        or _blob_sha256(asset.file_id) != asset.sha256
        or _blob_sha256(file.file_id) != asset.sha256
        or _blob_sha256(asset.text_file_id) != asset.text_sha256
    ):
        raise IntakeConflictError("The retained source evidence changed")
    return {
        "package_id": str(package.id),
        "asset_id": str(asset.id),
        "asset_sha256": asset.sha256,
        "asset_file_id": asset.file_id,
        "prepared_file_id": asset.text_file_id,
        "prepared_sha256": asset.text_sha256,
        "original_file_id": file.file_id,
        "manifest_file_id": package.manifest_file_id,
        "manifest_sha256": package.manifest_sha256,
        "role": member.role,
        "parent_document_id": str(member.parent_document_id)
        if member.parent_document_id
        else None,
        "relationships": member.relationships,
    }


def _parent_snapshot(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    parent_id: UUID,
) -> dict[str, Any]:
    from onyx.db.document_intake import IntakeConflictError, pending_intake_file_ids
    from onyx.db.document_lineage import file_lineage_snapshot

    file = session.get(UserFile, parent_id, populate_existing=True)
    membership = session.get(DocumentSet__UserFile, (batch.document_set_id, parent_id))
    publication = session.get(
        RegulatoryFilePublication, parent_id, populate_existing=True
    )
    request = session.get(UserFileProjectionRepair, parent_id, populate_existing=True)
    same_batch = session.scalar(
        select(RegulatoryDocumentIntakeFile.id).where(
            RegulatoryDocumentIntakeFile.intake_id == batch.id,
            RegulatoryDocumentIntakeFile.user_file_id == parent_id,
        )
    )
    if (
        file is None
        or membership is None
        or same_batch is not None
        or parent_id == member.user_file_id
        or file.status != UserFileStatus.COMPLETED
        or pending_intake_file_ids(session, [parent_id], include_approving=True)
        or (
            publication is not None
            and (
                publication.gate_closed
                or publication.writer_manifest is not None
                or (
                    publication.lease_expires_at is not None
                    and publication.lease_expires_at
                    > datetime.datetime.now(datetime.timezone.utc)
                )
            )
        )
        or (
            request is not None
            and request.status
            in {
                UserFileProjectionRepairStatus.PENDING,
                UserFileProjectionRepairStatus.RUNNING,
            }
        )
        or session.scalar(
            select(RegulatoryChunk.id)
            .where(RegulatoryChunk.user_file_id == parent_id)
            .limit(1)
        )
        is None
    ):
        raise IntakeConflictError(
            "Select a completed, idle main document in PC Külliyatı"
        )
    lineage = file_lineage_snapshot(session, batch.document_set_id, parent_id)
    if lineage["status"] != "current" or lineage["successors"]:
        raise IntakeConflictError(
            "A repealed or replaced document cannot be selected as the current parent"
        )
    return {
        "user_file_id": str(file.id),
        "display_name": file.name,
        "source_url": file.link_url,
        "file_id": file.file_id,
        "canonical_sha256": canonical_scope_digest(session, file.id),
        "document_set_id": batch.document_set_id,
        "generation_hash": file.regulatory_chunk_generation_hash,
        "publication_epoch": publication.epoch if publication else None,
    }


def has_active_publication_decisions(
    session: Session, batch: RegulatoryDocumentIntake
) -> bool:
    return any(
        row.disposition == "source_evidence" or row.existing_parent_file_id is not None
        for row in latest_publication_decisions(session, batch.id).values()
    )


def publication_decision_payload(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row.id),
            "document_id": str(row.document_id),
            "sequence": row.sequence,
            "previous_decision_id": str(row.previous_decision_id)
            if row.previous_decision_id
            else None,
            "disposition": row.disposition,
            "duplicate_of_document_id": str(row.duplicate_of_document_id)
            if row.duplicate_of_document_id
            else None,
            "existing_parent_file_id": str(row.existing_parent_file_id)
            if row.existing_parent_file_id
            else None,
            "existing_parent_snapshot": row.existing_parent_snapshot,
            "source_snapshot": row.source_snapshot,
            "reason": row.reason,
            "reviewed_by": str(row.reviewed_by),
        }
        for row in sorted(
            latest_publication_decisions(session, batch.id).values(),
            key=lambda row: str(row.document_id),
        )
    ]


def effective_parent_document_ids(
    members: Sequence[RegulatoryDocumentIntakeFile],
    decisions: dict[UUID, RegulatoryDocumentIntakeReviewDecision],
) -> dict[UUID, UUID | None]:
    """Resolve publication aliases without rewriting the retained source graph."""
    aliases = {
        identifier: row.duplicate_of_document_id
        for identifier, row in decisions.items()
        if row.disposition == "source_evidence"
    }
    parents: dict[UUID, UUID | None] = {}
    for member in members:
        if member.id in aliases:
            continue
        parent = member.parent_document_id
        parents[member.id] = aliases.get(parent, parent) if parent else None
    for identifier in parents:
        visited: set[UUID] = set()
        current: UUID | None = identifier
        while current is not None:
            if current in visited or current not in parents:
                raise ValueError(
                    "The publication rendition decision would create an invalid parent relationship"
                )
            visited.add(current)
            current = parents[current]
    return parents


def validate_publication_review(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    require_current_parent: bool = True,
) -> None:
    from onyx.db.document_intake import IntakeConflictError, intake_files

    members = {member.id: member for member in intake_files(session, batch)}
    decisions = latest_publication_decisions(session, batch.id)
    try:
        effective_parent_document_ids(list(members.values()), decisions)
    except ValueError as exc:
        raise IntakeConflictError(str(exc)) from exc
    for document_id, row in decisions.items():
        member = members.get(document_id)
        if member is None or _source_snapshot(
            session, batch, member
        ) != row.source_snapshot.get("source"):
            raise IntakeConflictError("The reviewed document source changed")
        if row.disposition == "source_evidence":
            file = session.get_one(UserFile, member.user_file_id)
            publication = session.get(RegulatoryFilePublication, file.id)
            if (
                file.status != UserFileStatus.CHUNKED
                or member.approval_attempt_id is not None
                or session.get(UserFileProjectionRepair, file.id) is not None
                or (
                    publication is not None
                    and (
                        publication.epoch != 0
                        or publication.gate_closed
                        or publication.writer_manifest is not None
                    )
                )
            ):
                raise IntakeConflictError(
                    "Retained source evidence must remain unpublished"
                )
            target = (
                members.get(row.duplicate_of_document_id)
                if row.duplicate_of_document_id
                else None
            )
            if (
                target is None
                or target.id == member.id
                or (
                    target.id in decisions
                    and decisions[target.id].disposition == "source_evidence"
                )
                or row.source_snapshot.get("canonical_sha256")
                != canonical_scope_digest(session, member.user_file_id)
                or row.source_snapshot.get("duplicate_source")
                != _source_snapshot(session, batch, target)
            ):
                raise IntakeConflictError("The retained duplicate rendition changed")
        elif (
            row.disposition != "legislation" or row.duplicate_of_document_id is not None
        ):
            raise IntakeConflictError("Invalid publication review disposition")
        if row.existing_parent_file_id is not None:
            if (
                member.parent_document_id is not None
                or row.disposition != "legislation"
            ):
                raise IntakeConflictError("A document cannot have two primary parents")
            if (
                require_current_parent
                and _parent_snapshot(
                    session, batch, member, row.existing_parent_file_id
                )
                != row.existing_parent_snapshot
            ):
                raise IntakeConflictError(
                    "The selected main document changed; select it again"
                )


def publication_review_digest(
    session: Session, batch: RegulatoryDocumentIntake
) -> str | None:
    from onyx.db.document_intake_edit import edit_review_digest

    receipt = edit_review_digest(session, batch)
    if receipt is None:
        return None
    return publication_digest(
        {
            "purpose": "intake_publication_review",
            "edit_review_sha256": receipt,
            "decisions": publication_decision_payload(session, batch),
        }
    )


def detach_source_evidence_memberships(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    from onyx.db.document_intake import intake_files

    excluded = source_evidence_file_ids(
        session, [m.user_file_id for m in intake_files(session, batch)]
    )
    if excluded:
        session.execute(
            delete(DocumentSet__UserFile).where(
                DocumentSet__UserFile.document_set_id == batch.document_set_id,
                DocumentSet__UserFile.user_file_id.in_(excluded),
            )
        )


def parent_labeling_contexts(
    session: Session, file_ids: Sequence[UUID]
) -> dict[UUID, dict[str, Any]]:
    """One scoped lookup, including absence, for the complete requested page."""
    if not file_ids:
        return {}
    newest = (
        select(
            RegulatoryDocumentIntakeReviewDecision.document_id,
            func.max(RegulatoryDocumentIntakeReviewDecision.sequence).label("sequence"),
        )
        .join(
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntakeFile.id
            == RegulatoryDocumentIntakeReviewDecision.document_id,
        )
        .where(RegulatoryDocumentIntakeFile.user_file_id.in_(file_ids))
        .group_by(RegulatoryDocumentIntakeReviewDecision.document_id)
        .subquery()
    )
    rows = session.execute(
        select(
            RegulatoryDocumentIntakeFile.user_file_id,
            RegulatoryDocumentIntakeReviewDecision,
        )
        .join(
            RegulatoryDocumentIntakeReviewDecision,
            RegulatoryDocumentIntakeReviewDecision.document_id
            == RegulatoryDocumentIntakeFile.id,
        )
        .join(
            newest,
            (newest.c.document_id == RegulatoryDocumentIntakeReviewDecision.document_id)
            & (newest.c.sequence == RegulatoryDocumentIntakeReviewDecision.sequence),
        )
        .where(
            RegulatoryDocumentIntakeReviewDecision.disposition == "legislation",
            RegulatoryDocumentIntakeReviewDecision.existing_parent_file_id.is_not(None),
        )
    )
    result = {}
    for file_id, row in rows:
        if row.existing_parent_snapshot is None:
            raise ValueError("The reviewed parent identity is unavailable")
        result[file_id] = {
            "decision_id": str(row.id),
            "user_file_id": str(row.existing_parent_file_id),
            "display_name": row.existing_parent_snapshot["display_name"],
            "source_url": row.existing_parent_snapshot.get("source_url"),
            "reviewer_reason": row.reason,
            "authority": "reviewed_parent_identity_only_not_operative_text_or_validity_evidence",
        }
    return result


def parent_labeling_context(session: Session, file_id: UUID) -> dict[str, Any] | None:
    return parent_labeling_contexts(session, [file_id]).get(file_id)


def _replace_labels(
    session: Session, batch: RegulatoryDocumentIntake, changed_file_ids: set[UUID]
) -> UUID:
    from onyx.db import document_intake as intake
    from onyx.db import regulatory_labeling as labeling
    from onyx.db.document_intake_edit import _copy_unchanged_label_results

    previous = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    user = session.get_one(User, batch.created_by)
    binding = intake.resolve_intake_labeling_provider_binding(
        session, batch.labeling_model_configuration_id, user=user
    )
    run, _ = labeling.create_labeling_run(
        session,
        document_set_id=batch.document_set_id,
        taxonomy=previous.taxonomy,
        model_configuration_id=batch.labeling_model_configuration_id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=batch.created_by,
        idempotency_key=uuid5(
            previous.id,
            publication_digest(publication_decision_payload(session, batch)),
        ),
        uses_current_labels=previous.uses_current_labels,
        target_file_ids=[
            m.user_file_id for m in effective_intake_files(session, batch)
        ],
    )
    _copy_unchanged_label_results(
        session,
        previous=previous,
        replacement=run,
        changed_file_ids=changed_file_ids,
        allow_incomplete=True,
    )
    batch.labeling_run_id = run.id
    batch.status, batch.stage, batch.error, batch.review_sha256 = (
        "preparing",
        "labeling",
        None,
        None,
    )
    for member in effective_intake_files(session, batch):
        member.status, member.error = "labeling", None
    session.flush()
    return run.id


def set_publication_review(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    document_id: UUID,
    expected_review_sha256: str,
    disposition: PublicationDisposition,
    duplicate_of_document_id: UUID | None,
    existing_parent_file_id: UUID | None,
    reason: str,
    user: User,
) -> UUID | None:
    from onyx.db import document_intake as intake

    with session.begin_nested():
        session.refresh(batch, with_for_update=True)
        session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
        intake.refresh_intake(session, batch)
        if (
            user.id != batch.created_by
            or not expected_review_sha256
            or publication_review_digest(session, batch) != expected_review_sha256
        ):
            raise intake.IntakeConflictError(
                "The publication review changed; reload before saving"
            )
        reason = reason.strip()
        if not reason or len(reason) > 2000:
            raise ValueError("Supply a review reason between 1 and 2000 characters")
        members = {member.id: member for member in intake.intake_files(session, batch)}
        member = members.get(document_id)
        if member is None:
            raise ValueError("Intake document not found")
        previous = latest_publication_decisions(session, batch.id).get(member.id)
        target: RegulatoryDocumentIntakeFile | None = None
        if disposition not in {"legislation", "source_evidence"}:
            raise ValueError("Invalid publication disposition")
        if disposition == "source_evidence":
            target = (
                members.get(duplicate_of_document_id)
                if duplicate_of_document_id
                else None
            )
            decisions = latest_publication_decisions(session, batch.id)
            if (
                target is None
                or target.id == member.id
                or existing_parent_file_id is not None
                or (
                    target.id in decisions
                    and decisions[target.id].disposition == "source_evidence"
                )
                or any(
                    row.duplicate_of_document_id == member.id
                    for row in decisions.values()
                )
                or len(effective_intake_files(session, batch)) <= 1
            ):
                raise ValueError(
                    "Select a different publication document from this batch; duplicate chains are not allowed"
                )
            if session.get(RegulatoryDocumentLineageReview, member.id) is not None:
                raise intake.IntakeConflictError(
                    "Clear the predecessor review before retaining this document as evidence"
                )
        elif duplicate_of_document_id is not None:
            raise ValueError("A legislation document cannot be a duplicate rendition")
        if (
            existing_parent_file_id is not None
            and member.parent_document_id is not None
        ):
            raise ValueError("This attachment already has a same-batch parent")
        for source_member in members.values():
            source_file = session.get_one(
                UserFile, source_member.user_file_id, with_for_update=True
            )
            publication = session.get(RegulatoryFilePublication, source_file.id)
            if (
                source_file.status != UserFileStatus.CHUNKED
                or source_file.user_id != batch.created_by
                or source_member.approval_attempt_id is not None
                or session.get(UserFileProjectionRepair, source_file.id) is not None
                or (
                    publication is not None
                    and (
                        publication.epoch != 0
                        or publication.gate_closed
                        or publication.writer_manifest is not None
                    )
                )
            ):
                raise intake.IntakeConflictError(
                    "Only unpublished source members can change publication scope"
                )
        parent = (
            _parent_snapshot(session, batch, member, existing_parent_file_id)
            if existing_parent_file_id
            else None
        )
        row = RegulatoryDocumentIntakeReviewDecision(
            id=uuid4(),
            intake_id=batch.id,
            document_id=member.id,
            sequence=previous.sequence + 1 if previous else 1,
            previous_decision_id=previous.id if previous else None,
            disposition=disposition,
            duplicate_of_document_id=duplicate_of_document_id,
            existing_parent_file_id=existing_parent_file_id,
            existing_parent_snapshot=parent,
            source_snapshot={
                "previous_labeling_run_id": str(batch.labeling_run_id)
                if batch.labeling_run_id
                else None,
                "accepted_publication_review_sha256": expected_review_sha256,
                "source": _source_snapshot(session, batch, member),
                "duplicate_source": _source_snapshot(session, batch, target)
                if disposition == "source_evidence" and target is not None
                else None,
                "canonical_sha256": canonical_scope_digest(
                    session, member.user_file_id
                ),
            },
            reason=reason,
            reviewed_by=user.id,
        )
        session.add(row)
        session.flush()
        validate_publication_review(session, batch)
        target_changed = (
            previous is None
            and disposition == "source_evidence"
            or previous is not None
            and previous.disposition != disposition
        )
        parent_changed = existing_parent_file_id is not None or (
            previous is not None and previous.existing_parent_file_id is not None
        )
        run_id = (
            _replace_labels(
                session, batch, {member.user_file_id} if parent_changed else set()
            )
            if target_changed or parent_changed
            else None
        )
        if disposition == "source_evidence":
            member.status, member.error = "source_evidence", None
        elif previous is not None and previous.disposition == "source_evidence":
            member.status = "labeling"
        if run_id is None and batch.status == "ready":
            batch.review_sha256 = intake.review_digest(session, batch)
        session.flush()
        return run_id


def existing_parent_candidates(
    session: Session,
    batch: RegulatoryDocumentIntake,
    document_id: UUID,
    search: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    member = session.get(RegulatoryDocumentIntakeFile, document_id)
    if member is None or member.intake_id != batch.id:
        raise ValueError("Intake document not found")
    if member.parent_document_id is not None:
        return []
    query = (
        select(UserFile)
        .join(DocumentSet__UserFile, UserFile.id == DocumentSet__UserFile.user_file_id)
        .where(
            DocumentSet__UserFile.document_set_id == batch.document_set_id,
            UserFile.status == UserFileStatus.COMPLETED,
        )
        .order_by(UserFile.name, UserFile.id)
    )
    if search and search.strip():
        query = query.where(
            UserFile.name.icontains(search.strip()[:500], autoescape=True)
        )
    results = []
    for file in session.scalars(query.limit(min(max(limit, 1), 50) * 5)):
        try:
            snapshot = _parent_snapshot(session, batch, member, file.id)
        except (ValueError, RuntimeError):
            continue
        results.append(
            {
                "user_file_id": str(file.id),
                "display_name": file.name,
                "source_url": snapshot["source_url"],
                "chunk_count": session.scalar(
                    select(func.count())
                    .select_from(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == file.id)
                )
                or 0,
            }
        )
        if len(results) >= min(max(limit, 1), 50):
            break
    return results


def file_attachment_relations(
    session: Session, document_set_id: int, file_id: UUID
) -> dict[str, list[dict[str, Any]]]:
    """Only published children related to this file, without scanning other intakes."""
    relations: dict[str, list[dict[str, Any]]] = {
        "existing_parents": [],
        "attachments": [],
    }
    newest = (
        select(
            RegulatoryDocumentIntakeReviewDecision.document_id,
            func.max(RegulatoryDocumentIntakeReviewDecision.sequence).label("sequence"),
        )
        .group_by(RegulatoryDocumentIntakeReviewDecision.document_id)
        .subquery()
    )
    parent_membership = aliased(DocumentSet__UserFile)
    rows = session.execute(
        select(RegulatoryDocumentIntakeReviewDecision, UserFile, RegulatorySourceAsset)
        .join(
            newest,
            (newest.c.document_id == RegulatoryDocumentIntakeReviewDecision.document_id)
            & (newest.c.sequence == RegulatoryDocumentIntakeReviewDecision.sequence),
        )
        .join(
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntakeFile.id
            == RegulatoryDocumentIntakeReviewDecision.document_id,
        )
        .join(
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
        )
        .join(UserFile, UserFile.id == RegulatoryDocumentIntakeFile.user_file_id)
        .join(
            RegulatorySourceAsset,
            RegulatorySourceAsset.id == RegulatoryDocumentIntakeFile.source_asset_id,
        )
        .join(
            DocumentSet__UserFile,
            (DocumentSet__UserFile.user_file_id == UserFile.id)
            & (DocumentSet__UserFile.document_set_id == document_set_id),
        )
        .join(
            parent_membership,
            (
                parent_membership.user_file_id
                == RegulatoryDocumentIntakeReviewDecision.existing_parent_file_id
            )
            & (parent_membership.document_set_id == document_set_id),
        )
        .join(
            RegulatoryFilePublication,
            RegulatoryFilePublication.user_file_id == UserFile.id,
        )
        .where(
            RegulatoryDocumentIntake.document_set_id == document_set_id,
            RegulatoryDocumentIntake.status == "completed",
            RegulatoryDocumentIntakeFile.status == "completed",
            RegulatoryDocumentIntakeFile.approval_attempt_id.is_not(None),
            UserFile.status == UserFileStatus.COMPLETED,
            RegulatoryFilePublication.gate_closed.is_(False),
            RegulatoryFilePublication.writer_manifest.is_(None),
            RegulatoryFilePublication.epoch > 0,
            RegulatoryDocumentIntakeReviewDecision.disposition == "legislation",
            or_(
                RegulatoryDocumentIntakeReviewDecision.existing_parent_file_id
                == file_id,
                RegulatoryDocumentIntakeFile.user_file_id == file_id,
            ),
        )
        .order_by(UserFile.name, UserFile.id)
    )
    for row, file, asset in rows:
        if file.id == file_id and row.existing_parent_snapshot is not None:
            parent = row.existing_parent_snapshot
            relations["existing_parents"].append(
                {
                    key: parent.get(key)
                    for key in ("user_file_id", "display_name", "source_url")
                }
            )
        if row.existing_parent_file_id == file_id:
            relations["attachments"].append(
                {
                    "user_file_id": str(file.id),
                    "display_name": file.name,
                    "source_url": asset.final_url or asset.original_url,
                }
            )
    return relations
