"""Read approved links to an existing parent independently of intake package roots."""

from collections import Counter
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, aliased

from onyx.db.document_intake_source_reference import intake_source_reference
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    FileRecord,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_amendment_targets import published_amendment_source_conditions
from onyx.regulatory.document_intake.source_lineage import (
    ReviewedFileRelations,
    ReviewedRelatedFile,
)


def load_reviewed_file_relations(
    session: Session,
    file_ids: Sequence[UUID],
    *,
    document_set_id: int | None = None,
) -> dict[UUID, ReviewedFileRelations]:
    if not file_ids:
        return {}
    if len(file_ids) > 100:
        raise ValueError("Reviewed relationship reads are limited to 100 files")
    decision = RegulatoryDocumentIntakeReviewDecision
    member = RegulatoryDocumentIntakeFile
    batch = RegulatoryDocumentIntake
    asset = RegulatorySourceAsset
    package = AmendmentSourcePackage
    child, parent = aliased(UserFile), aliased(UserFile)
    child_membership, parent_membership = (
        aliased(DocumentSet__UserFile),
        aliased(DocumentSet__UserFile),
    )
    newest = (
        select(decision.document_id, func.max(decision.sequence).label("sequence"))
        .group_by(decision.document_id)
        .subquery()
    )
    query = (
        select(decision, member, batch, asset, package, child, parent)
        .select_from(decision)
        .join(
            newest,
            (newest.c.document_id == decision.document_id)
            & (newest.c.sequence == decision.sequence),
        )
        .join(member, member.id == decision.document_id)
        .join(batch, batch.id == member.intake_id)
        .join(asset, asset.id == member.source_asset_id)
        .join(package, package.id == batch.source_package_id)
        .join(child, child.id == member.user_file_id)
        .join(parent, parent.id == decision.existing_parent_file_id)
        .join(
            child_membership,
            (child_membership.user_file_id == child.id)
            & (child_membership.document_set_id == batch.document_set_id),
        )
        .join(
            parent_membership,
            (parent_membership.user_file_id == parent.id)
            & (parent_membership.document_set_id == batch.document_set_id),
        )
        .where(
            decision.intake_id == batch.id,
            decision.disposition == "legislation",
            decision.duplicate_of_document_id.is_(None),
            member.parent_document_id.is_(None),
            member.status == "completed",
            member.approval_attempt_id.is_not(None),
            batch.status == "completed",
            package.status == "ready",
            asset.package_id == package.id,
            package.document_set_id == batch.document_set_id,
            package.created_by == batch.created_by,
            child.user_id == batch.created_by,
            or_(parent.id.in_(file_ids), child.id.in_(file_ids)),
        )
        .order_by(batch.document_set_id, parent.id, child.id)
    )
    if document_set_id is not None:
        query = query.where(batch.document_set_id == document_set_id)
    rows = session.execute(query.limit(2101)).all()
    if len(rows) > 2100:
        raise ValueError("Reviewed attachment relationships exceed the package bound")
    involved = {
        file.id
        for *_rest, child_file, parent_file in rows
        for file in (child_file, parent_file)
    }
    if not involved:
        return {}
    eligible = set(
        session.scalars(
            select(UserFile.id)
            .outerjoin(
                RegulatoryFilePublication,
                RegulatoryFilePublication.user_file_id == UserFile.id,
            )
            .where(UserFile.id.in_(involved), *published_amendment_source_conditions())
        )
    )
    original_names = {
        file_id: name
        for file_id, name in session.execute(
            select(FileRecord.file_id, FileRecord.display_name).where(
                FileRecord.file_id.in_([row[-1].file_id for row in rows])
            )
        ).all()
    }
    counts = Counter(
        (batch_row.document_set_id, child_file.id)
        for _decision, _member, batch_row, *_rest, child_file, _parent in rows
    )
    result: dict[UUID, ReviewedFileRelations] = {}
    wanted = set(file_ids)
    for (
        review,
        source_member,
        intake,
        source_asset,
        source_package,
        child_file,
        parent_file,
    ) in rows:
        parent_snapshot = review.existing_parent_snapshot
        if (
            child_file.id == parent_file.id
            or counts[(intake.document_set_id, child_file.id)] != 1
            or not {child_file.id, parent_file.id} <= eligible
            or source_package.issues
            or not source_package.manifest_file_id
            or not source_package.manifest_sha256
            or not intake.review_sha256
            or review.source_snapshot.get("source")
            != intake_source_reference(
                source_package, source_member, source_asset, child_file
            )
            or not isinstance(parent_snapshot, dict)
            or parent_snapshot.get("user_file_id") != str(parent_file.id)
            or parent_snapshot.get("file_id") != parent_file.file_id
            or parent_snapshot.get("document_set_id") != intake.document_set_id
            or parent_snapshot.get("source_url") != parent_file.link_url
        ):
            continue
        common = {
            "document_set_id": intake.document_set_id,
            "intake_id": intake.id,
            "review_decision_id": review.id,
        }
        if child_file.id in wanted:
            result.setdefault(child_file.id, ReviewedFileRelations()).parents.append(
                ReviewedRelatedFile(
                    user_file_id=parent_file.id,
                    title=parent_file.name,
                    original_name=original_names.get(parent_file.file_id),
                    source_url=parent_file.link_url,
                    **common,
                )
            )
        if parent_file.id in wanted:
            result.setdefault(
                parent_file.id, ReviewedFileRelations()
            ).attachments.append(
                ReviewedRelatedFile(
                    user_file_id=child_file.id,
                    title=child_file.name,
                    original_name=source_asset.display_name,
                    source_url=source_asset.final_url or source_asset.original_url,
                    **common,
                )
            )
    return result
