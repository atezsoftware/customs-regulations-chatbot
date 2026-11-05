"""Read persisted, source-verified parent and annex identities without reindexing."""

from collections import Counter
from collections.abc import Sequence
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.source_lineage import (
    IntakeFileSourceLineage,
    IntakeOriginalSource,
    IntakeRelatedFile,
)


def load_intake_source_lineages(
    session: Session, file_ids: Sequence[UUID]
) -> dict[UUID, IntakeFileSourceLineage]:
    if not file_ids:
        return {}
    if len(file_ids) > 100:
        raise ValueError("Source lineage reads are limited to 100 files")
    batches = select(RegulatoryDocumentIntakeFile.intake_id).where(
        RegulatoryDocumentIntakeFile.user_file_id.in_(file_ids)
    )
    rows = session.execute(
        select(
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntake,
            RegulatorySourceAsset,
            UserFile,
        )
        .join(
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
        )
        .join(
            RegulatorySourceAsset,
            RegulatorySourceAsset.id == RegulatoryDocumentIntakeFile.source_asset_id,
        )
        .join(UserFile, UserFile.id == RegulatoryDocumentIntakeFile.user_file_id)
        .join(
            AmendmentSourcePackage,
            AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
        )
        .join(
            DocumentSet__UserFile,
            (DocumentSet__UserFile.user_file_id == UserFile.id)
            & (
                DocumentSet__UserFile.document_set_id
                == RegulatoryDocumentIntake.document_set_id
            ),
        )
        .where(
            RegulatoryDocumentIntakeFile.intake_id.in_(batches),
            RegulatorySourceAsset.package_id
            == RegulatoryDocumentIntake.source_package_id,
            AmendmentSourcePackage.document_set_id
            == RegulatoryDocumentIntake.document_set_id,
            AmendmentSourcePackage.created_by == RegulatoryDocumentIntake.created_by,
            UserFile.user_id == RegulatoryDocumentIntake.created_by,
        )
        .limit(2101)
    ).all()
    if len(rows) > 2100:
        raise ValueError("Source lineage exceeds the intake package bound")
    counts = Counter(member.user_file_id for member, _batch, _asset, _file in rows)
    views: dict[UUID, IntakeFileSourceLineage] = {}
    members: dict[UUID, RegulatoryDocumentIntakeFile] = {}
    titles: dict[UUID, str] = {}
    links: dict[UUID, list[SourceLink]] = {}
    for member, batch, asset, file in rows:
        if counts[file.id] != 1 or member.role not in {"root", "annex"}:
            continue
        try:
            source = IntakeOriginalSource(
                asset_id=asset.id,
                sha256=asset.sha256,
                original_name=asset.display_name,
                original_url=asset.original_url,
                final_url=asset.final_url,
            )
            parsed = [SourceLink.model_validate(item) for item in member.relationships]
        except (ValidationError, TypeError):
            continue
        views[member.id] = IntakeFileSourceLineage(
            intake_id=batch.id,
            package_id=batch.source_package_id,
            document_set_id=batch.document_set_id,
            document_id=member.id,
            role=member.role,
            source=source,
        )
        members[member.id], titles[member.id], links[member.id] = (
            member,
            file.name,
            parsed,
        )
    for identifier, member in members.items():
        parent_id = member.parent_document_id
        if parent_id is None or parent_id not in views or member.role != "annex":
            continue
        child, parent = views[identifier], views[parent_id]
        if child.intake_id != parent.intake_id or child.package_id != parent.package_id:
            continue
        seen = {identifier}
        cursor = parent_id
        while cursor in members and cursor not in seen:
            seen.add(cursor)
            cursor = members[cursor].parent_document_id
        if cursor is not None:
            continue
        edges = [
            link
            for link in links[parent_id]
            if link.parent_asset_hash == parent.source.sha256
            and link.target_asset_hash == child.source.sha256
            and link.parent_asset_hash != link.target_asset_hash
            and bool(link.source_field.strip())
            and link in links[identifier]
        ]
        if not edges:
            continue
        child.parents.append(
            IntakeRelatedFile(
                user_file_id=members[parent_id].user_file_id,
                document_id=parent_id,
                title=titles[parent_id],
                source=parent.source,
                source_links=edges,
            )
        )
        parent.attachments.append(
            IntakeRelatedFile(
                user_file_id=member.user_file_id,
                document_id=identifier,
                title=titles[identifier],
                source=child.source,
                source_links=edges,
            )
        )
    wanted = set(file_ids)
    return {
        member.user_file_id: views[identifier]
        for identifier, member in members.items()
        if member.user_file_id in wanted
    }
