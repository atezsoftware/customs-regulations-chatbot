"""Resolve only completed, source-linked intake families inside an amendment scope."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_amendment_targets import published_amendment_source_conditions
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.amendments.structural_target import normalize_appendix_label
from onyx.regulatory.provision_identity import article_identity

_MAX_MEMBERS = 21
_MAX_STRUCTURAL_ROWS = 4096
_SHA = re.compile(r"^[a-f0-9]{64}$")
_APPENDIX_HEADING = re.compile(
    r"^(?:ek|annex|appendix)\s*[-–—:.]?\s*(?:[ivxlcdm]+|\d+[a-z]?|[a-z])$",
    re.IGNORECASE,
)
_SUBPROVISION = re.compile(r"^\s*\(?(?:\d{1,3}[.)]|[a-zçğıöşü]\))", re.IGNORECASE)


@dataclass(frozen=True)
class AmendmentSourceFamily:
    intake_id: UUID
    primary_file_id: UUID
    member_file_ids: tuple[UUID, ...]
    target_file_ids: tuple[UUID, ...]


def _source_reference(
    package: AmendmentSourcePackage,
    member: RegulatoryDocumentIntakeFile,
    asset: RegulatorySourceAsset,
    file: UserFile,
) -> dict[str, object]:
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


def _component(
    members: Sequence[RegulatoryDocumentIntakeFile], matched: set[UUID]
) -> tuple[RegulatoryDocumentIntakeFile, list[RegulatoryDocumentIntakeFile]] | None:
    by_id = {member.id: member for member in members}
    by_file = {member.user_file_id: member for member in members}
    if (
        len(by_id) != len(members)
        or len(by_file) != len(members)
        or not matched <= by_file.keys()
    ):
        return None
    roots = set()
    for file_id in matched:
        current = by_file[file_id]
        seen = set()
        while current.parent_document_id is not None:
            if current.id in seen:
                return None
            seen.add(current.id)
            parent = by_id.get(current.parent_document_id)
            if parent is None or parent.intake_id != current.intake_id:
                return None
            current = parent
        if current.role != "root":
            return None
        roots.add(current.id)
    if len(roots) != 1:
        return None
    root = by_id[next(iter(roots))]
    if root.user_file_id not in matched:
        return None
    included = {root.id}
    while True:
        added = {
            member.id for member in members if member.parent_document_id in included
        }
        if added <= included:
            break
        included |= added
    component = [member for member in members if member.id in included]
    if len(component) < 2 or any(
        member.intake_id != root.intake_id
        or (member.id != root.id and member.role != "annex")
        for member in component
    ):
        return None
    return root, component


def _source_linked(
    parent: RegulatoryDocumentIntakeFile,
    child: RegulatoryDocumentIntakeFile,
    assets: dict[UUID, RegulatorySourceAsset],
) -> bool:
    expected_parent = assets[parent.source_asset_id].sha256
    expected_child = assets[child.source_asset_id].sha256
    if expected_parent == expected_child:
        return False
    try:
        parent_links = [
            SourceLink.model_validate(item) for item in parent.relationships
        ]
        child_links = [SourceLink.model_validate(item) for item in child.relationships]
    except (ValidationError, TypeError):
        return False
    return any(
        link.parent_asset_hash == expected_parent
        and link.target_asset_hash == expected_child
        and bool(link.source_field.strip())
        and link in child_links
        for link in parent_links
    )


def _own_appendix(
    heading_path: list[str], metadata: dict[str, object], wanted: str
) -> bool:
    for heading in heading_path:
        if article_identity(heading) is not None or _SUBPROVISION.match(heading):
            return False
        if (
            _APPENDIX_HEADING.fullmatch(heading.strip())
            and normalize_appendix_label(heading) == wanted
        ):
            declared = metadata.get("appendix_label")
            return declared is None or (
                isinstance(declared, str)
                and normalize_appendix_label(declared) == wanted
            )
    return False


def resolve_amendment_source_family(
    db_session: Session,
    *,
    document_set_id: int,
    user_file_ids: Sequence[UUID],
    matched_file_ids: Sequence[UUID],
    appendix_label: str | None = None,
) -> AmendmentSourceFamily | None:
    """None preserves the caller's original ambiguity; no file/version is guessed."""
    matched, scope = set(matched_file_ids), set(user_file_ids)
    if not 1 <= len(matched) <= _MAX_MEMBERS or not matched <= scope:
        return None
    matched_members = list(
        db_session.scalars(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.user_file_id.in_(matched)
            )
        )
    )
    intake_ids = {member.intake_id for member in matched_members}
    if {member.user_file_id for member in matched_members} != matched or len(
        intake_ids
    ) != 1:
        return None
    batch = db_session.get(RegulatoryDocumentIntake, next(iter(intake_ids)))
    if (
        batch is None
        or batch.document_set_id != document_set_id
        or batch.status != "completed"
        or not _SHA.fullmatch(batch.review_sha256 or "")
    ):
        return None
    package = db_session.get(AmendmentSourcePackage, batch.source_package_id)
    if (
        package is None
        or package.document_set_id != document_set_id
        or package.created_by != batch.created_by
        or package.status != "ready"
        or package.issues
        or not _SHA.fullmatch(package.manifest_sha256 or "")
        or not package.manifest_file_id
    ):
        return None
    members = list(
        db_session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .limit(_MAX_MEMBERS + 1)
        )
    )
    if len(members) > _MAX_MEMBERS or len(members) != package.asset_count:
        return None
    assets = {
        asset.id: asset
        for asset in db_session.scalars(
            select(RegulatorySourceAsset)
            .where(RegulatorySourceAsset.package_id == package.id)
            .limit(_MAX_MEMBERS + 1)
        )
    }
    if (
        len(assets) != len(members)
        or set(assets) != {member.source_asset_id for member in members}
        or any(not _SHA.fullmatch(asset.sha256) for asset in assets.values())
    ):
        return None
    latest: dict[UUID, RegulatoryDocumentIntakeReviewDecision] = {}
    all_members = {member.id: member for member in members}
    for decision in db_session.scalars(
        select(RegulatoryDocumentIntakeReviewDecision)
        .where(RegulatoryDocumentIntakeReviewDecision.document_id.in_(all_members))
        .order_by(RegulatoryDocumentIntakeReviewDecision.sequence.desc())
    ):
        latest.setdefault(decision.document_id, decision)
    admitted = []
    for member in members:
        if member.status != "source_evidence":
            admitted.append(member)
            continue
        decision = latest.get(member.id)
        duplicate = (
            all_members.get(decision.duplicate_of_document_id)
            if decision is not None and decision.duplicate_of_document_id is not None
            else None
        )
        file = db_session.get(UserFile, member.user_file_id)
        asset = assets[member.source_asset_id]
        publication = db_session.get(RegulatoryFilePublication, member.user_file_id)
        if (
            member.user_file_id in matched
            or decision is None
            or decision.intake_id != batch.id
            or decision.disposition != "source_evidence"
            or decision.existing_parent_file_id is not None
            or duplicate is None
            or duplicate.id == member.id
            or duplicate.status != "completed"
            or file is None
            or file.user_id != batch.created_by
            or file.status != UserFileStatus.CHUNKED
            or member.approval_attempt_id is not None
            or db_session.get(UserFileProjectionRepair, member.user_file_id) is not None
            or (
                publication is not None
                and (
                    publication.epoch != 0
                    or publication.gate_closed
                    or publication.writer_manifest is not None
                    or publication.writer_manifest_sha256 is not None
                )
            )
            or db_session.get(DocumentSet__UserFile, (document_set_id, file.id))
            is not None
        ):
            return None
        duplicate_file = db_session.get(UserFile, duplicate.user_file_id)
        duplicate_decision = latest.get(duplicate.id)
        if (
            duplicate_file is None
            or duplicate_file.user_id != batch.created_by
            or (
                duplicate_decision is not None
                and duplicate_decision.disposition != "legislation"
            )
            or decision.source_snapshot.get("source")
            != _source_reference(package, member, asset, file)
            or decision.source_snapshot.get("duplicate_source")
            != _source_reference(
                package, duplicate, assets[duplicate.source_asset_id], duplicate_file
            )
            or decision.source_snapshot.get("canonical_sha256")
            != canonical_scope_digest(db_session, file.id)
        ):
            return None
    resolved = _component(admitted, matched)
    if resolved is None:
        return None
    primary, component = resolved
    files = {member.user_file_id for member in component}
    if not files <= scope or any(
        member.status != "completed" or member.approval_attempt_id is None
        for member in component
    ):
        return None
    eligible = set(
        db_session.scalars(
            select(UserFile.id)
            .join(
                RegulatoryFilePublication,
                RegulatoryFilePublication.user_file_id == UserFile.id,
            )
            .join(
                DocumentSet__UserFile, DocumentSet__UserFile.user_file_id == UserFile.id
            )
            .where(
                UserFile.id.in_(files),
                UserFile.user_id == batch.created_by,
                DocumentSet__UserFile.document_set_id == document_set_id,
                *published_amendment_source_conditions(),
            )
        )
    )
    if eligible != files:
        return None
    by_id = {member.id: member for member in component}
    for member in component:
        if member.id != primary.id:
            parent_id = member.parent_document_id
            parent = by_id.get(parent_id) if parent_id is not None else None
            if parent is None or not _source_linked(parent, member, assets):
                return None
    for member in component:
        decision = latest.get(member.id)
        parent_id = member.parent_document_id
        parent = by_id.get(parent_id) if parent_id is not None else None
        if decision is not None and (
            decision.intake_id != batch.id
            or decision.disposition != "legislation"
            or decision.duplicate_of_document_id is not None
            or decision.existing_parent_file_id
            not in {None, parent.user_file_id if parent else None}
        ):
            return None
    targets = {primary.user_file_id}
    if appendix_label is not None:
        wanted = normalize_appendix_label(appendix_label)
        rows = list(
            db_session.scalars(
                select(RegulatoryChunk)
                .where(
                    RegulatoryChunk.user_file_id.in_(files),
                    RegulatoryChunk.status == "active",
                    RegulatoryChunk.superseded_by_chunk_id.is_(None),
                    RegulatoryChunk.chunk_type.is_distinct_from(
                        "hierarchical_aggregate"
                    ),
                    RegulatoryChunk.chunk_metadata[
                        "chunk_variant"
                    ].astext.is_distinct_from("hierarchical_aggregate"),
                    RegulatoryChunk.chunk_metadata[
                        "bound_to_regulatory_chunk_id"
                    ].astext.is_(None),
                )
                .limit(_MAX_STRUCTURAL_ROWS + 1)
            )
        )
        if len(rows) > _MAX_STRUCTURAL_ROWS:
            return None
        targets = {
            row.user_file_id
            for row in rows
            if _own_appendix(list(row.heading_path), dict(row.chunk_metadata), wanted)
        }
    return AmendmentSourceFamily(
        batch.id, primary.user_file_id, tuple(sorted(files)), tuple(sorted(targets))
    )
