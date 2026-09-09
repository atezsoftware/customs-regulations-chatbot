"""Atomic grouped review checkpoints and staged publication contracts."""

import datetime
import hashlib
from collections.abc import Mapping
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.amendment_sources import (
    get_source_package,
    list_source_assets,
    require_ready_source_package,
)
from onyx.db.file_record import get_filerecord_by_file_id_optional
from onyx.db.models import (
    AmendmentBatch,
    AnnexChangeEvidence,
    AnnexChangeItem,
    AnnexChangeSet,
    RegulatoryChunk,
)
from onyx.db.regulatory_annexes import require_annex_file_scope
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
    AnnexReviewEvidence,
    AnnexReviewEvidenceScope,
)


def capture_canonical_scope(
    session: Session, user_file_id: UUID
) -> list[AnnexCanonicalSnapshot]:
    rows = session.scalars(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.user_file_id == user_file_id)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
    )
    return [
        AnnexCanonicalSnapshot(
            id=row.id,
            user_file_id=str(row.user_file_id),
            chunk_type=row.chunk_type,
            status=row.status,
            projection_ordinal=row.projection_ordinal,
            supersedes_chunk_id=row.supersedes_chunk_id,
            superseded_by_chunk_id=row.superseded_by_chunk_id,
            position=row.position,
            text=row.text,
            heading_path=row.heading_path,
            metadata=row.chunk_metadata,
            source=row.source,
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
        )
        for row in rows
    ]


def list_annex_changes(session: Session, batch_id: int) -> list[AnnexChangeSet]:
    return list(
        session.scalars(
            select(AnnexChangeSet)
            .where(AnnexChangeSet.batch_id == batch_id)
            .order_by(AnnexChangeSet.instruction_index)
        )
    )


def persist_annex_checkpoint(
    session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    draft: AnnexChangeDraft,
    environment: str,
) -> AnnexChangeSet | None:
    draft = AnnexChangeDraft.model_validate(draft.model_dump(mode="json"))
    batch = session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )
    if (
        batch is None
        or batch.status != "analyzing"
        or batch.lease_generation != lease_generation
    ):
        session.rollback()
        return None
    covered = set(
        batch.processed_instruction_indices or range(batch.processed_instruction_count)
    )
    indices = set(draft.instruction_indices)
    if not indices.issubset(range(batch.instruction_count)):
        raise ValueError("instruction scope mismatch")
    existing = session.scalar(
        select(AnnexChangeSet).where(
            AnnexChangeSet.batch_id == batch_id,
            AnnexChangeSet.instruction_index == draft.instruction_indices[0],
        )
    )
    if existing is not None:
        if (
            existing.environment != environment
            or existing.review_sha256 != context_hash(draft.model_dump(mode="json"))
        ):
            raise ValueError("checkpoint review changed")
        if (
            existing.instruction_indices != draft.instruction_indices
            or not indices.issubset(covered)
        ):
            raise ValueError("checkpoint coverage mismatch")
        return existing
    if indices.intersection(covered):
        raise ValueError("instruction already checkpointed")
    if draft.user_file_id is not None:
        require_annex_file_scope(session, batch.document_set_id, draft.user_file_id)
        if str(draft.user_file_id) not in batch.user_file_ids:
            raise ValueError("file outside frozen batch scope")
    if (
        draft.user_file_id is not None
        and draft.baseline_scope != capture_canonical_scope(session, draft.user_file_id)
    ):
        raise ValueError("canonical baseline changed")
    baseline_ids = {row.id for row in draft.baseline_scope}
    old_ids = [chunk_id for item in draft.items for chunk_id in item.old_chunk_ids]
    new_chunks = [chunk for item in draft.items for chunk in item.new_chunks]
    new_ids = [chunk.id for chunk in new_chunks]
    if not set(old_ids).issubset(baseline_ids) or len(old_ids) != len(set(old_ids)):
        raise ValueError("old canonical lineage scope mismatch")
    if len(new_ids) != len(set(new_ids)) or set(new_ids).intersection(baseline_ids):
        raise ValueError("prospective canonical identity collision")
    for chunk in new_chunks:
        UUID(chunk.id)
        if (
            chunk.user_file_id != str(draft.user_file_id)
            or session.get(RegulatoryChunk, chunk.id) is not None
        ):
            raise ValueError("prospective canonical scope mismatch")
    if draft.evidence:
        if draft.user_file_id is None:
            raise ValueError("evidence file scope missing")
        scope = load_review_evidence_scope(
            session,
            batch_id=batch_id,
            user_file_id=draft.user_file_id,
            environment=environment,
            created_by=batch.created_by,
        )
        validate_frozen_evidence(session, scope=scope, evidence=draft.evidence)
    if draft.source_package_id != batch.source_package_id:
        raise ValueError("source package scope mismatch")
    payload = draft.model_dump(mode="json")
    ready = (
        not draft.issues
        and draft.patch_plan is not None
        and draft.patch_plan.ready
        and draft.impact is not None
        and draft.impact.ready
    )
    if ready:
        validate_prepared_annex_change(
            session, batch=batch, draft=draft, environment=environment
        )
    change = AnnexChangeSet(
        batch_id=batch_id,
        instruction_index=draft.instruction_indices[0],
        instruction_indices=draft.instruction_indices,
        environment=environment,
        user_file_id=draft.user_file_id,
        status="pending" if ready else "blocked",
        review_payload=payload,
        review_sha256=context_hash(payload),
    )
    session.add(change)
    session.flush()
    for position, item in enumerate(draft.items):
        session.add(
            AnnexChangeItem(
                change_set_id=change.id,
                position=position,
                operation=item.operation,
                old_chunk_ids=item.old_chunk_ids,
                prospective_chunk_ids=[chunk.id for chunk in item.new_chunks],
                payload=item.model_dump(mode="json"),
            )
        )
    for evidence in draft.evidence:
        session.add(
            AnnexChangeEvidence(
                change_set_id=change.id,
                evidence_id=evidence.id,
                file_id=evidence.file_id,
                payload=evidence.model_dump(mode="json"),
            )
        )
    covered.update(indices)
    batch.processed_instruction_indices = sorted(covered)
    batch.processed_instruction_count = len(covered)
    batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
    session.commit()
    return change


def load_review_evidence_scope(
    session: Session,
    *,
    batch_id: int,
    user_file_id: UUID,
    environment: str,
    created_by: UUID | None,
) -> AnnexReviewEvidenceScope:
    batch = session.get(AmendmentBatch, batch_id)
    if (
        batch is None
        or batch.created_by != created_by
        or str(user_file_id) not in batch.user_file_ids
    ):
        raise ValueError("review evidence batch scope mismatch")
    file = require_annex_file_scope(session, batch.document_set_id, user_file_id)
    old_ids = {file.file_id}
    for row in capture_canonical_scope(session, user_file_id):
        image_id = row.metadata.get("image_file_id")
        if isinstance(image_id, str):
            old_ids.add(image_id)
        image_ids = row.metadata.get("image_file_ids")
        if isinstance(image_ids, list):
            old_ids.update(value for value in image_ids if isinstance(value, str))
    new_ids: list[str] = []
    if batch.source_package_id is not None:
        package = get_source_package(
            session,
            package_id=batch.source_package_id,
            document_set_id=batch.document_set_id,
            environment=environment,
        )
        if package is None or package.created_by != created_by:
            raise ValueError("source package owner scope mismatch")
        new_ids = [asset.file_id for asset in list_source_assets(session, package.id)]
    return AnnexReviewEvidenceScope(
        batch_id=batch_id,
        document_set_id=batch.document_set_id,
        user_file_id=user_file_id,
        created_by=created_by,
        environment=environment,
        old_original_file_ids=sorted(old_ids),
        new_original_file_ids=new_ids,
    )


def validate_frozen_evidence(
    session: Session,
    *,
    scope: AnnexReviewEvidenceScope,
    evidence: list[AnnexReviewEvidence],
) -> None:
    for item in evidence:
        record = get_filerecord_by_file_id_optional(item.file_id, session)
        if (
            record is None
            or not isinstance(record.file_metadata, Mapping)
            or cast(Mapping[str, object], record.file_metadata).get(
                "annex_review_scope"
            )
            != scope.storage_identity()
        ):
            raise ValueError("frozen evidence scope mismatch")
        metadata = cast(Mapping[str, object], record.file_metadata)
        if (
            metadata.get("sha256") != item.sha256
            or metadata.get("parent_sha256") != item.parent_sha256
            or metadata.get("parent_file_id") != item.parent_file_id
        ):
            raise ValueError("frozen evidence identity mismatch")


def get_annex_review_evidence(
    session: Session,
    *,
    change_set_id: UUID,
    evidence_id: UUID,
    document_set_id: int,
    created_by: UUID | None,
    environment: str,
) -> AnnexReviewEvidence:
    association = session.scalar(
        select(AnnexChangeEvidence)
        .join(AnnexChangeSet, AnnexChangeSet.id == AnnexChangeEvidence.change_set_id)
        .join(AmendmentBatch, AmendmentBatch.id == AnnexChangeSet.batch_id)
        .where(
            AnnexChangeSet.id == change_set_id,
            AnnexChangeEvidence.evidence_id == evidence_id,
            AmendmentBatch.document_set_id == document_set_id,
            AmendmentBatch.created_by == created_by,
            AnnexChangeSet.environment == environment,
        )
    )
    if association is None:
        raise ValueError("annex review evidence scope mismatch")
    return AnnexReviewEvidence.model_validate(association.payload)


def validate_prepared_annex_change(
    session: Session,
    *,
    batch: AmendmentBatch,
    draft: AnnexChangeDraft,
    environment: str,
) -> None:
    """Check the frozen preparation contract before a group can become pending."""
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    if (
        draft.user_file_id is None
        or draft.source_package_id is None
        or draft.baseline is None
        or draft.old_extraction is None
        or draft.new_extraction is None
        or draft.comparison is None
        or draft.patch_plan is None
        or draft.impact is None
        or draft.effective_date is None
        or not draft.evidence
    ):
        raise ValueError("prepared annex review is incomplete")
    package = require_ready_source_package(
        session,
        package_id=draft.source_package_id,
        document_set_id=batch.document_set_id,
        environment=environment,
    )
    if (
        draft.source_text_sha256 is None
        or hashlib.sha256(batch.raw_text.encode()).hexdigest()
        != draft.source_text_sha256
        or package.created_by != batch.created_by
        or package.manifest_sha256 != draft.source_manifest_sha256
        or batch.source_text_sha256 != draft.source_text_sha256
    ):
        raise ValueError("prepared source ownership or identity changed")
    actual_plan = prepare_annex_patch(
        baseline=draft.baseline,
        old=draft.old_extraction,
        new=draft.new_extraction,
        comparison=draft.comparison,
        effective_date=draft.effective_date,
        package_complete=True,
    )
    if not actual_plan.ready or actual_plan != draft.patch_plan:
        raise ValueError("prepared patch validation changed")
    sources = {asset.file_id for asset in list_source_assets(session, package.id)}
    if draft.new_extraction.evidence_view is not None and any(
        parent.file_id not in sources
        for parent in draft.new_extraction.evidence_view.parents
    ):
        raise ValueError("prepared NEW parent outside source package")
    old_ids = {chunk_id for item in draft.items for chunk_id in item.old_chunk_ids}
    if old_ids != {
        patch.old_chunk_id for patch in actual_plan.patches if patch.old_chunk_id
    }:
        raise ValueError("staged lineage differs from reviewed patch")
    expected_text = "\n".join(
        patch.new_text for patch in actual_plan.patches if patch.new_text is not None
    )
    new_chunks = [chunk for item in draft.items for chunk in item.new_chunks]
    if expected_text != "\n".join(chunk.text for chunk in new_chunks) or any(
        chunk.validity_start_date != draft.effective_date for chunk in new_chunks
    ):
        raise ValueError("staged canonical content differs from reviewed patch")
    allowed_ids = {row.id for row in draft.baseline_scope} | {
        chunk.id for chunk in new_chunks
    }
    if any(
        projection.canonical_chunk_id not in allowed_ids
        for projection in draft.impact.prepared.projections
    ) or any(
        source.canonical_chunk_id not in allowed_ids
        for snapshot in draft.impact.prepared.snapshots
        for source in snapshot.ordered_ranges
    ):
        raise ValueError("prepared context outside canonical scope")
    parent_ids = (
        {parent.file_id for parent in draft.old_extraction.evidence_view.parents}
        if draft.old_extraction.evidence_view
        else {
            original.file_id
            for original in draft.baseline.originals
            if original.available
        }
    )
    frozen_ids = {
        evidence.parent_file_id
        for evidence in draft.evidence
        if evidence.side == "old" and evidence.kind == "original"
    }
    if not parent_ids or not parent_ids.issubset(frozen_ids):
        raise ValueError("frozen OLD review originals missing")
