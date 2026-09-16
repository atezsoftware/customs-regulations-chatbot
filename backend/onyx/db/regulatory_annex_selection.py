"""Selections are independent immutable reviews; pending siblings never enter a delta."""

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import AnnexChangeItem, AnnexChangeSet, RegulatoryAnnexRevision
from onyx.db.regulatory_annex_changes import (
    _add_review_associations,
    capture_canonical_scope,
    require_current_annex_review,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexChangeItemDraft,
)


def selection_reviews(session: Session, parent_id: UUID) -> list[AnnexChangeSet]:
    rows = session.scalars(
        select(AnnexChangeSet)
        .where(
            AnnexChangeSet.review_payload["selection_parent_id"].astext
            == str(parent_id)
        )
        .order_by(AnnexChangeSet.review_revision.desc())
    ).all()
    latest: dict[UUID, AnnexChangeSet] = {}
    for row in rows:
        latest.setdefault(row.logical_group_id, row)
    return list(latest.values())


def chunk_review_page(
    session: Session, review: AnnexChangeSet, *, offset: int, limit: int
) -> tuple[int, list[list[AnnexChangeItem]], list[AnnexChangeSet]]:
    from onyx.regulatory.amendments.annexes.selective_impact import review_units

    groups = review_units(AnnexChangeDraft.model_validate(review.review_payload).items)
    page = groups[offset : offset + limit]
    positions = [position for group in page for position in group]
    records = {
        row.position: row
        for row in session.scalars(
            select(AnnexChangeItem).where(
                AnnexChangeItem.change_set_id == review.id,
                AnnexChangeItem.position.in_(positions),
            )
        )
    }
    return (
        len(groups),
        [[records[position] for position in group] for group in page],
        selection_reviews(session, review.id),
    )


def validate_selection(session: Session, draft: AnnexChangeDraft) -> None:
    if draft.selection_parent_id is None:
        raise ValueError("selection parent missing")
    parent = session.get(AnnexChangeSet, draft.selection_parent_id)
    if (
        parent is None
        or parent.batch_id != draft.batch_id
        or parent.user_file_id != draft.user_file_id
    ):
        raise ValueError("selection parent scope mismatch")
    original = AnnexChangeDraft.model_validate(parent.review_payload)
    items = list(
        session.scalars(
            select(AnnexChangeItem)
            .where(
                AnnexChangeItem.change_set_id == parent.id,
                AnnexChangeItem.id.in_(draft.selection_item_ids),
            )
            .order_by(AnnexChangeItem.position)
        )
    )
    if (
        not items
        or len(items) != len(set(draft.selection_item_ids))
        or len(draft.selection_item_ids) != len(set(draft.selection_item_ids))
    ):
        raise ValueError("invalid selection membership")
    if draft.items != [
        AnnexChangeItemDraft.model_validate(item.payload) for item in items
    ]:
        raise ValueError("selection changed an immutable operation")
    from onyx.regulatory.amendments.annexes.selective_impact import review_units

    positions = {item.position for item in items}
    if any(
        positions.intersection(group) and not set(group).issubset(positions)
        for group in review_units(original.items)
    ):
        raise ValueError("selection omits linked source operations")
    for field in (
        "comparison",
        "patch_plan",
        "old_extraction",
        "new_extraction",
        "baseline",
        "evidence",
        "new_evidence_remapping",
        "source_manifest_sha256",
    ):
        if getattr(draft, field) != getattr(original, field):
            raise ValueError("selection source proof changed")
    original_rows = {row.id: row for row in original.baseline_scope}
    current_rows = {row.id: row for row in draft.baseline_scope}
    for item in draft.items:
        for identifier in item.old_chunk_ids:
            if original_rows.get(identifier) != current_rows.get(identifier):
                raise ValueError("selected original chunk changed; review it again")


def create_selection(
    session: Session,
    *,
    parent_id: UUID,
    expected_sha256: str,
    item_ids: list[UUID],
    environment: str,
    user_id: UUID,
    tenant_id: str,
    database_identity: str,
) -> AnnexChangeSet:
    from onyx.db.regulatory_annex_preparation import queue_review_preparation
    from onyx.db.regulatory_annexes import get_effective_annex_revision

    parent = require_current_annex_review(
        session,
        change_set_id=parent_id,
        expected_review_sha256=expected_sha256,
        environment=environment,
    )
    original = AnnexChangeDraft.model_validate(parent.review_payload)
    if original.date_resolution and original.date_resolution.effective_end_date:
        raise ValueError("temporary changes require the complete dated review")
    if (
        parent.status not in ("pending", "blocked")
        or parent.publication_generation
        or original.selection_parent_id is not None
        or original.user_file_id is None
        or original.baseline is None
        or original.baseline.revision_id is None
        or original.effective_date is None
    ):
        raise ValueError("review does not allow a new chunk selection")
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("select distinct chunk operations")
    from onyx.regulatory.amendments.annexes.selective_impact import review_units

    identity_rows = session.execute(
        select(AnnexChangeItem.id, AnnexChangeItem.position).where(
            AnnexChangeItem.change_set_id == parent.id
        )
    ).all()
    identities = {row.position: row.id for row in identity_rows}
    groups = {
        identities[group[0]]: [identities[position] for position in group]
        for group in review_units(original.items)
    }
    if not set(item_ids).issubset(groups):
        raise ValueError("select complete review unit identities")
    item_ids = [identifier for key in item_ids for identifier in groups[key]]
    for previous in selection_reviews(session, parent.id):
        selected = AnnexChangeDraft.model_validate(
            previous.review_payload
        ).selection_item_ids
        if set(selected) & set(item_ids):
            if set(selected) == set(item_ids):
                return previous
            raise ValueError(
                "selection overlaps an existing decision; open that selection"
            )
    records = list(
        session.scalars(
            select(AnnexChangeItem)
            .where(
                AnnexChangeItem.change_set_id == parent.id,
                AnnexChangeItem.id.in_(item_ids),
            )
            .order_by(AnnexChangeItem.position)
        )
    )
    if len(records) != len(item_ids):
        raise ValueError("chunk selection outside review")
    baseline_revision = session.get(
        RegulatoryAnnexRevision, UUID(original.baseline.revision_id)
    )
    if baseline_revision is None:
        raise ValueError("annex source revision missing")
    effective = get_effective_annex_revision(
        session, baseline_revision.annex_id, original.effective_date
    )
    if effective is None:
        raise ValueError("effective annex source missing")
    draft = original.model_copy(
        update={
            "selection_parent_id": parent.id,
            "selection_item_ids": [item.id for item in records],
            "selection_source_revision_id": effective.id,
            "selection_source_sha256": effective.baseline_sha256,
            "items": [
                AnnexChangeItemDraft.model_validate(item.payload) for item in records
            ],
            "baseline_scope": capture_canonical_scope(session, original.user_file_id),
            "impact_strategy": "source_dependencies_v1",
            "dependency_impact": None,
            "baseline_context": None,
            "impact": None,
            "publication": None,
        }
    )
    validate_selection(session, draft)
    payload = draft.model_dump(mode="json")
    child = AnnexChangeSet(
        id=uuid4(),
        batch_id=parent.batch_id,
        logical_group_id=uuid4(),
        review_revision=1,
        instruction_index=parent.instruction_index,
        instruction_indices=parent.instruction_indices,
        environment=environment,
        user_file_id=parent.user_file_id,
        status="pending",
        review_payload=payload,
        review_sha256=context_hash(payload),
        publication_generation=0,
    )
    session.add(child)
    session.flush()
    _add_review_associations(session, change=child, draft=draft)
    # The durable preparation row freezes this exact selection before the worker starts.
    child = queue_review_preparation(
        session,
        review_id=child.id,
        expected_review_sha256=child.review_sha256,
        corrections=None,
        corrected_by=user_id,
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
        initial_checkpoint=draft,
    )
    return child


def rebase_selection(session: Session, draft: AnnexChangeDraft) -> AnnexChangeDraft:
    from onyx.db.regulatory_annexes import get_effective_annex_revision

    if draft.selection_parent_id is None:
        return draft
    if (
        draft.user_file_id is None
        or draft.effective_date is None
        or draft.baseline is None
        or draft.baseline.revision_id is None
    ):
        raise ValueError("selection rebase scope missing")
    rebased = draft.model_copy(
        update={"baseline_scope": capture_canonical_scope(session, draft.user_file_id)}
    )
    validate_selection(session, rebased)
    baseline = session.get(RegulatoryAnnexRevision, UUID(draft.baseline.revision_id))
    if baseline is None:
        raise ValueError("selection source missing")
    current = get_effective_annex_revision(
        session, baseline.annex_id, draft.effective_date
    )
    if current is None:
        raise ValueError("selection source no longer effective")
    return rebased.model_copy(
        update={
            "selection_source_revision_id": current.id,
            "selection_source_sha256": current.baseline_sha256,
        }
    )


def require_unpartitioned_review(session: Session, review: AnnexChangeSet) -> None:
    """Once decisions exist, their parent cannot supersede or reapply them."""
    if (
        session.scalar(
            select(AnnexChangeSet.id)
            .where(
                AnnexChangeSet.review_payload["selection_parent_id"].astext
                == str(review.id)
            )
            .limit(1)
        )
        is not None
    ):
        raise ValueError(
            "This review has chunk selections; continue through those selections"
        )
