from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentBatch,
    AnnexChangeItem,
    AnnexChangeSet,
    DocumentSet,
    RegulatoryAnnex,
    RegulatoryAnnexRevision,
)
from onyx.db.regulatory_annex_changes import (
    _add_review_associations,
    capture_canonical_scope,
)
from onyx.db.regulatory_annex_selection import (
    chunk_review_page,
    create_selection,
    rebase_selection,
    validate_selection,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexChangeDraft,
    AnnexChangeItemDraft,
)
from onyx.regulatory.amendments.annexes.publication import (
    prepare_legal_publication_timeline,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


def test_selection_is_durable_paginated_idempotent_and_excludes_pending_siblings(
    source_session: Session,
) -> None:
    session = source_session
    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(docset)
    session.flush()
    file = _file(session, docset)
    first = _chunk(session, file, 0, "old first")
    second = _chunk(session, file, 1, "old second")
    session.flush()
    snapshots = capture_canonical_scope(session, file.id)
    new = [
        row.model_copy(
            update={
                "id": str(uuid4()),
                "text": "new " + row.text,
                "supersedes_chunk_id": row.id,
                "projection_ordinal": -1,
                "validity_start_date": date(2026, 9, 15),
            }
        )
        for row in snapshots
    ]
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="EK-1",
        user_file_ids=[str(file.id)],
        status="analyzed",
        created_by=file.user_id,
    )
    session.add(batch)
    annex = RegulatoryAnnex(
        id=uuid4(), document_set_id=docset.id, user_file_id=file.id, label="EK-1"
    )
    session.add(annex)
    session.flush()
    revision = RegulatoryAnnexRevision(
        id=uuid4(),
        annex_id=annex.id,
        baseline_sha256="baseline",
        snapshot={},
        effective_start=date(2020, 1, 1),
        approved_at=datetime.now(timezone.utc),
    )
    session.add(revision)
    session.flush()
    draft = AnnexChangeDraft(
        batch_id=batch.id,
        instruction_indices=[0],
        instruction_texts=["EK-1 changed"],
        annex_label="EK-1",
        user_file_id=file.id,
        effective_date=date(2026, 9, 15),
        baseline_scope=snapshots,
        baseline=AnnexBaseline(
            revision_id=str(revision.id),
            baseline_sha256="baseline",
            canonical_text="old",
            elements=[],
            originals=[],
            visual_evidence_available=False,
        ),
        items=[
            AnnexChangeItemDraft(
                operation="replace",
                old_chunk_ids=[old.id],
                new_chunks=[replacement],
                old_positions=[i],
                new_positions=[i],
            )
            for i, (old, replacement) in enumerate(zip(snapshots, new, strict=True))
        ],
    )
    parent = AnnexChangeSet(
        id=uuid4(),
        batch_id=batch.id,
        logical_group_id=uuid4(),
        review_revision=1,
        instruction_index=0,
        instruction_indices=[0],
        environment="selection-test",
        user_file_id=file.id,
        status="pending",
        review_payload=draft.model_dump(mode="json"),
        review_sha256=context_hash(draft.model_dump(mode="json")),
        publication_generation=0,
    )
    session.add(parent)
    session.flush()
    _add_review_associations(session, change=parent, draft=draft)
    session.flush()
    records = list(
        session.scalars(
            select(AnnexChangeItem)
            .where(AnnexChangeItem.change_set_id == parent.id)
            .order_by(AnnexChangeItem.position)
        )
    )
    assert file.user_id is not None

    def choose(ids: list[UUID]) -> AnnexChangeSet:
        assert file.user_id is not None
        return create_selection(
            session,
            parent_id=parent.id,
            expected_sha256=parent.review_sha256,
            item_ids=ids,
            environment="selection-test",
            user_id=file.user_id,
            tenant_id="public",
            database_identity="isolated",
        )

    child = choose([records[0].id])
    assert child.preparation is not None and child.preparation.status == "queued"
    selected = AnnexChangeDraft.model_validate(child.review_payload)
    assert selected.items == [draft.items[0]]
    assert child.preparation.checkpoint == child.review_payload
    legal = prepare_legal_publication_timeline(selected)
    assert new[0].id in {row.id for row in legal.canonical_rows}
    assert new[1].id not in {row.id for row in legal.canonical_rows}
    assert (
        next(row for row in legal.canonical_rows if row.id == second.id).text
        == "old second"
    )
    assert choose([records[0].id]).id == child.id
    total, page, children = chunk_review_page(session, parent, offset=1, limit=1)
    assert (
        total == 2
        and [group[0].id for group in page] == [records[1].id]
        and children[0].id == child.id
    )
    with pytest.raises(ValueError, match="overlaps"):
        choose([r.id for r in records])
    second.text = "independent update"
    session.flush()
    rebased = rebase_selection(session, selected)
    assert (
        next(row for row in rebased.baseline_scope if row.id == second.id).text
        == "independent update"
    )
    first.text = "changed selected source"
    session.flush()
    with pytest.raises(ValueError, match="original chunk changed"):
        rebase_selection(session, selected)
    with pytest.raises(ValueError, match="immutable operation"):
        validate_selection(
            session, selected.model_copy(update={"items": [draft.items[1]]})
        )


def test_sequential_source_approvals_preserve_pending_then_previous_approved_elements(
    source_session: Session,
) -> None:
    from onyx.db.models import (
        RegulatoryAnnexElement,
        RegulatoryAnnexElementChunk,
        RegulatoryAnnexRevisionElement,
    )
    from onyx.db.regulatory_annex_activation import _activate_selected_sources
    from onyx.db.regulatory_annexes import get_effective_annex_revision
    from onyx.regulatory.amendments.annexes.models import (
        AnnexExtraction,
        AnnexNewElementEvidence,
        AnnexNewEvidenceRemapping,
        ExtractedAnnexElement,
    )
    from onyx.regulatory.amendments.annexes.publication_execution_models import (
        AnnexPublicationDelivery,
    )

    session = source_session
    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(docset)
    session.flush()
    file = _file(session, docset)
    old = [_chunk(session, file, i, f"old {i}") for i in range(2)]
    new = [_chunk(session, file, i + 2, f"new {i}") for i in range(2)]
    session.flush()
    annex = RegulatoryAnnex(
        id=uuid4(), document_set_id=docset.id, user_file_id=file.id, label="EK-1"
    )
    session.add(annex)
    session.flush()
    revision = RegulatoryAnnexRevision(
        id=uuid4(),
        annex_id=annex.id,
        baseline_sha256="old",
        snapshot={},
        effective_start=date(2020, 1, 1),
        approved_at=datetime.now(timezone.utc),
    )
    session.add(revision)
    session.flush()
    for i, chunk in enumerate(old):
        element = RegulatoryAnnexElement(id=uuid4(), annex_id=annex.id)
        session.add(element)
        session.flush()
        session.add(
            RegulatoryAnnexRevisionElement(
                revision_id=revision.id,
                element_id=element.id,
                position=i,
                payload=ExtractedAnnexElement(kind="text", text=chunk.text).model_dump(
                    mode="json"
                ),
            )
        )
        session.flush()
        session.add(
            RegulatoryAnnexElementChunk(
                revision_id=revision.id, element_id=element.id, chunk_id=chunk.id
            )
        )
    mappings = [
        AnnexNewElementEvidence(
            position=i,
            element_id=uuid4(),
            source_asset_id=uuid4(),
            parent_file_id="new",
            evidence_ids=[],
            image_file_ids=[],
        )
        for i in range(2)
    ]
    for chunk, mapping in zip(new, mappings, strict=True):
        chunk.chunk_metadata = {"annex_element_ids": [str(mapping.element_id)]}
    session.flush()
    snapshots = {row.id: row for row in capture_canonical_scope(session, file.id)}
    when = date(2026, 9, 15)
    for selected in range(2):
        current = get_effective_annex_revision(session, annex.id, when)
        assert current is not None
        draft = AnnexChangeDraft(
            instruction_indices=[0],
            instruction_texts=["Replace"],
            annex_label="EK-1",
            user_file_id=file.id,
            effective_date=when,
            selection_parent_id=uuid4(),
            selection_item_ids=[uuid4()],
            selection_source_revision_id=current.id,
            selection_source_sha256=current.baseline_sha256,
            new_extraction=AnnexExtraction(
                source_sha256="new",
                mime_type="text/plain",
                elements=[
                    ExtractedAnnexElement(kind="text", text=row.text) for row in new
                ],
            ),
            new_evidence_remapping=AnnexNewEvidenceRemapping(
                extraction_sha256="new", elements=mappings
            ),
            items=[
                AnnexChangeItemDraft(
                    operation="replace",
                    old_chunk_ids=[old[selected].id],
                    new_chunks=[snapshots[new[selected].id]],
                    old_positions=[selected],
                    new_positions=[selected],
                )
            ],
        )
        delivery = AnnexPublicationDelivery(
            intent_id=uuid4(),
            change_set_id=uuid4(),
            logical_group_id=uuid4(),
            review_revision=1,
            review_sha256=str(uuid4()),
            publication_generation=1,
            tenant_id="public",
            environment="local",
            database_identity="test",
        )
        _activate_selected_sources(
            session,
            delivery,
            draft,
            now=datetime.now(timezone.utc),
            user_file_id=file.id,
            effective_start=when,
            effective_end=None,
        )
        latest = get_effective_annex_revision(session, annex.id, when)
        assert latest is not None
        elements = list(
            session.scalars(
                select(RegulatoryAnnexRevisionElement)
                .where(RegulatoryAnnexRevisionElement.revision_id == latest.id)
                .order_by(RegulatoryAnnexRevisionElement.position)
            )
        )
        assert [element.payload["text"] for element in elements] == (
            ["new 0", "old 1"] if selected == 0 else ["new 0", "new 1"]
        )
        links = set(
            session.scalars(
                select(RegulatoryAnnexElementChunk.chunk_id).where(
                    RegulatoryAnnexElementChunk.revision_id == latest.id
                )
            )
        )
        assert links == (
            {new[0].id, old[1].id} if selected == 0 else {new[0].id, new[1].id}
        )
    session.refresh(revision)
    assert revision.effective_end == when
    original_payload = session.scalar(
        select(RegulatoryAnnexRevisionElement.payload).where(
            RegulatoryAnnexRevisionElement.revision_id == revision.id,
            RegulatoryAnnexRevisionElement.position == 0,
        )
    )
    assert original_payload is not None and original_payload["text"] == "old 0"
