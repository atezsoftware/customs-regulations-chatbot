from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, DocumentSet, RegulatoryChunk
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


def test_group_checkpoint_stages_without_canonical_mutation_and_counts_once(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_amendments import mark_batch_analyzed
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        persist_annex_checkpoint,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    first = _chunk(source_session, file, 0, "old one")
    second = _chunk(source_session, file, 1, "old two")
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="two annex instructions",
        user_file_ids=[str(file.id)],
        status="analyzing",
        stage="processing",
        lease_generation=1,
        instruction_count=2,
        processed_instruction_count=0,
    )
    source_session.add(batch)
    source_session.flush()
    draft = AnnexChangeDraft(
        instruction_indices=[0, 1],
        instruction_texts=["first", "second"],
        annex_label="EK-1",
        user_file_id=file.id,
        effective_date=date(2026, 9, 10),
        baseline_scope=capture_canonical_scope(source_session, file.id),
        issues=["original_unavailable"],
    )
    change = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="local-test",
    )
    assert change is not None and change.status == "blocked"
    repeated = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="local-test",
    )
    assert repeated is not None and repeated.id == change.id
    assert repeated.review_sha256 == change.review_sha256
    from sqlalchemy import update
    from sqlalchemy.exc import DBAPIError

    from onyx.db.models import AnnexChangeSet

    with pytest.raises(DBAPIError, match="immutable"):
        with source_session.begin_nested():
            source_session.execute(
                update(AnnexChangeSet)
                .where(AnnexChangeSet.id == change.id)
                .values(review_sha256="changed")
            )
    assert batch.processed_instruction_count == 2
    assert batch.processed_instruction_indices == [0, 1]
    assert mark_batch_analyzed(source_session, batch_id=batch.id, lease_generation=1)
    assert [
        (row.id, row.text)
        for row in source_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == file.id)
            .order_by(RegulatoryChunk.position)
        )
    ] == [(first.id, "old one"), (second.id, "old two")]


def test_stale_lease_cannot_checkpoint_annex(source_session: Session) -> None:
    from onyx.db.regulatory_annex_changes import persist_annex_checkpoint
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="annex",
        status="analyzing",
        lease_generation=2,
        instruction_count=1,
    )
    source_session.add(batch)
    source_session.flush()
    assert (
        persist_annex_checkpoint(
            source_session,
            batch_id=batch.id,
            lease_generation=1,
            environment="local-test",
            draft=AnnexChangeDraft(
                instruction_indices=[0],
                instruction_texts=["annex"],
                annex_label="EK-1",
                issues=["scope_unresolved"],
            ),
        )
        is None
    )


def test_checkpoint_rejects_changed_review_for_same_instruction(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_annex_changes import persist_annex_checkpoint
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="annex",
        status="analyzing",
        lease_generation=1,
        instruction_count=1,
    )
    source_session.add(batch)
    source_session.flush()
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=["annex"],
        annex_label="EK-1",
        issues=["scope_unresolved"],
    )
    assert (
        persist_annex_checkpoint(
            source_session,
            batch_id=batch.id,
            lease_generation=1,
            draft=draft,
            environment="local-test",
        )
        is not None
    )
    with pytest.raises(ValueError, match="review"):
        persist_annex_checkpoint(
            source_session,
            batch_id=batch.id,
            lease_generation=1,
            draft=draft.model_copy(update={"instruction_texts": ["changed"]}),
            environment="local-test",
        )


def test_frozen_original_is_independent_and_evidence_cannot_cross_scope(
    source_session: Session,
) -> None:
    from hashlib import sha256
    from io import BytesIO

    from onyx.configs.constants import FileOrigin
    from onyx.db.regulatory_annex_changes import (
        load_review_evidence_scope,
        validate_frozen_evidence,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments.annexes.evidence import freeze_review_original
    from onyx.regulatory.amendments.annexes.models import AnnexOriginalEvidence

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    batch = AmendmentBatch(
        document_set_id=docset.id, raw_text="annex", user_file_ids=[str(file.id)]
    )
    source_session.add(batch)
    source_session.commit()
    store = get_default_file_store()
    content = b"<h1>EK-1</h1><p>Original 5%</p>"
    file_id = store.save_file(
        BytesIO(content), "old.html", FileOrigin.OTHER, "text/html"
    )
    file.file_id = file_id
    source_session.commit()
    scope = load_review_evidence_scope(
        source_session,
        batch_id=batch.id,
        user_file_id=file.id,
        environment="local-test",
        created_by=None,
    )
    frozen = freeze_review_original(
        store,
        scope=scope,
        side="old",
        original=AnnexOriginalEvidence(
            file_id=file_id,
            sha256=sha256(content).hexdigest(),
            mime_type="text/html",
            available=True,
        ),
    )
    store.save_file(
        BytesIO(b"changed"), "old.html", FileOrigin.OTHER, "text/html", file_id=file_id
    )
    with store.read_file(frozen.file_id) as stream:
        assert stream.read() == content
    assert frozen.file_id != file_id and frozen.parent_file_id == file_id
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        get_annex_review_evidence,
        persist_annex_checkpoint,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    batch.status, batch.instruction_count, batch.lease_generation = "analyzing", 1, 1
    change = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        environment="local-test",
        draft=AnnexChangeDraft(
            instruction_indices=[0],
            instruction_texts=["annex"],
            annex_label="EK-1",
            user_file_id=file.id,
            baseline_scope=capture_canonical_scope(source_session, file.id),
            evidence=[frozen],
            issues=["incomplete"],
        ),
    )
    assert change is not None
    assert (
        get_annex_review_evidence(
            source_session,
            change_set_id=change.id,
            evidence_id=frozen.id,
            document_set_id=docset.id,
            created_by=None,
            environment="local-test",
        )
        == frozen
    )
    for wrong_set, wrong_user in [(docset.id + 1, None), (docset.id, uuid4())]:
        with pytest.raises(ValueError, match="scope"):
            get_annex_review_evidence(
                source_session,
                change_set_id=change.id,
                evidence_id=frozen.id,
                document_set_id=wrong_set,
                created_by=wrong_user,
                environment="local-test",
            )
    validate_frozen_evidence(source_session, scope=scope, evidence=[frozen])
    with pytest.raises(ValueError, match="scope"):
        validate_frozen_evidence(
            source_session,
            scope=scope.model_copy(update={"batch_id": batch.id + 1}),
            evidence=[frozen],
        )


def test_frozen_page_regions_match_exact_comparison_bytes(
    source_session: Session,
) -> None:
    from hashlib import sha256
    from io import BytesIO

    from PIL import Image

    from onyx.configs.constants import FileOrigin
    from onyx.db.regulatory_annex_changes import (
        load_review_evidence_scope,
        validate_frozen_evidence,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments.annexes.comparison import comparison_page_evidence
    from onyx.regulatory.amendments.annexes.evidence import (
        freeze_review_original,
        freeze_review_pages,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexOriginalEvidence

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    batch = AmendmentBatch(
        document_set_id=docset.id, raw_text="annex", user_file_ids=[str(file.id)]
    )
    source_session.add(batch)
    source_session.flush()
    store = get_default_file_store()
    buffer = BytesIO()
    Image.new("RGB", (160, 180), "white").save(buffer, format="PNG")
    content = buffer.getvalue()
    file.file_id = store.save_file(
        BytesIO(content), "source.png", FileOrigin.OTHER, "image/png"
    )
    source_session.commit()
    scope = load_review_evidence_scope(
        source_session,
        batch_id=batch.id,
        user_file_id=file.id,
        environment="local-test",
        created_by=None,
    )
    original = freeze_review_original(
        store,
        scope=scope,
        side="old",
        original=AnnexOriginalEvidence(
            file_id=file.file_id,
            sha256=sha256(content).hexdigest(),
            mime_type="image/png",
            available=True,
        ),
    )
    pages, evidence = freeze_review_pages(store, scope=scope, original=original)
    expected = comparison_page_evidence(pages[0])
    assert len(evidence) == len(expected)
    for frozen, compared in zip(evidence, expected):
        with store.read_file(frozen.file_id) as stream:
            assert stream.read() == compared.png
        assert frozen.parent_sha256 == original.sha256
        assert frozen.locator.page == 1
    validate_frozen_evidence(source_session, scope=scope, evidence=evidence)


@pytest.mark.parametrize(
    "operation,old_count,new_count",
    [
        ("replace", 1, 1),
        ("insert", 0, 1),
        ("remove", 1, 0),
        ("split", 1, 2),
        ("merge", 2, 1),
    ],
)
def test_checkpoint_retains_n_to_m_lineage_and_rejects_stale_baseline(
    source_session: Session,
    operation: str,
    old_count: int,
    new_count: int,
) -> None:
    from onyx.db.models import AnnexChangeItem
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        persist_annex_checkpoint,
    )
    from onyx.regulatory.amendments.annexes.models import (
        AnnexChangeDraft,
        AnnexChangeItemDraft,
    )

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    rows = [_chunk(source_session, file, index, f"old {index}") for index in range(2)]
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="annex",
        user_file_ids=[str(file.id)],
        status="analyzing",
        instruction_count=2,
        lease_generation=1,
    )
    source_session.add(batch)
    source_session.flush()
    baseline = capture_canonical_scope(source_session, file.id)
    new = [
        baseline[0].model_copy(
            update={
                "id": str(uuid4()),
                "text": f"new {index}",
                "projection_ordinal": index + 2,
            }
        )
        for index in range(new_count)
    ]
    item = AnnexChangeItemDraft.model_validate(
        dict(
            operation=operation,
            old_chunk_ids=[row.id for row in rows[:old_count]],
            new_chunks=[chunk.model_dump(mode="json") for chunk in new],
            old_positions=list(range(old_count)),
            new_positions=list(range(new_count)),
        )
    )
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=["annex"],
        annex_label="EK-1",
        user_file_id=file.id,
        baseline_scope=baseline,
        items=[item],
        issues=["not yet prepared"],
    )
    change = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="local-test",
    )
    assert change is not None
    stored = source_session.scalars(
        select(AnnexChangeItem).where(AnnexChangeItem.change_set_id == change.id)
    ).one()
    assert stored.old_chunk_ids == item.old_chunk_ids
    assert stored.prospective_chunk_ids == [chunk.id for chunk in new]
    assert all(source_session.get(RegulatoryChunk, chunk.id) is None for chunk in new)
    rows[0].text = "Concurrent approved edit"
    source_session.flush()
    stale = draft.model_copy(update={"instruction_indices": [1]})
    with pytest.raises(ValueError, match="baseline"):
        persist_annex_checkpoint(
            source_session,
            batch_id=batch.id,
            lease_generation=1,
            draft=stale,
            environment="local-test",
        )


def test_approvable_domain_contract_rejects_incomplete_preparation(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_annex_changes import validate_prepared_annex_change
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    with pytest.raises(ValueError, match="incomplete"):
        validate_prepared_annex_change(
            source_session,
            batch=AmendmentBatch(document_set_id=1, raw_text="annex"),
            draft=AnnexChangeDraft(
                instruction_indices=[0], instruction_texts=["annex"], annex_label="EK-1"
            ),
            environment="local-test",
        )


def test_complete_prepared_group_freezes_context_and_stages_before_approval(
    source_session: Session,
) -> None:
    from hashlib import sha256
    from io import BytesIO

    from onyx.configs.constants import FileOrigin
    from onyx.db.amendment_sources import create_source_package
    from onyx.db.models import RegulatorySourceAsset
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        load_review_evidence_scope,
        persist_annex_checkpoint,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
        context_hash,
    )
    from onyx.regulatory.amendments.annexes.evidence import (
        freeze_review_original,
        select_annex_evidence_view,
    )
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
    from onyx.regulatory.amendments.annexes.models import (
        AnnexBaseline,
        AnnexChangeDraft,
        AnnexOriginalEvidence,
        ContextSourceRange,
        ContextSourceSnapshot,
        ExtractedAnnexElement,
        FrozenContextProjection,
        PreparedContextView,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    row = _chunk(source_session, file, 0, "old")
    store = get_default_file_store()
    old_bytes = b'<h1>EK-1</h1><p id="rate">old</p>'
    new_bytes = b'<h1>EK-1</h1><p id="rate">new</p>'
    file.file_id = store.save_file(
        BytesIO(old_bytes), "old.html", FileOrigin.OTHER, "text/html"
    )
    new_id = store.save_file(
        BytesIO(new_bytes), "new.html", FileOrigin.OTHER, "text/html"
    )
    package, _ = create_source_package(
        source_session,
        document_set_id=docset.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash=sha256(new_bytes).hexdigest(),
        input_spec={},
        created_by=file.user_id,
    )
    asset = RegulatorySourceAsset(
        package_id=package.id,
        sha256=sha256(new_bytes).hexdigest(),
        file_id=new_id,
        mime_type="text/html",
        display_name="new.html",
        byte_count=len(new_bytes),
    )
    source_session.add(asset)
    source_session.flush()
    package.status, package.asset_count = "ready", 1
    package.manifest_file_id, package.manifest_sha256 = "manifest", "m" * 64
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="EK-1 replaced",
        user_file_ids=[str(file.id)],
        created_by=file.user_id,
        status="analyzing",
        instruction_count=1,
        lease_generation=1,
        source_package_id=package.id,
        source_text_sha256=sha256(b"EK-1 replaced").hexdigest(),
    )
    source_session.add(batch)
    source_session.flush()
    scope = load_review_evidence_scope(
        source_session,
        batch_id=batch.id,
        user_file_id=file.id,
        environment="local-test",
        created_by=file.user_id,
    )
    originals = [
        AnnexOriginalEvidence(
            file_id=original_id,
            sha256=sha256(content).hexdigest(),
            mime_type="text/html",
            available=True,
            canonical_chunk_ids=[row.id],
        )
        for original_id, content in [(file.file_id, old_bytes), (new_id, new_bytes)]
    ]
    views = [
        select_annex_evidence_view(
            extraction=extract_annex_structure(content, "text/html"),
            original=original,
            annex_label="EK-1",
            canonical_labels=["EK-1"],
            canonical_chunk_ids=[row.id],
        )
        for content, original in zip([old_bytes, new_bytes], originals)
    ]
    baseline = AnnexBaseline(
        baseline_sha256="baseline",
        canonical_text="old",
        elements=[
            ExtractedAnnexElement(canonical_chunk_id=row.id, kind="text", text="old")
        ],
        originals=[originals[0]],
        visual_evidence_available=False,
    )
    comparison = compare_annexes(old=views[0], new=views[1])
    plan = prepare_annex_patch(
        baseline=baseline,
        old=views[0],
        new=views[1],
        comparison=comparison,
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )
    assert plan.ready
    canonical_scope = capture_canonical_scope(source_session, file.id)
    items = stage_canonical_items(plan=plan, baseline_scope=canonical_scope)
    candidate = items[0].new_chunks[0]
    snapshot = ContextSourceSnapshot(
        sha256=context_hash("new"),
        selector="fixture",
        reference_date=date(2026, 9, 10),
        text="new",
        ordered_ranges=[
            ContextSourceRange(canonical_chunk_id=candidate.id, start=0, end=3)
        ],
    )
    prepared = PreparedContextView(
        snapshots=[snapshot],
        projections=[
            FrozenContextProjection(
                canonical_chunk_id=candidate.id,
                source_snapshot_sha256=snapshot.sha256,
                generation_path="normal",
                request_hashes=[],
                embedding_input_sha256=context_hash(["new"]),
                embedding_config_sha256=context_hash({"model": "fixture"}),
                embedding_config={"model": "fixture"},
                embedding_texts=["new"],
                canonical_text_sha256=context_hash("new"),
                metadata_sha256=context_hash({}),
            )
        ],
    )
    impact = compare_context_views(
        old=PreparedContextView(), new=prepared, direct_canonical_changes=[candidate.id]
    )
    evidence = [
        freeze_review_original(store, scope=scope, side=side, original=original)
        for side, original in zip(["old", "new"], originals)
    ]
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=[batch.raw_text],
        annex_label="EK-1",
        user_file_id=file.id,
        source_package_id=package.id,
        source_text_sha256=batch.source_text_sha256,
        source_manifest_sha256=package.manifest_sha256,
        effective_date=plan.effective_date,
        baseline_scope=canonical_scope,
        baseline=baseline,
        old_extraction=views[0],
        new_extraction=views[1],
        comparison=comparison,
        patch_plan=plan,
        items=items,
        impact=impact,
        evidence=evidence,
    )
    change = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="local-test",
    )
    assert change is not None and change.status == "pending"
    frozen = AnnexChangeDraft.model_validate(change.review_payload)
    assert frozen.impact is not None and frozen.impact.prepared == prepared
    assert frozen.items[0].new_chunks[0].id == candidate.id
    assert source_session.get(RegulatoryChunk, candidate.id) is None
    assert row.text == "old" and row.validity_end_date is None
    from onyx.db.regulatory_annex_changes import validate_prepared_annex_change

    batch.raw_text = "Edited source text"
    source_session.flush()
    with pytest.raises(ValueError, match="source"):
        validate_prepared_annex_change(
            source_session, batch=batch, draft=draft, environment="local-test"
        )
