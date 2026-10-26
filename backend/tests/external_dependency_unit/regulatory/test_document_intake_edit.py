"""Unpublished corrections retain history and cannot change an existing corpus file."""

import datetime
from copy import deepcopy
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
)
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def _ready(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    *,
    aggregate: bool = False,
    metadata_audit: bool = False,
) -> tuple[RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile]]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    root = next(member for member in members if member.role == "root")
    first = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == root.user_file_id)
    )
    assert first is not None
    second = RegulatoryChunk(
        id=f"rc_edit_neighbor_{uuid4().hex}",
        user_file_id=root.user_file_id,
        text="MADDE 2 — İlgili koşullar birlikte uygulanır.",
        position=1,
        projection_ordinal=1,
        heading_path=["root", "MADDE 2"],
        chunk_type="article",
        chunk_metadata={},
        source="indexed",
        status="active",
    )
    session.add(second)
    if metadata_audit:
        second.text += " Bu hüküm 01/01/2024 tarihinde yürürlüğe girer. Belge tarihi 31/12/2023'tür."
        for row in (first, second):
            row.validity_start_date = datetime.date(2024, 1, 1)
            row.chunk_metadata = {
                **row.chunk_metadata,
                "document_date": "2023-12-31",
                "intake_legal_metadata": {
                    "fields": {
                        field: {
                            "source": "llm",
                            "stale": False,
                            "source_chunk_id": second.id,
                            "evidence_quote": "01/01/2024"
                            if field == "validity_start_date"
                            else "31/12/2023",
                        }
                        for field in ("validity_start_date", "document_date")
                    },
                    "input_stale": False,
                    "warnings": [],
                },
            }
    if aggregate:
        session.add(
            RegulatoryChunk(
                id=f"rc_edit_aggregate_{uuid4().hex}",
                user_file_id=root.user_file_id,
                text=f"root\n\n{first.text}\n\n{second.text}",
                position=2,
                projection_ordinal=2,
                heading_path=["root"],
                chunk_type="hierarchical_aggregate",
                chunk_metadata={
                    "chunk_variant": "hierarchical_aggregate",
                    "hierarchy_root_path": ["root"],
                    "source_regulatory_chunk_ids": [first.id, second.id],
                },
                source="indexed",
                status="active",
            )
        )
    session.flush()
    seed = _completed_labels(session, batch, members)
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="intake-edit-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    monkeypatch.setattr(
        editing, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    run, _created = labeling.create_labeling_run(
        session,
        document_set_id=target.id,
        taxonomy=seed.taxonomy,
        model_configuration_id=model.id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=user.id,
        idempotency_key=uuid4(),
        target_file_ids=[member.user_file_id for member in members],
    )
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    for item in labeling.prepare_next_item_page(session, lease, limit=128):
        item.status = "completed"
        item.labels = ["Gümrük"]
        assignment: dict[str, object] = {
            "label": "Gümrük",
            "evidence": item.text_snapshot,
        }
        item.assignments = [assignment]
        item.request_hash = uuid4().hex * 2
        request: dict[str, object] = {"custom_id": item.request_hash}
        item.request_payload = request
    run.status, run.stage, run.completed_chunks = (
        "completed",
        "finished",
        run.total_chunks,
    )
    run.lease_token, run.lease_expires_at = None, None
    batch.labeling_run_id, batch.status, batch.stage = run.id, "preparing", "labeling"
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.review_sha256
    return batch, members


def _root_chunk(
    session: Session, members: list[RegulatoryDocumentIntakeFile]
) -> tuple[RegulatoryDocumentIntakeFile, RegulatoryChunk]:
    member = next(member for member in members if member.role == "root")
    chunk = session.scalar(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == member.user_file_id,
            RegulatoryChunk.position == 0,
        )
    )
    assert chunk is not None
    return member, chunk


def _partial(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_role: str = "root",
) -> tuple[RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile]]:
    batch, members = _ready(session, owner, monkeypatch)
    failed_member = next(member for member in members if member.role == failed_role)
    run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    failed_item = session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == run.id,
            RegulatoryLabelingItem.user_file_id == failed_member.user_file_id,
        )
    )
    assert failed_item
    failed_item.status, failed_item.error = "failed", "insufficient_evidence"
    failed_item.labels, failed_item.assignments = [], []
    run.status, run.error = "completed_with_errors", "One chunk needs review"
    run.completed_chunks -= 1
    run.failed_chunks = 1
    batch.status, batch.stage, batch.review_sha256 = "preparing", "labeling", None
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "partial" and batch.review_sha256 is None
    return batch, members


@pytest.mark.parametrize("failed_role", ["root", "annex"])
def test_terminal_partial_edit_receipt_relabels_changed_file_and_preserves_other_results(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    failed_role: str,
) -> None:
    batch, members = _partial(
        intake_session, intake_owner, monkeypatch, failed_role=failed_role
    )
    root, chunk = _root_chunk(intake_session, members)
    previous_id = batch.labeling_run_id
    previous_items = {
        item.id: (item.status, item.error, deepcopy(item.labels), item.text_snapshot)
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous_id
            )
        )
    }
    receipt = editing.edit_review_digest(intake_session, batch)
    assert receipt and batch.review_sha256 is None
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, expected_review_sha256=receipt)
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=receipt,
        updates={"text": "MADDE 1 — Onay öncesi düzeltilen beyan yükümlülüğü."},
    )
    assert run_id and run_id != previous_id
    replacement = intake_session.get_one(RegulatoryLabelingRun, run_id)
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    assert set(replacement.file_ids) == {str(member.user_file_id) for member in members}
    assert all(
        item.status == "pending"
        for item in items
        if item.user_file_id == root.user_file_id
    )
    annex = next(item for item in items if item.user_file_id != root.user_file_id)
    assert annex.status == ("completed" if failed_role == "root" else "pending")
    assert replacement.completed_chunks == (1 if failed_role == "root" else 0)
    assert {
        item.id: (item.status, item.error, item.labels, item.text_snapshot)
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous_id
            )
        )
    } == previous_items
    assert batch.review_sha256 is None
    assert editing.edit_review_digest(intake_session, batch) is None
    with pytest.raises(intake.IntakeConflictError):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=receipt,
            updates={"text": "Cannot edit while replacement labels are queued"},
        )
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, expected_review_sha256=receipt)


@pytest.mark.parametrize("changed_field", ["error", "status", "canonical_type"])
def test_partial_receipt_rejects_stale_label_outcome_or_canonical_review(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    batch, members = _partial(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    receipt = editing.edit_review_digest(intake_session, batch)
    assert receipt
    failed = intake_session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == batch.labeling_run_id,
            RegulatoryLabelingItem.status == "failed",
        )
    )
    assert failed
    if changed_field == "canonical_type":
        chunk.chunk_type = "free_text"
    else:
        setattr(
            failed,
            changed_field,
            "cancelled" if changed_field == "status" else "Changed failure",
        )
    intake_session.flush()
    assert editing.edit_review_digest(intake_session, batch) != receipt
    original = chunk.text
    with pytest.raises(intake.IntakeConflictError, match="review changed"):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=receipt,
            updates={"text": "Stale correction"},
        )
    assert chunk.text == original


def test_partial_receipt_requires_terminal_unleased_unpublished_run(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _partial(intake_session, intake_owner, monkeypatch)
    run = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    assert editing.edit_review_digest(intake_session, batch)
    run.lease_token = uuid4()
    run.lease_expires_at = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=5)
    intake_session.flush()
    assert editing.edit_review_digest(intake_session, batch) is None
    run.lease_token, run.lease_expires_at = None, None
    for status in ("queued", "running"):
        run.status = status
        intake_session.flush()
        assert editing.edit_review_digest(intake_session, batch) is None
    run.status = "completed_with_errors"
    members[-1].approval_attempt_id = uuid4()
    intake_session.flush()
    assert editing.edit_review_digest(intake_session, batch) is None


@pytest.mark.parametrize("field", ["chunk_type", "position"])
def test_ready_receipt_rejects_underlying_canonical_semantic_change(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    receipt = batch.review_sha256
    assert receipt and editing.edit_review_digest(intake_session, batch) == receipt
    setattr(chunk, field, "free_text" if field == "chunk_type" else 100)
    intake_session.flush()
    assert intake.review_digest(intake_session, batch) != receipt
    assert editing.edit_review_digest(intake_session, batch) is None
    with pytest.raises(intake.IntakeConflictError):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=receipt,
            updates={"text": "Stale correction"},
        )
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, expected_review_sha256=receipt)


def test_edit_relabels_changed_file_and_reuses_exact_current_annex_labels(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    digest = batch.review_sha256
    assert digest and batch.labeling_run_id
    previous_id = batch.labeling_run_id
    previous_items = {
        item.regulatory_chunk_id: deepcopy(item.labels)
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous_id
            )
        )
    }
    before_text, before_metadata = chunk.text, deepcopy(chunk.chunk_metadata)
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=digest,
        updates={
            "text": "MADDE 1 — Düzeltilmiş mevzuat metni.",
            "heading_path": ["Ana belge", "MADDE 1"],
        },
    )
    assert run_id and run_id != previous_id
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    assert set(run.file_ids) == {str(member.user_file_id) for member in members}
    assert run.model == "gemini-3.5-flash" and run.completed_chunks == 1
    assert len([item for item in items if item.status == "pending"]) == 2
    assert all(
        item.status == "pending" and not item.labels
        for item in items
        if item.user_file_id == root.user_file_id
    )
    annex = next(item for item in items if item.user_file_id != root.user_file_id)
    assert (
        annex.status == "completed"
        and annex.labels == previous_items[annex.regulatory_chunk_id]
    )
    assert labeling.current_labeling_items_for_search(
        intake_session, run=run, items=[annex]
    ) == frozenset({annex.regulatory_chunk_id})
    assert batch.status == "preparing" and batch.review_sha256 is None
    assert all(
        chunk.chunk_metadata.get(key) == value for key, value in before_metadata.items()
    )
    assert chunk.chunk_metadata["heading_path"] == chunk.heading_path
    assert (
        chunk.chunk_metadata["intake_legal_metadata"]["fields"]["heading_path"][
            "source"
        ]
        == "manual"
    )
    assert UUID(chunk.chunk_metadata["intake_edit_revision"])
    revisions = list(
        intake_session.scalars(
            select(RegulatoryCanonicalRevision).where(
                RegulatoryCanonicalRevision.canonical_chunk_id == chunk.id
            )
        )
    )
    assert {row.payload["text"] for row in revisions} == {before_text, chunk.text}
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, expected_review_sha256=digest)
    assert {
        item.regulatory_chunk_id: item.labels
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous_id
            )
        )
    } == previous_items
    lease = labeling.claim_run(
        intake_session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    pending = labeling.prepare_next_item_page(intake_session, lease, limit=128)
    assert len(pending) == 2
    for item in pending:
        item.status, item.labels = "completed", ["Gümrük"]
    run.status, run.completed_chunks = "completed", run.total_chunks
    run.lease_token, run.lease_expires_at = None, None
    intake_session.flush()
    intake.refresh_intake(intake_session, batch)
    assert batch.status == "ready" and batch.review_sha256 != digest
    assert batch.review_sha256
    deliveries = intake.approve_intake(
        intake_session, batch, expected_review_sha256=batch.review_sha256
    )
    assert {file_id for file_id, _token in deliveries} == {
        member.user_file_id for member in members
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"text": " "},
        {"text": "x" * (editing.MAX_EDIT_TEXT_CHARS + 1)},
        {"heading_path": None},
        {"chunk_type": "hierarchical_aggregate"},
        {"chunk_metadata": {"labels": ["Injected"]}},
        {
            "validity_start_date": datetime.date(2026, 2, 1),
            "validity_end_date": datetime.date(2026, 1, 1),
        },
    ],
)
def test_edit_rejects_invalid_or_generated_fields_without_mutation(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, Any],
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    digest, original = batch.review_sha256, chunk.text
    assert digest
    with pytest.raises(ValueError):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=digest,
            updates=updates,
        )
    assert (
        chunk.text == original
        and batch.review_sha256 == digest
        and batch.status == "ready"
    )


def test_noop_scope_stale_and_published_guards(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    digest = batch.review_sha256
    assert digest
    assert (
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=digest,
            updates={"text": chunk.text},
        )
        is None
    )
    annex = next(member for member in members if member.role == "annex")
    with pytest.raises(ValueError, match="does not belong"):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=annex.id,
            chunk_id=chunk.id,
            expected_review_sha256=digest,
            updates={"text": "Bad scope"},
        )
    with pytest.raises(intake.IntakeConflictError, match="review changed"):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256="0" * 64,
            updates={"text": "Stale"},
        )
    intake.approve_intake(intake_session, batch, expected_review_sha256=digest)
    with pytest.raises(intake.IntakeConflictError):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=digest,
            updates={"text": "After approval"},
        )


def test_edit_rebuilds_derived_text_but_rejects_direct_aggregate_edit(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch, aggregate=True)
    root, chunk = _root_chunk(intake_session, members)
    aggregate = intake_session.scalar(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.chunk_type == "hierarchical_aggregate",
        )
    )
    assert aggregate and batch.review_sha256
    with pytest.raises(ValueError, match="canonical chunks"):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=aggregate.id,
            expected_review_sha256=batch.review_sha256,
            updates={"text": "Overwrite derived"},
        )
    before = aggregate.text
    editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={"text": "YENİ KANONİK METİN"},
    )
    assert aggregate.text != before and "YENİ KANONİK METİN" in aggregate.text


def test_nullable_dates_and_chunk_type_survive_frozen_relabel_snapshot(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    assert batch.review_sha256
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={
            "validity_start_date": datetime.date(2026, 10, 26),
            "validity_end_date": None,
            "chunk_type": None,
        },
    )
    item = intake_session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == run_id,
            RegulatoryLabelingItem.regulatory_chunk_id == chunk.id,
        )
    )
    assert item and item.source_snapshot["validity_start_date"] == "2026-10-26"
    assert (
        item.source_snapshot["validity_end_date"] is None
        and item.source_snapshot["chunk_type"] is None
    )


def test_type_only_edit_changes_review_digest_but_noop_adds_no_marker(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    before = intake.review_digest(intake_session, batch)
    assert batch.review_sha256 == before
    assert (
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=before,
            updates={"chunk_type": chunk.chunk_type},
        )
        is None
    )
    assert "intake_edit_revision" not in chunk.chunk_metadata
    assert intake.review_digest(intake_session, batch) == before
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=before,
        updates={"chunk_type": "free_text"},
    )
    assert run_id and chunk.chunk_type == "free_text"
    assert UUID(chunk.chunk_metadata["intake_edit_revision"])
    assert intake.review_digest(intake_session, batch) != before


def test_manual_text_edit_normalizes_only_resolved_annex_navigation_before_relabel(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    assert batch.review_sha256
    graph = deepcopy(root.relationships)
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={
            "text": "Beyanname zorunludur.\n\n[Ekler için tıklayınız](https://example.gov/EK-1.html)"
        },
    )
    assert "tıklayınız" not in chunk.text
    assert "[EK-1](https://example.gov/EK-1.html)" in chunk.text
    assert root.relationships == graph
    item = intake_session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == run_id,
            RegulatoryLabelingItem.regulatory_chunk_id == chunk.id,
        )
    )
    assert item and item.text_snapshot == chunk.text


@pytest.mark.parametrize("manual_date", [False, True])
@pytest.mark.parametrize("context_field", ["text", "heading_path", "chunk_type"])
def test_context_edit_invalidates_cross_chunk_llm_dates_without_marking_unchanged_form_fields_manual(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    manual_date: bool,
    context_field: str,
) -> None:
    batch, members = _ready(
        intake_session, intake_owner, monkeypatch, metadata_audit=True
    )
    root, chunk = _root_chunk(intake_session, members)
    assert batch.review_sha256
    supplied_start = (
        datetime.date(2024, 2, 1) if manual_date else chunk.validity_start_date
    )
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={
            "text": "MADDE 1 — Düzeltilmiş beyan yükümlülüğü."
            if context_field == "text"
            else chunk.text,
            "heading_path": ["Düzeltilmiş ana belge", "MADDE 1"]
            if context_field == "heading_path"
            else list(chunk.heading_path),
            "chunk_type": "free_text"
            if context_field == "chunk_type"
            else chunk.chunk_type,
            "validity_start_date": supplied_start,
            "validity_end_date": None,
            "chunk_metadata": {
                "document_date": "2023-12-31",
                "document_number": "5000",
            },
        },
    )
    assert run_id
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == root.user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    assert rows[0].validity_start_date == (supplied_start if manual_date else None)
    assert rows[1].validity_start_date is None
    for row in rows:
        assert "document_date" not in row.chunk_metadata
        audit = row.chunk_metadata["intake_legal_metadata"]
        assert audit["input_stale"]
        assert (
            audit["fields"]["document_date"]["source"] == "llm"
            and audit["fields"]["document_date"]["stale"]
        )
    assert chunk.chunk_metadata["document_number"] == "5000"
    assert (
        chunk.chunk_metadata["intake_legal_metadata"]["fields"]["document_number"][
            "source"
        ]
        == "manual"
    )
    assert chunk.chunk_metadata["intake_legal_metadata"]["fields"][
        "validity_start_date"
    ]["source"] == ("manual" if manual_date else "llm")
    if context_field == "heading_path":
        assert chunk.chunk_metadata["heading_path"] == chunk.heading_path
        assert (
            chunk.chunk_metadata["intake_legal_metadata"]["fields"]["heading_path"][
                "source"
            ]
            == "manual"
        )


def _retry_labeling(
    session: Session,
    batch: RegulatoryDocumentIntake,
    members: list[RegulatoryDocumentIntakeFile],
    monkeypatch: pytest.MonkeyPatch,
) -> RegulatoryLabelingRun:
    previous = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    binding = LabelingProviderBinding.model_validate(previous.provider_binding)
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    batch.status, batch.stage = "partial", "labeling"
    session.flush()
    assert intake.retry_intake(session, batch) == []
    token = intake.claim_intake(session, batch.id)
    assert token is not None
    for member in members:
        intake.mark_intake_file_prepared(session, batch.id, token, member.id)
    run_id = intake.start_intake_labeling(session, batch.id, token)
    return session.get_one(RegulatoryLabelingRun, run_id)


def test_failed_edited_run_retry_reuses_annex_and_completed_main_item(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    assert batch.review_sha256
    edited_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={"text": "MADDE 1 — İncelenmiş yeni metin."},
    )
    assert edited_id is not None
    edited = intake_session.get_one(RegulatoryLabelingRun, edited_id)
    lease = labeling.claim_run(
        intake_session, run_id=edited.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    pending = labeling.prepare_next_item_page(intake_session, lease, limit=128)
    assert len(pending) == 2
    for item in pending:
        item.status = "failed" if item.regulatory_chunk_id == chunk.id else "completed"
        item.labels = [] if item.status == "failed" else ["Tamamlanan ana etiket"]
        item.assignments = []
    edited.status, edited.stage, edited.error = "failed", "finished", "Provider failed"
    edited.completed_chunks, edited.failed_chunks = 2, 1
    edited.lease_token, edited.lease_expires_at = None, None
    intake_session.flush()
    previous_items = {
        item.regulatory_chunk_id: (
            item.status,
            deepcopy(item.labels),
            deepcopy(item.assignments),
        )
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == edited.id
            )
        )
    }
    replacement = _retry_labeling(intake_session, batch, members, monkeypatch)
    assert replacement.id != edited.id
    assert replacement.taxonomy_id == edited.taxonomy_id
    assert replacement.model == edited.model == "gemini-3.5-flash"
    assert replacement.completed_chunks == 2
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id
            )
        )
    )
    assert {item.regulatory_chunk_id for item in items if item.status == "pending"} == {
        chunk.id
    }
    assert all(
        item.labels == previous_items[item.regulatory_chunk_id][1]
        for item in items
        if item.status == "completed"
    )
    assert chunk.text == "MADDE 1 — İncelenmiş yeni metin."
    assert {
        item.regulatory_chunk_id: (item.status, item.labels, item.assignments)
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == edited.id
            )
        )
    } == previous_items


@pytest.mark.parametrize("stale", [False, True])
def test_failed_initial_run_retry_copies_only_current_completed_results(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    stale: bool,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    previous = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    item = intake_session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == previous.id,
            RegulatoryLabelingItem.regulatory_chunk_id == chunk.id,
        )
    )
    assert item is not None
    if stale:
        chunk.text = "Başka incelemede değişmiş metin"
    else:
        item.status, item.labels = "failed", []
    previous.status, previous.stage = "failed", "finished"
    intake_session.flush()
    replacement = _retry_labeling(intake_session, batch, members, monkeypatch)
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id
            )
        )
    )
    if stale:
        assert replacement.completed_chunks == 1
        assert all(
            item.status == "pending"
            for item in items
            if item.user_file_id == root.user_file_id
        )
    else:
        assert replacement.completed_chunks == 2
        assert {
            item.regulatory_chunk_id for item in items if item.status == "pending"
        } == {chunk.id}
    assert all(
        item.status == "completed"
        for item in items
        if item.user_file_id != root.user_file_id
    )
