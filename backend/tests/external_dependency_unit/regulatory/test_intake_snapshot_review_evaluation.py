"""One snapshot shares fresh authority; later requests and writes never reuse it."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_review as review
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_self_repeal as source_only
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    UserFile,
)
from tests.external_dependency_unit.regulatory.test_intake_source_only_self_repeal_flow import (
    _create,
    _end,
    _RetainedStore,
    _seed,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def _completed_source_only_intake(
    session: Session, data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> tuple[RegulatoryDocumentIntake, _RetainedStore]:
    old_run, _, items, store, _ = _seed(session, data, monkeypatch)
    old = items[data.first_id]
    old.status, old.error = "failed", source_only.CONTINUED_ABSTENTION
    items[data.second_id].status = "completed"
    _end(old_run)
    session.flush()
    run = _create(session, data)
    assert (
        editing.reuse_completed_intake_labels(
            session, previous=old_run, replacement=run
        )
        == 2
    )
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=600
    )
    assert lease is not None
    labeling.project_derived_labels(session, lease)
    assert labeling.final_status(session, lease) == "completed"
    _end(run, "completed")
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.user_file_id == data.file_id
        )
    )
    assert member is not None
    batch = session.get_one(RegulatoryDocumentIntake, member.intake_id)
    batch.labeling_run_id = run.id
    batch.status, batch.stage = "preparing", "labeling"
    session.flush()
    return batch, store


def test_snapshot_reduces_real_source_authority_digests_without_changing_receipts(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        batch, store = _completed_source_only_intake(session, data, monkeypatch)
        with patch.object(
            intake, "review_digest", wraps=intake.review_digest
        ) as digest:
            with patch.object(store, "read_file", wraps=store.read_file) as reads:
                first = intake.intake_snapshot(session, batch)
                assert digest.call_count == 2 and reads.call_count >= 2
        expected = intake.review_digest(session, batch)
        expected_edit = editing.edit_review_digest(session, batch)
        expected_publication = review.publication_review_digest(session, batch)
        assert first["review_sha256"] == first["edit_review_sha256"] == expected
        assert first["edit_review_sha256"] == expected_edit
        assert first["publication_review_sha256"] == expected_publication
        with patch.object(
            intake, "review_digest", wraps=intake.review_digest
        ) as digest:
            second = intake.intake_snapshot(session, batch)
            assert digest.call_count == 1
            third = intake.intake_snapshot(session, batch)
            assert digest.call_count == 2
        assert second == third == first
        with Session(
            bind=session.connection(), join_transaction_mode="create_savepoint"
        ) as next_session:
            next_batch = next_session.get_one(RegulatoryDocumentIntake, batch.id)
            with patch.object(
                intake, "review_digest", wraps=intake.review_digest
            ) as digest:
                assert (
                    intake.intake_snapshot(next_session, next_batch)["review_sha256"]
                    == expected
                )
                assert digest.call_count == 1


def test_snapshot_materializes_all_reads_before_its_final_authority_check(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        batch, _ = _completed_source_only_intake(session, data, monkeypatch)
        intake.intake_snapshot(session, batch)
        original_digest = intake.review_digest
        validated = False
        later_queries: list[str] = []

        def digest(
            current_session: Session, current_batch: RegulatoryDocumentIntake
        ) -> str:
            nonlocal validated
            result = original_digest(current_session, current_batch)
            validated = True
            return result

        def executed(*arguments: object) -> None:
            if validated:
                later_queries.append(str(arguments[2]))

        monkeypatch.setattr(intake, "review_digest", digest)
        event.listen(data.database.engine, "before_cursor_execute", executed)
        try:
            intake.intake_snapshot(session, batch)
        finally:
            event.remove(data.database.engine, "before_cursor_execute", executed)
        assert validated and later_queries == []


@pytest.mark.parametrize(
    "change", ["source", "canonical", "label", "bulk", "core_bulk", "retained_bytes"]
)
def test_binding_changes_during_snapshot_revalidate_and_fail_closed(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        batch, store = _completed_source_only_intake(session, data, monkeypatch)
        intake.intake_snapshot(session, batch)
        original_context = editing.load_intake_edit_review_context

        def change_before_final_review(
            current_session: Session,
            current_batch: RegulatoryDocumentIntake,
        ) -> editing.IntakeEditReviewContext | None:
            context = original_context(current_session, current_batch)
            if change == "source":
                session.get_one(UserFile, data.file_id).file_id = str(uuid4())
            elif change == "canonical":
                row = session.get_one(RegulatoryChunk, data.first_id)
                row.chunk_metadata = {**row.chunk_metadata, "document_number": "6455"}
            elif change == "label":
                item = session.scalar(
                    select(RegulatoryLabelingItem).where(
                        RegulatoryLabelingItem.run_id == batch.labeling_run_id,
                        RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
                    )
                )
                assert item is not None
                item.labels = ["unsupported-label"]
            elif change in {"bulk", "core_bulk"}:
                statement = (
                    update(UserFile)
                    .where(UserFile.id == data.file_id)
                    .values(file_id=str(uuid4()))
                )
                if change == "core_bulk":
                    current_session.connection().execute(statement)
                    current_session.expire_all()
                else:
                    current_session.execute(statement)
            else:
                store.raw += b"changed"
            current_session.flush()
            return context

        monkeypatch.setattr(
            editing, "load_intake_edit_review_context", change_before_final_review
        )
        with patch.object(
            intake, "review_digest", wraps=intake.review_digest
        ) as digest:
            with pytest.raises(intake.IntakeConflictError):
                intake.intake_snapshot(session, batch)
            assert digest.call_count == 1


def test_new_snapshot_and_public_mutation_guards_always_reread_retained_source(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        batch, store = _completed_source_only_intake(session, data, monkeypatch)
        snapshot = intake.intake_snapshot(session, batch)
        expected = snapshot["review_sha256"]
        assert isinstance(expected, str)
        store.raw += b"changed"
        with pytest.raises(intake.IntakeConflictError):
            intake.intake_snapshot(session, batch)
        assert editing.edit_review_digest(session, batch) is None
        assert review.publication_review_digest(session, batch) is None
        with pytest.raises(intake.IntakeConflictError):
            intake.approve_intake(session, batch, expected)
