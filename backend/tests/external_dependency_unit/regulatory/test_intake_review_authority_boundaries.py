"""Normal intake GET and approval retain the authority of the reviewed receipts."""

from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntake,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_readers import STRUCTURAL_READER_KEY
from onyx.db.regulatory_labeling_semantic_repair import SEMANTIC_REPAIR_KEY
from onyx.document_index.publication_models import publication_digest
from tests.external_dependency_unit.regulatory.test_intake_labeling_review_authority import (
    _prepared_run,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def _ready_review(
    session: Session, data: LabelingData, *, reader: bool = False
) -> RegulatoryDocumentIntake:
    session.execute(
        delete(RegulatoryChunk).where(
            RegulatoryChunk.id.in_([data.derived_id, data.incomplete_id])
        )
    )
    run = _prepared_run(session, data, reader=reader, semantic=not reader)
    run.stage, run.completed_chunks = "finished", run.total_chunks
    run.lease_token, run.lease_expires_at = None, None
    session.get_one(UserFile, data.file_id).status = UserFileStatus.CHUNKED
    batch = session.scalar(
        select(RegulatoryDocumentIntake).where(
            RegulatoryDocumentIntake.document_set_id == data.document_set_id
        )
    )
    assert batch is not None and batch.preparation_policy is None
    batch.status, batch.stage, batch.labeling_run_id = "preparing", "labeling", run.id
    session.flush()
    assert intake.intake_snapshot(session, batch)["status"] == "ready"
    assert batch.review_sha256 == intake.review_digest(session, batch)
    assert (
        intake.intake_snapshot(session, batch)["review_sha256"] == batch.review_sha256
    )
    return batch


@pytest.mark.parametrize(
    "tamper", ["audit", "context", "remove_audit", "remove_proof", "child_audit"]
)
def test_normal_get_and_approve_reject_changed_or_removed_new_authority(
    labeling_data: LabelingData, tamper: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        reader = tamper in {"remove_proof", "child_audit"}
        batch = _ready_review(session, labeling_data, reader=reader)
        before = publication_digest(
            [
                row.model_dump(mode="json")
                for row in capture_canonical_scope(session, labeling_data.file_id)
            ]
        )
        frozen = batch.review_sha256
        assert frozen is not None and batch.labeling_run_id is not None
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == batch.labeling_run_id,
                RegulatoryLabelingItem.regulatory_chunk_id
                == (labeling_data.second_id if reader else labeling_data.first_id),
            )
        )
        assert item is not None
        if tamper == "remove_proof":
            projection = session.scalar(
                select(RegulatoryDerivedLabelProjection).where(
                    RegulatoryDerivedLabelProjection.run_id == batch.labeling_run_id,
                    RegulatoryDerivedLabelProjection.regulatory_chunk_id
                    == labeling_data.first_id,
                )
            )
            assert projection is not None
            projection.source_snapshot = {
                key: value
                for key, value in projection.source_snapshot.items()
                if key != STRUCTURAL_READER_KEY
            }
        elif tamper == "remove_audit":
            item.source_snapshot = {
                key: value
                for key, value in item.source_snapshot.items()
                if key != SEMANTIC_REPAIR_KEY
            }
        elif tamper == "context":
            item.context_snapshot = (item.context_snapshot or "") + " changed"
        else:
            item.source_snapshot = {**item.source_snapshot, SEMANTIC_REPAIR_KEY: None}
        session.flush()
        with pytest.raises(intake.IntakeConflictError):
            intake.intake_snapshot(session, batch)
        with pytest.raises(intake.IntakeConflictError):
            intake.approve_intake(session, batch, frozen)
        assert batch.status == "ready" and batch.review_sha256 == frozen
        assert (
            session.get_one(RegulatoryFilePublication, labeling_data.file_id).epoch == 0
        )
        assert session.scalar(select(UserFileProjectionRepair.user_file_id)) is None
        assert (
            publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, labeling_data.file_id)
                ]
            )
            == before
        )


def test_legacy_review_digest_payload_and_ready_get_are_unchanged(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        run = _prepared_run(session, labeling_data)
        run.lease_token, run.lease_expires_at = None, None
        batch = session.scalar(
            select(RegulatoryDocumentIntake).where(
                RegulatoryDocumentIntake.document_set_id
                == labeling_data.document_set_id
            )
        )
        assert batch is not None
        batch.labeling_run_id, batch.status, batch.stage = run.id, "ready", "review"
        for row in session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == labeling_data.file_id
            )
        ):
            row.chunk_metadata = {
                key: value
                for key, value in row.chunk_metadata.items()
                if key != "intake_preparation_policy"
            } | {"intake_chunking_version": "3"}
        session.flush()
        captured: list[dict[str, Any]] = []

        def capture(payload: dict[str, Any]) -> str:
            captured.append(payload)
            return publication_digest(payload)

        monkeypatch.setattr(intake, "publication_digest", capture)
        digest = intake.review_digest(session, batch)
        assert set(captured[-1]) == {"source_package_id", "files", "chunks"}
        assert digest == publication_digest(captured[-1])
        batch.review_sha256 = "f" * 64
        session.flush()
        assert intake.intake_snapshot(session, batch)["status"] == "ready"
        assert batch.review_sha256 == "f" * 64
