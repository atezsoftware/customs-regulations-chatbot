"""Committed labeling outcomes reach intake review without an API GET or broker."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
from onyx.db import document_intake as intake
from onyx.db import document_intake_labeling_recovery as recovery
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_labeling_semantic_repair import SEMANTIC_REPAIR_KEY
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY
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

ENVIRONMENT = "repair-test"


def _completed(session: Session, data: LabelingData) -> UUID:
    session.execute(
        delete(RegulatoryChunk).where(
            RegulatoryChunk.id.in_([data.derived_id, data.incomplete_id])
        )
    )
    run = _prepared_run(session, data, semantic=True)
    run.status, run.stage = "running", "projecting"
    batch = session.scalars(
        select(RegulatoryDocumentIntake).where(
            RegulatoryDocumentIntake.created_by == data.user_id,
            RegulatoryDocumentIntake.document_set_id == data.document_set_id,
        )
    ).one()
    batch.status, batch.stage, batch.labeling_run_id = "preparing", "labeling", run.id
    session.get_one(UserFile, data.file_id).status = UserFileStatus.CHUNKED
    for member in session.scalars(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.intake_id == batch.id
        )
    ):
        member.status = "labeling"
    session.flush()
    assert run.lease_token is not None
    lease = labeling.RunLease(run.id, run.lease_generation, run.lease_token)
    assert labeling.final_status(session, lease) == "completed"
    labeling.release_run(session, lease, status="completed", finished=True)
    session.flush()
    return batch.id


@pytest.fixture
def completed_batch(labeling_data: LabelingData) -> Generator[UUID, None, None]:
    assert POSTGRES_HOST in {"localhost", "127.0.0.1"} and str(POSTGRES_PORT) == "25432"
    assert POSTGRES_DB.startswith("rg_intake_test_")
    with Session(labeling_data.database.engine) as session:
        identifier = _completed(session, labeling_data)
        session.commit()
    try:
        yield identifier
    finally:
        with Session(labeling_data.database.engine) as session:
            packages = select(AmendmentSourcePackage.id).where(
                AmendmentSourcePackage.created_by == labeling_data.user_id
            )
            session.execute(
                delete(RegulatoryDocumentIntake).where(
                    RegulatoryDocumentIntake.created_by == labeling_data.user_id
                )
            )
            session.execute(
                delete(RegulatorySourceAsset).where(
                    RegulatorySourceAsset.package_id.in_(packages)
                )
            )
            session.execute(
                delete(AmendmentSourcePackage).where(
                    AmendmentSourcePackage.id.in_(packages)
                )
            )
            session.execute(
                delete(RegulatoryFilePublication).where(
                    RegulatoryFilePublication.user_file_id.in_(
                        select(UserFile.id).where(
                            UserFile.user_id == labeling_data.user_id
                        )
                    )
                )
            )
            session.execute(
                delete(UserFile).where(
                    UserFile.user_id == labeling_data.user_id,
                    UserFile.id != labeling_data.file_id,
                )
            )
            session.commit()


def test_completion_crash_replay_and_duplicate_delivery_need_no_get(
    labeling_data: LabelingData, completed_batch: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    get = MagicMock(side_effect=AssertionError("Recovery must not call the review API"))
    monkeypatch.setattr(intake, "intake_snapshot", get)
    with Session(labeling_data.database.engine) as session:
        source = canonical_scope_digest(session, labeling_data.file_id)
        assert recovery.reconcile_terminal_intake_labeling(
            session, environment=ENVIRONMENT
        ) == [completed_batch]
        assert (
            session.get_one(RegulatoryDocumentIntake, completed_batch).status == "ready"
        )
        # A worker crash before commit must leave the durable intent recoverable.
        session.rollback()
    with Session(labeling_data.database.engine) as session:
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        assert (batch.status, batch.stage, batch.review_sha256) == (
            "preparing",
            "labeling",
            None,
        )
        assert recovery.reconcile_terminal_intake_labeling(
            session, environment=ENVIRONMENT
        ) == [completed_batch]
        session.commit()
    with Session(labeling_data.database.engine) as session:
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        assert (batch.status, batch.stage, batch.error) == ("ready", "review", None)
        digest = batch.review_sha256
        assert digest and digest == intake.review_digest(session, batch)
        assert (
            recovery.reconcile_terminal_intake_labeling(
                session, environment=ENVIRONMENT
            )
            == []
        )
        assert batch.review_sha256 == digest
        assert canonical_scope_digest(session, labeling_data.file_id) == source
        publication = session.get_one(RegulatoryFilePublication, labeling_data.file_id)
        assert publication.epoch == 0 and publication.writer_manifest is None
        assert (
            session.scalar(select(func.count()).select_from(RegulatoryIndexingJob)) == 0
        )
    get.assert_not_called()


@pytest.mark.parametrize("terminal", ["failed", "completed_with_errors", "cancelled"])
def test_terminal_failures_become_partial_without_ready_or_retry(
    labeling_data: LabelingData, completed_batch: UUID, terminal: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        run.status, run.error = terminal, "Unresolved source evidence"
        session.commit()
    with Session(labeling_data.database.engine) as session:
        assert recovery.reconcile_terminal_intake_labeling(
            session, environment=ENVIRONMENT
        ) == [completed_batch]
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        assert (batch.status, batch.stage) == ("partial", "labeling")
        assert (
            batch.error == "Unresolved source evidence" and batch.review_sha256 is None
        )


@pytest.mark.parametrize(
    "case",
    [
        "environment",
        "owner",
        "model",
        "document_set",
        "running",
        "batch_lease",
        "run_lease",
    ],
)
def test_noncurrent_or_owned_work_is_not_reconciled(
    labeling_data: LabelingData, completed_batch: UUID, case: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        if case == "environment":
            session.get_one(
                AmendmentSourcePackage, batch.source_package_id
            ).environment = "foreign"
        elif case == "owner":
            run.requested_by_id = None
        elif case == "model":
            run.model_configuration_id = None
        elif case == "document_set":
            run.document_set_id = labeling_data.other_document_set_id
        elif case == "running":
            run.status = "running"
        elif case == "batch_lease":
            batch.lease_token, batch.lease_expires_at = (
                uuid4(),
                datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        else:
            run.lease_token, run.lease_expires_at = (
                uuid4(),
                datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        assert (
            recovery.reconcile_terminal_intake_labeling(
                session, environment=ENVIRONMENT
            )
            == []
        )
        batch = session.get_one(RegulatoryDocumentIntake, completed_batch)
        assert (batch.status, batch.stage, batch.review_sha256) == (
            "preparing",
            "labeling",
            None,
        )


def test_locked_batch_is_skipped_and_recovers_after_unlock(
    labeling_data: LabelingData, completed_batch: UUID
) -> None:
    with (
        Session(labeling_data.database.engine) as owner,
        Session(labeling_data.database.engine) as recovery_session,
    ):
        owner.get_one(RegulatoryDocumentIntake, completed_batch, with_for_update=True)
        assert (
            recovery.reconcile_terminal_intake_labeling(
                recovery_session, environment=ENVIRONMENT
            )
            == []
        )
        recovery_session.commit()
        owner.rollback()
        assert recovery.reconcile_terminal_intake_labeling(
            recovery_session, environment=ENVIRONMENT
        ) == [completed_batch]


@pytest.mark.parametrize("authority", ["semantic", "metadata"])
def test_bad_authority_rolls_back_only_its_batch_and_preserves_other_completion(
    labeling_data: LabelingData, completed_batch: UUID, authority: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        file = UserFile(
            id=uuid4(),
            user_id=labeling_data.user_id,
            file_id=str(uuid4()),
            name="Other law",
            file_type="text/markdown",
        )
        session.add(file)
        session.flush()
        session.add(
            DocumentSet__UserFile(
                document_set_id=labeling_data.other_document_set_id,
                user_file_id=file.id,
            )
        )
        identifiers = [str(uuid4()), str(uuid4())]
        for position, (identifier, body) in enumerate(
            zip(
                identifiers,
                [
                    "Importers must declare the customs value of all imported goods.",
                    "Exporters must retain the certificate of origin for five years.",
                ],
            )
        ):
            session.add(
                RegulatoryChunk(
                    id=identifier,
                    user_file_id=file.id,
                    position=position,
                    projection_ordinal=position,
                    text=body,
                    heading_path=["Other law"],
                    chunk_type="article",
                    chunk_metadata={"chunk_variant": "atomic"},
                    status="active",
                    source="indexed",
                )
            )
        session.flush()
        other = labeling_data._replace(
            document_set_id=labeling_data.other_document_set_id,
            file_id=file.id,
            first_id=identifiers[0],
            second_id=identifiers[1],
            derived_id=str(uuid4()),
            incomplete_id=str(uuid4()),
        )
        good_id = _completed(session, other)
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id
            )
        )
        assert item is not None
        if authority == "semantic":
            snapshot: dict[str, Any] = dict(item.source_snapshot)
            snapshot[SEMANTIC_REPAIR_KEY] = {
                **snapshot[SEMANTIC_REPAIR_KEY],
                "output_response_sha256": "f" * 64,
            }
            item.source_snapshot = snapshot
        else:
            row = session.get_one(RegulatoryChunk, labeling_data.first_id)
            row.chunk_metadata = {
                **row.chunk_metadata,
                "article_no": "33",
                METADATA_AUDIT_KEY: {
                    "fields": {
                        "article_no": {
                            "source": "llm",
                            "source_chunk_id": labeling_data.second_id,
                            "evidence_quote": "certificate of origin",
                        }
                    }
                },
            }
        session.commit()
    with Session(labeling_data.database.engine) as session:
        assert set(
            recovery.reconcile_terminal_intake_labeling(
                session, environment=ENVIRONMENT
            )
        ) == {completed_batch, good_id}
        bad, good = [
            session.get_one(RegulatoryDocumentIntake, identifier)
            for identifier in (completed_batch, good_id)
        ]
        assert (bad.status, bad.stage, bad.review_sha256) == (
            "failed",
            "labeling",
            None,
        )
        assert "source evidence changed" in (bad.error or "")
        assert (good.status, good.stage) == ("ready", "review")
        assert good.review_sha256 == intake.review_digest(session, good)
