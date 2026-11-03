"""History summaries retain receipt bytes without loading frozen LLM payloads."""

from collections.abc import Generator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy import JSON, Column, MetaData, String, Table, Uuid, create_engine
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_review as review
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import RegulatoryDocumentIntake, RegulatoryLabelingRun, UserFile
from onyx.document_index.publication_models import publication_digest

RUN = UUID(int=4)
FILE = UUID(int=5)


@pytest.fixture
def history_session() -> Generator[Session, None, None]:
    metadata = MetaData()
    # Missing payload columns make an accidental full ORM read fail immediately.
    items = Table(
        "regulatory_labeling_item",
        metadata,
        Column("id", Uuid, primary_key=True),
        Column("run_id", Uuid),
        Column("regulatory_chunk_id", String),
        Column("user_file_id", Uuid),
        Column("status", String),
        Column("error", String),
        Column("request_hash", String),
        Column("labels", JSON),
    )
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            items.insert(),
            [
                {
                    "id": UUID(int=2),
                    "run_id": RUN,
                    "regulatory_chunk_id": "chunk-b",
                    "user_file_id": FILE,
                    "status": "failed",
                    "error": "Kaynak bulunamadı",
                    "request_hash": None,
                    "labels": [],
                },
                {
                    "id": UUID(int=1),
                    "run_id": RUN,
                    "regulatory_chunk_id": "chunk-a",
                    "user_file_id": FILE,
                    "status": "completed",
                    "error": None,
                    "request_hash": "a" * 64,
                    "labels": ["second", "first", "second"],
                },
                {
                    "id": UUID(int=3),
                    "run_id": UUID(int=99),
                    "regulatory_chunk_id": "foreign-chunk",
                    "user_file_id": UUID(int=99),
                    "status": "submitted",
                    "error": None,
                    "request_hash": None,
                    "labels": ["foreign"],
                },
            ],
        )
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def test_history_label_map_preserves_arrays_and_run_scope(
    history_session: Session,
) -> None:
    batch = RegulatoryDocumentIntake(labeling_run_id=RUN)
    labels = intake._labels(history_session, batch)
    assert labels == {"chunk-a": ["second", "first", "second"], "chunk-b": []}
    labels["chunk-a"].append("changed")
    assert intake._labels(history_session, batch)["chunk-a"] == [
        "second",
        "first",
        "second",
    ]
    batch.labeling_run_id = None
    assert intake._labels(history_session, batch) == {}


@pytest.mark.parametrize("pending", ["none", "item", "shard"])
def test_edit_receipt_projection_preserves_exact_digest_and_pending_guards(
    monkeypatch: pytest.MonkeyPatch, history_session: Session, pending: str
) -> None:
    batch = RegulatoryDocumentIntake(
        status="failed",
        stage="metadata",
        error="Saved locator evidence changed",
        labeling_run_id=RUN,
        document_set_id=15,
    )
    run = RegulatoryLabelingRun(
        id=RUN,
        status="completed_with_errors",
        file_ids=[str(FILE)],
        error="One unresolved label",
    )
    file = UserFile(id=FILE, status=UserFileStatus.CHUNKED)
    session = MagicMock(spec=Session)
    session.get.side_effect = lambda model, _id: (
        run if model is RegulatoryLabelingRun else file
    )
    session.execute.side_effect = history_session.execute
    session.scalar.return_value = UUID(int=8) if pending == "shard" else None
    session.scalars.side_effect = AssertionError("Full item payloads must not load")
    monkeypatch.setattr(labeling, "get_active_run_id", lambda *_args: None)
    monkeypatch.setattr(
        review,
        "effective_intake_files",
        lambda *_args: [SimpleNamespace(user_file_id=FILE, approval_attempt_id=None)],
    )
    monkeypatch.setattr(editing, "get_user_file_projection_repair", lambda *_args: None)
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "canonical")
    if pending == "item":
        history_session.connection().exec_driver_sql(
            "UPDATE regulatory_labeling_item SET status='submitted' WHERE regulatory_chunk_id='chunk-a'"
        )
    digest = editing.edit_review_digest(cast(Session, session), batch)
    if pending != "none":
        assert digest is None
        return
    assert digest == publication_digest(
        {
            "purpose": "unpublished_intake_edit",
            "batch_status": "failed",
            "batch_stage": "metadata",
            "batch_error": "Saved locator evidence changed",
            "canonical_review": "canonical",
            "run_id": str(RUN),
            "status": "completed_with_errors",
            "error": "One unresolved label",
            "items": [
                {
                    "id": str(UUID(int=1)),
                    "chunk_id": "chunk-a",
                    "file_id": str(FILE),
                    "status": "completed",
                    "error": None,
                    "request_hash": "a" * 64,
                },
                {
                    "id": str(UUID(int=2)),
                    "chunk_id": "chunk-b",
                    "file_id": str(FILE),
                    "status": "failed",
                    "error": "Kaynak bulunamadı",
                    "request_hash": None,
                },
            ],
        }
    )
