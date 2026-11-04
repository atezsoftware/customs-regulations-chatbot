"""All-target revision checks and atomic canonical edits use real owned PostgreSQL."""

from datetime import date, timedelta
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_chunk_edits import ChunkEditConflict
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import PublicationScope
from onyx.regulatory import writer_publication as writer
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.chunk_corrections import ChunkCorrectionBatch
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize(
    "boundary", ["ok", "stale", "foreign", "derived", "missing", "empty"]
)
def test_owned_batch_is_atomic_and_preserves_other_canonical_material(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    foreign_id = None
    with Session(labeling_data.database.engine) as session:
        parent = session.get_one(UserFile, labeling_data.file_id)
        parent.status = UserFileStatus.CHUNKED
        first = session.get_one(RegulatoryChunk, labeling_data.first_id)
        second = session.get_one(RegulatoryChunk, labeling_data.second_id)
        second.validity_start_date = date(2022, 1, 1)
        second.validity_end_date = date(2026, 10, 17)
        second.status = "superseded"
        second.superseded_by_chunk_id = first.id
        if boundary == "foreign":
            foreign = UserFile(
                id=uuid4(),
                user_id=labeling_data.user_id,
                file_id=str(uuid4()),
                name="Foreign regulation",
                file_type="text/markdown",
            )
            session.add(foreign)
            session.flush()
            foreign_id = foreign.id
            second.user_file_id = foreign.id
        elif boundary == "empty":
            second.text = ""
        session.commit()
        second_id = labeling_data.derived_id if boundary == "derived" else second.id
        inspected_second = session.get_one(RegulatoryChunk, second_id)
        expected_second = inspected_second.updated_at
        if boundary == "stale":
            expected_second -= timedelta(seconds=1)
        updates = [
            {
                "chunk_id": first.id,
                "expected_updated_at": first.updated_at,
                "text": "Reviewed exact source correction",
                "chunk_metadata": {
                    "image_file_id": "retained-source-page",
                    "reviewed": True,
                },
            },
            {
                "chunk_id": "missing-source-id" if boundary == "missing" else second_id,
                "expected_updated_at": expected_second,
                "heading_path": ["Original source chapter"],
            },
        ]
        before = capture_canonical_scope(session, parent.id)
        before_versions = {
            chunk_id: updated_at
            for chunk_id, updated_at in session.execute(
                select(RegulatoryChunk.id, RegulatoryChunk.updated_at).where(
                    RegulatoryChunk.user_file_id == parent.id
                )
            )
        }
    scope = PublicationScope(
        tenant_id=labeling_data.database.schema,
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(
        labeling_data.file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
    )
    prepare, publish = (
        Mock(side_effect=AssertionError("deferred batch must not call provider/index")),
        Mock(side_effect=AssertionError("deferred batch must not publish")),
    )
    monkeypatch.setattr(
        "onyx.regulatory.writer_projection.prepare_owned_correction", prepare
    )
    monkeypatch.setattr(writer, "execute_writer_publication", publish)
    try:
        request = ChunkCorrectionBatch.model_validate({"updates": updates})
        if boundary == "ok":
            writer.correct_owned_chunks(owner, MagicMock(), request)
        else:
            with pytest.raises(ChunkEditConflict):
                writer.correct_owned_chunks(owner, MagicMock(), request)
        with Session(labeling_data.database.engine) as session:
            after = capture_canonical_scope(session, labeling_data.file_id)
            after_versions = {
                chunk_id: updated_at
                for chunk_id, updated_at in session.execute(
                    select(RegulatoryChunk.id, RegulatoryChunk.updated_at).where(
                        RegulatoryChunk.user_file_id == labeling_data.file_id
                    )
                )
            }
            if boundary == "ok":
                expected = [
                    row.model_copy(
                        update={
                            "text": "Reviewed exact source correction",
                            "metadata": {
                                "image_file_id": "retained-source-page",
                                "reviewed": True,
                            },
                        }
                    )
                    if row.id == labeling_data.first_id
                    else row.model_copy(
                        update={"heading_path": ["Original source chapter"]}
                    )
                    if row.id == labeling_data.second_id
                    else row
                    for row in before
                ]
                assert after == expected
                for chunk_id, inspected_at in before_versions.items():
                    if chunk_id not in {
                        labeling_data.first_id,
                        labeling_data.second_id,
                    }:
                        assert after_versions[chunk_id] == inspected_at
            else:
                assert after == before
                assert after_versions == before_versions
        prepare.assert_not_called()
        publish.assert_not_called()
    finally:
        authority.release(owner)
        if foreign_id is not None:
            with Session(labeling_data.database.engine) as session:
                session.execute(delete(UserFile).where(UserFile.id == foreign_id))
                session.commit()
