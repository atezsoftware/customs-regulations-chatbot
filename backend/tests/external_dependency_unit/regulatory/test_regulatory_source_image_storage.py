"""A source image creates only a storage record and retains the parent ACL."""

import hashlib
import io
from typing import cast

import pytest
from fastapi import UploadFile
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from onyx.access.access import user_can_access_chat_file
from onyx.configs.constants import FileOrigin
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    Connector,
    FileContent,
    FileRecord,
    RegulatoryChunk,
    RegulatoryIndexingJob,
    RegulatoryTemporalProjection,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.server.features.regulatory import api
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_actual_storage_preserves_canonical_and_does_not_ingest(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = io.BytesIO()
    Image.new("RGB", (16, 24), "white").save(content, format="PNG")
    raw = content.getvalue()
    store = PostgresBackedFileStore()
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    image_id: str | None = None
    try:
        with Session(labeling_data.database.engine) as session:
            parent = session.get(UserFile, labeling_data.file_id)
            user = session.get(User, labeling_data.user_id)
            assert parent is not None and user is not None
            parent.status = UserFileStatus.COMPLETED
            session.commit()
            tables = (
                UserFile,
                Connector,
                RegulatoryChunk,
                RegulatoryIndexingJob,
                RegulatoryTemporalProjection,
                FileRecord,
                FileContent,
            )
            before_counts = [
                session.scalar(select(func.count()).select_from(model))
                for model in tables
            ]
            before_chunks = list(
                session.execute(
                    select(
                        RegulatoryChunk.id,
                        RegulatoryChunk.text,
                        RegulatoryChunk.chunk_metadata,
                    )
                )
            )
            receipt = api.upload_source_image(
                parent.id,
                UploadFile(
                    io.BytesIO(raw),
                    filename="source-page.png",
                    headers=Headers({"content-type": "image/png"}),
                ),
                user,
                session,
            )
            image_id = receipt.file_id
            session.expire_all()
            after_counts = [
                session.scalar(select(func.count()).select_from(model))
                for model in tables
            ]
            assert after_counts[:-2] == before_counts[:-2]
            assert after_counts[-2:] == [
                count + 1 for count in before_counts[-2:] if count is not None
            ]
            assert (
                list(
                    session.execute(
                        select(
                            RegulatoryChunk.id,
                            RegulatoryChunk.text,
                            RegulatoryChunk.chunk_metadata,
                        )
                    )
                )
                == before_chunks
            )
            record = session.get(FileRecord, image_id)
            assert record is not None
            assert record.file_origin == FileOrigin.OTHER
            assert record.file_type == "image/png"
            assert isinstance(record.file_metadata, dict)
            metadata = cast(dict[str, object], record.file_metadata)
            assert metadata["regulatory_user_file_id"] == str(parent.id)
            assert metadata["purpose"] == "regulatory_source_image"
            with store.read_file(image_id, mode="b") as downloaded:
                assert downloaded.read() == raw
            assert receipt.sha256 == hashlib.sha256(raw).hexdigest()
            # Storage metadata alone grants no public/chat-file access.
            assert not user_can_access_chat_file(image_id, user, session)
            assert parent.status == UserFileStatus.COMPLETED
    finally:
        if image_id is not None:
            store.delete_file(image_id)
