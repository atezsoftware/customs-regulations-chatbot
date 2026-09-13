from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import RegulatoryChunk, User, UserRole
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.regulatory import publication_reads
from onyx.server.features.regulatory import api
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.fixture
def chunk_client(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, MagicMock], None, None]:
    def session_dependency() -> Generator[Session, None, None]:
        with Session(labeling_data.database.engine) as session:
            yield session

    def user_dependency() -> User:
        with Session(labeling_data.database.engine) as session:
            user = session.get(User, labeling_data.user_id)
            assert user is not None
            session.expunge(user)
            return user

    store = MagicMock()
    store.observe.return_value = ReadObservation(
        scope=PublicationScope(
            tenant_id=labeling_data.database.schema,
            environment="local-test",
            database_identity="labeling-test",
        ),
        committed_epoch=0,
    )
    store.unavailable.return_value = frozenset()
    monkeypatch.setattr(publication_reads, "public_read_store", lambda: store)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router)
    app.dependency_overrides[get_session] = session_dependency
    app.dependency_overrides[current_user] = user_dependency
    with TestClient(app) as client:
        yield client, store


def test_chunk_pages_load_only_requested_rows_and_preserve_legacy_response(
    labeling_data: LabelingData, chunk_client: tuple[TestClient, MagicMock]
) -> None:
    client, _store = chunk_client
    path = f"/regulatory/files/{labeling_data.file_id}/chunks"
    with Session(labeling_data.database.engine) as session:
        expected = list(
            session.scalars(
                select(RegulatoryChunk.id)
                .where(RegulatoryChunk.user_file_id == labeling_data.file_id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            )
        )
    loaded: list[str] = []

    def record_load(_session: Session, instance: object) -> None:
        if isinstance(instance, RegulatoryChunk):
            loaded.append(instance.id)

    event.listen(Session, "loaded_as_persistent", record_load)
    try:
        for offset in (0, 2, len(expected)):
            loaded.clear()
            response = client.get(f"{path}/page", params={"offset": offset, "limit": 2})
            assert response.status_code == 200
            page = response.json()
            assert page["total"] == len(expected)
            assert page["offset"] == offset
            assert page["limit"] == 2
            ids = [item["id"] for item in page["items"]]
            assert ids == expected[offset : offset + 2]
            assert loaded == ids
    finally:
        event.remove(Session, "loaded_as_persistent", record_load)
    legacy = client.get(path)
    assert legacy.status_code == 200
    assert [item["id"] for item in legacy.json()] == expected


def test_chunk_page_keeps_publication_and_admin_guards(
    labeling_data: LabelingData, chunk_client: tuple[TestClient, MagicMock]
) -> None:
    client, store = chunk_client
    path = f"/regulatory/files/{labeling_data.file_id}/chunks/page"
    store.unavailable.return_value = frozenset({labeling_data.file_id})
    assert client.get(path, params={"offset": 100}).status_code == 503
    store.unavailable.return_value = frozenset()
    with Session(labeling_data.database.engine) as session:
        user = session.get(User, labeling_data.user_id)
        assert user is not None
        user.role = UserRole.BASIC
        user.effective_permissions = []
        session.commit()
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("query", [{"offset": -1}, {"limit": 0}, {"limit": 101}])
def test_chunk_page_rejects_unbounded_requests(
    labeling_data: LabelingData,
    chunk_client: tuple[TestClient, MagicMock],
    query: dict[str, int],
) -> None:
    client, _store = chunk_client
    path = f"/regulatory/files/{labeling_data.file_id}/chunks/page"
    assert client.get(path, params=query).status_code == 422
