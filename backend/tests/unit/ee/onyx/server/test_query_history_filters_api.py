from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ee.onyx.server.query_history import api
from onyx.auth.users import current_user
from onyx.configs.constants import QueryHistoryType
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.error_handling.exceptions import register_onyx_exception_handlers


@pytest.fixture
def history_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, User]:
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    register_onyx_exception_handlers(app)
    user = User(
        id=uuid4(), effective_permissions=[Permission.FULL_ADMIN_PANEL_ACCESS.value]
    )
    app.dependency_overrides[current_user] = lambda: user
    app.dependency_overrides[get_session] = lambda: MagicMock()
    monkeypatch.setattr(api.client_app, "send_task", MagicMock())
    monkeypatch.setattr(api, "register_task", MagicMock())
    reads = MagicMock()
    reads.page.return_value = []
    reads.count.return_value = 0
    reads.settings.return_value = SimpleNamespace(
        query_history_type=QueryHistoryType.NORMAL
    )
    monkeypatch.setattr(api, "get_page_of_chat_sessions", reads.page)
    monkeypatch.setattr(api, "get_total_filtered_chat_sessions_count", reads.count)
    monkeypatch.setattr(api, "get_response_usage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api, "load_settings", reads.settings)
    return TestClient(app), reads, user


def test_history_http_passes_same_filters_to_page_and_count(
    history_client: tuple[TestClient, MagicMock, User],
) -> None:
    client, reads, _ = history_client
    user_id = uuid4()
    response = client.get(
        "/api/admin/chat-session-history",
        params={
            "user_id": str(user_id),
            "user_email": "person@example.com",
            "user_query": "Çağrı Şahin",
            "api_key_query": "Customer Portal",
            "q": "duty %_",
            "page_num": 2,
            "page_size": 20,
            "session_rating": 4,
            "feedback_type": "like",
        },
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"items": [], "total_items": 0}
    page_kwargs = reads.page.call_args.kwargs.copy()
    assert page_kwargs.pop("page_num") == 2
    assert page_kwargs.pop("page_size") == 20
    assert page_kwargs == reads.count.call_args.kwargs
    assert page_kwargs["user_id"] == user_id
    assert page_kwargs["user_email"] == "person@example.com"
    assert page_kwargs["user_query"] == "Çağrı Şahin"
    assert page_kwargs["api_key_query"] == "Customer Portal"
    assert page_kwargs["q"] == "duty %_"


@pytest.mark.parametrize(
    "identity", ["user_id", "user_email", "user_query", "api_key_query"]
)
def test_anonymized_history_rejects_identity_filter_oracle(
    history_client: tuple[TestClient, MagicMock, User], identity: str
) -> None:
    client, reads, _ = history_client
    reads.settings.return_value.query_history_type = QueryHistoryType.ANONYMIZED
    value = str(uuid4()) if identity == "user_id" else "private identity"
    response = client.get("/api/admin/chat-session-history", params={identity: value})
    assert response.status_code == 403
    reads.page.assert_not_called()
    reads.count.assert_not_called()


@pytest.mark.parametrize("mode", [QueryHistoryType.NORMAL, QueryHistoryType.ANONYMIZED])
def test_history_unfiltered_contract_and_content_search_remain_available(
    history_client: tuple[TestClient, MagicMock, User], mode: QueryHistoryType
) -> None:
    client, reads, _ = history_client
    reads.settings.return_value.query_history_type = mode
    assert client.get("/api/admin/chat-session-history").json() == {
        "items": [],
        "total_items": 0,
    }
    assert (
        client.get("/api/admin/chat-session-history", params={"q": "duty"}).status_code
        == 200
    )


def test_history_still_requires_admin_and_respects_disabled_setting(
    history_client: tuple[TestClient, MagicMock, User],
) -> None:
    client, reads, user = history_client
    user.effective_permissions = [Permission.BASIC_ACCESS.value]
    assert (
        client.get(
            "/api/admin/chat-session-history", params={"user_query": "someone"}
        ).status_code
        == 403
    )
    user.effective_permissions = [Permission.FULL_ADMIN_PANEL_ACCESS.value]
    reads.settings.return_value.query_history_type = QueryHistoryType.DISABLED
    assert client.get("/api/admin/chat-session-history").status_code == 403
    reads.page.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [
        {"q": "x" * 501},
        {"user_query": "x" * 256},
        {"user_id": "bad-id"},
        {"page_num": -1},
    ],
)
def test_history_validates_filter_bounds(
    history_client: tuple[TestClient, MagicMock, User], params: dict[str, str | int]
) -> None:
    client, reads, _ = history_client
    assert (
        client.get("/api/admin/chat-session-history", params=params).status_code == 422
    )
    reads.page.assert_not_called()


def test_export_dispatch_preserves_date_and_applied_filters(
    history_client: tuple[TestClient, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = history_client
    send = MagicMock()
    monkeypatch.setattr(api.client_app, "send_task", send)
    monkeypatch.setattr(api, "register_task", MagicMock())
    response = client.post(
        "/api/admin/query-history/start-export",
        params={
            "start": "2026-11-01T00:00:00Z",
            "end": "2026-11-09T00:00:00Z",
            "user_query": "Çağrı",
            "api_key_query": "Tariff",
            "q": "duty",
            "feedback_type": "like",
            "session_rating": 4,
        },
    )
    assert response.status_code == 200
    kwargs = send.call_args.kwargs
    assert kwargs["expires"] > 0
    assert kwargs["kwargs"]["filters"] == {
        "user_query": "Çağrı",
        "api_key_query": "Tariff",
        "q": "duty",
        "feedback_type": "like",
        "session_rating": 4,
    }
    assert kwargs["kwargs"]["start"].isoformat() == "2026-11-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "identity", ["user_id", "user_email", "user_query", "api_key_query"]
)
def test_anonymized_export_rejects_identity_filters(
    history_client: tuple[TestClient, MagicMock, User],
    identity: str,
) -> None:
    client, reads, _ = history_client
    reads.settings.return_value.query_history_type = QueryHistoryType.ANONYMIZED
    value = str(uuid4()) if identity == "user_id" else "Tariff"
    assert (
        client.post(
            "/api/admin/query-history/start-export", params={identity: value}
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    "params",
    [
        {"user_query": "x" * 256},
        {"api_key_query": "x" * 256},
        {"session_rating": 5},
        {"user_id": "bad-id"},
        {"q": "x" * 501},
    ],
)
def test_export_validates_filter_bounds(
    history_client: tuple[TestClient, MagicMock, User],
    params: dict[str, str | int],
) -> None:
    client, _, _ = history_client
    assert (
        client.post("/api/admin/query-history/start-export", params=params).status_code
        == 422
    )


def test_download_streams_large_utf8_csv_in_bounded_blocks(
    history_client: tuple[TestClient, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import io

    _, _, user = history_client
    payload = ('Çağrı,"çok satırlı\nrapor"\n' * 10000).encode("utf-8")
    store = MagicMock()
    store.has_file.return_value = True
    store.read_file.return_value = io.BytesIO(payload)
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    response = api.download_query_history_csv("report-id", user, MagicMock())

    async def chunks() -> list[bytes]:
        return [
            chunk.encode() if isinstance(chunk, str) else bytes(chunk)
            async for chunk in response.body_iterator
        ]

    parts = asyncio.run(chunks())
    assert b"".join(parts) == payload
    assert len(parts) <= (len(payload) + 65535) // 65536
    assert response.media_type == "text/csv"


def test_exact_api_selection_reaches_table_count_and_csv(
    history_client: tuple[TestClient, MagicMock, User],
) -> None:
    client, reads, _ = history_client
    assert (
        client.get(
            "/api/admin/chat-session-history", params={"api_key_id": 17}
        ).status_code
        == 200
    )
    assert reads.page.call_args.kwargs["api_key_id"] == 17
    assert reads.count.call_args.kwargs["api_key_id"] == 17
    assert (
        client.post(
            "/api/admin/query-history/start-export", params={"api_key_id": 17}
        ).status_code
        == 200
    )
    assert (
        cast(MagicMock, api.client_app.send_task).call_args.kwargs["kwargs"]["filters"][
            "api_key_id"
        ]
        == 17
    )


def test_api_picker_is_private_and_anonymized_history_cannot_list_keys(
    history_client: tuple[TestClient, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, reads, _ = history_client
    monkeypatch.setattr(
        api, "get_query_history_api_key_options", lambda _session: [], raising=False
    )
    response = client.get("/api/admin/query-history/api-keys")
    assert response.status_code == 200
    assert response.json() == []
    assert response.headers["cache-control"] == "private, no-store"
    reads.settings.return_value.query_history_type = QueryHistoryType.ANONYMIZED
    assert client.get("/api/admin/query-history/api-keys").status_code == 403
    assert (
        client.get(
            "/api/admin/chat-session-history", params={"api_key_id": 17}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/admin/query-history/start-export", params={"api_key_id": 17}
        ).status_code
        == 403
    )
