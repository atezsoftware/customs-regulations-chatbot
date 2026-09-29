from collections.abc import Awaitable, Callable
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.manage.answer_graph.api import admin_router


class _User:
    def __init__(self, *, admin: bool) -> None:
        self.id = UUID("00000000-0000-0000-0000-000000000001")
        self.effective_permissions = (
            [Permission.FULL_ADMIN_PANEL_ACCESS.value] if admin else []
        )


def _client(*, admin: bool, scoped: bool = False) -> TestClient:
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(admin_router)
    app.dependency_overrides[current_user] = lambda: _User(admin=admin)
    app.dependency_overrides[get_session] = lambda: MagicMock()
    if scoped:

        @app.middleware("http")
        async def restrict_token(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            request.state.token_scopes = [Permission.READ_CHAT]
            return await call_next(request)

    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/admin/answer-graphs/by-message/1",
        "/admin/answer-graphs/00000000-0000-0000-0000-000000000002/nodes",
        "/admin/answer-graphs/00000000-0000-0000-0000-000000000002/edges",
        "/admin/answer-graphs/00000000-0000-0000-0000-000000000002/nodes/x",
    ],
)
def test_graph_routes_require_full_admin_access(path: str) -> None:
    assert _client(admin=False).get(path).status_code == 403
    assert _client(admin=True, scoped=True).get(path).status_code == 403


def test_admin_overview_has_no_store_header() -> None:
    with (
        patch(
            "onyx.server.manage.answer_graph.api.answer_graph_message_exists",
            return_value=True,
        ),
        patch(
            "onyx.server.manage.answer_graph.api.get_answer_graph_run_by_message",
            return_value=None,
        ),
    ):
        response = _client(admin=True).get("/admin/answer-graphs/by-message/1")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["status"] == "NO_TRACE"
