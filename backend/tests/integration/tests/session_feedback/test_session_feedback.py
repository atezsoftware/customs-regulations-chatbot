from uuid import uuid4

import httpx
import pytest

from onyx.auth.constants import API_KEY_PREFIX
from onyx.db.enums import Permission
from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.http_client import client
from tests.integration.common_utils.managers.api_key import APIKeyManager
from tests.integration.common_utils.managers.chat import ChatSessionManager
from tests.integration.common_utils.managers.pat import PATManager
from tests.integration.common_utils.managers.query_history import QueryHistoryManager
from tests.integration.common_utils.test_models import DATestAPIKey, DATestUser


def _put(
    session_id: str, headers: dict[str, str], body: dict[str, object]
) -> httpx.Response:
    return client.put(
        f"{API_SERVER_URL}/chat/chat-session/{session_id}/feedback",
        headers=headers,
        json=body,
    )


def _detail(session_id: str, user: DATestUser, shared: bool = False) -> httpx.Response:
    return client.get(
        f"{API_SERVER_URL}/chat/get-chat-session/{session_id}",
        headers=user.headers,
        params={"is_shared": str(shared).lower()},
    )


def test_owner_feedback_persistence_and_shared_privacy(
    admin_user: DATestUser, basic_user: DATestUser
) -> None:
    session = ChatSessionManager.create(user_performing_action=basic_user)
    session_id = str(session.id)
    first = _put(session_id, basic_user.headers, {"rating": 2, "expected_version": 0})
    assert first.status_code == 200
    saved = first.json()
    assert saved["label"] == "Eksik"
    assert saved["feedback_text"] is None
    assert saved["version"] == 1

    retry = _put(session_id, basic_user.headers, {"rating": 2})
    assert retry.json() == saved
    changed = _put(
        session_id,
        basic_user.headers,
        {"rating": 3, "feedback_text": "  İstisna eksik.  ", "expected_version": 1},
    )
    assert changed.status_code == 200
    assert changed.json()["feedback_text"] == "İstisna eksik."
    assert changed.json()["version"] == 2
    assert changed.json()["time_created"] == saved["time_created"]
    assert _detail(session_id, basic_user).json()["session_feedback"] == changed.json()

    stale = _put(session_id, basic_user.headers, {"rating": 1, "expected_version": 1})
    assert stale.status_code == 409
    assert _detail(session_id, basic_user).json()["session_feedback"] == changed.json()
    assert _put(session_id, admin_user.headers, {"rating": 4}).status_code == 404

    sharing = client.patch(
        f"{API_SERVER_URL}/chat/chat-session/{session_id}",
        headers=basic_user.headers,
        json={"sharing_status": "public"},
    )
    assert sharing.status_code == 200
    shared = _detail(session_id, admin_user, shared=True)
    assert shared.status_code == 200
    assert shared.json()["session_feedback"] is None
    assert (
        _detail(session_id, basic_user, shared=True).json()["session_feedback"] is None
    )

    cleared = _put(session_id, basic_user.headers, {"rating": 4, "expected_version": 2})
    assert cleared.status_code == 200
    assert cleared.json()["feedback_text"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"rating": 0},
        {"rating": 5},
        {"rating": True},
        {"rating": 2.0},
        {"rating": "3"},
        {"rating": 4, "feedback_text": "x" * 5001},
        {"rating": 4, "feedback_text": "bad\x00text"},
        {"rating": 4, "expected_version": True},
    ],
)
def test_invalid_feedback_is_rejected(
    admin_user: DATestUser, body: dict[str, object]
) -> None:
    session = ChatSessionManager.create(user_performing_action=admin_user)
    assert _put(str(session.id), admin_user.headers, body).status_code == 422
    assert _detail(str(session.id), admin_user).json()["session_feedback"] is None


def test_api_key_identity_and_pat_scopes(
    admin_user: DATestUser, basic_user: DATestUser, api_key: DATestAPIKey
) -> None:
    service_user = admin_user.model_copy(
        update={"id": str(api_key.user_id), "headers": api_key.headers, "cookies": {}}
    )
    session = ChatSessionManager.create(user_performing_action=service_user)
    saved = _put(str(session.id), api_key.headers, {"rating": 1})
    assert saved.status_code == 200
    assert saved.json()["label"] == "Yanlış"
    assert (
        _detail(str(session.id), service_user).json()["session_feedback"]
        == saved.json()
    )
    assert _put(str(session.id), admin_user.headers, {"rating": 4}).status_code == 404
    revoked_key = APIKeyManager.create(user_performing_action=admin_user)
    APIKeyManager.delete(revoked_key, user_performing_action=admin_user)
    assert _put(str(session.id), revoked_key.headers, {"rating": 4}).status_code == 403

    own_session = ChatSessionManager.create(user_performing_action=basic_user)
    assert _put(str(own_session.id), api_key.headers, {"rating": 4}).status_code == 404
    write_pat = PATManager.create(
        name=f"feedback-write-{uuid4()}",
        expiration_days=1,
        user_performing_action=basic_user,
        scopes=[Permission.WRITE_CHAT],
    )
    read_pat = PATManager.create(
        name=f"feedback-read-{uuid4()}",
        expiration_days=1,
        user_performing_action=basic_user,
        scopes=[Permission.READ_CHAT],
    )
    assert write_pat.token is not None
    assert read_pat.token is not None
    assert (
        _put(
            str(own_session.id),
            PATManager.get_auth_headers(write_pat.token),
            {"rating": 3},
        ).status_code
        == 200
    )
    assert (
        _put(
            str(own_session.id),
            PATManager.get_auth_headers(read_pat.token),
            {"rating": 4},
        ).status_code
        == 403
    )
    with httpx.Client(timeout=30) as anonymous:
        url = f"{API_SERVER_URL}/chat/chat-session/{own_session.id}/feedback"
        missing = anonymous.put(url, json={"rating": 4})
        assert missing.status_code == 403
        assert missing.json()["detail"] == "Access denied. User is not authenticated."
        invalid_key = anonymous.put(
            url,
            headers={"Authorization": f"Bearer {API_KEY_PREFIX}invalid"},
            json={"rating": 4},
        )
        assert invalid_key.status_code == 403
        assert invalid_key.json() == missing.json()


def test_admin_rating_filter_and_detail(admin_user: DATestUser) -> None:
    session = ChatSessionManager.create(user_performing_action=admin_user)
    saved = _put(
        str(session.id), admin_user.headers, {"rating": 2, "feedback_text": "Eksik"}
    )
    assert saved.status_code == 200
    response = QueryHistoryManager.get_query_history_page(
        user_performing_action=admin_user, session_rating=2, page_size=100
    )
    item_ids = {str(item.id) for item in response.items}
    assert str(session.id) in item_ids
    assert response.total_items == len(response.items)
    other = QueryHistoryManager.get_query_history_page(
        user_performing_action=admin_user, session_rating=1, page_size=100
    )
    assert str(session.id) not in {str(item.id) for item in other.items}
    detail = QueryHistoryManager.get_chat_session_admin(
        chat_session_id=session.id, user_performing_action=admin_user
    )
    assert detail.session_feedback is not None
    assert detail.session_feedback.model_dump(mode="json") == saved.json()
