from io import BytesIO
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from fastapi import Request
from sqlalchemy.orm import Session

from onyx.access import access
from onyx.configs.constants import DEFAULT_PERSONA_ID, FileOrigin
from onyx.db import regulatory_public_reads
from onyx.db.models import ChatSession, ChatSessionSharedStatus, FileRecord, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.models import ChatAttachmentMetadata, ChatFileType
from onyx.regulatory import publication_reads
from onyx.server.features.chat_files import api as chat_files_api
from onyx.server.query_and_chat import chat_backend


def _attachment(owner: User, session: ChatSession | None) -> FileRecord:
    file_id = str(uuid4())
    metadata = ChatAttachmentMetadata(
        user_id=owner.id,
        chat_session_id=session.id if session else None,
        plaintext_file_id=f"chat_attachment_text_{file_id}",
        chat_file_type=ChatFileType.DOC,
        token_count=2,
        size=4,
        text_chars=4,
        sha256="a" * 64,
    )
    return FileRecord(
        file_id=file_id,
        display_name="private.pdf",
        file_origin=FileOrigin.CHAT_UPLOAD,
        file_type="application/pdf",
        file_metadata=metadata.model_dump(mode="json"),
    )


@pytest.mark.parametrize("is_owner", [False, True])
def test_shared_session_cannot_broaden_private_attachment_access(
    is_owner: bool,
) -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    viewer = owner if is_owner else User(id=uuid4(), email="viewer@example.test")
    shared_session = ChatSession(
        id=uuid4(),
        user_id=owner.id,
        persona_id=DEFAULT_PERSONA_ID,
        project_id=None,
        deleted=False,
        shared_status=ChatSessionSharedStatus.PUBLIC,
    )
    record = _attachment(owner, shared_session)
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = record
    db_session.scalar.return_value = shared_session
    db_session.query.side_effect = AssertionError("Legacy public access must not run")
    db_session.execute.side_effect = AssertionError("Legacy public access must not run")

    assert (
        access.user_can_access_chat_file(record.file_id, viewer, db_session) is is_owner
    )
    db_session.query.assert_not_called()
    db_session.execute.assert_not_called()


def test_deleted_session_cannot_serve_private_attachment() -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    session = ChatSession(id=uuid4(), user_id=owner.id)
    record = _attachment(owner, session)
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = record
    db_session.scalar.return_value = None

    assert not access.user_can_access_chat_file(record.file_id, owner, db_session)
    db_session.query.assert_not_called()
    db_session.execute.assert_not_called()


def test_plaintext_cache_never_falls_through_to_public_session_access() -> None:
    db_session = MagicMock(spec=Session)
    owner = User(id=uuid4(), email="owner@example.test")

    assert not access.user_can_access_chat_file(
        f"chat_attachment_text_{uuid4()}", owner, db_session
    )
    db_session.get.assert_not_called()
    db_session.query.assert_not_called()
    db_session.execute.assert_not_called()


def test_malformed_private_metadata_cannot_fall_through_to_public_access() -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    record = _attachment(owner, None)
    record.file_metadata = {"kind": "session_attachment"}
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = record

    assert not access.user_can_access_chat_file(record.file_id, owner, db_session)
    db_session.query.assert_not_called()
    db_session.execute.assert_not_called()


def test_unrelated_files_keep_existing_legacy_acl_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = None
    db_session.query.return_value.scalar.return_value = True
    monkeypatch.setattr(
        regulatory_public_reads, "regulatory_source_image_owners", lambda *_: None
    )

    assert access.user_can_access_chat_file("legacy-file", owner, db_session)
    db_session.query.assert_called_once()


@pytest.mark.parametrize("parsed", [False, True])
def test_private_original_legacy_url_uses_no_store_without_corpus_scans(
    monkeypatch: pytest.MonkeyPatch,
    parsed: bool,
) -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    record = _attachment(owner, None)
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = record
    file_store = Mock()
    file_store.read_file_record.return_value = record
    file_store.read_file.return_value = BytesIO(b"private bytes")
    monkeypatch.setattr(chat_backend, "get_default_file_store", lambda: file_store)
    monkeypatch.setattr(chat_files_api, "get_default_file_store", lambda: file_store)
    forbidden_scan = Mock(
        side_effect=AssertionError("Private attachment requested corpus scan")
    )
    monkeypatch.setattr(chat_backend, "get_file_id_by_user_file_id", forbidden_scan)
    monkeypatch.setattr(
        regulatory_public_reads, "public_file_source_owners", forbidden_scan
    )
    monkeypatch.setattr(regulatory_public_reads, "protected_file_ids", forbidden_scan)
    monkeypatch.setattr(publication_reads, "observe_publication_read", forbidden_scan)
    monkeypatch.setattr(publication_reads, "require_publication_files", forbidden_scan)
    request = Request(
        {
            "type": "http",
            "headers": [(b"if-none-match", f'"{record.file_id}"'.encode())],
        }
    )

    response = chat_backend.fetch_chat_file(
        record.file_id,
        request=request,
        parsed=parsed,
        user=owner,
        db_session=db_session,
    )

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "ETag" not in response.headers


@pytest.mark.parametrize("target", ["foreign", "plaintext"])
def test_legacy_private_file_denial_precedes_any_corpus_scan(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    owner = User(id=uuid4(), email="owner@example.test")
    viewer = User(id=uuid4(), email="viewer@example.test")
    record = _attachment(owner, None)
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = record
    forbidden_scan = Mock(
        side_effect=AssertionError("Private attachment requested corpus scan")
    )
    monkeypatch.setattr(publication_reads, "observe_publication_read", forbidden_scan)
    monkeypatch.setattr(chat_backend, "get_file_id_by_user_file_id", forbidden_scan)
    monkeypatch.setattr(
        regulatory_public_reads, "public_file_source_owners", forbidden_scan
    )
    file_id = (
        f"chat_attachment_text_{record.file_id}"
        if target == "plaintext"
        else record.file_id
    )

    with pytest.raises(OnyxError) as caught:
        chat_backend.fetch_chat_file(
            file_id,
            request=Request({"type": "http", "headers": []}),
            parsed=False,
            user=viewer,
            db_session=db_session,
        )
    assert caught.value.error_code is OnyxErrorCode.NOT_FOUND
