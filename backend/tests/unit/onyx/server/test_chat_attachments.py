import io
from collections.abc import Iterator
from typing import Any, NamedTuple, cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Table, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from onyx.auth.users import current_user
from onyx.chat.chat_utils import load_chat_file
from onyx.configs.constants import FileOrigin, MessageType
from onyx.db import chat as chat_db
from onyx.db import chat_attachments
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import ChatSessionSharedStatus, Permission, UserFileStatus
from onyx.db.models import ChatMessage, ChatSession, FileRecord, User, UserFile
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError, register_onyx_exception_handlers
from onyx.file_store.models import ChatFileType, FileDescriptor
from onyx.file_store.utils import verify_user_files
from onyx.server.features.chat_files import api


@compiles(JSONB, "sqlite")
def _sqlite_jsonb(_element: object, _compiler: object, **_kwargs: object) -> str:
    return "JSON"


class AttachmentEnvironment(NamedTuple):
    client: TestClient
    session: Session
    owner: User
    other_user_id: UUID
    chat: ChatSession
    blobs: dict[str, bytes]
    store: Mock


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[AttachmentEnvironment]:
    # Exercise the real owner/session/JSON predicates without external services.
    # SQLite does not emulate PostgreSQL row locks.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    owner = User(id=uuid4(), effective_permissions=[Permission.BASIC_ACCESS.value])
    other_user_id = uuid4()
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE user (id CHAR(36) PRIMARY KEY)")
        for user_id in (owner.id, other_user_id):
            connection.exec_driver_sql(
                "INSERT INTO user (id) VALUES (?)", (str(user_id),)
            )
        cast(Table, ChatSession.__table__).create(connection)
        cast(Table, FileRecord.__table__).create(connection)
    session = Session(engine, expire_on_commit=False)
    chat = ChatSession(user_id=owner.id, persona_id=0)
    session.add(chat)
    session.commit()
    blobs: dict[str, bytes] = {}
    store = Mock()

    def save_file(
        content: io.BytesIO,
        display_name: str,
        file_origin: FileOrigin,
        file_type: str,
        file_id: str,
        file_metadata: dict[str, Any] | None = None,
    ) -> str:
        blobs[file_id] = content.read()
        session.add(
            FileRecord(
                file_id=file_id,
                display_name=display_name,
                file_origin=file_origin,
                file_type=file_type,
                file_metadata=file_metadata,
                bucket_name="test",
                object_key=file_id,
            )
        )
        session.flush()
        return file_id

    def read_file(file_id: str, mode: str = "b") -> io.BytesIO:
        assert mode == "b"
        return io.BytesIO(blobs[file_id])

    def delete_file(file_id: str, error_on_missing: bool = True) -> None:
        if error_on_missing:
            assert file_id in blobs
        blobs.pop(file_id, None)
        record = session.get(FileRecord, file_id)
        if record is not None:
            session.delete(record)
            session.flush()

    store.save_file.side_effect = save_file
    store.read_file.side_effect = read_file
    store.delete_file.side_effect = delete_file
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    monkeypatch.setattr(chat_attachments, "get_default_file_store", lambda: store)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router)
    app.dependency_overrides[current_user] = lambda: owner
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield AttachmentEnvironment(
            client, session, owner, other_user_id, chat, blobs, store
        )
    session.close()
    engine.dispose()


def _upload(
    env: AttachmentEnvironment,
    session_id: UUID | None = None,
    *,
    filename: str = "invoice.csv",
    content: bytes = b"item,amount\ncustoms,120\n",
) -> dict[str, Any]:
    response = env.client.post(
        "/chat/files/upload",
        files=[("files", (filename, content, "application/octet-stream"))],
        data={"chat_session_id": str(session_id)} if session_id else {},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_draft_claim_is_canonical_and_retains_complete_text(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    uploaded = _upload(env)["files"][0]
    assert uploaded["type"] == "tabular"
    assert uploaded["is_session_attachment"] is True
    assert uploaded["token_count"] > 0
    descriptor: FileDescriptor = {
        "id": uploaded["id"],
        "type": ChatFileType.IMAGE,
        "name": "forged.png",
        "user_file_id": str(env.other_user_id),
    }
    canonical = chat_attachments.canonicalize_and_bind_chat_attachments(
        env.session, [descriptor, descriptor], env.chat.id, env.owner.id
    )
    assert canonical == [
        {
            "id": uploaded["id"],
            "type": ChatFileType.TABULAR,
            "name": "invoice.csv",
            "is_session_attachment": True,
        }
    ]
    loaded = chat_attachments.load_chat_session_attachments(
        env.session, env.chat.id, env.owner.id
    )
    assert len(loaded) == 1
    assert "customs" in (loaded[0].content_text or "")
    assert "120" in (loaded[0].content_text or "")
    assert loaded[0].content == b"item,amount\ncustoms,120\n"


@pytest.mark.parametrize(
    "invalid_kind", ["foreign_owner", "other_session", "inactive", "corpus", "missing"]
)
def test_claim_rejects_entire_batch_before_binding_any_draft(
    environment: AttachmentEnvironment, invalid_kind: str
) -> None:
    env = environment
    valid = _upload(env)["files"][0]
    invalid = _upload(env, filename="second.txt", content=b"Second file")["files"][0]
    record = env.session.get(FileRecord, invalid["id"])
    assert record is not None
    metadata = chat_attachments.attachment_metadata(record)
    if invalid_kind == "foreign_owner":
        metadata.user_id = env.other_user_id
    elif invalid_kind == "other_session":
        metadata.chat_session_id = uuid4()
    elif invalid_kind == "inactive":
        metadata.active = False
    elif invalid_kind == "corpus":
        record.file_origin = FileOrigin.CONNECTOR
    elif invalid_kind == "missing":
        invalid["id"] = str(uuid4())
    record.file_metadata = metadata.model_dump(mode="json")
    env.session.commit()
    descriptors: list[FileDescriptor] = [
        {"id": file["id"], "type": ChatFileType.PLAIN_TEXT} for file in (valid, invalid)
    ]
    with pytest.raises(OnyxError, match="Attachment not found"):
        chat_attachments.canonicalize_and_bind_chat_attachments(
            env.session, descriptors, env.chat.id, env.owner.id
        )
    valid_record = env.session.get(FileRecord, valid["id"])
    assert valid_record is not None
    assert chat_attachments.attachment_metadata(valid_record).chat_session_id is None


def test_session_upload_listing_unlink_and_original_download(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    attachment = _upload(env, env.chat.id)["files"][0]
    path = f"/chat/sessions/{env.chat.id}/files"
    assert env.client.get(path).json() == [attachment]
    assert env.client.delete(f"/chat/files/{attachment['id']}").status_code == 409
    assert env.client.delete(f"{path}/{attachment['id']}").status_code == 204
    assert env.client.get(path).json() == []
    response = env.client.get(f"/chat/files/{attachment['id']}")
    assert response.status_code == 200
    assert response.content == b"item,amount\ncustoms,120\n"
    assert response.headers["cache-control"] == "private, no-store"
    assert (
        chat_attachments.load_chat_session_attachments(
            env.session, env.chat.id, env.owner.id
        )
        == []
    )


@pytest.mark.parametrize("owner_id", [None, "other"])
def test_public_or_unowned_session_never_grants_file_api_access(
    environment: AttachmentEnvironment, owner_id: str | None
) -> None:
    env = environment
    env.chat.user_id = env.other_user_id if owner_id else None
    env.chat.shared_status = ChatSessionSharedStatus.PUBLIC
    env.session.commit()
    path = f"/chat/sessions/{env.chat.id}/files"
    assert env.client.get(path).status_code == 404
    assert (
        env.client.post(
            "/chat/files/upload",
            files={"files": ("note.txt", b"private")},
            data={"chat_session_id": str(env.chat.id)},
        ).status_code
        == 404
    )


def test_draft_download_and_delete_require_exact_file_owner(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    attachment = _upload(env)["files"][0]
    path = f"/chat/files/{attachment['id']}"
    assert env.client.get(path).status_code == 200
    env.owner.id, original_owner = env.other_user_id, env.owner.id
    assert env.client.get(path).status_code == 404
    assert env.client.delete(path).status_code == 404
    env.owner.id = original_owner
    assert env.client.delete(path).status_code == 204
    assert env.blobs == {}


def test_abandoned_drafts_are_recoverable_and_listed_only_to_owner(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment
    monkeypatch.setattr(chat_attachments, "MAX_SESSION_ATTACHMENTS", 1)
    draft = _upload(env)["files"][0]
    _upload(env, env.chat.id)
    assert env.client.get("/chat/files").json() == [draft]
    assert _upload(env)["files"] == []
    env.owner.id, original_owner = env.other_user_id, env.owner.id
    assert env.client.get("/chat/files").json() == []
    env.owner.id = original_owner
    assert env.client.delete(f"/chat/files/{draft['id']}").status_code == 204
    assert _upload(env)["files"]


def test_quota_rejection_and_unsupported_file_preserve_partial_success(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment
    monkeypatch.setattr(chat_attachments, "MAX_SESSION_ATTACHMENTS", 1)
    response = env.client.post(
        "/chat/files/upload",
        files=[
            ("files", ("a.txt", b"one")),
            ("files", ("a.exe", b"invalid")),
            ("files", ("b.txt", b"two")),
        ],
    )
    assert response.status_code == 200
    result = response.json()
    assert [file["name"] for file in result["files"]] == ["a.txt"]
    assert len(result["rejected_files"]) == 2
    assert "at most" in result["rejected_files"][1]["reason"].lower()


@pytest.mark.parametrize(
    "quota", ["MAX_SESSION_ATTACHMENT_BYTES", "MAX_SESSION_ATTACHMENT_TEXT_CHARS"]
)
def test_combined_session_capacity_applies_to_draft_claim(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch, quota: str
) -> None:
    env = environment
    _upload(env, env.chat.id, filename="session.txt", content=b"12345")
    draft = _upload(env, filename="draft.txt", content=b"67890")["files"][0]
    monkeypatch.setattr(chat_attachments, quota, 9)
    with pytest.raises(OnyxError, match="Combined|Retained originals"):
        chat_attachments.canonicalize_and_bind_chat_attachments(
            env.session,
            [{"id": draft["id"], "type": ChatFileType.PLAIN_TEXT}],
            env.chat.id,
            env.owner.id,
        )


def test_session_cleanup_deletes_unmessaged_and_inactive_files(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    active = _upload(env, env.chat.id)["files"][0]
    inactive = _upload(env, env.chat.id, filename="removed.txt", content=b"old")[
        "files"
    ][0]
    chat_attachments.remove_chat_session_attachment(
        env.session, env.chat.id, inactive["id"], env.owner.id
    )
    draft = _upload(env, filename="unbound.txt", content=b"draft")["files"][0]
    removed = chat_attachments.delete_chat_session_attachment_files(
        env.session, env.chat.id
    )
    assert removed == {active["id"], inactive["id"]}
    assert set(env.blobs) == {draft["id"], f"chat_attachment_text_{draft['id']}"}


def test_chat_read_scope_does_not_allow_upload(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    env.owner.effective_permissions = [Permission.READ_CHAT.value]
    assert env.client.get(f"/chat/sessions/{env.chat.id}/files").status_code == 200
    assert (
        env.client.post(
            "/chat/files/upload", files={"files": ("note.txt", b"read only")}
        ).status_code
        == 403
    )


def test_removing_bound_files_does_not_reset_retained_original_storage_quota(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment
    monkeypatch.setattr(chat_attachments, "MAX_SESSION_ATTACHMENT_BYTES", 9)
    monkeypatch.setattr(chat_attachments, "MAX_SESSION_ATTACHMENTS", 1)
    original = _upload(env, env.chat.id, filename="original.txt", content=b"12345")[
        "files"
    ][0]
    path = f"/chat/sessions/{env.chat.id}/files"
    assert env.client.delete(f"{path}/{original['id']}").status_code == 204
    assert env.client.get(path).json() == []

    rejected = _upload(env, env.chat.id, filename="new.txt", content=b"67890")
    assert rejected["files"] == []
    assert "Retained originals exceed" in rejected["rejected_files"][0]["reason"]
    assert "start a new session" in rejected["rejected_files"][0]["reason"]
    response = env.client.get(f"/chat/files/{original['id']}")
    assert response.status_code == 200
    assert response.content == b"12345"

    # Inactive files consume storage, but do not consume the active file limit.
    accepted = _upload(env, env.chat.id, filename="small.txt", content=b"6789")
    assert len(accepted["files"]) == 1


def test_draft_claim_also_counts_removed_session_originals(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment
    original = _upload(env, env.chat.id, filename="original.txt", content=b"12345")[
        "files"
    ][0]
    assert (
        env.client.delete(
            f"/chat/sessions/{env.chat.id}/files/{original['id']}"
        ).status_code
        == 204
    )
    draft = _upload(env, filename="draft.txt", content=b"67890")["files"][0]
    monkeypatch.setattr(chat_attachments, "MAX_SESSION_ATTACHMENT_BYTES", 9)

    with pytest.raises(OnyxError, match="Retained originals exceed"):
        chat_attachments.canonicalize_and_bind_chat_attachments(
            env.session,
            [{"id": draft["id"], "type": ChatFileType.PLAIN_TEXT}],
            env.chat.id,
            env.owner.id,
        )
    draft_record = env.session.get(FileRecord, draft["id"])
    assert draft_record is not None
    assert chat_attachments.attachment_metadata(draft_record).chat_session_id is None


@pytest.mark.parametrize("target", ["foreign_raw", "plaintext", "owned_raw"])
def test_generic_verifier_rejects_private_id_with_owned_user_file_id(
    environment: AttachmentEnvironment, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = environment
    cast(Table, UserFile.__table__).create(env.session.connection())
    existing_file = UserFile(
        id=uuid4(),
        user_id=env.owner.id,
        file_id=str(uuid4()),
        name="existing.txt",
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
    )
    env.session.add(existing_file)
    env.session.commit()
    # Suppress unrelated UserFile project/persona eager loads in this minimal DB.
    monkeypatch.setattr(
        "onyx.file_store.utils.get_user_file_by_id",
        lambda file_id, db_session: db_session.get(UserFile, file_id),
    )
    valid_descriptor: FileDescriptor = {
        "id": existing_file.file_id,
        "type": ChatFileType.PLAIN_TEXT,
        "name": existing_file.name,
        "user_file_id": str(existing_file.id),
    }
    verify_user_files([valid_descriptor], env.owner.id, env.session)
    private = _upload(env)["files"][0]
    file_id = private["id"]
    if target == "foreign_raw":
        record = env.session.get(FileRecord, file_id)
        assert record is not None
        metadata = chat_attachments.attachment_metadata(record)
        record.file_metadata = metadata.model_copy(
            update={"user_id": env.other_user_id}
        ).model_dump(mode="json")
        env.session.commit()
    elif target == "plaintext":
        file_id = f"chat_attachment_text_{file_id}"
    forged: FileDescriptor = {
        "id": file_id,
        "type": ChatFileType.PLAIN_TEXT,
        "user_file_id": str(existing_file.id),
        "is_session_attachment": False,
    }
    with pytest.raises(OnyxError) as caught:
        verify_user_files([forged], env.owner.id, env.session)
    expected = (
        OnyxErrorCode.INVALID_INPUT
        if target == "owned_raw"
        else OnyxErrorCode.NOT_FOUND
    )
    assert caught.value.error_code is expected
    env.store.read_file.assert_not_called()


@pytest.mark.parametrize("target", ["owned_raw", "foreign_raw", "plaintext"])
def test_generic_loader_never_reads_private_session_bytes(
    environment: AttachmentEnvironment, target: str
) -> None:
    env = environment
    private = _upload(env)["files"][0]
    file_id = private["id"]
    if target == "foreign_raw":
        record = env.session.get(FileRecord, file_id)
        assert record is not None
        metadata = chat_attachments.attachment_metadata(record)
        record.file_metadata = metadata.model_copy(
            update={"user_id": env.other_user_id}
        ).model_dump(mode="json")
        env.session.commit()
    elif target == "plaintext":
        file_id = f"chat_attachment_text_{file_id}"
    descriptor: FileDescriptor = {
        "id": file_id,
        "type": ChatFileType.PLAIN_TEXT,
        "user_file_id": str(uuid4()),
    }
    with pytest.raises(OnyxError) as caught:
        load_chat_file(descriptor, env.session)
    assert caught.value.error_code is OnyxErrorCode.NOT_FOUND
    env.store.read_file.assert_not_called()


def test_slack_seed_omits_private_files_by_stored_identity(
    environment: AttachmentEnvironment,
) -> None:
    env = environment
    cast(Table, ChatMessage.__table__).create(env.session.connection())
    attachment = _upload(env, env.chat.id)["files"][0]
    ordinary: FileDescriptor = {
        "id": "legacy-message-file",
        "type": ChatFileType.PLAIN_TEXT,
        "is_session_attachment": True,
    }
    env.session.add(
        ChatMessage(
            chat_session_id=env.chat.id,
            message="source message",
            token_count=2,
            message_type=MessageType.USER,
            files=[
                {"id": attachment["id"], "type": ChatFileType.PLAIN_TEXT},
                {
                    "id": f"chat_attachment_text_{attachment['id']}",
                    "type": ChatFileType.PLAIN_TEXT,
                    "is_session_attachment": False,
                },
                ordinary,
            ],
        )
    )
    destination = ChatSession(user_id=env.other_user_id, persona_id=0)
    env.session.add(destination)
    env.session.commit()

    chat_db.add_chats_to_session_from_slack_thread(
        env.session, env.chat.id, destination.id
    )

    copied_message = env.session.scalar(
        select(ChatMessage).where(
            ChatMessage.chat_session_id == destination.id,
            ChatMessage.message_type == MessageType.USER,
        )
    )
    assert copied_message is not None
    assert copied_message.files == [ordinary]
    assert attachment["id"] in env.blobs
    assert f"chat_attachment_text_{attachment['id']}" in env.blobs


@pytest.mark.parametrize("forged_marker", [False, True])
def test_deleting_copied_session_preserves_source_private_files(
    environment: AttachmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    forged_marker: bool,
) -> None:
    env = environment
    cast(Table, ChatMessage.__table__).create(env.session.connection())
    attachment = _upload(env, env.chat.id)["files"][0]
    plaintext_id = f"chat_attachment_text_{attachment['id']}"
    ordinary_id = env.store.save_file(
        content=io.BytesIO(b"ordinary old message upload"),
        display_name="ordinary.txt",
        file_origin=FileOrigin.CHAT_UPLOAD,
        file_type="text/plain",
        file_id="legacy-message-file",
    )
    destination = ChatSession(user_id=env.other_user_id, persona_id=0)
    env.session.add(destination)
    env.session.flush()
    env.session.add(
        ChatMessage(
            chat_session_id=destination.id,
            message="copied or forged message",
            token_count=3,
            message_type=MessageType.USER,
            files=[
                {
                    "id": attachment["id"],
                    "type": ChatFileType.PLAIN_TEXT,
                    "is_session_attachment": forged_marker,
                },
                {"id": plaintext_id, "type": ChatFileType.PLAIN_TEXT},
                {
                    "id": ordinary_id,
                    "type": ChatFileType.PLAIN_TEXT,
                    "is_session_attachment": True,
                },
            ],
        )
    )
    env.session.commit()
    # Graph and orphan-search cleanup are unrelated to file ownership.
    monkeypatch.setattr(chat_db, "delete_answer_graphs_for_session", lambda *_: None)
    monkeypatch.setattr(chat_db, "delete_orphaned_search_docs", lambda *_: None)
    monkeypatch.setattr(chat_db, "get_default_file_store", lambda: env.store)

    chat_db.delete_messages_and_files_from_chat_session(destination.id, env.session)

    assert (
        env.session.scalar(
            select(ChatMessage.id).where(ChatMessage.chat_session_id == destination.id)
        )
        is None
    )
    assert ordinary_id not in env.blobs
    assert env.blobs[attachment["id"]] == b"item,amount\ncustoms,120\n"
    assert "customs" in env.blobs[plaintext_id].decode()
    chat_db.delete_messages_and_files_from_chat_session(env.chat.id, env.session)
    assert env.blobs == {}


def test_scanned_pdf_upload_keeps_original_without_text_extraction(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.onyx.file_processing.test_chat_attachments import _pdf

    def reject_text_extraction(*_args: object) -> str:
        raise AssertionError("PDF upload must not extract or pre-read the document")

    monkeypatch.setattr(api, "extract_chat_attachment_text", reject_text_extraction)
    content = _pdf("")
    result = _upload(environment, filename="scan.pdf", content=content)
    assert result["rejected_files"] == []
    uploaded = result["files"][0]
    assert environment.blobs[uploaded["id"]] == content
    assert len(environment.blobs) == 2
    downloaded = environment.client.get(f"/chat/files/{uploaded['id']}")
    assert downloaded.content == content


@pytest.mark.parametrize("kind", ["malformed", "encrypted", "page_limit"])
def test_native_pdf_upload_still_rejects_unsafe_documents(
    environment: AttachmentEnvironment, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from onyx.file_processing import chat_attachments as parser
    from tests.unit.onyx.file_processing.test_chat_attachments import _pdf

    if kind == "page_limit":
        monkeypatch.setattr(parser, "MAX_CHAT_ATTACHMENT_PAGES", 1)
    original = (
        b"not a pdf"
        if kind == "malformed"
        else _pdf("", password="secret")
        if kind == "encrypted"
        else _pdf("", "")
    )
    result = _upload(environment, filename="unsafe.pdf", content=original)
    assert result["files"] == []
    assert len(result["rejected_files"]) == 1
    assert environment.blobs == {}
