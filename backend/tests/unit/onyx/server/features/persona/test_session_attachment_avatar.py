from io import BytesIO
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Request
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import persona as persona_db
from onyx.db.models import FileRecord, Persona, User
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.models import ChatAttachmentMetadata, ChatFileType
from onyx.server.features.persona import api
from onyx.server.features.persona.models import PersonaUpsertRequest


def private_attachment() -> FileRecord:
    file_id = str(uuid4())
    metadata = ChatAttachmentMetadata(
        user_id=uuid4(),
        plaintext_file_id=f"chat_attachment_text_{file_id}",
        chat_file_type=ChatFileType.DOC,
        token_count=3,
        size=12,
        text_chars=12,
        sha256="a" * 64,
    )
    return FileRecord(
        file_id=file_id,
        display_name="private.pdf",
        file_origin=FileOrigin.CHAT_UPLOAD,
        file_type="application/pdf",
        file_metadata=metadata.model_dump(mode="json"),
    )


@pytest.mark.parametrize("cached", [False, True])
def test_persona_avatar_rejects_rebound_private_attachment_before_serving_or_cache(
    monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    user = User(id=uuid4())
    record = private_attachment()
    persona = Persona(id=13, user_id=user.id, uploaded_image_id=record.file_id)
    session = MagicMock(spec=Session)
    session.get.return_value = record
    store = MagicMock()
    store.read_file.return_value = BytesIO(b"private text")
    monkeypatch.setattr(api, "get_persona_by_id", lambda **_: persona)
    monkeypatch.setattr(api, "get_filerecord_by_file_id_optional", lambda *_: record)
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    request = Request(
        {
            "type": "http",
            "headers": [(b"if-none-match", f'"{record.file_id}"'.encode())]
            if cached
            else [],
        }
    )
    with pytest.raises(OnyxError):
        api.get_persona_avatar(persona.id, request, user, session)
    store.read_file.assert_not_called()


def test_persona_save_rejects_private_attachment_image_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = private_attachment()
    session = MagicMock(spec=Session)
    session.get.return_value = record
    request = PersonaUpsertRequest(
        name="Assistant",
        description="",
        document_set_ids=[],
        tool_ids=[],
        uploaded_image_id=record.file_id,
        system_prompt="",
        task_prompt="",
        datetime_aware=False,
    )
    monkeypatch.setattr(
        persona_db,
        "upsert_persona",
        MagicMock(side_effect=AssertionError("Private image binding reached storage")),
    )
    with pytest.raises(OnyxError):
        persona_db.create_update_persona(None, request, User(id=uuid4()), session)
