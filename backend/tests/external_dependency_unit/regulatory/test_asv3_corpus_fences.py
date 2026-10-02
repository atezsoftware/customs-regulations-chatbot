"""Exercise actual PostgreSQL owner/date fences and retained-evidence validation."""

from datetime import date
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import RunContext
from onyx.asv3.source_tools import original_evidence
from onyx.configs.constants import FileOrigin, MessageType
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusScopeUnavailable,
    read_source_chunks,
    require_source,
)
from onyx.db.asv3_runs import save_asv3_checkpoint
from onyx.db.models import ChatMessage, ChatSession, RegulatoryChunk, UserFile
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.server import asv3_citations
from tests.external_dependency_unit.conftest import create_test_user


@pytest.mark.usefixtures("tenant_context")
def test_actual_owner_date_and_text_revalidation_fences(db_session: Session) -> None:
    owner = create_test_user(db_session, "asv3_corpus_owner")
    stranger = create_test_user(db_session, "asv3_corpus_stranger")
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="Scope test",
        file_type="text/plain",
    )
    db_session.add(source)
    db_session.flush()
    old = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        text="Old legal rule",
        position=0,
        projection_ordinal=0,
        validity_start_date=date(2025, 1, 1),
        validity_end_date=date(2026, 1, 1),
        status="superseded",
        heading_path=["MADDE 1"],
    )
    current = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        text="Current legal rule",
        position=1,
        projection_ordinal=1,
        validity_start_date=date(2026, 1, 1),
        heading_path=["MADDE 1"],
    )
    db_session.add_all([old, current])
    db_session.commit()
    filters = IndexFilters(
        access_control_list=[],
        attached_document_ids=[str(source.id)],
        as_of_date=date(2025, 12, 31),
    )
    try:
        authorized, rows, more = read_source_chunks(
            db_session, user=owner, filters=filters, source_id=source.id
        )
        assert [row.id for row in rows] == [old.id] and not more
        boundary_filters = filters.model_copy(update={"as_of_date": date(2026, 1, 1)})
        _, boundary, _ = read_source_chunks(
            db_session, user=owner, filters=boundary_filters, source_id=source.id
        )
        assert [row.id for row in boundary] == [current.id]
        with pytest.raises(PermissionError):
            require_source(
                db_session, user=stranger, filters=filters, source_id=source.id
            )
        broker = CorpusBroker(owner, filters)
        retained = evidence_for_chunk(authorized, rows[0])
        broker.revalidate_evidence([retained], RunContext())
        old.text = "Changed retained text"
        db_session.commit()
        with pytest.raises(CorpusScopeUnavailable, match="changed"):
            broker.revalidate_evidence([retained], RunContext())
        source.user_id = stranger.id
        db_session.commit()
        with pytest.raises(PermissionError):
            broker.revalidate_evidence([retained], RunContext())
    finally:
        db_session.rollback()
        db_session.delete(source)
        db_session.commit()
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_native_preview_revalidates_saved_owner_and_original_bytes(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = create_test_user(db_session, "asv3_native_owner")
    stranger = create_test_user(db_session, "asv3_native_stranger")
    store = get_default_file_store()
    raw = b"Header\nOriginal native row\n"
    file_id = store.save_file(
        BytesIO(raw), "native.txt", FileOrigin.USER_FILE, "text/plain"
    )
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=file_id,
        name="native.txt",
        file_type="text/plain",
    )
    chat = ChatSession(id=uuid4(), user_id=owner.id)
    db_session.add_all([source, chat])
    db_session.flush()
    message = ChatMessage(
        chat_session_id=chat.id,
        message="",
        token_count=0,
        message_type=MessageType.ASSISTANT,
    )
    db_session.add(message)
    db_session.commit()
    try:
        filters = IndexFilters(
            access_control_list=[], attached_document_ids=[str(source.id)]
        )
        broker = CorpusBroker(owner, filters)
        native = original_evidence(
            str(source.id), "Original native row", sha256(raw).hexdigest(), {"row": 2}
        )
        attached = broker.attach_native_citation(native, 1, message.id, RunContext())
        save_asv3_checkpoint(
            message_id=message.id,
            user_id=owner.id,
            snapshot={
                "run_id": str(uuid4()),
                "sequence": 1,
                "scope": filters.model_dump(mode="json"),
                "evidence": {
                    "version": 1,
                    "records": [
                        {"citation": 1, "item": attached.model_dump(mode="json")}
                    ],
                },
            },
        )
        monkeypatch.setattr(
            asv3_citations,
            "get_tokenizer",
            lambda **_kwargs: SimpleNamespace(encode=lambda text: text.split()),
        )
        preview = asv3_citations.get_native_citation(
            message.id, 1, user=owner, db_session=db_session
        )
        assert preview.content == "Original native row" and preview.num_tokens == 3
        with pytest.raises(OnyxError) as denied:
            asv3_citations.get_native_citation(
                message.id, 1, user=stranger, db_session=db_session
            )
        assert denied.value.status_code == 404
        store.save_file(
            BytesIO(b"Changed original"),
            "native.txt",
            FileOrigin.USER_FILE,
            "text/plain",
            file_id=file_id,
        )
        with pytest.raises(OnyxError) as changed:
            asv3_citations.get_native_citation(
                message.id, 1, user=owner, db_session=db_session
            )
        assert changed.value.status_code == 404
    finally:
        db_session.rollback()
        db_session.delete(chat)
        db_session.delete(source)
        db_session.commit()
        store.delete_file(file_id)
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()
