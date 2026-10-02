"""Exercise actual PostgreSQL owner/date fences and retained-evidence validation."""

from collections.abc import Generator
from datetime import date
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import RunContext
from onyx.asv3.source_tools import original_evidence
from onyx.configs.constants import DocumentSource, FileOrigin, MessageType
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    PC_CORPUS_NAME,
    CorpusScopeUnavailable,
    find_sources,
    read_source_chunks,
    require_source,
    resolve_pc_corpus_scope,
)
from onyx.db.asv3_runs import save_asv3_checkpoint
from onyx.db.models import (
    ChatMessage,
    ChatSession,
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    UserFile,
)
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.server import asv3_citations
from tests.external_dependency_unit.conftest import create_test_user


@pytest.fixture
def pc_corpus(db_session: Session) -> Generator[DocumentSet, None, None]:
    if (
        db_session.scalar(
            select(DocumentSet.id).where(DocumentSet.name == PC_CORPUS_NAME)
        )
        is not None
    ):
        raise RuntimeError(
            "PC scope tests require an isolated validation DB without a pre-existing PC corpus"
        )
    corpus = DocumentSet(name=PC_CORPUS_NAME, is_public=False)
    db_session.add(corpus)
    db_session.commit()
    try:
        yield corpus
    finally:
        db_session.rollback()
        db_session.delete(corpus)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_source_resolution_matches_turkish_query_to_ascii_canonical_filename(
    db_session: Session, pc_corpus: DocumentSet
) -> None:
    owner = create_test_user(db_session, "asv3_source_names")
    pc_corpus.user_id = owner.id
    sources = [
        UserFile(
            id=uuid4(),
            user_id=owner.id,
            file_id=uuid4().hex,
            name=name,
            file_type="text/plain",
        )
        for name in (
            "Kanunlar/4458_gumruk_kanunu.md",
            "Genelgeler/Gümrükler genel müdürlüğü/genelge_gumruk_kanunu_uygulamasi.md",
            "Genelgeler/Gümrükler genel müdürlüğü/genelge_kabahatler_kanunu.md",
            "Kanunlar/gumruk_kanunu_outside_pc.md",
            "literal/source_100%_special.md",
        )
    ]
    db_session.add_all(sources)
    db_session.flush()
    db_session.add_all(
        DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=source.id)
        for source in [*sources[:3], sources[4]]
    )
    db_session.commit()
    try:
        filters = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=IndexFilters(access_control_list=[]),
        )
        matches, more = find_sources(
            db_session, user=owner, filters=filters, query="Gümrük Kanunu"
        )
        assert [source.id for source in matches] == [row.id for row in sources[:3]]
        assert not more
        page, more = find_sources(
            db_session, user=owner, filters=filters, query="GUMRUK KANUNU", limit=1
        )
        assert page[0].id == sources[0].id and more
        literal, _ = find_sources(
            db_session, user=owner, filters=filters, query="100%_special"
        )
        assert [source.id for source in literal] == [sources[4].id]
    finally:
        db_session.rollback()
        for source in sources:
            db_session.delete(source)
        pc_corpus.user_id = None
        db_session.commit()
        db_session.delete(owner)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_actual_owner_date_and_text_revalidation_fences(
    db_session: Session, pc_corpus: DocumentSet
) -> None:
    owner = create_test_user(db_session, "asv3_corpus_owner")
    stranger = create_test_user(db_session, "asv3_corpus_stranger")
    pc_corpus.user_id = owner.id
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="Scope test",
        file_type="text/plain",
    )
    db_session.add(source)
    db_session.flush()
    db_session.add(
        DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=source.id)
    )
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
    filters = resolve_pc_corpus_scope(
        db_session,
        user=owner,
        filters=IndexFilters(
            access_control_list=[],
            source_type=[DocumentSource.USER_FILE],
            regulatory_chunks_only=True,
            attached_document_ids=[str(source.id)],
            as_of_date=date(2025, 12, 31),
        ),
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
        assert retained.search_doc is not None
        retrieved = broker.hydrate_search_centers([retained.search_doc], RunContext())
        originals = retrieved[(str(source.id), old.projection_ordinal)]
        assert len(originals) == 1 and originals[0].text == old.text
        assert originals[0].chunk_id == old.id
        assert originals[0].metadata["article_closure_complete"] is False
        wrong_ordinal = retained.search_doc.model_copy(
            update={"chunk_ind": current.projection_ordinal}
        )
        assert (
            broker.hydrate_search_centers([wrong_ordinal], RunContext())[
                (str(source.id), current.projection_ordinal)
            ]
            == []
        )
        broker.revalidate_evidence([retained], RunContext())
        old.text = "Changed retained text"
        db_session.commit()
        with pytest.raises(CorpusScopeUnavailable, match="changed"):
            broker.revalidate_evidence([retained], RunContext())
        source.user_id = stranger.id
        pc_corpus.user_id = stranger.id
        db_session.commit()
        with pytest.raises(PermissionError):
            broker.revalidate_evidence([retained], RunContext())
    finally:
        db_session.rollback()
        db_session.delete(source)
        pc_corpus.user_id = None
        db_session.commit()
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_native_preview_revalidates_saved_owner_and_original_bytes(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, pc_corpus: DocumentSet
) -> None:
    owner = create_test_user(db_session, "asv3_native_owner")
    stranger = create_test_user(db_session, "asv3_native_stranger")
    pc_corpus.user_id = owner.id
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
    db_session.add(
        DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=source.id)
    )
    message = ChatMessage(
        chat_session_id=chat.id,
        message="",
        token_count=0,
        message_type=MessageType.ASSISTANT,
    )
    db_session.add(message)
    db_session.commit()
    try:
        filters = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=IndexFilters(
                access_control_list=[],
                source_type=[DocumentSource.USER_FILE],
                attached_document_ids=[str(source.id)],
            ),
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
        pc_corpus.user_id = None
        db_session.commit()
        store.delete_file(file_id)
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_pc_membership_is_not_bypassed_by_sets_attachments_or_changed_identity(
    db_session: Session, pc_corpus: DocumentSet
) -> None:
    owner = create_test_user(db_session, "asv3_pc_owner")
    stranger = create_test_user(db_session, "asv3_pc_stranger")
    pc_corpus.user_id = owner.id
    other = DocumentSet(name="ASv3 other " + uuid4().hex, is_public=True)
    inside = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="inside",
        file_type="text/plain",
    )
    outside = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="outside",
        file_type="text/plain",
    )
    inaccessible = UserFile(
        id=uuid4(),
        user_id=stranger.id,
        file_id=uuid4().hex,
        name="private",
        file_type="text/plain",
    )
    db_session.add_all([other, inside, outside, inaccessible])
    db_session.flush()
    db_session.add_all(
        [
            DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=inside.id),
            DocumentSet__UserFile(document_set_id=other.id, user_file_id=outside.id),
        ]
    )
    db_session.commit()
    try:
        base = IndexFilters(
            access_control_list=[],
            source_type=[DocumentSource.USER_FILE],
            regulatory_chunks_only=True,
        )
        scoped = resolve_pc_corpus_scope(db_session, user=owner, filters=base)
        assert scoped.asv3_document_set_id == pc_corpus.id
        sources, _ = find_sources(db_session, user=owner, filters=scoped)
        assert {row.id for row in sources} == {inside.id}
        attached = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=base.model_copy(
                update={"attached_document_ids": [str(outside.id)]}
            ),
        )
        assert find_sources(db_session, user=owner, filters=attached)[0] == []
        disjoint = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=base.model_copy(update={"document_set": [other.name]}),
        )
        assert find_sources(db_session, user=owner, filters=disjoint)[0] == []
        for identifier in (outside.id, inaccessible.id):
            with pytest.raises(PermissionError):
                require_source(
                    db_session, user=owner, filters=scoped, source_id=identifier
                )
        pc_corpus.name = "Renamed " + uuid4().hex
        db_session.commit()
        with pytest.raises(PermissionError, match="Pinned PC"):
            find_sources(db_session, user=owner, filters=scoped)
    finally:
        db_session.rollback()
        for item in (inside, outside, inaccessible, other):
            db_session.delete(item)
        pc_corpus.user_id = None
        db_session.commit()
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()
