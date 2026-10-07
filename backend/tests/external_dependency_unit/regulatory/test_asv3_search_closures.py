"""Actual PostgreSQL owner/date/parent fences for source-local closure hydration."""

from datetime import date
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusClosureRead,
    read_search_source_closures,
    resolve_pc_corpus_scope,
)
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    User,
    UserFile,
)
from onyx.db.regulatory_chunks import (
    RegulatoryChunkSiblingCandidate,
    _load_compact_sibling_candidates,
    select_bounded_same_provision_siblings,
)
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.regulatory.test_asv3_corpus_fences import (
    pc_corpus as pc_corpus,
)


@pytest.mark.usefixtures("tenant_context")
def test_actual_db_local_closure_keeps_owner_date_and_annex_scope(
    db_session: Session,
    pc_corpus: DocumentSet,
) -> None:
    owner = create_test_user(db_session, "asv3_closure_owner")
    stranger = create_test_user(db_session, "asv3_closure_stranger")
    pc_corpus.user_id = owner.id
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="Closure fence test",
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
        position=0,
        projection_ordinal=0,
        text="Old frozen rule",
        heading_path=["Law", "Ek 1", "MADDE 3"],
        status="superseded",
        validity_start_date=date(2025, 1, 1),
        validity_end_date=date(2026, 1, 1),
    )
    current = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        position=0,
        projection_ordinal=1,
        text="New rule opening",
        heading_path=["Law", "Ek 1", "MADDE 3", "(1)"],
        validity_start_date=date(2026, 1, 1),
    )
    tail = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        position=1,
        projection_ordinal=2,
        text="Operative continuation",
        heading_path=["Law", "Ek 1", "MADDE 3", "(2)"],
        validity_start_date=date(2026, 1, 1),
    )
    other = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        position=2,
        projection_ordinal=3,
        text="Other annex rule",
        heading_path=["Law", "Ek 2", "MADDE 3"],
        validity_start_date=date(2026, 1, 1),
    )
    aggregate = RegulatoryChunk(
        id=uuid4().hex,
        user_file_id=source.id,
        position=50,
        projection_ordinal=4,
        text="Aggregate text must never be cited",
        chunk_type="hierarchical_aggregate",
        heading_path=current.heading_path,
        validity_start_date=date(2026, 1, 1),
    )
    db_session.add_all([old, current, tail, other, aggregate])
    db_session.commit()
    try:
        filters = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=IndexFilters(
                access_control_list=[],
                source_type=[DocumentSource.USER_FILE],
                regulatory_chunks_only=True,
                attached_document_ids=[str(source.id)],
                as_of_date=date(2026, 1, 1),
            ),
        )

        def read(
            center_id: str,
            *,
            scoped_filters: IndexFilters = filters,
            scoped_user: User = owner,
            max_chars: int = 64000,
            local_groups: bool = False,
        ) -> CorpusClosureRead:
            return read_search_source_closures(
                db_session,
                user=scoped_user,
                filters=scoped_filters,
                source_id=source.id,
                index=None,
                check_active=lambda: None,
                center_ids=(center_id,),
                max_chars=max_chars,
                local_groups=local_groups,
            )

        result = read(current.id)
        assert {chunk.id for chunk in result.chunks} == {current.id, tail.id}
        assert result.complete[current.id]
        group = read(aggregate.id, local_groups=True)
        assert [chunk.id for chunk in group.chunks] == [current.id]
        assert group.members[aggregate.id] == (current.id,)
        assert group.complete[aggregate.id]
        assert group.center_ordinals[aggregate.id] == aggregate.projection_ordinal
        with pytest.raises(PermissionError):
            read(aggregate.id, local_groups=True, scoped_user=stranger)
        historical = read(
            old.id,
            scoped_filters=filters.model_copy(
                update={"as_of_date": date(2025, 12, 31)}
            ),
        )
        assert [chunk.text for chunk in historical.chunks] == ["Old frozen rule"]
        with pytest.raises(PermissionError):
            read(current.id, scoped_user=stranger)
        partial = read(current.id, max_chars=len(current.text))
        assert [chunk.id for chunk in partial.chunks] == [current.id]
        assert partial.continuation[current.id] == (tail.id,)
        assert not partial.complete[current.id]
    finally:
        db_session.rollback()
        pc_corpus.user_id = None
        db_session.flush()
        for item in (old, current, tail, other, aggregate):
            db_session.delete(item)
        db_session.delete(source)
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
def test_actual_db_sibling_projection_preserves_full_selection_without_vectors(
    db_session: Session,
) -> None:
    owner = create_test_user(db_session, "asv3_sibling_projection_owner")
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="Sibling projection test",
        file_type="text/plain",
    )
    db_session.add(source)
    db_session.flush()
    metadata_rows = [
        {
            "article_no": 3,
            "article_title": "Rule",
            "paragraph_no": 1,
            "clause_label": "a",
            "image_file_id": "image-original",
        },
        {"article_no": 3, "paragraph_no": 2},
        {
            "article_no": 3,
            "article_title": None,
            "paragraph_no": False,
            "clause_label": {"legacy": "value"},
        },
        {},
    ]
    chunks = [
        RegulatoryChunk(
            id=uuid4().hex,
            user_file_id=source.id,
            position=position,
            projection_ordinal=position + 10,
            text=f"Unabridged operative rule {position}" * 20,
            heading_path=["Law", "Ek 1" if position < 2 else "Ek 2", "MADDE 3"],
            chunk_metadata={
                **metadata,
                "embeddings": [0.1] * 8192,
                "frozen_payload": "not selected" * 1024,
            },
            chunk_type="article",
            status="active" if position != 3 else "superseded",
            validity_start_date=date(2025, 1, 1),
            validity_end_date=date(2026, 1, 1) if position == 3 else None,
        )
        for position, metadata in enumerate(metadata_rows)
    ]
    db_session.add_all(chunks)
    db_session.commit()
    statements: list[str] = []

    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    try:
        expected = [
            RegulatoryChunkSiblingCandidate(
                regulatory_chunk_id=chunk.id,
                user_file_id=source.id,
                position=chunk.position,
                text=chunk.text,
                status=chunk.status,
                heading_path=tuple(chunk.heading_path),
                article_no=str(metadata["article_no"])
                if metadata.get("article_no") is not None
                else None,
                article_title=str(metadata["article_title"])
                if metadata.get("article_title") is not None
                else None,
                paragraph_no=str(metadata["paragraph_no"])
                if metadata.get("paragraph_no") is not None
                else None,
                clause_label=str(metadata["clause_label"])
                if metadata.get("clause_label") is not None
                else None,
                image_file_id=cast(str | None, metadata.get("image_file_id")),
                chunk_type=chunk.chunk_type,
                validity_start_date=chunk.validity_start_date,
                validity_end_date=chunk.validity_end_date,
                projection_ordinal=chunk.projection_ordinal,
            )
            for chunk, metadata in zip(chunks, metadata_rows)
        ]
        connection = db_session.connection()
        event.listen(connection, "before_cursor_execute", capture)
        try:
            actual = _load_compact_sibling_candidates(db_session, [source.id])
        finally:
            event.remove(connection, "before_cursor_execute", capture)
        assert actual == expected
        assert len(statements) == 1
        selected_columns = statements[0].split("FROM", 1)[0]
        assert "chunk_metadata," not in selected_columns
        assert " AS article_no" in selected_columns
        assert " AS image_file_id" in selected_columns
        for as_of in (None, date(2025, 12, 31), date(2026, 1, 1)):
            assert select_bounded_same_provision_siblings(
                actual,
                [chunks[0].id],
                query="operative",
                as_of_date=as_of,
            ) == select_bounded_same_provision_siblings(
                expected,
                [chunks[0].id],
                query="operative",
                as_of_date=as_of,
            )
        assert _load_compact_sibling_candidates(db_session, []) == []
    finally:
        db_session.rollback()
        for chunk in chunks:
            db_session.delete(chunk)
        db_session.delete(source)
        db_session.delete(owner)
        db_session.commit()
