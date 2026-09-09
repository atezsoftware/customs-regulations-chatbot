from datetime import date
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


def test_context_only_history_retains_canonical_identity_and_validity(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_context_projections import (
        activate_context_projection,
        get_effective_context_projection,
        persist_context_view,
        retire_context_projection,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceRange,
        ContextSourceSnapshot,
        FrozenContextProjection,
        PreparedContextView,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    chunk = _chunk(source_session, file, 0, "Unchanged legal text")
    chunk.validity_start_date = date(2026, 1, 1)
    chunk.validity_end_date = date(2027, 1, 1)

    def prepare(context: str) -> list[str]:
        text = context + chunk.text
        config: dict[str, str | int | float | bool | None] = {
            "model": "embedding",
            "dimension": 3,
        }
        snapshot = ContextSourceSnapshot(
            sha256=context_hash(context),
            selector="fixture",
            reference_date=date(2026, 1, 1),
            text=context,
            ordered_ranges=[
                ContextSourceRange(
                    canonical_chunk_id=chunk.id, start=0, end=len(context)
                )
            ],
        )
        projection = FrozenContextProjection(
            canonical_chunk_id=chunk.id,
            source_snapshot_sha256=snapshot.sha256,
            generation_path="normal",
            request_hashes=[],
            embedding_input_sha256=context_hash([text]),
            embedding_config_sha256=context_hash(config),
            embedding_config=config,
            embedding_texts=[text],
            canonical_text_sha256=context_hash(chunk.text),
            metadata_sha256="meta",
            doc_summary=context,
        )
        return persist_context_view(
            source_session,
            user_file_id=file.id,
            view=PreparedContextView(projections=[projection], snapshots=[snapshot]),
        )

    first = prepare("old context")
    assert first == prepare("old context")
    second = prepare("new context")
    assert first != second
    activate_context_projection(
        source_session,
        UUID(first[0]),
        effective_start=date(2026, 1, 1),
        effective_end=None,
    )
    retire_context_projection(
        source_session, UUID(first[0]), effective_end=date(2026, 9, 10)
    )
    activate_context_projection(
        source_session,
        UUID(second[0]),
        effective_start=date(2026, 9, 10),
        effective_end=None,
    )
    before = get_effective_context_projection(
        source_session, chunk.id, as_of_date=date(2026, 9, 9)
    )
    after = get_effective_context_projection(
        source_session, chunk.id, as_of_date=date(2026, 9, 10)
    )
    assert before is not None and before.id == UUID(first[0])
    assert after is not None and after.id == UUID(second[0])
    assert before.canonical_chunk_id == after.canonical_chunk_id == chunk.id
    assert (
        get_effective_context_projection(
            source_session, chunk.id, as_of_date=date(2027, 1, 1)
        )
        is None
    )
    assert chunk.text == "Unchanged legal text"


def test_context_snapshot_cannot_cross_file_scope(source_session: Session) -> None:
    import pytest

    from onyx.db.regulatory_context_projections import persist_context_view
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceRange,
        ContextSourceSnapshot,
        PreparedContextView,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    other = _file(source_session, document_set)
    foreign = _chunk(source_session, other, 0, "private")
    snapshot = ContextSourceSnapshot(
        sha256="a" * 64,
        selector="fixture",
        reference_date=date.today(),
        text="private",
        ordered_ranges=[
            ContextSourceRange(canonical_chunk_id=foreign.id, start=0, end=7)
        ],
    )
    with pytest.raises(ValueError, match="scope"):
        persist_context_view(
            source_session,
            user_file_id=file.id,
            view=PreparedContextView(snapshots=[snapshot]),
        )
