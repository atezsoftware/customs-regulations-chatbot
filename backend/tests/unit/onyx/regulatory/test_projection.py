import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.models import RegulatoryChunk
from onyx.indexing.models import DocAwareChunk
from onyx.llm.constants import LlmProviderNames
from onyx.regulatory import projection
from onyx.regulatory.projection import (
    _build_document_shell,
    _contextualize_chunks,
    _row_context_text,
    _rows_in_structural_order,
    _rows_to_doc_aware_chunks,
    project_user_file_to_index,
)


def _settings(*, current: bool, future: bool = False) -> MagicMock:
    settings = MagicMock()
    settings.id = 1 if current else 2
    settings.status.is_current.return_value = current
    settings.status.is_future.return_value = future
    settings.enable_contextual_rag = False
    return settings


def test_structural_order_does_not_reassign_immutable_projection_ordinals() -> None:
    imported_later = SimpleNamespace(
        id="imported-20",
        source="indexed",
        position=20,
        projection_ordinal=20,
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )
    imported_earlier = SimpleNamespace(
        id="imported-10",
        source="indexed",
        position=10,
        projection_ordinal=10,
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )
    first_amendment = SimpleNamespace(
        id="amendment-a",
        source="amendment",
        position=10,
        projection_ordinal=1_000_000_064,
        created_at=datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc),
    )
    second_amendment = SimpleNamespace(
        id="amendment-b",
        source="amendment",
        position=20,
        projection_ordinal=1_000_000_066,
        created_at=datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc),
    )

    ordered = _rows_in_structural_order(
        cast(
            list[RegulatoryChunk],
            [second_amendment, imported_later, first_amendment, imported_earlier],
        )
    )

    assert [row.id for row in ordered] == [
        "amendment-a",
        "imported-10",
        "amendment-b",
        "imported-20",
    ]
    assert [row.projection_ordinal for row in ordered] == [
        1_000_000_064,
        10,
        1_000_000_066,
        20,
    ]


def test_contextual_projection_rejects_incomplete_eligible_chunks() -> None:
    canonical_document = MagicMock()
    chunks = [cast(DocAwareChunk, MagicMock()), cast(DocAwareChunk, MagicMock())]
    rows = [cast(RegulatoryChunk, MagicMock()), cast(RegulatoryChunk, MagicMock())]
    for position, (chunk, row) in enumerate(zip(chunks, rows)):
        chunk.chunk_id = position
        chunk.content = f"chunk-{position}"
        chunk.title_prefix = ""
        chunk.metadata_suffix_semantic = ""
        chunk.doc_summary = ""
        chunk.chunk_context = ""
        chunk.source_document = canonical_document
        row.id = f"row-{position}"
        row.position = position
        row.text = f"row text {position}"
        row.heading_path = []
        row.validity_start_date = None
        row.validity_end_date = None

    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Regulation"
    settings = _settings(current=False, future=True)
    settings.enable_contextual_rag = True
    llm = MagicMock()
    llm.config.model_name = "contextual-model"
    llm.config.model_provider = "provider"
    embedder = MagicMock()
    embedder.embedding_model.tokenizer.encode.return_value = [1]

    def leave_one_chunk_incomplete(*, chunks: list[DocAwareChunk], **_: object) -> None:
        chunks[0].chunk_context = "complete context"

    with (
        patch(
            "onyx.regulatory.projection.require_contextual_rag_llm",
            return_value=llm,
        ),
        patch("onyx.regulatory.projection.get_tokenizer", return_value=MagicMock()),
        patch(
            "onyx.indexing.indexing_pipeline.add_contextual_summaries",
            side_effect=leave_one_chunk_incomplete,
        ) as add_summaries,
        patch("onyx.regulatory.projection.USE_CHUNK_SUMMARY", True),
        pytest.raises(RuntimeError, match="1/2 eligible chunks"),
    ):
        _contextualize_chunks(
            chunks=chunks,
            rows=rows,
            user_file=user_file,
            embedder=embedder,
            search_settings=settings,
        )

    assert add_summaries.call_args.kwargs["raise_on_failure"] is True
    assert all(chunk.source_document is canonical_document for chunk in chunks)


def test_vertex_contextual_projection_uses_offline_budget_tokenizer() -> None:
    llm = MagicMock()
    llm.config.model_provider = "vertex_ai"
    llm.config.model_name = "gemini-3.6-flash"
    tokenizer = MagicMock()

    with (
        patch(
            "onyx.regulatory.projection.get_contextual_token_budget_tokenizer",
            return_value=tokenizer,
        ) as get_vertex_tokenizer,
        patch("onyx.regulatory.projection.get_tokenizer") as get_generic_tokenizer,
    ):
        selected = projection._get_contextual_tokenizer(llm)

    assert selected is tokenizer
    get_vertex_tokenizer.assert_called_once_with(
        model_provider=LlmProviderNames.VERTEX_AI,
        model_name="gemini-3.6-flash",
    )
    get_generic_tokenizer.assert_not_called()


def test_projection_and_context_use_reverse_article_anchor() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Belge"
    document = _build_document_shell(user_file)
    row = MagicMock()
    row.id = "row-4a"
    row.text = "Eklenen hüküm."
    row.heading_path = ["Belge", "MADDE 4", "4A Maddesi:", "(1)"]
    row.chunk_metadata = {"article_no": "4"}
    row.validity_start_date = None
    row.validity_end_date = None
    row.projection_ordinal = 1_000_000_064

    with patch("onyx.regulatory.projection.extract_blurb", return_value=row.text):
        chunks = _rows_to_doc_aware_chunks(document, [row], MagicMock())

    assert chunks[0].chunk_id == 1_000_000_064
    assert chunks[0].heading_path == ["Belge", "4A Maddesi:", "(1)"]
    assert _row_context_text(row) == "Belge > 4A Maddesi: > (1)\nEklenen hüküm."


def test_projection_and_context_repair_legacy_article_metadata_lineage() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Belge"
    document = _build_document_shell(user_file)
    row = MagicMock()
    row.id = "row-75-intro"
    row.text = "2. Kapsamlı teminatın tutarı:"
    row.heading_path = ["Belge", "Teminatlar", "2. Kapsamlı teminatın tutarı"]
    row.chunk_metadata = {
        "article_no": "75",
        "paragraph_no": None,
        "clause_label": None,
    }
    row.chunk_type = "numbered_section"
    row.validity_start_date = None
    row.validity_end_date = None
    row.projection_ordinal = 75

    with patch("onyx.regulatory.projection.extract_blurb", return_value=row.text):
        chunks = _rows_to_doc_aware_chunks(document, [row], MagicMock())

    expected_path = [
        "Belge",
        "Teminatlar",
        "MADDE 75",
        "2. Kapsamlı teminatın tutarı",
    ]
    assert chunks[0].heading_path == expected_path
    assert _row_context_text(row) == (
        "Belge > Teminatlar > MADDE 75 > 2. Kapsamlı teminatın tutarı\n"
        "2. Kapsamlı teminatın tutarı:"
    )


def test_projection_preserves_image_and_source_links() -> None:
    from chonkie import SentenceChunker

    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection

    row = RegulatoryChunk(
        id="image-row",
        user_file_id=uuid4(),
        text="Approved image transcript",
        position=0,
        projection_ordinal=0,
        heading_path=["EK-1"],
        chunk_type="image",
        chunk_metadata={
            "image_file_id": "original-image",
            "source_links": {"0": "https://example.gov/annex"},
        },
    )
    document = Document(
        id=str(row.user_file_id),
        source=DocumentSource.USER_FILE,
        metadata={},
        semantic_identifier="Annex",
        sections=[TextSection(text=row.text, link=None)],
    )
    chunks = _rows_to_doc_aware_chunks(
        document,
        [row],
        SentenceChunker(
            tokenizer_or_token_counter=lambda text: len(text.split()),
            chunk_size=100,
            chunk_overlap=0,
            return_type="texts",
        ),
    )
    assert chunks[0].image_file_id == "original-image"
    assert chunks[0].source_links == {0: "https://example.gov/annex"}


@pytest.mark.parametrize(
    "include_chunked,include_failed,current_id,count",
    [(False, False, None, 2), (True, False, 41, 3), (False, True, 41, 0)],
)
def test_projection_routes_exact_scope_and_status_policy_to_owned_writer(
    include_chunked: bool, include_failed: bool, current_id: int | None, count: int
) -> None:
    from sqlalchemy.orm import Session

    from onyx.db.models import UserFile

    session = MagicMock(spec=Session)
    session.new, session.dirty, session.deleted = set(), set(), set()
    file = UserFile(id=uuid4(), name="regulation.md")
    events: list[str] = []
    session.rollback.side_effect = lambda: events.append("rollback")
    session.refresh.side_effect = lambda _: events.append("refresh")

    def republish(file_id: object, tenant_id: str, **kwargs: object) -> int:
        assert file_id == file.id and tenant_id == "tenant-a"
        assert kwargs == {
            "include_chunked": include_chunked,
            "include_failed": include_failed,
            "current_search_settings_id": current_id,
        }
        events.append("owned_publish")
        return count

    with patch(
        "onyx.regulatory.writer_publication.republish_user_file", side_effect=republish
    ):
        result = project_user_file_to_index(
            session,
            file,
            "tenant-a",
            include_chunked=include_chunked,
            include_failed=include_failed,
            current_search_settings_id=current_id,
        )
    assert result == count
    assert events == ["rollback", "owned_publish", "refresh"]
    session.refresh.assert_called_once_with(file)


@pytest.mark.parametrize("pending", ["new", "dirty", "deleted"])
def test_projection_rejects_unstaged_canonical_mutation(pending: str) -> None:
    from sqlalchemy.orm import Session

    from onyx.db.models import UserFile

    session = MagicMock(spec=Session)
    session.new, session.dirty, session.deleted = set(), set(), set()
    setattr(session, pending, {object()})
    with patch("onyx.regulatory.writer_publication.republish_user_file") as publish:
        with pytest.raises(ValueError, match="staged under publication ownership"):
            project_user_file_to_index(session, UserFile(id=uuid4()), "tenant-a")
    publish.assert_not_called()
    session.rollback.assert_not_called()


def test_owned_projection_failure_propagates_without_refreshing_stale_file() -> None:
    from sqlalchemy.orm import Session

    from onyx.db.models import UserFile

    session = MagicMock(spec=Session)
    session.new, session.dirty, session.deleted = set(), set(), set()
    with patch(
        "onyx.regulatory.writer_publication.republish_user_file",
        side_effect=ValueError("PRESENT changed"),
    ):
        with pytest.raises(ValueError, match="PRESENT changed"):
            project_user_file_to_index(
                session, UserFile(id=uuid4()), "tenant-a", current_search_settings_id=41
            )
    session.rollback.assert_called_once_with()
    session.refresh.assert_not_called()
