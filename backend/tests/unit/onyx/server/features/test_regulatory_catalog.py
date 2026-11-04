"""Catalog routing, identity hydration and retrieval isolation."""

from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.routing import APIRoute
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.regulatory_catalog import (
    catalog_chunk_page_offset,
    catalog_folder_path,
    normalize_catalog_path,
)
from onyx.error_handling.exceptions import OnyxError
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.server.features.regulatory import catalog_api as api
from onyx.server.features.regulatory.catalog_models import (
    CatalogFile,
    CatalogSearchRequest,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot

MODULE = "onyx.server.features.regulatory.catalog_api"


@pytest.fixture
def catalog_file() -> CatalogFile:
    return CatalogFile(
        id=uuid4(),
        name="Gümrük/Kanunlar/4458.md",
        folder_path="Gümrük/Kanunlar",
        user_id=uuid4(),
        file_id="stored-original",
        status="COMPLETED",
        file_type="text/markdown",
        created_at=datetime.now(timezone.utc),
        chunk_count=7,
        document_sets=[],
        pending_review=False,
    )


@pytest.fixture
def retrieval() -> Iterator[tuple[MagicMock, MagicMock, MagicMock, MagicMock]]:
    index = MagicMock()
    with (
        patch(f"{MODULE}.get_current_search_settings"),
        patch(
            f"{MODULE}.get_reranker_configuration", return_value=MagicMock(enabled=True)
        ),
        patch(f"{MODULE}.EmbeddingModel.from_db_model"),
        patch(f"{MODULE}.get_default_document_index", return_value=index),
        patch(f"{MODULE}.get_current_tenant_id", return_value="tenant-catalog"),
        patch(f"{MODULE}.get_query_embedding", return_value=[0.1, 0.2]) as embed,
        patch(f"{MODULE}.observe_publication_read"),
        patch(
            f"{MODULE}.filter_publication_read", side_effect=lambda _, hits, _fn: hits
        ),
        patch(f"{MODULE}.require_publication_files") as require,
        patch(f"{MODULE}.rerank_regulatory_candidates_in_batches") as rerank,
    ):

        def score_candidates(
            *, chunks: list[InferenceChunk], **_kwargs: object
        ) -> RerankResult:
            return RerankResult(
                ordered_chunks=list(reversed(chunks)),
                scores_by_chunk={
                    (chunk.document_id, chunk.chunk_id): rank / 100
                    for rank, chunk in enumerate(chunks)
                },
                submitted_count=len(chunks),
                result_count=len(chunks),
                outcome=RerankOutcome.SUCCESS,
                fallback_used=False,
            )

        rerank.side_effect = score_candidates
        yield index, embed, require, rerank


def test_every_catalog_route_requires_admin_permission() -> None:
    routes = [route for route in api.router.routes if isinstance(route, APIRoute)]
    assert len(routes) == 6
    for route in routes:
        guards = [
            dep.call
            for dep in route.dependant.dependencies
            if getattr(dep.call, "_is_require_permission", False)
        ]
        assert len(guards) == 1
        assert (
            getattr(guards[0], "_required_permission")
            is Permission.FULL_ADMIN_PANEL_ACCESS
        )
        assert route.tags


@pytest.mark.parametrize(
    ("name", "normalized", "folder"),
    [
        ("Gümrük\\Kanunlar\\4458.md", "Gümrük/Kanunlar/4458.md", "Gümrük/Kanunlar"),
        ("/mevzuat///2026/a.md/", "mevzuat/2026/a.md", "mevzuat/2026"),
        ("4458.md", "4458.md", ""),
        ("100%_literal/a.md", "100%_literal/a.md", "100%_literal"),
    ],
)
def test_archive_folder_identity_is_preserved(
    name: str, normalized: str, folder: str
) -> None:
    assert normalize_catalog_path(name) == normalized
    assert catalog_folder_path(name) == folder


def test_file_page_preserves_global_total_and_pagination(
    catalog_file: CatalogFile,
) -> None:
    session = MagicMock()
    with (
        patch(
            f"{MODULE}.list_catalog_files", return_value=([MagicMock()], 103)
        ) as listing,
        patch(
            f"{MODULE}._file_snapshots", return_value={catalog_file.id: catalog_file}
        ),
    ):
        result = api.get_files(
            q="4458",
            folder="Gümrük",
            recursive=True,
            offset=50,
            limit=25,
            _=MagicMock(),
            db_session=session,
        )
    assert result.total == 103
    assert result.offset == 50
    assert result.items == [catalog_file]
    listing.assert_called_once_with(
        session, q="4458", folder="Gümrük", recursive=True, offset=50, limit=25
    )


def test_folder_children_are_global_and_have_independent_pagination() -> None:
    with patch(
        f"{MODULE}.list_catalog_folders", return_value=([("Kanunlar", 86)], 6)
    ) as listing:
        result = api.get_folders(
            parent="Gümrük\\",
            q="2026",
            offset=2,
            limit=3,
            _=MagicMock(),
            db_session=MagicMock(),
        )
    assert result.items[0].path == "Gümrük/Kanunlar"
    assert result.items[0].file_count == 86
    assert result.total == 6
    assert result.offset == 2
    assert listing.call_args.kwargs["q"] == "2026"


def _chunk(file_id: UUID, canonical_id: str, chunk_id: int = 0) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=chunk_id,
        blurb="indexed blurb",
        content="indexed projection",
        source_links=None,
        image_file_id=None,
        section_continuation=False,
        document_id=str(file_id),
        source_type=DocumentSource.FILE,
        semantic_identifier="indexed title",
        title="indexed title",
        boost=0,
        score=12.5,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        regulatory_chunk_id=canonical_id,
    )


def _snapshot(canonical_id: str, file_id: UUID) -> RegulatoryChunkSnapshot:
    return RegulatoryChunkSnapshot(
        id=canonical_id,
        user_file_id=str(file_id),
        text="Canonical Madde 1",
        position=0,
        chunk_type="article",
        heading_path=["Madde 1"],
        chunk_metadata={},
        validity_start_date=None,
        validity_end_date=None,
        status="active",
        source="indexed",
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_search_reranks_only_published_canonical_hybrid_candidates(
    retrieval: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    catalog_file: CatalogFile,
) -> None:
    index, embedding, require, rerank = retrieval
    canonical = MagicMock()
    canonical.id = "canonical-1"
    source = MagicMock()
    source.id = catalog_file.id
    source.name = catalog_file.name
    hit = _chunk(source.id, canonical.id)
    orphan = _chunk(source.id, "deleted-source")
    mismatched = _chunk(uuid4(), "wrong-file")
    index.hybrid_retrieval.return_value = [hit, hit, orphan, mismatched]
    snapshot = _snapshot(canonical.id, source.id)
    with (
        patch(
            f"{MODULE}.catalog_search_rows",
            return_value={
                canonical.id: (canonical, source),
                "wrong-file": (canonical, source),
            },
        ),
        patch(f"{MODULE}._file_snapshots", return_value={source.id: catalog_file}),
        patch(f"{MODULE}.RegulatoryChunkSnapshot.from_model", return_value=snapshot),
    ):
        result = api.search_chunks(
            CatalogSearchRequest(query="gümrük"),
            _=MagicMock(),
            db_session=MagicMock(),
        )
    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].chunk.id == canonical.id
    assert result.total_is_exact is False
    selected = index.hybrid_retrieval
    selected.assert_called_once()
    filters = selected.call_args.kwargs["filters"]
    assert filters.regulatory_chunks_only is True
    assert filters.tenant_id == "tenant-catalog"
    assert filters.access_control_list is None
    assert selected.call_args.kwargs["num_to_retrieve"] == 100
    embedding.assert_called_once()
    submitted = rerank.call_args.kwargs["chunks"]
    assert len(submitted) == 1
    assert submitted[0].content == snapshot.text
    assert result.items[0].score == 0
    assert result.candidate_limit == 100
    assert result.result_limit == 20
    assert result.per_file_limit == 8
    assert (
        sum(
            method.call_count
            for method in (
                index.keyword_retrieval,
                index.semantic_retrieval,
                index.hybrid_retrieval,
            )
        )
        == 1
    )
    require.assert_called_once()


def test_search_caps_candidates_and_applies_diversity_after_reranking(
    retrieval: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    catalog_file: CatalogFile,
) -> None:
    index, _, _, rerank = retrieval
    files = [catalog_file.model_copy(update={"id": uuid4()}) for _ in range(3)]
    file_order = [files[0]] * 80 + [files[1]] * 10 + [files[2]] * 11
    chunks = [_chunk(file.id, f"canonical-{i}", i) for i, file in enumerate(file_order)]
    index.hybrid_retrieval.return_value = chunks
    rows = {
        chunk.regulatory_chunk_id: (
            MagicMock(id=chunk.regulatory_chunk_id, user_file_id=file.id),
            file,
        )
        for chunk, file in zip(chunks, file_order)
    }
    with (
        patch(f"{MODULE}.catalog_search_rows", return_value=rows),
        patch(
            f"{MODULE}._file_snapshots", return_value={file.id: file for file in files}
        ),
        patch(
            f"{MODULE}.RegulatoryChunkSnapshot.from_model",
            side_effect=lambda chunk: _snapshot(chunk.id, chunk.user_file_id),
        ),
    ):
        result = api.search_catalog_chunks(
            CatalogSearchRequest(query="gümrük"), MagicMock()
        )
    assert len(rerank.call_args.kwargs["chunks"]) == 100
    assert len(result.items) == result.total == 20
    assert [hit.file.id for hit in result.items] == (
        [files[2].id] * 8 + [files[1].id] * 8 + [files[0].id] * 4
    )
    assert result.items[0].chunk.id == "canonical-99"
    assert result.items[0].score == 0.99
    assert result.has_more is False


def test_search_does_not_return_unreranked_fallbacks(
    retrieval: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    catalog_file: CatalogFile,
) -> None:
    index, _, require, rerank = retrieval
    chunk = _chunk(catalog_file.id, "canonical")
    index.hybrid_retrieval.return_value = [chunk]
    rerank.side_effect = None
    rerank.return_value = RerankResult(
        ordered_chunks=[chunk],
        scores_by_chunk={},
        submitted_count=1,
        result_count=0,
        outcome=RerankOutcome.TIMEOUT,
        fallback_used=True,
    )
    with (
        patch(
            f"{MODULE}.catalog_search_rows",
            return_value={"canonical": (MagicMock(), catalog_file)},
        ),
        patch(
            f"{MODULE}._file_snapshots", return_value={catalog_file.id: catalog_file}
        ),
        patch(
            f"{MODULE}.RegulatoryChunkSnapshot.from_model",
            return_value=_snapshot("canonical", catalog_file.id),
        ),
        pytest.raises(OnyxError, match="Yeniden sıralama"),
    ):
        api.search_catalog_chunks(CatalogSearchRequest(query="gümrük"), MagicMock())
    require.assert_not_called()


def test_publication_filter_removes_sources_before_external_reranking(
    retrieval: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    catalog_file: CatalogFile,
) -> None:
    index, _, _, rerank = retrieval
    index.hybrid_retrieval.return_value = [_chunk(catalog_file.id, "canonical")]
    with (
        patch(
            f"{MODULE}.catalog_search_rows",
            return_value={"canonical": (MagicMock(), catalog_file)},
        ),
        patch(
            f"{MODULE}._file_snapshots", return_value={catalog_file.id: catalog_file}
        ),
        patch(
            f"{MODULE}.RegulatoryChunkSnapshot.from_model",
            return_value=_snapshot("canonical", catalog_file.id),
        ),
        patch(f"{MODULE}.filter_publication_read", return_value=[]),
    ):
        result = api.search_catalog_chunks(
            CatalogSearchRequest(query="gümrük"), MagicMock()
        )
    assert result.items == []
    rerank.assert_not_called()


def test_search_requires_enabled_reranker_before_embedding(
    retrieval: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    _, embed, _, rerank = retrieval
    with (
        patch(
            f"{MODULE}.get_reranker_configuration",
            return_value=MagicMock(enabled=False),
        ),
        pytest.raises(OnyxError, match="etkinleştirin"),
    ):
        api.search_catalog_chunks(CatalogSearchRequest(query="gümrük"), MagicMock())
    embed.assert_not_called()
    rerank.assert_not_called()


def test_chunk_location_handles_tied_positions_history_and_file_membership() -> None:
    file_id, other_id = uuid4(), uuid4()
    engine = create_engine("sqlite://")
    with Session(engine) as session:
        session.execute(
            text(
                "CREATE TABLE regulatory_chunk (id TEXT, user_file_id TEXT, position INTEGER)"
            )
        )
        session.execute(
            text("INSERT INTO regulatory_chunk VALUES (:id, :file, :position)"),
            [
                {"id": f"chunk-{i:02}", "file": file_id.hex, "position": i // 2}
                for i in range(31)
            ]
            + [{"id": "other", "file": other_id.hex, "position": 0}],
        )
        assert catalog_chunk_page_offset(session, file_id, "chunk-25", limit=25) == 25
        assert catalog_chunk_page_offset(session, file_id, "chunk-24", limit=25) == 0
        assert catalog_chunk_page_offset(session, file_id, "other", limit=25) is None


@pytest.mark.parametrize("offset", [None, 25])
def test_chunk_location_validates_file_membership_and_publication(
    offset: int | None,
) -> None:
    with (
        patch(f"{MODULE}.catalog_chunk_page_offset", return_value=offset),
        patch(f"{MODULE}.observe_publication_read"),
        patch(f"{MODULE}.require_publication_files") as require,
    ):
        if offset is None:
            with pytest.raises(OnyxError, match="Chunk not found"):
                api.locate_chunk(
                    uuid4(), "chunk", limit=25, _=MagicMock(), db_session=MagicMock()
                )
            require.assert_not_called()
        else:
            result = api.locate_chunk(
                uuid4(), "chunk", limit=25, _=MagicMock(), db_session=MagicMock()
            )
            assert result.offset == 25
            require.assert_called_once()


def test_missing_file_fails_explicitly() -> None:
    with (
        patch(f"{MODULE}.get_catalog_file", return_value=None),
        pytest.raises(OnyxError),
    ):
        api.get_file(uuid4(), _=MagicMock(), db_session=MagicMock())


def test_chunked_file_index_reuses_durable_scoped_request(
    catalog_file: CatalogFile,
) -> None:
    file = MagicMock(id=catalog_file.id, status=UserFileStatus.CHUNKED)
    session = MagicMock()
    with (
        patch(f"{MODULE}.get_catalog_file", return_value=file),
        patch(f"{MODULE}.pending_intake_file_ids", return_value=set()),
        patch(f"{MODULE}.catalog_chunk_counts", return_value={file.id: 3}),
        patch(f"{MODULE}.get_current_tenant_id", return_value="tenant-catalog"),
        patch(f"{MODULE}._file_snapshots", return_value={file.id: catalog_file}),
        patch(
            "onyx.server.features.document_set.api._request_user_file_indexing"
        ) as request,
    ):
        result = api.index_file(file.id, _=MagicMock(), db_session=session)
    assert result.id == file.id
    request.assert_called_once_with([file], session, "tenant-catalog")


@pytest.mark.parametrize("reason", ["missing", "completed", "review", "empty"])
def test_index_refuses_unsafe_files_before_queueing(reason: str) -> None:
    file_id = uuid4()
    file = MagicMock(id=file_id, status=UserFileStatus.CHUNKED)
    if reason == "completed":
        file.status = UserFileStatus.COMPLETED
    with (
        patch(
            f"{MODULE}.get_catalog_file",
            return_value=None if reason == "missing" else file,
        ),
        patch(
            f"{MODULE}.pending_intake_file_ids",
            return_value={file_id} if reason == "review" else set(),
        ),
        patch(
            f"{MODULE}.catalog_chunk_counts",
            return_value={} if reason == "empty" else {file_id: 3},
        ),
        patch(
            "onyx.server.features.document_set.api._request_user_file_indexing"
        ) as request,
        pytest.raises(OnyxError),
    ):
        api.index_file(file_id, _=MagicMock(), db_session=MagicMock())
    request.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "a", "mode": "llm"},
        {"query": "a", "mode": "bm25"},
        {"query": "a", "mode": "semantic"},
        {"query": "a", "offset": 20},
        {"query": "a", "limit": 21},
    ],
)
def test_search_rejects_blank_queries_generation_modes_and_unbounded_offsets(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CatalogSearchRequest.model_validate(payload)
