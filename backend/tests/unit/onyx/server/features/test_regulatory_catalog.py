"""Catalog routing, identity hydration and retrieval isolation."""

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Literal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute
from pydantic import ValidationError

from onyx.db.enums import Permission, UserFileStatus
from onyx.db.regulatory_catalog import catalog_folder_path, normalize_catalog_path
from onyx.error_handling.exceptions import OnyxError
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
def retrieval() -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    index = MagicMock()
    with (
        patch(f"{MODULE}.get_current_search_settings"),
        patch(f"{MODULE}.EmbeddingModel.from_db_model"),
        patch(f"{MODULE}.get_default_document_index", return_value=index),
        patch(f"{MODULE}.get_current_tenant_id", return_value="tenant-catalog"),
        patch(f"{MODULE}.get_query_embedding", return_value=[0.1, 0.2]) as embed,
        patch(f"{MODULE}.observe_publication_read"),
        patch(
            f"{MODULE}.filter_publication_read", side_effect=lambda _, hits, _fn: hits
        ),
        patch(f"{MODULE}.require_publication_files") as require,
    ):
        yield index, embed, require


def test_every_catalog_route_requires_admin_permission() -> None:
    routes = [route for route in api.router.routes if isinstance(route, APIRoute)]
    assert len(routes) == 5
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


@pytest.mark.parametrize("mode", ["bm25", "semantic", "hybrid"])
def test_search_uses_only_selected_retrieval_and_canonical_identity(
    retrieval: tuple[MagicMock, MagicMock, MagicMock],
    catalog_file: CatalogFile,
    mode: Literal["bm25", "semantic", "hybrid"],
) -> None:
    index, embedding, require = retrieval
    canonical = MagicMock()
    canonical.id = "canonical-1"
    source = MagicMock()
    source.id = catalog_file.id
    hit = MagicMock(
        regulatory_chunk_id=canonical.id, document_id=str(source.id), score=12.5
    )
    orphan = MagicMock(regulatory_chunk_id="deleted-source", document_id=str(source.id))
    mismatched = MagicMock(regulatory_chunk_id="wrong-file", document_id=str(uuid4()))
    for method in (
        index.keyword_retrieval,
        index.semantic_retrieval,
        index.hybrid_retrieval,
    ):
        method.return_value = [hit, hit, orphan, mismatched]
    snapshot = RegulatoryChunkSnapshot(
        id=canonical.id,
        user_file_id=str(source.id),
        text="Madde 1",
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
            CatalogSearchRequest(query="gümrük", mode=mode),
            _=MagicMock(),
            db_session=MagicMock(),
        )
    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].chunk.id == canonical.id
    assert result.total_is_exact is False
    selected = {
        "bm25": index.keyword_retrieval,
        "semantic": index.semantic_retrieval,
        "hybrid": index.hybrid_retrieval,
    }[mode]
    selected.assert_called_once()
    filters = selected.call_args.kwargs["filters"]
    assert filters.regulatory_chunks_only is True
    assert filters.tenant_id == "tenant-catalog"
    assert filters.access_control_list is None
    assert selected.call_args.kwargs["num_to_retrieve"] == 500
    if mode == "bm25":
        embedding.assert_not_called()
    else:
        embedding.assert_called_once()
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
    [{"query": " "}, {"query": "a", "mode": "llm"}, {"query": "a", "offset": 500}],
)
def test_search_rejects_blank_queries_generation_modes_and_unbounded_offsets(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CatalogSearchRequest.model_validate(payload)
