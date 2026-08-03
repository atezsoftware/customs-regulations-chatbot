from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from onyx.context.search.models import ChunkIndexRequest, IndexFilters
from onyx.context.search.retrieval import search_runner


def _request(*, regulatory_chunks_only: bool) -> ChunkIndexRequest:
    return ChunkIndexRequest(
        query="unknown controlling terminology",
        filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=regulatory_chunks_only,
        ),
    )


def test_regulatory_hybrid_search_falls_back_to_keyword_when_embedding_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    embedding_error = RuntimeError("embedding provider unavailable")
    monkeypatch.setattr(
        search_runner,
        "get_query_embedding",
        MagicMock(side_effect=embedding_error),
    )
    expected_chunks = [MagicMock()]
    document_index = MagicMock()
    document_index.keyword_retrieval.return_value = expected_chunks

    result = search_runner._embed_and_hybrid_search(
        _request(regulatory_chunks_only=True),
        document_index,
    )

    assert result == expected_chunks
    document_index.keyword_retrieval.assert_called_once()


def test_non_regulatory_hybrid_search_preserves_embedding_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    embedding_error = RuntimeError("embedding provider unavailable")
    monkeypatch.setattr(
        search_runner,
        "get_query_embedding",
        MagicMock(side_effect=embedding_error),
    )
    document_index = MagicMock()

    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        search_runner._embed_and_hybrid_search(
            _request(regulatory_chunks_only=False),
            document_index,
        )

    document_index.keyword_retrieval.assert_not_called()
