from typing import Any

import pytest

from onyx.context.search.models import ChunkIndexRequest, ChunkSearchRequest
from onyx.context.search.pipeline import search_pipeline


def test_pipeline_preserves_formal_turkish_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: ChunkIndexRequest | None = None

    def capture_search_request(**kwargs: Any) -> list[Any]:
        nonlocal captured_request
        captured_request = kwargs["query_request"]
        return []

    monkeypatch.setattr(
        "onyx.context.search.pipeline.search_chunks", capture_search_request
    )
    monkeypatch.setattr(
        "onyx.context.search.pipeline.fetch_ee_implementation_or_noop",
        lambda *_args: lambda **kwargs: kwargs["chunks"],
    )

    search_pipeline(
        chunk_search_request=ChunkSearchRequest(
            query="ve ile Geçici Madde 2 İmar",
            bypass_acl=True,
        ),
        document_index=object(),  # type: ignore[arg-type]
        user=None,  # type: ignore[arg-type]
        persona_search_info=None,
        prefetched_federated_retrieval_infos=[],
    )

    assert captured_request is not None
    assert captured_request.query == "ve ile Geçici Madde 2 İmar"
    assert captured_request.query_keywords is None
