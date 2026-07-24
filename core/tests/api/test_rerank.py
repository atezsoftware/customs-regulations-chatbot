import os
from unittest.mock import patch

import httpx

from fs_explorer_api.rerank import ChunkReranker, get_reranker, reset_reranker_singleton


def test_rerank_short_circuits_for_zero_or_one_document() -> None:
    client = ChunkReranker(api_key="test", client=httpx.Client())
    assert client.rerank("q", [], top_n=5) == []
    assert client.rerank("q", ["only doc"], top_n=5) == [(0, 1.0)]


def test_rerank_returns_ordered_index_score_pairs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    raw_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    reranker = ChunkReranker(api_key="test", client=raw_client)

    result = reranker.rerank("query", ["doc a", "doc b", "doc c"], top_n=2)

    assert result == [(2, 0.9), (0, 0.4)]


def test_rerank_returns_none_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    raw_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    reranker = ChunkReranker(api_key="test", client=raw_client)

    assert reranker.rerank("query", ["a", "b"], top_n=2) is None


def test_rerank_returns_none_on_malformed_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    raw_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    reranker = ChunkReranker(api_key="test", client=raw_client)

    assert reranker.rerank("query", ["a", "b"], top_n=2) is None


def test_rerank_returns_none_on_out_of_range_index() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"index": 99, "relevance_score": 0.9}]}
        )

    raw_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.test"
    )
    reranker = ChunkReranker(api_key="test", client=raw_client)

    assert reranker.rerank("query", ["a", "b"], top_n=2) is None


class TestGetReranker:
    def setup_method(self) -> None:
        reset_reranker_singleton()

    def teardown_method(self) -> None:
        reset_reranker_singleton()

    def test_disabled_returns_none(self) -> None:
        with patch.dict(
            os.environ,
            {"FS_EXPLORER_RERANK_ENABLED": "false", "OPENROUTER_API_KEY": "key"},
        ):
            assert get_reranker() is None

    def test_missing_api_key_returns_none(self) -> None:
        with patch.dict(
            os.environ, {"FS_EXPLORER_RERANK_ENABLED": "true"}, clear=False
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            assert get_reranker() is None

    def test_enabled_with_key_builds_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FS_EXPLORER_RERANK_ENABLED": "true",
                "OPENROUTER_API_KEY": "key",
                "FS_EXPLORER_RERANK_MODEL": "cohere/rerank-4-fast",
            },
        ):
            reranker = get_reranker()
        assert reranker is not None
        assert reranker.model == "cohere/rerank-4-fast"
        # Singleton: a second call returns the same instance without
        # re-reading env vars.
        assert get_reranker() is reranker
