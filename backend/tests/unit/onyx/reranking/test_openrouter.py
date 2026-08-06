from unittest.mock import MagicMock

import httpx
import pytest

from onyx.reranking.constants import OPENROUTER_RERANK_URL
from onyx.reranking.models import (
    InvalidRerankResponse,
    RerankProviderError,
    RerankRateLimited,
    RerankTimeout,
)
from onyx.reranking.openrouter import OpenRouterRerankClient


@pytest.fixture
def http() -> MagicMock:
    return MagicMock(spec=httpx.Client)


@pytest.fixture
def client(http: MagicMock) -> OpenRouterRerankClient:
    return OpenRouterRerankClient(http=http)


def _response(
    payload: object, *, status_code: int = 200, headers: dict[str, str] | None = None
) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = httpx.Headers(headers)
    response.json.return_value = payload
    return response


def test_rerank_posts_fixed_private_request(
    client: OpenRouterRerankClient, http: MagicMock
) -> None:
    http.post.return_value = _response(
        {"results": [{"index": 1, "relevance_score": 0.9}]}
    )

    result = client.rerank(
        api_key="secret",
        model="cohere/rerank-v3.5",
        query="soru",
        documents=["a", "b"],
        top_n=2,
    )

    assert [(item.index, item.relevance_score) for item in result] == [(1, 0.9)]
    http.post.assert_called_once_with(
        OPENROUTER_RERANK_URL,
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
        json={
            "model": "cohere/rerank-v3.5",
            "query": "soru",
            "documents": ["a", "b"],
            "top_n": 2,
            "provider": {"zdr": True, "data_collection": "deny"},
        },
    )


def test_rerank_orders_equal_scores_by_original_index(
    client: OpenRouterRerankClient, http: MagicMock
) -> None:
    http.post.return_value = _response(
        {
            "results": [
                {"index": 2, "relevance_score": 0.7},
                {"index": 0, "relevance_score": 0.7},
                {"index": 1, "relevance_score": 0.9},
            ]
        }
    )

    result = client.rerank(
        api_key="k", model="m", query="q", documents=["a", "b", "c"], top_n=3
    )

    assert [item.index for item in result] == [1, 0, 2]


@pytest.mark.parametrize(
    "results",
    [
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.8},
        ],
        [{"index": True, "relevance_score": 0.9}],
        [{"index": 2, "relevance_score": 0.9}],
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
        [{"index": 0}],
        [{"relevance_score": 0.9}],
    ],
)
def test_rerank_rejects_invalid_result_items(
    client: OpenRouterRerankClient, http: MagicMock, results: list[dict[str, object]]
) -> None:
    http.post.return_value = _response({"results": results})

    with pytest.raises(InvalidRerankResponse):
        client.rerank(api_key="k", model="m", query="q", documents=["a", "b"], top_n=2)


@pytest.mark.parametrize("payload", [{}, {"results": None}, {"results": {}}, []])
def test_rerank_rejects_invalid_response_envelope(
    client: OpenRouterRerankClient, http: MagicMock, payload: object
) -> None:
    http.post.return_value = _response(payload)

    with pytest.raises(InvalidRerankResponse):
        client.rerank(api_key="k", model="m", query="q", documents=["a"], top_n=1)


def test_timeout_is_typed_and_not_retried(
    client: OpenRouterRerankClient, http: MagicMock
) -> None:
    http.post.side_effect = httpx.ReadTimeout("slow")

    with pytest.raises(RerankTimeout):
        client.rerank(api_key="k", model="m", query="q", documents=["a"], top_n=1)

    assert http.post.call_count == 1


def test_rate_limit_carries_retry_after(
    client: OpenRouterRerankClient, http: MagicMock
) -> None:
    http.post.return_value = _response(
        {"error": {"message": "limited"}},
        status_code=429,
        headers={"Retry-After": "900"},
    )

    with pytest.raises(RerankRateLimited) as exc_info:
        client.rerank(api_key="k", model="m", query="q", documents=["a"], top_n=1)

    assert exc_info.value.retry_after_seconds == 900
    assert http.post.call_count == 1


@pytest.mark.parametrize("status_code", [400, 401, 402, 403, 404, 422, 500, 503])
def test_provider_errors_are_typed_and_not_retried(
    client: OpenRouterRerankClient, http: MagicMock, status_code: int
) -> None:
    http.post.return_value = _response(
        {"error": {"message": "do not log this provider body"}},
        status_code=status_code,
    )

    with pytest.raises(RerankProviderError) as exc_info:
        client.rerank(api_key="k", model="m", query="q", documents=["a"], top_n=1)

    assert exc_info.value.status_code == status_code
    assert http.post.call_count == 1
