from unittest.mock import MagicMock

import httpx
import pytest

from onyx.reranking.constants import SILICONFLOW_RERANK_URL
from onyx.reranking.models import InvalidRerankResponse, RerankProviderError
from onyx.reranking.siliconflow import (
    SILICONFLOW_RERANK_MODELS,
    SiliconFlowRerankClient,
    normalize_siliconflow_model,
)


def _response(payload: object, *, status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = httpx.Headers()
    response.json.return_value = payload
    return response


def test_rerank_posts_siliconflow_contract_and_orders_scores() -> None:
    http = MagicMock(spec=httpx.Client)
    http.post.return_value = _response(
        {
            "id": "request-id",
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ],
            "tokens": {"input_tokens": 12, "output_tokens": 0},
        }
    )
    client = SiliconFlowRerankClient(http=http)

    result = client.rerank(
        api_key="secret",
        model="qwen/qwen3-reranker-8b",
        query="soru",
        documents=["a", "b"],
        top_n=2,
    )

    assert [(item.index, item.relevance_score) for item in result] == [
        (1, 0.9),
        (0, 0.2),
    ]
    http.post.assert_called_once_with(
        SILICONFLOW_RERANK_URL,
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
        json={
            "model": "Qwen/Qwen3-Reranker-8B",
            "query": "soru",
            "documents": ["a", "b"],
            "top_n": 2,
            "return_documents": False,
        },
    )


def test_model_ids_are_fixed_and_case_normalized() -> None:
    assert SILICONFLOW_RERANK_MODELS == (
        "Qwen/Qwen3-Reranker-8B",
        "Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen3-Reranker-0.6B",
    )
    assert (
        normalize_siliconflow_model(" qwen/qwen3-reranker-8b ")
        == "Qwen/Qwen3-Reranker-8B"
    )
    with pytest.raises(ValueError):
        normalize_siliconflow_model("unknown/model")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": None},
        {"results": [{"index": 2, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": float("nan")}]},
    ],
)
def test_invalid_response_is_rejected(payload: object) -> None:
    http = MagicMock(spec=httpx.Client)
    http.post.return_value = _response(payload)

    with pytest.raises(InvalidRerankResponse):
        SiliconFlowRerankClient(http=http).rerank(
            api_key="k",
            model=SILICONFLOW_RERANK_MODELS[0],
            query="q",
            documents=["a"],
            top_n=1,
        )


def test_provider_error_is_typed_without_exposing_body() -> None:
    http = MagicMock(spec=httpx.Client)
    http.post.return_value = _response(
        {"message": "sensitive provider response"}, status_code=401
    )

    with pytest.raises(RerankProviderError) as exc_info:
        SiliconFlowRerankClient(http=http).rerank(
            api_key="k",
            model=SILICONFLOW_RERANK_MODELS[0],
            query="q",
            documents=["a"],
            top_n=1,
        )

    assert exc_info.value.status_code == 401
