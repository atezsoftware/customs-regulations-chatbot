from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig
from onyx.regulatory.indexing_jobs.openrouter_batch import (
    HttpxOpenRouterBatchGateway,
    OpenRouterBatchContractError,
    OpenRouterBatchJobStatus,
    OpenRouterEmbeddingBatchRequest,
    parse_openrouter_embedding_results,
)


def _config() -> OpenRouterBatchConfig:
    return OpenRouterBatchConfig(
        api_url="https://openrouter.test/api/beta/batches",
        model_name="openai/text-embedding-3-large",
        effective_dimension=3,
        request_input_size=2,
        max_requests=10,
        max_inputs=40_000,
        max_bytes=1_000_000,
        completion_horizon_seconds=86_400,
    )


def _client(
    responses: Iterator[httpx.Response],
) -> tuple[httpx.Client, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = next(responses)
        response.request = request
        return response

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_gateway_submits_embedding_batch_with_provider_contract() -> None:
    responses = iter(
        [httpx.Response(202, json={"id": "batch-1", "status": "validating"})]
    )
    client, sent_requests = _client(responses)
    gateway = HttpxOpenRouterBatchGateway(
        config=_config(), api_key_provider=lambda: "secret", client=client
    )
    request = OpenRouterEmbeddingBatchRequest(
        custom_id="regulatory-0", inputs=["context\nchunk one", "chunk two"]
    )

    state = gateway.submit([request], submission_key="submission-a")

    assert state.remote_batch_id == "batch-1"
    assert state.status is OpenRouterBatchJobStatus.PENDING
    assert sent_requests[0].headers["Authorization"] == "Bearer secret"
    assert list(sent_requests[0].read())
    assert sent_requests[0].url.path == "/api/beta/batches"
    assert json.loads(sent_requests[0].content) == {
        "endpoint": "/v1/embeddings",
        "model": "openai/text-embedding-3-large",
        "requests": [
            {
                "custom_id": "regulatory-0",
                "body": {
                    "input": ["context\nchunk one", "chunk two"],
                    "dimensions": 3,
                },
            }
        ],
        "metadata": {"submission_key": "submission-a"},
    }


def test_completed_results_are_mapped_out_of_order_and_partial_errors_survive() -> None:
    raw_results = [
        {
            "custom_id": "request-b",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "openai/text-embedding-3-large",
                    "data": [
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    ],
                },
            },
        },
        {
            "custom_id": "request-a",
            "response": {"status_code": 429, "body": {"error": "rate limited"}},
        },
    ]

    parsed = parse_openrouter_embedding_results(
        raw_results,
        expected_custom_ids={"request-a", "request-b"},
        expected_model="openai/text-embedding-3-large",
        expected_dimension=3,
    )

    assert parsed["request-b"].vectors == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    assert parsed["request-a"].error_code == "http_429"


def test_result_parser_rejects_duplicate_custom_ids() -> None:
    raw = [
        {"custom_id": "request-a", "response": {"status_code": 500}},
        {"custom_id": "request-a", "response": {"status_code": 500}},
    ]

    with pytest.raises(OpenRouterBatchContractError, match="duplicate"):
        parse_openrouter_embedding_results(
            raw,
            expected_custom_ids={"request-a"},
            expected_model="openai/text-embedding-3-large",
            expected_dimension=3,
        )


def test_reconcile_uses_persisted_submission_identity() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "batch-found",
                            "status": "in_progress",
                            "metadata": {"submission_key": "submission-a"},
                        }
                    ]
                },
            )
        ]
    )
    client, _requests = _client(responses)
    gateway = HttpxOpenRouterBatchGateway(
        config=_config(), api_key_provider=lambda: "secret", client=client
    )

    state = gateway.reconcile_submission("submission-a")

    assert state is not None
    assert state.remote_batch_id == "batch-found"
    assert state.status is OpenRouterBatchJobStatus.RUNNING
