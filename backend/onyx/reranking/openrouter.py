import json
import math
from collections.abc import Sequence
from typing import Any

import httpx

from onyx.reranking.constants import (
    MAX_RERANK_REQUEST_BYTES,
    MAX_RERANK_REQUEST_TOKENS,
    OPENROUTER_RERANK_URL,
    RERANK_CONNECT_TIMEOUT_SECONDS,
    RERANK_POOL_TIMEOUT_SECONDS,
    RERANK_READ_TIMEOUT_SECONDS,
    RERANK_WRITE_TIMEOUT_SECONDS,
)
from onyx.reranking.models import (
    InvalidRerankResponse,
    RerankPayloadTooLarge,
    RerankProviderError,
    RerankRateLimited,
    RerankScore,
    RerankTimeout,
)
from onyx.utils.retry_after import parse_retry_after_seconds


def _build_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=RERANK_CONNECT_TIMEOUT_SECONDS,
            read=RERANK_READ_TIMEOUT_SECONDS,
            write=RERANK_WRITE_TIMEOUT_SECONDS,
            pool=RERANK_POOL_TIMEOUT_SECONDS,
        ),
        transport=httpx.HTTPTransport(retries=0),
    )


class OpenRouterRerankClient:
    def __init__(self, *, http: httpx.Client | None = None) -> None:
        self.http = http or _build_http_client()

    def rerank(
        self,
        *,
        api_key: str,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankScore]:
        if not documents or top_n < 1 or top_n > len(documents):
            raise ValueError("top_n must select from a non-empty document list")
        request_payload = {
            "model": model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "provider": {"zdr": True, "data_collection": "deny"},
        }
        serialized_request = json.dumps(
            request_payload, ensure_ascii=False, separators=(",", ":")
        )
        if (
            len(serialized_request.encode("utf-8")) > MAX_RERANK_REQUEST_BYTES
            or (len(serialized_request) + 3) // 4 > MAX_RERANK_REQUEST_TOKENS
        ):
            raise RerankPayloadTooLarge()
        try:
            response = self.http.post(
                OPENROUTER_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
        except httpx.TimeoutException as error:
            raise RerankTimeout() from error
        except httpx.RequestError as error:
            raise RerankProviderError(status_code=None) from error

        retry_after = parse_retry_after_seconds(response.headers.get("Retry-After"))
        if response.status_code == 429:
            raise RerankRateLimited(retry_after_seconds=retry_after)
        if not 200 <= response.status_code < 300:
            raise RerankProviderError(
                status_code=response.status_code,
                retry_after_seconds=retry_after,
            )

        try:
            payload: Any = response.json()
        except (TypeError, ValueError) as error:
            raise InvalidRerankResponse() from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise InvalidRerankResponse()

        scores: list[RerankScore] = []
        seen_indices: set[int] = set()
        for item in payload["results"]:
            if (
                not isinstance(item, dict)
                or "index" not in item
                or "relevance_score" not in item
            ):
                raise InvalidRerankResponse()
            index = item["index"]
            relevance_score = item["relevance_score"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise InvalidRerankResponse()
            if index < 0 or index >= len(documents) or index in seen_indices:
                raise InvalidRerankResponse()
            if isinstance(relevance_score, bool) or not isinstance(
                relevance_score, (int, float)
            ):
                raise InvalidRerankResponse()
            score = float(relevance_score)
            if not math.isfinite(score):
                raise InvalidRerankResponse()
            seen_indices.add(index)
            scores.append(RerankScore(index=index, relevance_score=score))

        return sorted(scores, key=lambda item: (-item.relevance_score, item.index))
