"""Cross-encoder reranking via OpenRouter's hosted rerank models.

Sits between retrieval (Postgres/pgvector candidate fetch) and the LLM: a
larger, cheap candidate pool can be over-fetched and then narrowed to a
smaller, higher-precision set with a real cross-encoder instead of the
linear semantic/metadata-score heuristics in `search/ranker.py` and
`amendments/ranker.py`. Always fails open — any network error, timeout, or
malformed response returns `None` so callers fall back to those heuristics
rather than breaking retrieval.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_RERANK_MODEL = "cohere/rerank-4-pro"
DEFAULT_RERANK_BATCH_SIZE = 100


class ChunkReranker:
    """Thin sync client for OpenRouter's `POST /rerank` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        timeout: float = 8.0,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.batch_size = max(batch_size, 1)
        self._client = client or httpx.Client(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]] | None:
        """Return (index, relevance_score) pairs ordered best-match-first,
        or `None` on any failure. An empty candidate list short-circuits
        without a network call; even a single retrieved chunk is sent to the
        configured cross-encoder so every search candidate follows the same
        scoring path.

        Scores are returned (not just index order) so callers merging
        multiple independently-reranked result sets — e.g. one `semantic_search`
        call spanning several linked corpora — can re-sort the union by a
        consistent score instead of falling back to the old heuristic score,
        which would silently undo the rerank ordering."""
        if not documents:
            return []
        pairs: list[tuple[int, float]] = []
        requested_top_n = max(min(top_n, len(documents)), 1)
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            try:
                response = self._client.post(
                    "/rerank",
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": batch,
                        # The caller normally asks for the complete ranking so
                        # downstream diversification can choose globally. For
                        # smaller top-N requests, no batch can contribute more
                        # than that many items to the global result.
                        "top_n": min(requested_top_n, len(batch)),
                    },
                )
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError):
                return None
            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, list):
                return None
            for result in results:
                if not isinstance(result, dict):
                    return None
                index = result.get("index")
                score = result.get("relevance_score")
                if not isinstance(index, int) or not (0 <= index < len(batch)):
                    return None
                if not isinstance(score, (int, float)):
                    return None
                pairs.append((start + index, float(score)))
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:requested_top_n] or None


_reranker: ChunkReranker | None = None
_reranker_built = False


def get_reranker() -> ChunkReranker | None:
    """Lazily-built process-wide singleton.

    Returns `None` (reranking disabled) when `FS_EXPLORER_RERANK_ENABLED` is
    falsy or `OPENROUTER_API_KEY` isn't set, so every call site can treat a
    `None` reranker exactly like a failed rerank call — fall back silently.
    """
    global _reranker, _reranker_built
    if _reranker_built:
        return _reranker
    _reranker_built = True
    if os.getenv("FS_EXPLORER_RERANK_ENABLED", "true").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    _reranker = ChunkReranker(
        api_key=api_key,
        model=os.getenv("FS_EXPLORER_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        timeout=float(os.getenv("FS_EXPLORER_RERANK_TIMEOUT_SECONDS", "8")),
        batch_size=int(
            os.getenv(
                "FS_EXPLORER_RERANK_BATCH_SIZE", str(DEFAULT_RERANK_BATCH_SIZE)
            )
        ),
    )
    return _reranker


def reset_reranker_singleton() -> None:
    """Test-only hook to force `get_reranker()` to rebuild on next call."""
    global _reranker, _reranker_built
    _reranker = None
    _reranker_built = False
