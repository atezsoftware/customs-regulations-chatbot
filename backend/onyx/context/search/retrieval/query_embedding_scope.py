import contextvars
import hashlib
import json
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import dataclass

from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE
from shared_configs.model_server_models import Embedding


def embedding_model_key(model: EmbeddingModel) -> str:
    """Digest the complete query encoder configuration without exporting secrets."""
    identity = {
        "provider": model.provider_type.value if model.provider_type else None,
        "model": model.model_name,
        "model_server_endpoint": model.embed_server_endpoint,
        "api_url": model.api_url,
        "api_version": model.api_version,
        "deployment": model.deployment_name,
        "dimensions": model.reduced_dimension,
        "normalize": model.normalize,
        "query_prefix": model.query_prefix,
        "passage_prefix": model.passage_prefix,
        "retrim_content": model.retrim_content,
        "tokenizer": id(model.tokenizer) if model.retrim_content else None,
        "max_context_length": DOC_EMBEDDING_CONTEXT_SIZE,
        "search_settings_id": model.search_settings_id,
        "credential": hashlib.sha256(
            model.api_key.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        if model.api_key is not None
        else None,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8", errors="surrogatepass"
        )
    ).hexdigest()


class ParallelQueryEmbeddingScope:
    """Bound cold probes, optionally sharing exact embeddings within one request."""

    def __init__(
        self, *, cache_queries: bool = False, max_cached_queries: int = 256
    ) -> None:
        if (
            isinstance(max_cached_queries, bool)
            or not isinstance(max_cached_queries, int)
            or max_cached_queries < 1
        ):
            raise ValueError("Embedding cache capacity must be a positive integer")
        self.cache_queries = cache_queries
        self._max_cached_queries = max_cached_queries
        self._cache_lock = threading.Lock()
        self._embeddings: dict[tuple[str, str], Future[Embedding | None]] = {}
        self._cold_probe_slots = threading.BoundedSemaphore(4)

    def embed(
        self,
        query: str,
        embedding_model: EmbeddingModel | None,
        execute: Callable[[], Embedding | None],
        check_active: Callable[[], None],
    ) -> Embedding | None:
        check_active()
        # Unknown/custom encoders keep their existing execution contract.
        if not self.cache_queries or type(embedding_model) is not EmbeddingModel:
            return execute()
        identity = embedding_model_key(embedding_model)
        key = (
            identity,
            hashlib.sha256(query.encode("utf-8", errors="surrogatepass")).hexdigest(),
        )
        owner = False
        with self._cache_lock:
            future = self._embeddings.get(key)
            if future is None and len(self._embeddings) < self._max_cached_queries:
                future = Future()
                self._embeddings[key] = future
                owner = True
        if future is None:
            return execute()
        if owner:
            try:
                check_active()
                embedding = execute()
                check_active()
                if identity != embedding_model_key(embedding_model):
                    raise ValueError("Embedding configuration changed during execution")
                future.set_result(list(embedding) if embedding is not None else None)
                if embedding is None:
                    with self._cache_lock:
                        self._embeddings.pop(key, None)
            except BaseException as error:
                future.set_exception(error)
                with self._cache_lock:
                    self._embeddings.pop(key, None)
                raise
        while True:
            check_active()
            try:
                result = future.result(timeout=0.05)
                check_active()
                return list(result) if result is not None else None
            except FutureTimeout:
                if future.done():
                    raise

    @contextmanager
    def cold_probe(self, check_active: Callable[[], None]) -> Iterator[None]:
        check_active()
        while not self._cold_probe_slots.acquire(timeout=0.05):
            check_active()
        try:
            check_active()
            yield
        finally:
            self._cold_probe_slots.release()


@dataclass(frozen=True)
class QueryEmbeddingScopeBinding:
    scope: ParallelQueryEmbeddingScope
    check_active: Callable[[], None]


_query_embedding_scope: contextvars.ContextVar[QueryEmbeddingScopeBinding | None] = (
    contextvars.ContextVar("experimental_parallel_query_embeddings", default=None)
)


def current_parallel_query_embedding_scope() -> QueryEmbeddingScopeBinding | None:
    return _query_embedding_scope.get()


@contextmanager
def experimental_parallel_query_embeddings(
    scope: ParallelQueryEmbeddingScope,
    *,
    check_active: Callable[[], None],
) -> Iterator[None]:
    check_active()
    token = _query_embedding_scope.set(QueryEmbeddingScopeBinding(scope, check_active))
    try:
        yield
    finally:
        _query_embedding_scope.reset(token)
