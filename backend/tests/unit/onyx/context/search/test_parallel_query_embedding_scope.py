import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from onyx.asv3.models import RunStopped
from onyx.context.search.models import ChunkIndexRequest, IndexFilters
from onyx.context.search.retrieval import search_runner
from onyx.context.search.retrieval.query_embedding_scope import (
    ParallelQueryEmbeddingScope,
    current_parallel_query_embedding_scope,
    experimental_parallel_query_embeddings,
)
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from shared_configs.model_server_models import Embedding


@pytest.fixture(autouse=True)
def reset_embedding_circuit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(search_runner, "_regulatory_embedding_circuit_open_until", 0.0)
    monkeypatch.setattr(search_runner, "_regulatory_embedding_probe_succeeded", False)
    monkeypatch.setattr(search_runner, "_regulatory_embedding_failure_revision", 0)


def request(query: str, *, regulatory: bool = True) -> ChunkIndexRequest:
    return ChunkIndexRequest(
        query=query,
        filters=IndexFilters(
            access_control_list=None, regulatory_chunks_only=regulatory
        ),
    )


def no_cancellation() -> None:
    pass


def scoped_embedding(
    query: str,
    scope: ParallelQueryEmbeddingScope,
    check_active: Callable[[], None] = no_cancellation,
) -> Embedding | None:
    with experimental_parallel_query_embeddings(scope, check_active=check_active):
        return search_runner._get_regulatory_query_embedding(
            request(query), db_session=None, embedding_model=None
        )


def test_independent_cold_queries_rendezvous_outside_global_lock_and_keep_vectors(
    monkeypatch: MonkeyPatch,
) -> None:
    queries = [f"independent query {index}" for index in range(4)]
    expected = {query: [float(index)] for index, query in enumerate(queries)}
    rendezvous = threading.Barrier(4)
    model = MagicMock()

    def embed(query: str, **kwargs: object) -> Embedding:
        assert kwargs == {"db_session": None, "embedding_model": model}
        rendezvous.wait(timeout=5)
        return expected[query]

    get_embedding = MagicMock(side_effect=embed)
    monkeypatch.setattr(search_runner, "get_query_embedding", get_embedding)
    scope = ParallelQueryEmbeddingScope()
    with experimental_parallel_query_embeddings(scope, check_active=no_cancellation):
        # The real retrieval thread pool copies the explicit scope into each lane.
        def lane(query: str) -> Embedding | None:
            return search_runner._get_regulatory_query_embedding(
                request(query), db_session=None, embedding_model=model
            )

        results = run_functions_tuples_in_parallel([(lane, (q,)) for q in queries])

    assert results == [expected[query] for query in queries]
    assert get_embedding.call_count == 4
    assert search_runner._regulatory_embedding_probe_succeeded
    assert current_parallel_query_embedding_scope() is None


def test_successful_provider_keeps_ordinary_hot_queries_concurrent(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_runner, "_regulatory_embedding_probe_succeeded", True)
    rendezvous = threading.Barrier(5)

    def embed(query: str, **_kwargs: object) -> Embedding:
        rendezvous.wait(timeout=5)
        return [float(query)]

    monkeypatch.setattr(search_runner, "get_query_embedding", embed)
    scope = ParallelQueryEmbeddingScope()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scoped_embedding, str(i), scope) for i in range(5)]
        assert [future.result(timeout=5) for future in futures] == [
            [float(i)] for i in range(5)
        ]


def test_fifth_cold_probe_waits_without_adding_or_reusing_query_calls(
    monkeypatch: MonkeyPatch,
) -> None:
    scope = ParallelQueryEmbeddingScope()
    four_entered = threading.Event()
    queued = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    calls: list[str] = []
    active = maximum = 0

    def embed(query: str, **_kwargs: object) -> Embedding:
        nonlocal active, maximum
        with state_lock:
            calls.append(query)
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_entered.set()
        assert release.wait(timeout=5)
        with state_lock:
            active -= 1
        return [float(query)]

    check_count = 0

    def queued_check() -> None:
        nonlocal check_count
        check_count += 1
        if check_count >= 4:
            queued.set()

    monkeypatch.setattr(search_runner, "get_query_embedding", embed)
    with ThreadPoolExecutor(max_workers=5) as executor:
        first = [executor.submit(scoped_embedding, str(i), scope) for i in range(4)]
        try:
            assert four_entered.wait(timeout=5)
            fifth = executor.submit(scoped_embedding, "4", scope, queued_check)
            assert queued.wait(timeout=5)
            assert sorted(calls) == ["0", "1", "2", "3"]
        finally:
            release.set()
        assert [future.result(timeout=5) for future in first] == [
            [float(i)] for i in range(4)
        ]
        assert fifth.result(timeout=5) == [4.0]
    assert sorted(calls) == ["0", "1", "2", "3", "4"]
    assert maximum == 4


def test_cancelled_queued_probe_never_calls_provider_or_leaks_admission(
    monkeypatch: MonkeyPatch,
) -> None:
    scope = ParallelQueryEmbeddingScope()
    four_entered = threading.Barrier(5)
    release = threading.Event()
    waiting = threading.Event()
    cancelled = threading.Event()
    calls: list[str] = []
    call_lock = threading.Lock()

    def embed(query: str, **_kwargs: object) -> Embedding:
        with call_lock:
            calls.append(query)
        four_entered.wait(timeout=5)
        assert release.wait(timeout=5)
        return [1.0]

    checks = 0

    def check_active() -> None:
        nonlocal checks
        checks += 1
        if cancelled.is_set():
            raise RunStopped("cancelled while queued")
        if checks >= 4:
            waiting.set()

    monkeypatch.setattr(search_runner, "get_query_embedding", embed)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scoped_embedding, str(i), scope) for i in range(4)]
        try:
            four_entered.wait(timeout=5)
            stopped = executor.submit(
                scoped_embedding, "cancelled", scope, check_active
            )
            assert waiting.wait(timeout=5)
            cancelled.set()
            with pytest.raises(RunStopped, match="cancelled while queued"):
                stopped.result(timeout=5)
        finally:
            release.set()
        assert [future.result(timeout=5) for future in futures] == [[1.0]] * 4

    assert "cancelled" not in calls
    monkeypatch.setattr(search_runner, "_regulatory_embedding_probe_succeeded", False)
    monkeypatch.setattr(search_runner, "get_query_embedding", lambda *_a, **_kw: [2.0])
    assert scoped_embedding("next cold query", scope) == [2.0]


def test_late_success_keeps_newer_failure_circuit_open(
    monkeypatch: MonkeyPatch,
) -> None:
    both_entered = threading.Barrier(2)
    release_success = threading.Event()
    monkeypatch.setattr(search_runner.time, "monotonic", lambda: 100.0)

    def embed(query: str, **_kwargs: object) -> Embedding:
        both_entered.wait(timeout=5)
        if query == "failure":
            raise RuntimeError("provider failed")
        assert release_success.wait(timeout=5)
        return [0.25]

    provider = MagicMock(side_effect=embed)
    monkeypatch.setattr(search_runner, "get_query_embedding", provider)
    scope = ParallelQueryEmbeddingScope()
    with ThreadPoolExecutor(max_workers=2) as executor:
        slow = executor.submit(scoped_embedding, "success", scope)
        fast = executor.submit(scoped_embedding, "failure", scope)
        try:
            with pytest.raises(RuntimeError, match="provider failed"):
                fast.result(timeout=5)
        finally:
            release_success.set()
        assert slow.result(timeout=5) == [0.25]

    assert not search_runner._regulatory_embedding_probe_succeeded
    assert search_runner._regulatory_embedding_circuit_open_until == 160.0
    assert scoped_embedding("later", scope) is None
    assert provider.call_count == 2


def test_scoped_provider_runtime_error_keeps_existing_lexical_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    provider = MagicMock(side_effect=RuntimeError("provider unavailable"))
    monkeypatch.setattr(search_runner, "get_query_embedding", provider)
    index = MagicMock()
    expected = [MagicMock()]
    index.keyword_retrieval.return_value = expected
    with experimental_parallel_query_embeddings(
        ParallelQueryEmbeddingScope(), check_active=no_cancellation
    ):
        assert search_runner._embed_and_hybrid_search(request("one"), index) == expected
        assert search_runner._embed_and_hybrid_search(request("two"), index) == expected
    assert provider.call_count == 1
    assert index.keyword_retrieval.call_count == 2


def test_scope_cancellation_does_not_become_lexical_retrieval(
    monkeypatch: MonkeyPatch,
) -> None:
    cancelled = threading.Event()

    def check_active() -> None:
        if cancelled.is_set():
            raise RunStopped("cancelled")

    def embed(*_args: object, **_kwargs: object) -> Embedding:
        cancelled.set()
        return [0.25]

    monkeypatch.setattr(search_runner, "get_query_embedding", embed)
    index = MagicMock()
    with experimental_parallel_query_embeddings(
        ParallelQueryEmbeddingScope(), check_active=check_active
    ):
        with pytest.raises(RunStopped, match="cancelled"):
            search_runner._embed_and_hybrid_search(request("one"), index)
    index.keyword_retrieval.assert_not_called()
    index.hybrid_retrieval.assert_not_called()


def test_legacy_cold_probe_stays_serialized_without_explicit_scope(
    monkeypatch: MonkeyPatch,
) -> None:
    first_entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    calls: list[str] = []

    def embed(query: str, **_kwargs: object) -> Embedding:
        calls.append(query)
        if query == "first":
            first_entered.set()
            assert release.wait(timeout=5)
        return [1.0]

    def legacy(query: str) -> Embedding | None:
        if query == "second":
            second_started.set()
        return search_runner._get_regulatory_query_embedding(
            request(query), db_session=None, embedding_model=None
        )

    monkeypatch.setattr(search_runner, "get_query_embedding", embed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(legacy, "first")
        try:
            assert first_entered.wait(timeout=5)
            second = executor.submit(legacy, "second")
            assert second_started.wait(timeout=5)
            assert calls == ["first"]
            assert search_runner._regulatory_embedding_circuit_lock.locked()
        finally:
            release.set()
        assert first.result(timeout=5) == [1.0]
        assert second.result(timeout=5) == [1.0]
    assert calls == ["first", "second"]


def test_non_regulatory_search_preserves_provider_failure_even_in_scope(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_runner,
        "get_query_embedding",
        MagicMock(side_effect=RuntimeError("provider failed")),
    )
    index = MagicMock()
    with experimental_parallel_query_embeddings(
        ParallelQueryEmbeddingScope(), check_active=no_cancellation
    ):
        with pytest.raises(RuntimeError, match="provider failed"):
            search_runner._embed_and_hybrid_search(
                request("one", regulatory=False), index
            )
    index.keyword_retrieval.assert_not_called()
    assert search_runner._regulatory_embedding_failure_revision == 0


def test_unexpected_provider_exception_does_not_open_failure_circuit(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_runner,
        "get_query_embedding",
        MagicMock(side_effect=ValueError("invalid configuration")),
    )
    with pytest.raises(ValueError, match="invalid configuration"):
        scoped_embedding("one", ParallelQueryEmbeddingScope())
    assert search_runner._regulatory_embedding_circuit_open_until == 0.0


def test_nested_explicit_scopes_restore_binding_and_clear_on_exception() -> None:
    outer = ParallelQueryEmbeddingScope()
    inner = ParallelQueryEmbeddingScope()
    with experimental_parallel_query_embeddings(outer, check_active=no_cancellation):
        outer_binding = current_parallel_query_embedding_scope()
        assert outer_binding is not None and outer_binding.scope is outer
        with pytest.raises(ValueError, match="leave inner"):
            with experimental_parallel_query_embeddings(
                inner, check_active=no_cancellation
            ):
                inner_binding = current_parallel_query_embedding_scope()
                assert inner_binding is not None and inner_binding.scope is inner
                raise ValueError("leave inner")
        assert current_parallel_query_embedding_scope() is outer_binding
    assert current_parallel_query_embedding_scope() is None
