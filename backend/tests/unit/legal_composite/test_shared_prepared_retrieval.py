from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from onyx.asv3.models import RunStopped
from onyx.context.search.models import InferenceChunk
from onyx.legal_composite.shared_search import SharedPreparedRetrieval
from tests.unit.legal_composite.test_prepared_source_search import chunk


def test_identical_parallel_scopes_share_raw_hits_without_mutation() -> None:
    cache = SharedPreparedRetrieval()
    barrier, lock = Barrier(12), Lock()
    calls = 0

    def execute() -> list[InferenceChunk]:
        nonlocal calls
        with lock:
            calls += 1
        return [chunk("authorized-source")]

    def search(_index: int) -> tuple[list[InferenceChunk], bool]:
        barrier.wait(timeout=3)
        return cache.run("same-query-acl-encoder-date-ids", execute, lambda: None)

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(search, range(12)))
    assert calls == 1
    assert sum(shared for _, shared in results) == 11
    results[0][0][0].metadata["changed"] = True
    assert "changed" not in results[1][0][0].metadata


def test_distinct_scope_failure_and_capacity_do_not_drop_searches() -> None:
    cache = SharedPreparedRetrieval(max_entries=1)
    seen = []

    def execute() -> list[InferenceChunk]:
        seen.append(True)
        return [chunk("authorized-source")]

    for key in ("scope-one", "scope-two", "scope-two"):
        assert cache.run(key, execute, lambda: None)[0]
    assert len(seen) == 3
    cache = SharedPreparedRetrieval()

    def fail() -> list[InferenceChunk]:
        raise TimeoutError("provider failed")

    with pytest.raises(TimeoutError, match="provider failed"):
        cache.run("same-scope", fail, lambda: None)
    assert cache.run("same-scope", execute, lambda: None)[0]


def test_cancelled_subscriber_cannot_receive_cached_hits() -> None:
    cache = SharedPreparedRetrieval()
    cache.run("scope", lambda: [chunk("source")], lambda: None)

    def cancelled() -> None:
        raise RunStopped("cancelled")

    with pytest.raises(RunStopped, match="cancelled"):
        cache.run("scope", lambda: [], cancelled)
