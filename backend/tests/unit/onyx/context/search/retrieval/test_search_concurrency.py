from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event, Lock

import pytest


def test_shared_search_budget_bounds_nested_lanes_and_releases_on_failure() -> None:
    from onyx.context.search.retrieval.concurrency import (
        limit_search_concurrency,
        search_slot,
    )

    lock = Lock()
    full = Event()
    release = Event()
    active = peak = 0

    def lane(index: int) -> int:
        nonlocal active, peak
        with search_slot():
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 12:
                    full.set()
            try:
                assert release.wait(5)
                if index == 0:
                    raise ValueError("fixture failure")
                return index
            finally:
                with lock:
                    active -= 1

    with limit_search_concurrency(12), ThreadPoolExecutor(max_workers=50) as pool:
        futures = [pool.submit(copy_context().run, lane, index) for index in range(50)]
        try:
            assert full.wait(3)
            assert peak == 12
        finally:
            release.set()
        with pytest.raises(ValueError, match="fixture failure"):
            futures[0].result(timeout=5)
        assert [f.result(timeout=5) for f in futures[1:]] == list(range(1, 50))
    assert peak == 12 and active == 0


def test_waiting_search_aborts_on_resource_pressure() -> None:
    from onyx.context.search.retrieval.concurrency import (
        limit_search_concurrency,
        search_slot,
    )
    from onyx.regulatory.amendments.memory_budget import ResourcePressure

    def unavailable() -> None:
        raise ResourcePressure("memory_pressure")

    with limit_search_concurrency(1, check_resources=unavailable):
        with pytest.raises(ResourcePressure, match="memory_pressure"):
            with search_slot():
                pytest.fail("must not start work")


def test_ordinary_search_has_no_amendment_gate() -> None:
    from onyx.context.search.retrieval.concurrency import search_slot

    with search_slot():
        with search_slot():
            pass
