"""Related catalogue singleflight keeps unrelated pages independent and bounded."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from onyx.asv3.models import RunContext, RunStopped
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource
from tests.unit.onyx.asv3.test_legal_source_navigation import (
    AYM_SOURCE_NAME,
    broker,
    original,
    source,
)


def context(mode: str = "parallel") -> RunContext:
    services: dict[str, Any] = {"research_profile": "experimental"}
    if mode == "parallel":
        services["experimental_parallel"] = True
    elif mode == "hosted":
        services.update(
            experimental_parallel=False,
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-task",
        )
    return RunContext(services=services)


@pytest.mark.parametrize("mode", ["parallel", "hosted", "plain"])
def test_cold_page_only_blocks_cached_unrelated_page_in_protected_plain_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    current, warm, cold = broker(), original(article="241"), original(article="27")
    envelope = context(mode)
    monkeypatch.setattr(
        current, "related_catalog_sources", lambda *_a, **_k: ([], False)
    )
    cached = current.related_sources_for_evidence(warm, envelope)
    entered, release = Event(), Event()

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        entered.set()
        assert release.wait(3)
        return [], False

    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(current.related_sources_for_evidence, cold, envelope)
        assert entered.wait(1)
        hit = pool.submit(current.related_sources_for_evidence, warm, envelope)
        try:
            if mode == "plain":
                assert not hit.done()
            else:
                assert hit.result(timeout=1) == cached
        finally:
            release.set()
        assert pending.result(timeout=1) is not None
        assert hit.result(timeout=1) == cached


def test_different_cold_pages_overlap_but_never_exceed_two_catalogue_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, envelope = broker(), context()
    entered_two, release = Event(), Event()
    lock = Lock()
    active, maximum, count = 0, 0, 0

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal active, maximum, count
        with lock:
            active += 1
            count += 1
            maximum = max(maximum, active)
            if active == 2:
                entered_two.set()
        try:
            assert release.wait(3)
            return [], False
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = [
            pool.submit(
                current.related_sources_for_evidence, original(article=str(n)), envelope
            )
            for n in range(1, 7)
        ]
        try:
            assert entered_two.wait(1)
            with lock:
                assert active == maximum == count == 2
        finally:
            release.set()
        assert all(future.result(timeout=2) is not None for future in pending)
    assert count == 6 and maximum == 2
    assert current._related_sources_flights == {}


def test_duplicate_cold_page_is_singleflight_and_returned_copies_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, item, envelope = broker(), original(), context()
    candidate = source(AYM_SOURCE_NAME)
    entered, release = Event(), Event()
    calls = 0

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(3)
        return [candidate], True

    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [
            pool.submit(current.related_sources_for_evidence, item, envelope)
            for _ in range(8)
        ]
        try:
            assert entered.wait(1)
        finally:
            release.set()
        records = [future.result(timeout=2) for future in pending]
    assert calls == 1
    assert records[0] is not None and all(row == records[0] for row in records)
    assert records[0]["has_more"] is True and records[0]["next_offset"] == 50
    records[0]["candidates"] = []
    assert records[1] is not None and records[1]["candidates"]
    assert current._related_sources_flights == {}


@pytest.mark.parametrize(
    "error",
    [
        None,
        PermissionError("denied"),
        CorpusScopeUnavailable("unavailable"),
        SQLAlchemyError("db"),
    ],
)
def test_parallel_page_content_pagination_and_failure_status_match_plain(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None
) -> None:
    item, candidate = original(), source(AYM_SOURCE_NAME)
    results = []
    calls: list[tuple[Any, dict[str, Any]]] = []

    def lookup(*args: Any, **kwargs: Any) -> tuple[list[Any], bool]:
        calls.append((args[0], kwargs))
        if error is not None:
            raise error
        return [candidate], True

    for mode in ("plain", "parallel"):
        current = broker()
        monkeypatch.setattr(current, "related_catalog_sources", lookup)
        assert item.search_doc is not None
        selected = CorpusSource(
            UUID(item.source_id), item.search_doc.semantic_identifier, "file"
        )
        results.append(
            current.related_sources_for_provision(
                selected, [item], ("241", None), context(mode), offset=50
            )
        )
    assert results[0] == results[1]
    assert results[0] is not None and results[0]["absence_proven"] is False
    assert calls[0] == calls[1]
    assert calls[0][1] == {"offset": 50, "limit": 50}


def test_waiting_subscriber_cancellation_does_not_cancel_owner_or_leak_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, item = broker(), original()
    owner, subscriber = context(), context()
    entered, release = Event(), Event()
    calls = 0

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(3)
        return [], False

    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(current.related_sources_for_evidence, item, owner)
        assert entered.wait(1)
        follower = pool.submit(current.related_sources_for_evidence, item, subscriber)
        subscriber.cancel()
        try:
            with pytest.raises(RunStopped, match="cancelled"):
                follower.result(timeout=1)
        finally:
            release.set()
        assert first.result(timeout=1) is not None
    assert calls == 1 and current._related_sources_flights == {}
    assert current.related_sources_for_evidence(item, context()) is not None
    assert calls == 1


def test_cancelled_late_lookup_is_not_cached_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, item, owner = broker(), original(), context()

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        owner.cancel()
        return [], False

    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with pytest.raises(RunStopped, match="cancelled"):
        current.related_sources_for_evidence(item, owner)
    assert not current._related_sources and not current._related_sources_flights
    assert current._related_sources_slots.acquire(blocking=False)
    assert current._related_sources_slots.acquire(blocking=False)
    assert not current._related_sources_slots.acquire(blocking=False)
    current._related_sources_slots.release()
    current._related_sources_slots.release()
    monkeypatch.setattr(
        current, "related_catalog_sources", lambda *_a, **_k: ([], False)
    )
    assert current.related_sources_for_evidence(item, context()) is not None


def test_cancelled_miss_waiting_for_catalogue_capacity_never_queries_or_leaks_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, item, owner = broker(), original(), context()
    assert current._related_sources_slots.acquire(blocking=False)
    assert current._related_sources_slots.acquire(blocking=False)
    waiting = Event()
    active_check = owner.check_active
    checks = 0
    calls = 0

    def check() -> None:
        nonlocal checks
        checks += 1
        if checks >= 3:
            waiting.set()
        active_check()

    def lookup(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal calls
        calls += 1
        return [], False

    monkeypatch.setattr(owner, "check_active", check)
    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(current.related_sources_for_evidence, item, owner)
        try:
            assert waiting.wait(1)
            owner.cancel()
            with pytest.raises(RunStopped, match="cancelled"):
                pending.result(timeout=1)
        finally:
            current._related_sources_slots.release()
            current._related_sources_slots.release()
    assert calls == 0 and not current._related_sources_flights
    assert current.related_sources_for_evidence(item, context()) is not None
    assert calls == 1
