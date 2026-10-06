"""A run-local writer coalesces pending snapshots without advancing durability early."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.parallel_checkpoint_writer import ParallelCheckpointWriter
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


def test_pending_coalesces_inflight_detaches_and_flush_waits_for_actual_commit() -> (
    None
):
    entered_first, release_first = threading.Event(), threading.Event()
    entered_latest, release_latest = threading.Event(), threading.Event()
    committed: list[dict[str, JsonValue]] = []
    threads: list[threading.Thread] = []

    def persist(snapshot: dict[str, JsonValue]) -> None:
        threads.append(threading.current_thread())
        if snapshot["revision"] == 1:
            entered_first.set()
            assert release_first.wait(3)
        else:
            assert snapshot["revision"] == 3
            entered_latest.set()
            assert release_latest.wait(3)
        committed.append(copy.deepcopy(snapshot))

    writer = ParallelCheckpointWriter(persist)
    first: dict[str, JsonValue] = {
        "revision": 1,
        "source": {"text": "Exact original.", "citations": [1, 2]},
    }
    original = copy.deepcopy(first)
    one = writer.submit(first)
    assert entered_first.wait(1)
    source = first["source"]
    assert isinstance(source, dict)
    source["text"] = "Producer changed its alias."
    citations = source["citations"]
    assert isinstance(citations, list)
    citations.append(99)
    two = writer.submit({"revision": 2, "source": "Intermediate exact state."})
    three = writer.submit({"revision": 3, "source": "Latest exact state."})
    assert (one, two, three) == (1, 2, 3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(writer.flush, two)
        try:
            assert not waiting.done()
            assert committed == []
            release_first.set()
            assert entered_latest.wait(1)
            assert committed == [original]
            assert not waiting.done()
            release_latest.set()
            waiting.result(timeout=2)
            writer.flush(three)
        finally:
            release_first.set()
            release_latest.set()
            writer.close()
    assert committed == [original, {"revision": 3, "source": "Latest exact state."}]
    assert len({thread.ident for thread in threads}) == 1
    assert all(not thread.daemon for thread in threads)
    assert all(thread is not threading.current_thread() for thread in threads)
    assert all(not thread.is_alive() for thread in threads)


def test_writer_uses_constructor_tenant_and_trace_context_for_all_producers() -> None:
    trace: ContextVar[str] = ContextVar("checkpoint_writer_test_trace", default="unset")
    seen: list[tuple[str | None, str, int | None]] = []
    tenant_token = CURRENT_TENANT_ID_CONTEXTVAR.set("checkpoint-owner")
    trace_token = trace.set("original-run-trace")

    def persist(_snapshot: dict[str, JsonValue]) -> None:
        seen.append(
            (
                CURRENT_TENANT_ID_CONTEXTVAR.get(),
                trace.get(),
                threading.current_thread().ident,
            )
        )

    writer = ParallelCheckpointWriter(persist)
    CURRENT_TENANT_ID_CONTEXTVAR.reset(tenant_token)
    trace.reset(trace_token)
    first = writer.submit({"owner": "first"})
    writer.flush(first)

    def submit_elsewhere() -> int:
        tenant = CURRENT_TENANT_ID_CONTEXTVAR.set("different-producer")
        current_trace = trace.set("different-producer-trace")
        try:
            return writer.submit({"owner": "second"})
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(tenant)
            trace.reset(current_trace)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            second = pool.submit(submit_elsewhere).result(timeout=2)
        writer.flush(second)
    finally:
        writer.close()
    assert [(tenant, identifier) for tenant, identifier, _thread in seen] == [
        ("checkpoint-owner", "original-run-trace"),
        ("checkpoint-owner", "original-run-trace"),
    ]
    assert len({thread for _tenant, _identifier, thread in seen}) == 1


def test_first_storage_failure_poison_reaches_waiters_and_all_entrypoints() -> None:
    entered, release = threading.Event(), threading.Event()
    failure = RuntimeError("Storage commit failed.")
    calls: list[dict[str, JsonValue]] = []
    threads: list[threading.Thread] = []

    def persist(snapshot: dict[str, JsonValue]) -> None:
        calls.append(snapshot)
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(3)
        raise failure

    writer = ParallelCheckpointWriter(persist)
    revision = writer.submit({"stage": "inflight"})
    assert entered.wait(1)
    later = writer.submit({"stage": "pending"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(writer.flush, later)
        release.set()
        with pytest.raises(RuntimeError, match="Storage commit failed") as waiting:
            waiter.result(timeout=2)
        assert waiting.value is failure
    for operation in (
        lambda: writer.submit({"stage": "after-failure"}),
        lambda: writer.flush(revision),
        writer.flush,
        writer.raise_if_failed,
        writer.close,
        writer.close,
    ):
        with pytest.raises(RuntimeError, match="Storage commit failed") as raised:
            operation()
        assert raised.value is failure
    assert calls == [{"stage": "inflight"}]
    assert len(threads) == 1 and not threads[0].is_alive()


def test_close_race_drains_every_admitted_latest_snapshot_and_joins() -> None:
    entered, release = threading.Event(), threading.Event()
    race = threading.Barrier(3)
    committed: list[dict[str, JsonValue]] = []
    worker: list[threading.Thread] = []

    def persist(snapshot: dict[str, JsonValue]) -> None:
        worker.append(threading.current_thread())
        if snapshot["stage"] == "inflight":
            entered.set()
            assert release.wait(3)
        committed.append(snapshot)

    writer = ParallelCheckpointWriter(persist)
    writer.submit({"stage": "inflight"})
    assert entered.wait(1)
    writer.submit({"stage": "pending"})

    def close() -> None:
        race.wait(2)
        writer.close()

    def late_submit() -> int | RuntimeError:
        race.wait(2)
        try:
            return writer.submit({"stage": "racing"})
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        closing, submitting = pool.submit(close), pool.submit(late_submit)
        try:
            race.wait(2)
            late = submitting.result(timeout=2)
            assert not closing.done()
            release.set()
            closing.result(timeout=2)
        finally:
            release.set()
            writer.close()
    assert committed == [
        {"stage": "inflight"},
        {"stage": "racing" if type(late) is int else "pending"},
    ]
    assert worker and all(not thread.is_alive() for thread in worker)
    writer.close()
    with pytest.raises(RuntimeError, match="clos"):
        writer.submit({"stage": "closed"})


def test_empty_close_and_flush_do_not_fabricate_storage_work() -> None:
    calls: list[dict[str, JsonValue]] = []
    writer = ParallelCheckpointWriter(calls.append)
    writer.flush()
    writer.flush(0)
    writer.raise_if_failed()
    writer.close()
    writer.close()
    assert calls == []


@pytest.mark.parametrize("revision", [-1, True, 1.5, 1])
def test_flush_rejects_unissued_or_invalid_revision(revision: object) -> None:
    writer = ParallelCheckpointWriter(lambda _snapshot: None)
    try:
        with pytest.raises(ValueError):
            writer.flush(cast(int, revision))
    finally:
        writer.close()
