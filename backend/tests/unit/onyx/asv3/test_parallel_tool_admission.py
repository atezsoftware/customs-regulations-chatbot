import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import monotonic
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import parallel_execution
from onyx.asv3 import registry as registry_module
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.parallel_execution import ParallelExecutionSlots
from onyx.asv3.registry import CapabilityRegistry

Handler = Callable[[dict[str, JsonValue], RunContext], ToolOutcome]


def context(
    mode: str,
    slots: ParallelExecutionSlots,
    *,
    budget: SharedBudget | None = None,
    owner: str = "hosted-question",
) -> RunContext:
    services: dict[str, object] = {
        "research_profile": "experimental" if mode != "deep" else "deep",
        "experimental_parallel": mode in {"parallel", "deep"},
        "parallel_execution_slots": slots,
        "selected_model": object(),
    }
    if mode == "hosted":
        services.update(
            lean_native_mode=True,
            serial_session_diagnostics=True,
            task_id=owner,
        )
    return RunContext(
        run_id="same-authorized-run",
        scope={"tenant_id": "captured", "as_of_date": "2026-10-06"},
        services=services,
        budget=budget,
    )


def spec(name: str, handler: Handler, *, orchestrates: bool = False) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object"},
        handler=handler,
        orchestrates=orchestrates,
    )


def original() -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.FOUND,
        summary="unchanged exact result",
        data={"has_more": False, "next_position": 91},
        evidence=[
            EvidenceItem(
                source_id="authorized-source",
                chunk_id="canonical-91",
                text="MADDE 53 — özgün koşulların tamamı.\n",
                metadata={"read_as_of_date": "2026-10-06", "article_no": "53"},
            )
        ],
    )


@pytest.mark.parametrize("mode", ["parallel", "hosted", "plain", "deep"])
def test_canonical_read_passes_four_held_searches_only_in_parallel_paths(
    mode: str,
) -> None:
    lock, four_entered, release, read_entered = (
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    started = 0
    captured = context(mode, ParallelExecutionSlots())
    selected_model = captured.services["selected_model"]
    expected = original()

    def search(_args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        nonlocal started
        assert child is captured and child.services["selected_model"] is selected_model
        with lock:
            started += 1
            if started == 4:
                four_entered.set()
        assert release.wait(5)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="same search result")

    def read(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        assert args == {"source_id": "authorized-source", "chunk_id": "canonical-91"}
        assert child is captured and child.services["selected_model"] is selected_model
        assert child.scope == {"tenant_id": "captured", "as_of_date": "2026-10-06"}
        read_entered.set()
        return expected

    registry = CapabilityRegistry([spec("search", search), spec("read_chunk", read)])
    with ThreadPoolExecutor(max_workers=5) as pool:
        searches = [
            pool.submit(registry.dispatch, CapabilityCall(name="search"), captured)
            for _ in range(4)
        ]
        read_future = None
        try:
            assert four_entered.wait(5)
            read_future = pool.submit(
                registry.dispatch,
                CapabilityCall(
                    name="read_chunk",
                    arguments={
                        "source_id": "authorized-source",
                        "chunk_id": "canonical-91",
                    },
                ),
                captured,
            )
            if mode in {"parallel", "hosted"}:
                assert read_entered.wait(5)
                assert read_future.result(timeout=5) == expected
                assert not any(future.done() for future in searches)
            else:
                assert not read_entered.wait(0.1)
        finally:
            release.set()
        assert read_future is not None and read_future.result(timeout=5) == expected
        assert all(
            future.result(timeout=5).status == OutcomeStatus.FOUND
            for future in searches
        )
    assert captured.budget.snapshot()["tools"] == 5
    assert captured.services["selected_model"] is selected_model


def test_canonical_lane_is_bounded_to_two_across_owned_hosted_contexts() -> None:
    slots, budget = ParallelExecutionSlots(), SharedBudget()
    captured = [
        context("hosted", slots, budget=budget, owner=f"q{index}") for index in range(6)
    ]
    lock, two_entered, release = threading.Lock(), threading.Event(), threading.Event()
    running = peak = calls = 0

    def read(_args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        nonlocal running, peak, calls
        with lock:
            running += 1
            calls += 1
            peak = max(peak, running)
            if running == 2:
                two_entered.set()
        try:
            assert release.wait(5)
            return original()
        finally:
            with lock:
                running -= 1

    registry = CapabilityRegistry([spec("read_provision", read)])
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(registry.dispatch, CapabilityCall(name="read_provision"), child)
            for child in captured
        ]
        try:
            assert two_entered.wait(5)
            with lock:
                assert calls == peak == 2
        finally:
            release.set()
        assert all(future.result(timeout=5) == original() for future in futures)
    assert calls == 6 and peak == 2 and budget.snapshot()["tools"] == 6


def test_cancellation_while_queued_never_runs_handler_or_leaks_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = ParallelExecutionSlots()
    captured = context("hosted", slots)
    queued = threading.Event()
    original_acquire = slots.canonical_io.acquire

    def acquire(blocking: bool = True, timeout: float | None = None) -> bool:
        queued.set()
        return original_acquire(blocking=blocking, timeout=timeout)

    calls: list[str] = []
    registry = CapabilityRegistry(
        [spec("read_chunk", lambda _args, _child: calls.append("read") or original())]
    )
    assert slots.canonical_io.acquire(blocking=False)
    assert slots.canonical_io.acquire(blocking=False)
    monkeypatch.setattr(slots.canonical_io, "acquire", acquire)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                registry.dispatch, CapabilityCall(name="read_chunk"), captured
            )
            assert queued.wait(5)
            captured.cancel()
            assert future.result(timeout=5).status == OutcomeStatus.CANCELLED
        assert calls == []
        assert not slots.canonical_io.acquire(blocking=False)
    finally:
        slots.canonical_io.release()
        slots.canonical_io.release()
    assert slots.canonical_io.acquire(blocking=False)
    assert slots.canonical_io.acquire(blocking=False)
    assert not slots.canonical_io.acquire(blocking=False)
    slots.canonical_io.release()
    slots.canonical_io.release()


def test_cancellation_immediately_after_acquisition_releases_its_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = ParallelExecutionSlots()
    captured = context("parallel", slots)
    original_acquire = slots.canonical_io.acquire

    def acquire(blocking: bool = True, timeout: float | None = None) -> bool:
        acquired = original_acquire(blocking=blocking, timeout=timeout)
        if acquired:
            captured.cancel()
        return acquired

    monkeypatch.setattr(slots.canonical_io, "acquire", acquire)
    calls: list[str] = []
    registry = CapabilityRegistry(
        [spec("read_chunk", lambda _args, _child: calls.append("read") or original())]
    )
    assert (
        registry.dispatch(CapabilityCall(name="read_chunk"), captured).status
        == OutcomeStatus.CANCELLED
    )
    assert calls == []
    assert original_acquire(blocking=False)
    assert original_acquire(blocking=False)
    assert not original_acquire(blocking=False)
    slots.canonical_io.release()
    slots.canonical_io.release()


def test_orchestration_takes_no_outer_permit_even_when_both_lanes_are_full() -> None:
    slots = ParallelExecutionSlots()
    captured = context("parallel", slots)
    registry = CapabilityRegistry(
        [spec("orchestrate", lambda _args, _child: original(), orchestrates=True)]
    )
    for _ in range(4):
        assert captured.budget.tool_slots.acquire(blocking=False)
    for _ in range(2):
        assert slots.canonical_io.acquire(blocking=False)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert (
                pool.submit(
                    registry.dispatch, CapabilityCall(name="orchestrate"), captured
                ).result(timeout=5)
                == original()
            )
    finally:
        for _ in range(4):
            captured.budget.tool_slots.release()
        for _ in range(2):
            slots.canonical_io.release()
    assert captured.budget.snapshot()["tools"] == 1


def test_queue_and_handler_spans_have_separate_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = ParallelExecutionSlots()
    captured = context("hosted", slots)
    lock, admission_entered, handler_entered, release_handler = (
        threading.Lock(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    spans: list[dict[str, object]] = []

    @contextmanager
    def span(operation: str, payload: dict[str, object]) -> Generator[None, None, None]:
        row: dict[str, object] = {
            "operation": operation,
            "payload": payload,
            "start": monotonic(),
        }
        with lock:
            spans.append(row)
        if operation == "asv3.tool_admission":
            admission_entered.set()
        try:
            yield
        finally:
            row["end"] = monotonic()

    monkeypatch.setattr(parallel_execution, "graph_step", span)
    monkeypatch.setattr(registry_module, "graph_step", span)

    def read(_args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        handler_entered.set()
        assert release_handler.wait(5)
        return original()

    registry = CapabilityRegistry([spec("read_chunk", read)])
    assert slots.canonical_io.acquire(blocking=False)
    assert slots.canonical_io.acquire(blocking=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            registry.dispatch, CapabilityCall(name="read_chunk"), captured
        )
        try:
            assert admission_entered.wait(5)
            assert not handler_entered.wait(0.05)
            assert len(spans) == 1 and "end" not in spans[0]
            slots.canonical_io.release()
            assert handler_entered.wait(5)
            assert len(spans) == 2
            assert spans[0]["operation"] == "asv3.tool_admission"
            assert spans[0]["payload"] == {"tool": "read_chunk", "lane": "canonical_io"}
            assert cast(float, spans[0]["end"]) <= cast(float, spans[1]["start"])
            assert (
                spans[1]["operation"] == "asv3.tool_handler" and "end" not in spans[1]
            )
        finally:
            release_handler.set()
            slots.canonical_io.release()
        assert future.result(timeout=5) == original()
    assert cast(float, spans[1]["start"]) <= cast(float, spans[1]["end"])
