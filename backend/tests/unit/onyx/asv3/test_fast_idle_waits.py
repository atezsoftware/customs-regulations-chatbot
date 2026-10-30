from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from typing import Generic, TypeVar, cast

import pytest
from pydantic import JsonValue

import onyx.asv3.harness as harness_module
import onyx.asv3.sandbox as sandbox_module
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry

Result = TypeVar("Result")


class ManualPool(Generic[Result]):
    def __init__(self) -> None:
        self.jobs: list[tuple[Future[Result], Callable[[], Result]]] = []
        self.closed = False

    def submit(self, function: Callable[..., Result], *args: object) -> Future[Result]:
        future: Future[Result] = Future()
        self.jobs.append((future, lambda: function(*args)))
        return future

    def finish(self, index: int) -> None:
        future, function = self.jobs[index]
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(function())
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait is False and cancel_futures is True
        self.closed = True
        for future, _ in self.jobs:
            if not future.done():
                future.cancel()


class CountingContext(RunContext):
    def __init__(self, *, fast: object) -> None:
        super().__init__(services={"asv3_fast": fast})
        self.active_checks = 0

    def check_active(self) -> None:
        self.active_checks += 1
        super().check_active()


def program_steps() -> dict[str, dict[str, JsonValue]]:
    return {
        "first": {"tool": "read", "arguments": {"key": "first"}},
        "second": {"tool": "read", "arguments": {"key": "second"}},
    }


@pytest.mark.parametrize(
    "fast,expected_checks", [(True, 13), (False, 19), ("true", 19)]
)
def test_compose_idle_waits_keep_fresh_checks_without_rescanning_unchanged_steps(
    monkeypatch: pytest.MonkeyPatch, fast: object, expected_checks: int
) -> None:
    context = CountingContext(fast=fast)
    pool: ManualPool[ToolOutcome] = ManualPool()
    waits = 0
    calls: list[str] = []

    def dispatch(call: CapabilityCall) -> ToolOutcome:
        key = str(call.arguments["key"])
        calls.append(key)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary=key,
            data={"original": f"{key}: full conditional provision"},
        )

    def wait(
        futures: Iterable[Future[ToolOutcome]], *, timeout: float, return_when: str
    ) -> tuple[set[Future[ToolOutcome]], set[Future[ToolOutcome]]]:
        nonlocal waits
        waits += 1
        pending = set(futures)
        assert timeout == 0.05 and return_when == FIRST_COMPLETED
        if waits <= 3:
            assert context.active_checks >= waits
            assert all(not future.done() for future in pending)
            return set(), pending
        pool.finish(waits - 4)
        return {future for future in pending if future.done()}, set()

    monkeypatch.setattr(sandbox_module, "wait", wait)
    result = sandbox_module._continuous_compose(
        program_steps(),
        {"first": set(), "second": set()},
        context,
        cast(ThreadPoolExecutor, pool),
        1,
        dispatch,
    )
    assert waits == 5 and context.active_checks == expected_checks
    assert calls == list(result) == ["first", "second"]
    assert [item.data for item in result.values()] == [
        {"original": "first: full conditional provision"},
        {"original": "second: full conditional provision"},
    ]


@pytest.mark.parametrize("boundary", ["idle", "result", "deadline"])
@pytest.mark.parametrize("fast", [False, True])
def test_compose_fresh_stop_fences_reject_idle_or_completed_results(
    monkeypatch: pytest.MonkeyPatch, boundary: str, fast: bool
) -> None:
    context = CountingContext(fast=fast)
    pool: ManualPool[ToolOutcome] = ManualPool()
    calls: list[str] = []

    def dispatch(call: CapabilityCall) -> ToolOutcome:
        calls.append(call.name)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="late original")

    def wait(
        futures: Iterable[Future[ToolOutcome]], *, timeout: float, return_when: str
    ) -> tuple[set[Future[ToolOutcome]], set[Future[ToolOutcome]]]:
        pending = set(futures)
        assert timeout == 0.05 and return_when == FIRST_COMPLETED
        if boundary == "result":
            pool.finish(0)
        if boundary == "deadline":
            context.deadline = float("-inf")
        else:
            context.cancel()
        return {future for future in pending if future.done()}, pending

    monkeypatch.setattr(sandbox_module, "wait", wait)
    steps: dict[str, dict[str, JsonValue]] = {
        "first": {"tool": "read"},
        "dependent": {"tool": "must_not_start"},
    }
    with pytest.raises(
        RunStopped, match="deadline" if boundary == "deadline" else "cancelled"
    ):
        sandbox_module._continuous_compose(
            steps,
            {"first": set(), "dependent": {"first"}},
            context,
            cast(ThreadPoolExecutor, pool),
            1,
            dispatch,
        )
    assert len(pool.jobs) == 1
    assert calls == (["read"] if boundary == "result" else [])


def test_fast_compose_checks_between_each_completed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = CountingContext(fast=True)
    pool: ManualPool[ToolOutcome] = ManualPool()
    accepted: list[str] = []

    def dispatch(call: CapabilityCall) -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=call.name)

    class StopAfterResult(Future[ToolOutcome]):
        def result(self, timeout: float | None = None) -> ToolOutcome:
            accepted.append("first")
            context.cancel()
            return super().result(timeout)

    first: Future[ToolOutcome] = StopAfterResult()
    second: Future[ToolOutcome] = Future()

    def submit(
        function: Callable[..., ToolOutcome], *args: object
    ) -> Future[ToolOutcome]:
        future = first if not pool.jobs else second
        pool.jobs.append((future, lambda: function(*args)))
        return future

    monkeypatch.setattr(pool, "submit", submit)

    def wait(
        futures: Iterable[Future[ToolOutcome]], *, timeout: float, return_when: str
    ) -> tuple[list[Future[ToolOutcome]], set[Future[ToolOutcome]]]:
        assert timeout == 0.05 and return_when == FIRST_COMPLETED
        assert list(futures) == [first, second]
        pool.finish(0)
        pool.finish(1)
        return [first, second], set()

    monkeypatch.setattr(sandbox_module, "wait", wait)
    with pytest.raises(RunStopped, match="cancelled"):
        sandbox_module._continuous_compose(
            program_steps(),
            {"first": set(), "second": set()},
            context,
            cast(ThreadPoolExecutor, pool),
            2,
            dispatch,
        )
    assert accepted == ["first"]


@pytest.mark.parametrize("fast", [False, True, "true"])
def test_harness_waits_on_completion_only_for_strict_fast_and_keeps_receipt_order(
    monkeypatch: pytest.MonkeyPatch, fast: object
) -> None:
    pool: ManualPool[ToolReceipt] = ManualPool()
    committed: list[str] = []
    waiting: list[str] = []
    snapshots: list[dict[str, JsonValue]] = []

    def factory(*, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
        assert max_workers == 4 and thread_name_prefix == "asv3-tool"
        return cast(ThreadPoolExecutor, pool)

    def complete_next() -> None:
        # The second declared result completes first and becomes durable promptly.
        pool.finish(1 if not committed else 0)

    def wait(
        futures: Iterable[Future[ToolReceipt]], *, timeout: float, return_when: str
    ) -> tuple[set[Future[ToolReceipt]], set[Future[ToolReceipt]]]:
        waiting.append("future")
        assert fast is True and timeout == 0.02 and return_when == FIRST_COMPLETED
        pending = set(futures)
        complete_next()
        return {future for future in pending if future.done()}, set()

    def sleep(seconds: float) -> None:
        waiting.append("sleep")
        assert fast is not True and seconds == 0.02
        complete_next()

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="same full source",
            data={"key": args["key"]},
        )

    monkeypatch.setattr(harness_module, "ThreadPoolExecutor", factory)
    monkeypatch.setattr(harness_module, "wait", wait)
    monkeypatch.setattr(harness_module.time, "sleep", sleep)
    harness = Harness(
        request="ordered reads",
        context=RunContext(services={"asv3_fast": fast}),
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=read,
                )
            ]
        ),
        decide=lambda _view: Decision(answer="done"),
        on_receipt=lambda receipt: committed.append(receipt.call.call_id),
        checkpoint=snapshots.append,
    )
    calls = [
        CapabilityCall(name="read", call_id=key, arguments={"key": key})
        for key in ["first", "second"]
    ]
    result = harness._dispatch(calls)
    assert waiting == ["future" if fast is True else "sleep"] * 2
    assert [r.call.call_id for r in result] == ["first", "second"]
    assert committed == ["second", "first"]
    assert [r.outcome.data for r in result] == [{"key": "first"}, {"key": "second"}]
    assert snapshots[-1]["pending_calls"] == []
    assert pool.closed


@pytest.mark.parametrize(
    "stop", ["cancel", "completed_cancel", "research_deadline", "deadline"]
)
def test_fast_harness_wait_preserves_stop_and_child_cancellation(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    pool: ManualPool[ToolReceipt] = ManualPool()
    context = RunContext(services={"asv3_fast": True})
    children: list[RunContext] = []
    calls_run: list[str] = []
    original_child = context.child

    def child() -> RunContext:
        created = original_child()
        children.append(created)
        return created

    def factory(*, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
        assert max_workers == 4 and thread_name_prefix == "asv3-tool"
        return cast(ThreadPoolExecutor, pool)

    def wait(
        futures: Iterable[Future[ToolReceipt]], *, timeout: float, return_when: str
    ) -> tuple[set[Future[ToolReceipt]], set[Future[ToolReceipt]]]:
        assert timeout == 0.02 and return_when == FIRST_COMPLETED
        if stop == "completed_cancel":
            pool.finish(0)
            context.cancel()
        elif stop == "cancel":
            context.cancel()
        elif stop == "deadline":
            context.deadline = float("-inf")
        else:
            context.research_deadline = float("-inf")
        return set(), set(futures)

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        calls_run.append(str(args["key"]))
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="unaccepted original")

    monkeypatch.setattr(harness_module, "ThreadPoolExecutor", factory)
    monkeypatch.setattr(harness_module, "wait", wait)
    monkeypatch.setattr(context, "child", child)
    harness = Harness(
        request="stopped wait",
        context=context,
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=read,
                )
            ]
        ),
        decide=lambda _view: Decision(answer="done"),
    )
    calls = [
        CapabilityCall(name="read", arguments={"key": key})
        for key in ["first", "second"]
    ]
    if stop == "research_deadline":
        result = harness._dispatch(calls)
        assert [r.outcome.status for r in result] == [OutcomeStatus.TRUNCATED] * 2
    else:
        with pytest.raises(
            RunStopped, match="cancelled" if "cancel" in stop else "deadline"
        ):
            harness._dispatch(calls)
        assert harness.receipts == []
    assert calls_run == (["first"] if stop == "completed_cancel" else [])
    assert all(
        future.cancelled()
        for index, (future, _) in enumerate(pool.jobs)
        if stop != "completed_cancel" or index != 0
    )
    assert len(children) == 2 and all(child.is_cancelled() for child in children)
    assert pool.closed
