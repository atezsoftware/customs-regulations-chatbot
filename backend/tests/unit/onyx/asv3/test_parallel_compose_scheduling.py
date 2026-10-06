import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.models import (
    Artifact,
    EvidenceItem,
    OriginalEvidenceRead,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import _continuous_compose_enabled, compose


def context(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
    mode: str = "parallel",
) -> RunContext:
    services: dict[str, object] = {
        "research_profile": "experimental",
        "experimental_parallel": mode == "parallel",
    }
    if mode == "hosted":
        services.update(
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-question",
        )
    services["registry"] = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={"type": "object"},
                handler=handler,
            )
        ]
    )
    return RunContext(services=services)


@pytest.mark.parametrize("mode", ["parallel", "hosted", "plain"])
def test_fast_branch_continues_before_unrelated_slow_branch_only_when_enabled(
    mode: str,
) -> None:
    slow_entered, fast_finished, dependent_entered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    release = threading.Event()
    tenant: ContextVar[str] = ContextVar("compose_tenant", default="wrong")

    def read(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        assert tenant.get() == "captured"
        assert child is captured
        name = str(args["name"])
        if name == "slow":
            slow_entered.set()
            assert release.wait(5)
        elif name == "fast":
            fast_finished.set()
        else:
            assert args["value"] == "exact original"
            dependent_entered.set()
        original = EvidenceItem(
            source_id="authorized-source",
            chunk_id=name,
            text=f"{name}: özgün hüküm koşullarının tamamı",
            metadata={"read_as_of_date": "2026-10-06", "article_no": "53"},
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary=name,
            data={"value": "exact original"},
            evidence=[original],
            original_reads=[
                OriginalEvidenceRead(
                    citation={"slow": 1, "fast": 2, "dependent": 3}[name],
                    text_hash=original.text_hash,
                    start_char=0,
                    end_char=len(original.text),
                )
            ],
            artifacts=[Artifact(artifact_id=name, name=name)],
        )

    captured = context(read, mode)
    program: dict[str, JsonValue] = {
        "max_parallel": 2,
        "steps": [
            {"id": "slow", "tool": "read", "arguments": {"name": "slow"}},
            {"id": "fast", "tool": "read", "arguments": {"name": "fast"}},
            {
                "id": "dependent",
                "tool": "read",
                "depends_on": ["fast"],
                "arguments": {
                    "name": "dependent",
                    "value": {"$ref": "fast.data.value"},
                },
            },
        ],
    }

    def run() -> ToolOutcome:
        token = tenant.set("captured")
        try:
            return compose(program, captured)
        finally:
            tenant.reset(token)

    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(run)
        try:
            assert slow_entered.wait(5) and fast_finished.wait(5)
            if mode == "plain":
                assert not dependent_entered.wait(0.1)
            else:
                assert dependent_entered.wait(5)
                assert not future.done()
        finally:
            release.set()
        result = future.result(timeout=5)
    assert result.status == OutcomeStatus.FOUND
    assert dependent_entered.is_set()
    assert captured.budget.snapshot()["tools"] == 3
    assert {item.chunk_id for item in result.evidence} == {"slow", "fast", "dependent"}
    assert all(
        item.text == f"{item.chunk_id}: özgün hüküm koşullarının tamamı"
        and item.source_id == "authorized-source"
        and item.metadata["read_as_of_date"] == "2026-10-06"
        for item in result.evidence
    )
    assert {read.citation for read in result.original_reads} == {1, 2, 3}
    if mode != "plain":
        assert list(cast(dict, result.data["steps"])) == ["slow", "fast", "dependent"]
        assert [item.chunk_id for item in result.evidence] == [
            "slow",
            "fast",
            "dependent",
        ]
        assert [read.citation for read in result.original_reads] == [1, 2, 3]
        assert [item.artifact_id for item in result.artifacts] == [
            "slow",
            "fast",
            "dependent",
        ]


@pytest.mark.parametrize(
    "updates,depth,expected",
    [
        ({}, 0, True),
        ({"research_profile": "deep"}, 0, False),
        ({"research_profile": "asv3"}, 0, False),
        ({"experimental_parallel": False}, 0, False),
        ({"experimental_parallel": "true"}, 0, False),
        (
            {
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
                "task_id": "hosted-owner",
            },
            0,
            True,
        ),
        (
            {
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
                "task_id": "hosted-owner",
            },
            1,
            False,
        ),
        (
            {
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
                "task_id": " ",
            },
            0,
            False,
        ),
        (
            {"experimental_parallel": False, "serial_session_diagnostics": True},
            0,
            False,
        ),
    ],
)
def test_continuous_scheduler_policy_isolation(
    updates: dict[str, object], depth: int, expected: bool
) -> None:
    captured = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            **updates,
        },
        depth=depth,
    )
    assert _continuous_compose_enabled(captured) is expected


def test_parallel_limit_does_not_change_call_count() -> None:
    lock = threading.Lock()
    two_entered, release = threading.Event(), threading.Event()
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
            return ToolOutcome(status=OutcomeStatus.FOUND, summary="original")
        finally:
            with lock:
                running -= 1

    captured = context(read)
    program: dict[str, JsonValue] = {
        "max_parallel": 2,
        "steps": [{"id": str(index), "tool": "read"} for index in range(6)],
    }
    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(
            compose,
            program,
            captured,
        )
        try:
            assert two_entered.wait(5)
            with lock:
                assert calls == peak == 2
        finally:
            release.set()
        assert future.result(timeout=5).status == OutcomeStatus.FOUND
    assert calls == 6 and peak == 2
    assert captured.budget.snapshot()["tools"] == 6


@pytest.mark.parametrize(
    "status,allowed",
    [
        (OutcomeStatus.FOUND, True),
        (OutcomeStatus.PARTIAL, True),
        (OutcomeStatus.VERSION_UNKNOWN, True),
        (OutcomeStatus.AMBIGUOUS, True),
        (OutcomeStatus.DENIED, False),
        (OutcomeStatus.NOT_FOUND, False),
        (OutcomeStatus.UNAVAILABLE, False),
        (OutcomeStatus.CANCELLED, False),
    ],
)
def test_dependency_status_conditions_and_reference_semantics(
    status: OutcomeStatus, allowed: bool
) -> None:
    calls: list[str] = []

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        name = str(args["name"])
        calls.append(name)
        if name == "root":
            return ToolOutcome(status=status, summary="root", data={"value": "4"})
        assert name == "match" and args["value"] == "4"
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="matched")

    result = compose(
        {
            "steps": [
                {"id": "root", "tool": "read", "arguments": {"name": "root"}},
                {
                    "id": "match",
                    "tool": "read",
                    "depends_on": ["root"],
                    "when": {"value": {"$ref": "root.data.value"}, "equals": "4"},
                    "arguments": {
                        "name": "match",
                        "value": {"$ref": "root.data.value"},
                    },
                },
                {
                    "id": "skip",
                    "tool": "read",
                    "depends_on": ["root"],
                    "when": {"value": {"$ref": "root.data.value"}, "equals": "9"},
                },
                {"id": "blocked", "tool": "read", "depends_on": ["skip"]},
            ]
        },
        context(read),
    )
    steps = cast(dict, result.data["steps"])
    assert calls == (["root", "match"] if allowed else ["root"])
    assert steps["skip"]["status"] == steps["blocked"]["status"] == "cancelled"
    assert steps["match"]["status"] == ("found" if allowed else "cancelled")


def test_cancelled_program_never_launches_dependent_or_returns_late_original() -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls: list[str] = []

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        calls.append(str(args["name"]))
        entered.set()
        try:
            assert release.wait(5)
            return ToolOutcome(status=OutcomeStatus.FOUND, summary="late original")
        finally:
            finished.set()

    captured = context(read)
    program: dict[str, JsonValue] = {
        "steps": [
            {"id": "root", "tool": "read", "arguments": {"name": "root"}},
            {
                "id": "dependent",
                "tool": "read",
                "depends_on": ["root"],
                "arguments": {"name": "dependent"},
            },
        ]
    }
    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(
            compose,
            program,
            captured,
        )
        try:
            assert entered.wait(5)
            captured.cancel()
            with pytest.raises(RunStopped, match="cancelled"):
                future.result(timeout=5)
        finally:
            release.set()
            assert finished.wait(5)
    assert calls == ["root"]
