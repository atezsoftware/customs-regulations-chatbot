"""Fast execution preserves originals and scopes while removing scheduling waits."""

import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3 import corpus_tools
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.native_token_cache import native_token_cache_enabled
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import _continuous_compose_enabled, compose
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db import asv3_corpus as corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_cropped_hydration_source_batch import (
    fixture as batch_fixture,
)


@pytest.mark.parametrize("fast", [None, False, True, "true", 1])
@pytest.mark.parametrize("cropped", [False, True])
def test_fast_flags_leave_legacy_execution_policy_unchanged(
    monkeypatch: pytest.MonkeyPatch, fast: object, cropped: bool
) -> None:
    captured = RunContext(services={"asv3_fast": fast, "asv3_exp_cropped": cropped})
    broker = corpus_tools.CorpusBroker(
        cast(User, object()), IndexFilters(access_control_list=[])
    )
    batch = MagicMock(return_value={})
    monkeypatch.setattr(broker, "_hydrate_search_centers_batch", batch)
    broker.hydrate_search_centers([], captured)
    assert batch.call_count == int(cropped or fast is True)
    assert native_token_cache_enabled(captured.services) is (cropped or fast is True)
    assert _continuous_compose_enabled(captured) is (fast is True)


def test_fast_hydration_returns_identical_legacy_originals_with_both_live_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = batch_fixture(monkeypatch, 1)
    sources = cast(dict[UUID, CorpusSource], state["sources"])
    originals = cast(dict[UUID, CorpusChunk], state["originals"])
    key = next(iter(sources))
    chunk = originals[key]
    source = sources[key]
    docs = [
        SearchDoc(
            document_id=str(key),
            chunk_ind=chunk.projection_ordinal,
            semantic_identifier="Source",
            link=None,
            blurb="projection is not the original",
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": chunk.id},
            match_highlights=[],
        )
    ]
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock(spec=Session)),
    )
    monkeypatch.setattr(corpus_tools, "require_source", lambda *_a, **_kw: source)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_a: None)

    def stream(*_args: object, **_kwargs: object) -> Generator[CorpusChunk, None, None]:
        yield chunk

    monkeypatch.setattr(corpus_tools, "iter_source_chunks_by_ids", stream)
    broker = corpus_tools.CorpusBroker(
        cast(User, state["kwargs"]["user"]),
        cast(IndexFilters, state["kwargs"]["filters"]),
    )
    old = broker.hydrate_search_centers(docs, RunContext())
    fast = broker.hydrate_search_centers(docs, RunContext(services={"asv3_fast": True}))
    assert old == fast
    assert fast[(str(key), chunk.projection_ordinal)][0].text == chunk.text
    assert len(state["calls"]) == state["publication"].call_count == 2
    assert state["calls"][0][0] is not state["calls"][1][0]


@pytest.mark.parametrize(
    "defect",
    [None, "hash", "ordinal", "entry_acl", "exit_acl", "publication", "cancel"],
)
def test_fast_final_revalidation_preserves_hash_acl_publication_and_cancel_fences(
    monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    state = batch_fixture(monkeypatch, 1)
    sources = cast(dict[UUID, CorpusSource], state["sources"])
    originals = cast(dict[UUID, CorpusChunk], state["originals"])
    key = next(iter(sources))
    original = originals[key]
    retained = corpus_tools.evidence_for_chunk(sources[key], original)
    captured = RunContext(services={"asv3_fast": True})
    if defect == "hash":
        originals[key] = replace(original, text="changed original")
    if defect == "ordinal":
        originals[key] = replace(original, projection_ordinal=99)
    if defect in {"entry_acl", "exit_acl"}:
        find = MagicMock(wraps=corpus.find_sources)
        calls = 0

        def changed(*args: object, **kwargs: object) -> tuple[list[CorpusSource], bool]:
            nonlocal calls
            calls += 1
            rows, more = cast(tuple[list[CorpusSource], bool], find(*args, **kwargs))
            if calls == (1 if defect == "entry_acl" else 2):
                return [], False
            return rows, more

        monkeypatch.setattr(corpus, "find_sources", changed)
    if defect == "publication":
        state["publication"].side_effect = [
            None,
            PermissionError("revoked publication"),
        ]
    if defect == "cancel":
        captured.cancel()
    broker = corpus_tools.CorpusBroker(
        cast(User, state["kwargs"]["user"]),
        cast(IndexFilters, state["kwargs"]["filters"]),
    )
    if defect is None:
        broker.revalidate_evidence([retained], captured)
        assert len(state["calls"]) == state["publication"].call_count == 2
        assert state["calls"][0][0] is not state["calls"][1][0]
    else:
        with pytest.raises((CorpusScopeUnavailable, PermissionError, RunStopped)):
            broker.revalidate_evidence([retained], captured)


def context(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
    *,
    fast: bool,
    cropped: bool,
) -> RunContext:
    return RunContext(
        services={
            "asv3_fast": fast,
            "asv3_exp_cropped": cropped,
            "registry": CapabilityRegistry(
                [
                    ToolSpec(
                        name="read",
                        description="read originals",
                        parameters={"type": "object"},
                        handler=handler,
                    )
                ]
            ),
        }
    )


def program() -> dict[str, JsonValue]:
    return {
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


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("cropped", [False, True])
def test_only_fast_releases_a_ready_dependency_before_an_unrelated_slow_read(
    fast: bool, cropped: bool
) -> None:
    slow, immediate, dependent, release = (threading.Event() for _ in range(4))
    calls: list[str] = []

    def read(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        assert child is captured
        name = str(args["name"])
        calls.append(name)
        if name == "slow":
            slow.set()
            assert release.wait(5)
        elif name == "fast":
            immediate.set()
        else:
            assert args["value"] == "original retained"
            dependent.set()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary=name,
            data={"value": "original retained"},
        )

    captured = context(read, fast=fast, cropped=cropped)
    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(compose, program(), captured)
        try:
            assert slow.wait(5) and immediate.wait(5)
            if fast:
                assert dependent.wait(5)
                assert not future.done()
            else:
                assert not dependent.wait(0.1)
        finally:
            release.set()
        result = future.result(timeout=5)
    assert result.status == OutcomeStatus.FOUND
    assert sorted(calls) == ["dependent", "fast", "slow"]
    assert captured.budget.snapshot()["tools"] == 3
    if fast:
        assert list(cast(dict[str, JsonValue], result.data["steps"])) == [
            "slow",
            "fast",
            "dependent",
        ]


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("status", [OutcomeStatus.DENIED, OutcomeStatus.ERROR])
def test_fast_failed_dependencies_do_not_launch_child_reads(
    cropped: bool, status: OutcomeStatus
) -> None:
    calls: list[str] = []

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        calls.append(str(args["name"]))
        if status == OutcomeStatus.ERROR:
            raise RuntimeError("original read failed")
        return ToolOutcome(status=status, summary="not available")

    captured = context(read, fast=True, cropped=cropped)
    result = compose(program(), captured)
    assert sorted(calls) == ["fast", "slow"]
    assert result.status == OutcomeStatus.PARTIAL
    steps = cast(dict[str, dict[str, JsonValue]], result.data["steps"])
    assert steps["dependent"]["status"] == "cancelled"


@pytest.mark.parametrize("cropped", [False, True])
def test_cancelled_fast_program_rejects_late_results_and_never_starts_dependents(
    cropped: bool,
) -> None:
    entered, release, finished = (threading.Event() for _ in range(3))
    calls: list[str] = []

    def read(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        name = str(args["name"])
        calls.append(name)
        entered.set()
        try:
            assert release.wait(5)
            return ToolOutcome(status=OutcomeStatus.FOUND, summary="late original")
        finally:
            finished.set()

    captured = context(read, fast=True, cropped=cropped)
    with ThreadPoolExecutor(max_workers=1) as control:
        future = control.submit(compose, program(), captured)
        try:
            assert entered.wait(5)
            captured.cancel()
            with pytest.raises(RunStopped, match="cancelled"):
                future.result(timeout=5)
        finally:
            release.set()
            assert finished.wait(5)
    assert "dependent" not in calls
