"""Exact read sharing retains per-caller authority and cancellation."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import date
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3 import corpus_tools
from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.shared_reads import SharedReads
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_candidate_inventory import asv3_source_inventory_scope
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationIndexSnapshot
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

SOURCE = "3c3b3448-1f67-46a0-84f5-16980309c088"
OTHER_SOURCE = "2f107efa-1d8d-47a6-93da-5dde8dc90d26"


def context() -> RunContext:
    return RunContext(
        run_id="one-run",
        scope={"tenant": "one-tenant", "as_of": "2026-10-05"},
        budget=SharedBudget(unlimited_execution=True),
        services={"research_profile": "experimental"},
        deadline=float("inf"),
        research_deadline=float("inf"),
    )


def producer(root: RunContext) -> Callable[[RunContext], RunContext]:
    def create(caller: RunContext) -> RunContext:
        return RunContext(
            run_id=caller.run_id,
            scope=caller.scope,
            services=caller.services,
            budget=caller.budget,
            cancelled=root.is_cancelled,
            corpus_only=caller.corpus_only,
            deadline=caller.deadline,
            research_deadline=caller.research_deadline,
        )

    return create


def original(source: str = SOURCE) -> EvidenceItem:
    return EvidenceItem(
        source_id=source,
        chunk_id="canonical-chunk",
        text="The original requires both conditions.",
        search_doc=SearchDoc(
            document_id=source,
            chunk_ind=0,
            semantic_identifier="Original",
            blurb="Original",
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={},
            match_highlights=[],
        ),
    )


def outcome(status: OutcomeStatus = OutcomeStatus.FOUND) -> ToolOutcome:
    return ToolOutcome(
        status=status,
        summary="Actual source result",
        data={
            "has_more": True,
            "next_position": 30,
            "scan_truncated": True,
            "subunit_verified": False,
        },
        evidence=[original()],
    )


def test_concurrent_identical_reads_execute_once_and_isolate_caller_payloads() -> None:
    root = context()
    entered = threading.Event()
    joined = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    executions = 0
    fences = 0

    def fence(_source: str, _caller: RunContext) -> str:
        nonlocal fences
        with lock:
            fences += 1
            if fences == 2:
                joined.set()
        return "captured-user-index"

    def execute(_producer: RunContext) -> ToolOutcome:
        nonlocal executions
        executions += 1
        entered.set()
        assert release.wait(2)
        return outcome()

    reads = SharedReads(fence=fence, producer_context=producer(root))
    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            reads.run,
            "read_chunk",
            arguments,
            root.child(),
            execute,
        )
        assert entered.wait(2)
        second = pool.submit(
            reads.run,
            "read_chunk",
            arguments,
            root.child(),
            execute,
        )
        assert joined.wait(2)
        release.set()
        one, two = first.result(timeout=2), second.result(timeout=2)
    assert executions == 1
    assert fences == 2
    one.evidence[0].question_ids.append("first-need")
    one.data["next_position"] = 100
    assert two.evidence[0].question_ids == []
    assert two.data["next_position"] == 30
    assert two.data["shared_read_reuse"] in {"inflight", "completed"}


@pytest.mark.parametrize(
    "status", [OutcomeStatus.FOUND, OutcomeStatus.PARTIAL, OutcomeStatus.TRUNCATED]
)
def test_exact_reuse_preserves_partial_status_cursor_and_original_hash(
    status: OutcomeStatus,
) -> None:
    root = context()
    reads = SharedReads(
        fence=lambda _s, _c: "captured-index", producer_context=producer(root)
    )
    calls = 0

    def execute(_context: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return outcome(status)

    first = reads.run("read_source_range", {"source_id": SOURCE}, root, execute)
    second = reads.run(
        "read_source_range",
        {"source_id": SOURCE, "start": 0, "limit": 30},
        root,
        execute,
    )
    assert calls == 1
    assert second.status == first.status == status
    assert second.data == {**first.data, "shared_read_reuse": "completed"}
    assert second.evidence == first.evidence
    assert second.evidence[0].text_hash == original().text_hash


def test_material_source_range_scope_run_and_index_inputs_never_merge() -> None:
    root = context()
    index = "index-one"
    reads = SharedReads(fence=lambda _s, _c: index, producer_context=producer(root))
    calls = 0

    def execute(_current: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return outcome()

    reads.run("read_source_range", {"source_id": SOURCE}, root, execute)
    reads.run("read_source_range", {"source_id": SOURCE, "start": 1}, root, execute)
    reads.run("read_source_range", {"source_id": SOURCE, "limit": 1}, root, execute)
    reads.run("read_source_range", {"source_id": OTHER_SOURCE}, root, execute)
    different_scope = root.child()
    different_scope.scope["as_of"] = "2025-10-05"
    reads.run("read_source_range", {"source_id": SOURCE}, different_scope, execute)
    different_run = root.child()
    different_run.run_id = "another-run"
    reads.run("read_source_range", {"source_id": SOURCE}, different_run, execute)
    index = "index-two"
    reads.run("read_source_range", {"source_id": SOURCE}, root, execute)
    assert calls == 7


def test_provision_structured_start_subunits_and_query_results_are_distinct() -> None:
    root = context()
    reads = SharedReads(fence=lambda _s, _c: "index", producer_context=producer(root))
    calls = 0

    def execute(_current: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return outcome()

    base: dict[str, JsonValue] = {"source_id": SOURCE, "article": "17"}
    for extra in ({}, {"start": 0}, {"paragraph": "2"}, {"clause": "a"}):
        reads.run("read_provision", {**base, **extra}, root, execute)
    for target in ("first issue", "second issue"):
        reads.run(
            "search_corpus",
            {"source_id": SOURCE, "query": "same query", "evidence_target": target},
            root,
            execute,
        )
    assert calls == 6


@pytest.mark.parametrize(
    "status",
    [
        OutcomeStatus.ERROR,
        OutcomeStatus.DENIED,
        OutcomeStatus.NOT_FOUND,
        OutcomeStatus.CANCELLED,
    ],
)
def test_failure_and_negative_results_are_not_permanent_cache_entries(
    status: OutcomeStatus,
) -> None:
    root = context()
    reads = SharedReads(fence=lambda _s, _c: "index", producer_context=producer(root))
    calls = 0

    def execute(_current: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return (
            ToolOutcome(status=status, summary="Actual failure")
            if calls == 1
            else outcome()
        )

    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    assert reads.run("read_chunk", arguments, root, execute).status == status
    assert (
        reads.run("read_chunk", arguments, root, execute).status == OutcomeStatus.FOUND
    )
    assert (
        reads.run("read_chunk", arguments, root, execute).status == OutcomeStatus.FOUND
    )
    assert calls == 2


def test_every_reuse_rechecks_source_fence_and_cannot_return_revoked_evidence() -> None:
    root = context()
    authorized = True
    calls = 0

    def fence(_source: str, _caller: RunContext) -> str | ToolOutcome:
        return (
            "index"
            if authorized
            else ToolOutcome(
                status=OutcomeStatus.DENIED, summary="Source access revoked"
            )
        )

    def execute(_current: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return outcome()

    reads = SharedReads(fence=fence, producer_context=producer(root))
    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    reads.run("read_chunk", arguments, root, execute)
    authorized = False
    denied = reads.run("read_chunk", arguments, root, execute)
    assert denied.status == OutcomeStatus.DENIED
    assert denied.evidence == []
    assert calls == 1


def test_mismatched_original_source_and_changed_text_cannot_be_cached() -> None:
    root = context()
    reads = SharedReads(fence=lambda _s, _c: "index", producer_context=producer(root))
    calls = 0

    def wrong_source(_current: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        result = outcome()
        assert result.evidence[0].search_doc is not None
        result.evidence[0].search_doc.document_id = OTHER_SOURCE
        return result

    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    reads.run("read_chunk", arguments, root, wrong_source)
    reads.run("read_chunk", arguments, root, wrong_source)
    assert calls == 2
    assert reads._reads == {}

    def changed_text(_current: RunContext) -> ToolOutcome:
        result = outcome()
        result.evidence[0].text = "Changed without updating its original hash"
        return result

    with pytest.raises(ValueError, match="hash"):
        reads.run("read_chunk", arguments, root, changed_text)
    assert reads._reads == {}


def test_follower_cancellation_does_not_abort_producer_or_other_caller() -> None:
    root = context()
    follower = root.child()
    entered = threading.Event()
    joined = threading.Event()
    release = threading.Event()
    fences = 0

    def fence(_source: str, _caller: RunContext) -> str:
        nonlocal fences
        fences += 1
        if fences == 2:
            joined.set()
        return "index"

    def execute(current: RunContext) -> ToolOutcome:
        entered.set()
        assert release.wait(2)
        current.check_active()
        return outcome()

    reads = SharedReads(fence=fence, producer_context=producer(root))
    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reads.run, "read_chunk", arguments, root.child(), execute)
        assert entered.wait(2)
        second = pool.submit(reads.run, "read_chunk", arguments, follower, execute)
        assert joined.wait(2)
        follower.cancel()
        with pytest.raises(RunStopped):
            second.result(timeout=2)
        release.set()
        assert first.result(timeout=2).status == OutcomeStatus.FOUND
    assert (
        reads.run("read_chunk", arguments, root, execute).data["shared_read_reuse"]
        == "completed"
    )


def test_cancelled_leader_subscriber_does_not_cancel_shared_run_acquisition() -> None:
    root = context()
    leader = root.child()
    reads = SharedReads(fence=lambda _s, _c: "index", producer_context=producer(root))

    def execute(current: RunContext) -> ToolOutcome:
        leader.cancel()
        current.check_active()
        return outcome()

    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    with pytest.raises(RunStopped):
        reads.run("read_chunk", arguments, leader, execute)
    assert (
        reads.run("read_chunk", arguments, root, execute).status == OutcomeStatus.FOUND
    )


def test_underlying_timeout_reaches_waiters_and_allows_a_later_retry() -> None:
    root = context()
    entered = threading.Event()
    joined = threading.Event()
    release = threading.Event()
    fences = 0

    def fence(_source: str, _caller: RunContext) -> str:
        nonlocal fences
        fences += 1
        if fences == 2:
            joined.set()
        return "index"

    def execute(_current: RunContext) -> ToolOutcome:
        entered.set()
        assert release.wait(2)
        raise TimeoutError("Source operation failed")

    reads = SharedReads(fence=fence, producer_context=producer(root))
    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reads.run, "read_chunk", arguments, root.child(), execute)
        assert entered.wait(2)
        second = pool.submit(reads.run, "read_chunk", arguments, root.child(), execute)
        assert joined.wait(2)
        release.set()
        try:
            with pytest.raises(TimeoutError, match="Source operation failed"):
                first.result(timeout=2)
            with pytest.raises(TimeoutError, match="Source operation failed"):
                second.result(timeout=2)
            assert second.done()
        finally:
            if not second.done():
                root.cancel()
    assert reads._reads == {}
    assert (
        reads.run("read_chunk", arguments, root, lambda _c: outcome()).status
        == OutcomeStatus.FOUND
    )


def test_global_cancellation_drops_late_result_instead_of_retaining_it() -> None:
    root = context()
    reads = SharedReads(fence=lambda _s, _c: "index", producer_context=producer(root))

    def execute(_current: RunContext) -> ToolOutcome:
        root.cancel()
        return outcome()

    with pytest.raises(RunStopped):
        reads.run(
            "read_chunk", {"source_id": SOURCE, "chunk_id": "a"}, root.child(), execute
        )
    assert reads._reads == {}


def test_registry_joins_before_tool_slot_and_only_charges_physical_execution() -> None:
    root = context()
    root.budget.tool_slots = threading.BoundedSemaphore(1)
    fences = 0
    joined = threading.Event()
    lock = threading.Lock()
    executions = 0

    def fence(_source: str, _caller: RunContext) -> str:
        nonlocal fences
        with lock:
            fences += 1
            if fences == 2:
                joined.set()
        return "index"

    def execute(_arguments: dict[str, JsonValue], _current: RunContext) -> ToolOutcome:
        nonlocal executions
        executions += 1
        return outcome()

    root.services["shared_reads"] = SharedReads(
        fence=fence, producer_context=producer(root)
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_chunk",
                description="Exact chunk",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                    },
                    "required": ["source_id", "chunk_id"],
                    "additionalProperties": False,
                },
                handler=execute,
            )
        ]
    )
    arguments: dict[str, JsonValue] = {"source_id": SOURCE, "chunk_id": "a"}
    root.budget.tool_slots.acquire()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            registry.dispatch,
            CapabilityCall(name="read_chunk", arguments=arguments),
            root.child(),
        )
        second = pool.submit(
            registry.dispatch,
            CapabilityCall(name="read_chunk", arguments=arguments),
            root.child(),
        )
        assert joined.wait(2)
        root.budget.tool_slots.release()
        assert first.result(timeout=2).status == OutcomeStatus.FOUND
        assert second.result(timeout=2).status == OutcomeStatus.FOUND
    assert executions == 1
    assert root.budget.snapshot()["tools"] == 1


@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_registry_baselines_never_activate_installed_shared_reads(profile: str) -> None:
    root = context()
    root.services["research_profile"] = profile

    def forbidden_fence(_source: str, _caller: RunContext) -> str:
        pytest.fail("Baseline must not use shared read hook")

    root.services["shared_reads"] = SharedReads(
        fence=forbidden_fence, producer_context=producer(root)
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_chunk",
                description="Exact original",
                parameters={"type": "object"},
                handler=lambda _a, _c: outcome(),
            )
        ]
    )
    call = CapabilityCall(
        name="read_chunk", arguments={"source_id": SOURCE, "chunk_id": "a"}
    )
    assert registry.dispatch(call, root).status == OutcomeStatus.FOUND
    assert registry.dispatch(call, root).status == OutcomeStatus.FOUND
    assert root.budget.snapshot()["tools"] == 2


def broker_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CorpusBroker, MagicMock, MagicMock]:
    source = CorpusSource(UUID(SOURCE), "Canonical source", "original-file")
    session = MagicMock()
    authorization = MagicMock(return_value=source)
    snapshot = PublicationIndexSnapshot(
        index_name="captured-index",
        index_uuid="one-index-uuid",
        search_settings_id=7,
        model_provider="test-provider",
        model_name="test-embedding",
        vector_dimension=10,
        embedding_config_sha256="a" * 64,
        multitenant=True,
    )
    resolution = MagicMock(return_value=snapshot)
    monkeypatch.setattr(
        corpus_tools, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(corpus_tools, "require_source", authorization)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", resolution)
    return (
        CorpusBroker(
            User(id=UUID("c7ff80b1-ff7f-4820-afb8-7b11d55893f9")),
            IndexFilters(access_control_list=[], as_of_date=date(2026, 10, 5)),
        ),
        authorization,
        resolution,
    )


def test_broker_fence_captures_index_before_first_key_and_authorizes_every_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, authorization, resolution = broker_boundary(monkeypatch)
    root = context()
    first = broker.shared_read_fence(SOURCE, root)
    second = broker.shared_read_fence(SOURCE, root)
    assert isinstance(first, str) and first == second
    assert authorization.call_count == 2
    resolution.assert_called_once()
    assert UUID(SOURCE) in broker.query_indexes
    authorization.side_effect = PermissionError("No access")
    denied = broker.shared_read_fence(SOURCE, root)
    assert isinstance(denied, ToolOutcome)
    assert denied.status == OutcomeStatus.DENIED
    assert denied.evidence == []


def test_broker_fence_distinguishes_actual_tenant_user_date_file_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, authorization, _resolution = broker_boundary(monkeypatch)
    root = context()
    fences = {broker.shared_read_fence(SOURCE, root)}
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("another-tenant")
    try:
        fences.add(broker.shared_read_fence(SOURCE, root))
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    broker.user.id = UUID("7927a42e-ad78-47f7-a47f-75fda9e9c4a4")
    fences.add(broker.shared_read_fence(SOURCE, root))
    broker.filters.as_of_date = date(2025, 10, 5)
    fences.add(broker.shared_read_fence(SOURCE, root))
    authorization.return_value = CorpusSource(
        UUID(SOURCE), "Canonical source", "another-original-file"
    )
    fences.add(broker.shared_read_fence(SOURCE, root))
    broker.query_indexes[UUID(SOURCE)] = broker.query_indexes[UUID(SOURCE)].model_copy(
        update={"index_uuid": "another-index-uuid"}
    )
    fences.add(broker.shared_read_fence(SOURCE, root))
    assert len(fences) == 6


def test_broker_fence_rejects_conflicting_inventory_index_without_source_text_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, _authorization, resolution = broker_boundary(monkeypatch)
    root = context()
    first = broker.shared_read_fence(SOURCE, root)
    assert isinstance(first, str)
    with asv3_source_inventory_scope(scope_key="captured-scope") as inventory:
        inventory.query_indexes[UUID(SOURCE)] = broker.query_indexes[
            UUID(SOURCE)
        ].model_copy(update={"index_uuid": "wrong-index-uuid"})
        rejected = broker.shared_read_fence(SOURCE, root)
    assert isinstance(rejected, ToolOutcome)
    assert rejected.status == OutcomeStatus.UNAVAILABLE
    assert rejected.evidence == []
    resolution.assert_called_once()
