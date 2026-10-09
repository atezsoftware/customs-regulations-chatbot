"""Observed canonical reads can lead a full search batch without extra workers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import (
    CanonicalAcquirer,
    CanonicalEvidenceStage,
)
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.routing import SourceLaneRouter

SOURCE_ID = str(uuid4())
CHUNK_ID = str(uuid4())


def original(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        source_id=SOURCE_ID,
        chunk_id=chunk_id,
        text=text,
        search_doc=SearchDoc(
            document_id=SOURCE_ID,
            chunk_ind=0,
            semantic_identifier="Fictional Zeron rules",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": chunk_id},
            match_highlights=[],
        ),
    )


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="en",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id=need,
                question=f"What supports the {need} outcome?",
                governing_source="Fictional rules",
                conditions_to_check=[],
            )
            for need in ("rule", "condition", "exception")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def registry(
    search: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
    read: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        [
            ToolSpec(
                name="search_corpus",
                description="Search one authorized source lane.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "expand_query": {"type": "boolean"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=search,
            ),
            *[
                ToolSpec(
                    name=name,
                    description="Read an exact authorized original target.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            target: {"type": "string"},
                        },
                        "required": ["source_id", target],
                        "additionalProperties": False,
                    },
                    handler=read,
                )
                for name, target in (
                    ("read_provision", "article"),
                    ("read_chunk", "chunk_id"),
                    ("read_chunk_context", "chunk_id"),
                )
            ],
        ]
    )


def search_action(query: str = "Find applicable Zeron rules") -> SourceAction:
    return SourceAction(
        need_ids=["rule"],
        tool="search_corpus",
        arguments={"query": query},
    )


def read_action(need: str = "condition") -> SourceAction:
    return SourceAction(
        need_ids=[need],
        tool="read_provision",
        source_kind=SourceKind.REGULATION,
        arguments={"source_id": SOURCE_ID, "article": "17"},
    )


@pytest.mark.parametrize("cancel_search", [False, True])
def test_verified_read_precedes_blocked_all_type_searches_with_same_worker_limit(
    monkeypatch: pytest.MonkeyPatch, cancel_search: bool
) -> None:
    ledger = EvidenceLedger()
    context = RunContext(timeout_seconds=30, budget=SharedBudget(max_inflight_tools=12))
    ledger.add([original(CHUNK_ID, "Previously observed Zeron original.")], context)
    read_retained, release_searches = Event(), Event()
    all_searches_started, all_searches_returned = Event(), Event()
    lock = Lock()
    active = maximum_active = finished = 0
    started_kinds: list[SourceKind] = []
    search_stages: list[CanonicalEvidenceStage] = []
    later_chunk = str(uuid4())
    real_add = ledger.add

    def add(items: list[EvidenceItem], child: RunContext) -> list[int]:
        numbers = real_add(items, child)
        if any(item.chunk_id == later_chunk for item in items):
            read_retained.set()
        return numbers

    monkeypatch.setattr(ledger, "add", add)

    def read(_arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            child.check_research_active()
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="New canonical condition original.",
                evidence=[original(later_chunk, "Zeron's exact condition.")],
            )
        finally:
            with lock:
                active -= 1

    def search_for(
        kind: SourceKind,
    ) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
        def search(_arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
            nonlocal active, maximum_active, finished
            stage = child.services["legal_composite_original_stage"]
            assert isinstance(stage, CanonicalEvidenceStage)
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                started_kinds.append(kind)
                search_stages.append(stage)
                if len(started_kinds) == len(SourceKind):
                    all_searches_started.set()
            try:
                assert release_searches.wait(timeout=5)
                return ToolOutcome(
                    status=OutcomeStatus.NOT_FOUND,
                    summary="Bounded lane query finished without absence proof.",
                )
            finally:
                with lock:
                    active -= 1
                    finished += 1
                    if finished == len(SourceKind):
                        all_searches_returned.set()

        return search

    catalogue = SourceLaneCatalogue(
        user_id=uuid4(),
        scope_sha256="synthetic-priority-scope",
        records=(
            SourceClassification(
                source_id=UUID(SOURCE_ID),
                name="Fictional Zeron rules",
                kind=SourceKind.UNKNOWN,
                method="synthetic-canonical",
                uncertain=True,
                observed_document_types=(),
            ),
        ),
        complete=True,
    )
    router = SourceLaneRouter(catalogue, lambda kind: registry(search_for(kind), read))
    instance = CanonicalAcquirer(
        registry(search_for(SourceKind.UNKNOWN), read),
        context,
        ledger,
        WorkflowPolicy(max_parallel_tools=12, max_search_calls=12),
        expand_actions=router.expand,
        registry_for_action=router.registry,
        prioritize_observed_reads=True,
    )
    actions = [search_action(), read_action()]
    frozen_actions = [action.model_dump(mode="json") for action in actions]
    with ThreadPoolExecutor(max_workers=1) as coordinator:
        result = coordinator.submit(instance.acquire, actions, plan())
        try:
            assert read_retained.wait(timeout=3)
            assert all_searches_started.wait(timeout=3)
            assert not result.done()
            assert not release_searches.is_set()
            assert set(started_kinds) == set(SourceKind)
            assert len(started_kinds) == 12
            assert maximum_active == 12
            retained = ledger.get(2)
            assert retained is not None
            assert retained.text == "Zeron's exact condition."
            assert retained.question_ids == ["condition"]
            if cancel_search:
                context.research_deadline = 0
                with pytest.raises(RunStopped, match="deadline"):
                    result.result(timeout=3)
                for stage in search_stages:
                    with pytest.raises(RunStopped, match="closed"):
                        stage.retain([original(str(uuid4()), "Late original.")])
        finally:
            release_searches.set()
        if not cancel_search:
            receipts = result.result(timeout=3)
            assert len(receipts) == 13
            assert sum(row["tool"] == "search_corpus" for row in receipts) == 12
        assert all_searches_returned.wait(timeout=3)
    assert instance.search_calls == 12
    assert ledger.citation_numbers() == (1, 2)
    assert [action.model_dump(mode="json") for action in actions] == frozen_actions
    assert len(instance.last_receipts) == 13
    assert sum(row["status"] == "truncated" for row in instance.last_receipts) == (
        12 if cancel_search else 0
    )


@pytest.mark.parametrize("prioritize", [False, True])
def test_default_order_and_duplicate_reuse_need_bindings_remain_intact(
    prioritize: bool,
) -> None:
    ledger = EvidenceLedger()
    context = RunContext()
    ledger.add([original(CHUNK_ID, "An observed original.")], context)
    order: list[str] = []

    def search(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        order.append(f"search:{arguments['query']}")
        return ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="Bounded search.")

    def read(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        order.append("provision" if "article" in arguments else "context")
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical original.",
            evidence=[original(CHUNK_ID, "An observed original.")],
        )

    tools = registry(search, read)
    policy = WorkflowPolicy(max_parallel_tools=1)
    if prioritize:
        instance = CanonicalAcquirer(
            tools, context, ledger, policy, prioritize_observed_reads=True
        )
    else:
        instance = CanonicalAcquirer(tools, context, ledger, policy)
    actions = [
        search_action("first"),
        search_action("second"),
        read_action("rule"),
        read_action("condition"),
        SourceAction(
            need_ids=["condition"],
            tool="read_chunk_context",
            arguments={"source_id": SOURCE_ID, "chunk_id": CHUNK_ID},
        ),
    ]
    receipts = instance.acquire(actions, plan())
    assert order == (
        ["provision", "context", "search:first", "search:second"]
        if prioritize
        else ["search:first", "search:second", "provision", "context"]
    )
    assert len(receipts) == 4
    assert next(row for row in receipts if row["tool"] == "read_provision")[
        "need_ids"
    ] == ["rule", "condition"]
    original_calls = list(order)
    reused = instance.acquire([read_action("exception")], plan())
    assert reused[0]["reused"] is True
    assert order == original_calls
    retained = ledger.get(1)
    assert retained is not None
    assert set(retained.question_ids) == {"rule", "condition", "exception"}


@pytest.mark.parametrize(
    "tool,arguments,original_change",
    [
        ("read_provision", {"source_id": str(uuid4()), "article": "17"}, None),
        ("read_provision", {"source_id": SOURCE_ID, "article": 17}, None),
        ("read_provision", {"source_id": SOURCE_ID, "article": ""}, None),
        ("read_chunk", {"source_id": SOURCE_ID, "chunk_id": "unseen"}, None),
        ("read_chunk_context", {"source_id": SOURCE_ID, "chunk_id": 17}, None),
        ("read_provision", {"source_id": SOURCE_ID, "article": "17"}, "derived"),
        ("read_provision", {"source_id": SOURCE_ID, "article": "17"}, "truncated"),
        ("read_provision", {"source_id": SOURCE_ID, "article": "17"}, "binding"),
    ],
)
def test_unobserved_malformed_or_noncanonical_targets_are_not_prioritized(
    tool: str, arguments: dict[str, JsonValue], original_change: str | None
) -> None:
    ledger = EvidenceLedger()
    context = RunContext()
    item = original(CHUNK_ID, "Observed passage.")
    if original_change == "binding":
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_chunk_id"] = "different-original"
    elif original_change is not None:
        item.metadata[original_change] = True
    ledger.add([item], context)
    order: list[str] = []

    def search(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        order.append("search")
        return ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="Bounded search.")

    def read(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        order.append("read")
        return ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="Read attempt.")

    instance = CanonicalAcquirer(
        registry(search, read),
        context,
        ledger,
        WorkflowPolicy(max_parallel_tools=1),
        prioritize_observed_reads=True,
    )
    receipts = instance.acquire(
        [
            search_action(),
            SourceAction(need_ids=["rule"], tool=tool, arguments=arguments),
        ],
        plan(),
    )
    assert order[0] == "search"
    assert len(receipts) == 2
    if not isinstance(arguments.get("article", ""), str) or not isinstance(
        arguments.get("chunk_id", ""), str
    ):
        assert order == ["search"]
        assert any(row["status"] == "invalid" for row in receipts)
