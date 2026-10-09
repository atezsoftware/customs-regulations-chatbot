"""Material discovery retains independent lanes and admits whole physical batches."""

from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from threading import Barrier, Lock
from typing import cast
from unittest.mock import Mock
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
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
from onyx.legal_composite import acquisition
from onyx.legal_composite.acquisition import (
    AcquisitionTimingSnapshot,
    CanonicalAcquirer,
    CanonicalEvidenceStage,
    InvalidSourceAction,
    TaskDurationStats,
)
from onyx.legal_composite.dependencies import (
    CompositeDependencyExpander,
    HostDependencyStage,
    assess_dependencies,
    material_dependency_gaps,
)
from onyx.legal_composite.engine import LegalCompositeEngine, ModelGateway
from onyx.legal_composite.models import (
    AnswerReview,
    DraftAnswer,
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.routing import SourceLaneRouter
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    material_target,
    review,
    setup_composite_expander,
)
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    plan as dependency_plan,
)

Handler = Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
OWNER = UUID(int=100)
SOURCE_IDS = {kind: UUID(int=index + 1) for index, kind in enumerate(SourceKind)}
UNCERTAIN = UUID(int=101)
SECOND_STATUTE = UUID(int=102)


def research_plan() -> ResearchPlan:
    return ResearchPlan(
        language="en",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id=identity,
                question=f"Which fictional source supports issue {identity}?",
                governing_source="Fictional instrument",
                conditions_to_check=[],
                source_kinds=[SourceKind.STATUTE],
            )
            for identity in ("a", "b")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def search(query: str = "fictional dependency", need: str = "a") -> SourceAction:
    return SourceAction(
        need_ids=[need],
        tool="search_corpus",
        arguments={
            "query": query,
            "coverage_item": "observed relation",
            "evidence_target": "operative source",
        },
    )


def catalogue() -> SourceLaneCatalogue:
    records = [
        SourceClassification(
            source_id=source_id,
            name=f"Fictional {kind.value} source",
            kind=kind,
            method="synthetic",
            uncertain=False,
            observed_document_types=(),
        )
        for kind, source_id in SOURCE_IDS.items()
    ]
    records.extend(
        SourceClassification(
            source_id=source_id,
            name=f"Fictional source {source_id.int}",
            kind=SourceKind.STATUTE,
            method="synthetic",
            uncertain=uncertain,
            observed_document_types=(),
        )
        for source_id, uncertain in ((UNCERTAIN, True), (SECOND_STATUTE, False))
    )
    return SourceLaneCatalogue(
        user_id=OWNER,
        scope_sha256="frozen-synthetic-authorized-scope",
        records=tuple(records),
        complete=True,
    )


def evidence(source_id: UUID) -> EvidenceItem:
    text = f"Complete fictional source {source_id.int}. " * 500
    return EvidenceItem(
        source_id=str(source_id),
        chunk_id=f"synthetic-{source_id.int}",
        text=text,
        search_doc=SearchDoc(
            document_id=str(source_id),
            chunk_ind=0,
            semantic_identifier=f"Fictional source {source_id.int}",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={},
            match_highlights=[],
        ),
    )


def no_result(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.NOT_FOUND,
        summary="Synthetic empty candidate window; corpus absence is not established.",
    )


def source_priority(source_id: UUID) -> int:
    return {
        SOURCE_IDS[SourceKind.STATUTE]: 100,
        SECOND_STATUTE: 99,
        SOURCE_IDS[SourceKind.UNKNOWN]: -1,
        UNCERTAIN: -2,
    }.get(source_id, 10)


def spec(name: str, handler: Handler) -> ToolSpec:
    if name == "search_corpus":
        properties: dict[str, JsonValue] = {
            "query": {"type": "string", "minLength": 1},
            "coverage_item": {"type": "string"},
            "evidence_target": {"type": "string"},
            "expand_query": {"type": "boolean"},
        }
        required = ["query", "coverage_item", "evidence_target"]
    elif name == "dependency_related_sources":
        properties = {"edge_id": {"type": "string"}}
        required = ["edge_id"]
    elif name == "resolve_source":
        properties = {"query": {"type": "string"}}
        required = ["query"]
    else:
        properties = {
            "source_id": {"type": "string"},
            "article": {"type": "string"},
        }
        required = ["source_id", "article"]
    return ToolSpec(
        name=name,
        description="Controlled fictional canonical capability.",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        handler=handler,
    )


@dataclass(frozen=True)
class Invocation:
    tool: str
    kind: SourceKind | None
    arguments: dict[str, JsonValue]
    need_ids: tuple[str, ...]


@dataclass
class Harness:
    acquirer: CanonicalAcquirer
    router: SourceLaneRouter
    calls: list[Invocation] = field(default_factory=list)


def harness(
    *,
    opt_in: bool = True,
    return_originals: bool = False,
    max_search_calls: int = 96,
    max_tools: int = 192,
    search_barrier: Barrier | None = None,
) -> Harness:
    frozen_catalogue = catalogue()
    calls: list[Invocation] = []
    lock = Lock()

    def handler_for(name: str, kind: SourceKind | None) -> Handler:
        def execute(
            arguments: dict[str, JsonValue], context: RunContext
        ) -> ToolOutcome:
            stage = context.services["legal_composite_original_stage"]
            assert isinstance(stage, CanonicalEvidenceStage)
            with lock:
                calls.append(
                    Invocation(name, kind, deepcopy(arguments), stage.need_ids)
                )
            if (
                name == "search_corpus"
                and kind is not None
                and search_barrier is not None
            ):
                search_barrier.wait(timeout=3)
            if not return_originals or kind is None or name != "search_corpus":
                return no_result(arguments, context)
            allowed = frozen_catalogue.source_ids(kind)
            selected = max(allowed, key=source_priority)
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="One independently ranked source in this authorized lane.",
                evidence=[evidence(selected)],
            )

        return execute

    def build(kind: SourceKind) -> CapabilityRegistry:
        return CapabilityRegistry(
            [spec("search_corpus", handler_for("search_corpus", kind))]
        )

    router = SourceLaneRouter(frozen_catalogue, build)
    host_registry = CapabilityRegistry(
        spec(name, handler_for(name, None))
        for name in (
            "search_corpus",
            "dependency_related_sources",
            "resolve_source",
            "read_provision",
        )
    )
    acquirer = CanonicalAcquirer(
        host_registry,
        RunContext(
            corpus_only=True,
            budget=SharedBudget(max_tools=max_tools, max_inflight_tools=12),
        ),
        EvidenceLedger(),
        WorkflowPolicy(
            max_parallel_tools=12,
            max_search_calls=max_search_calls,
            max_tools=max_tools,
        ),
        registry_for_action=router.registry,
        expand_actions=router.expand,
        host_registry=host_registry,
        source_kinds={str(row.source_id): row.kind for row in frozen_catalogue.records},
        expand_host_search_lanes=opt_in,
    )
    return Harness(acquirer, router, calls)


def mixed_actions() -> list[SourceAction]:
    return [
        search("query one"),
        search("query two", "b"),
        SourceAction(
            need_ids=["a"],
            tool="dependency_related_sources",
            arguments={"edge_id": "relation-one"},
        ),
        SourceAction(
            need_ids=["a"],
            tool="resolve_source",
            arguments={"query": "source identity"},
        ),
        SourceAction(
            need_ids=["b"],
            tool="read_provision",
            arguments={
                "source_id": str(SOURCE_IDS[SourceKind.STATUTE]),
                "article": "7",
            },
        ),
    ]


def forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("Physical preflight performed acquisition work")


def test_independent_candidate_windows_recover_family_below_pooled_top_window() -> None:
    instance = harness(return_originals=True, search_barrier=Barrier(12))
    frozen_catalogue = instance.router.catalogue
    assert isinstance(frozen_catalogue, SourceLaneCatalogue)
    unknown = SOURCE_IDS[SourceKind.UNKNOWN]
    for kind in SourceKind:
        assert {unknown, UNCERTAIN} <= set(frozen_catalogue.source_ids(kind))
    pooled_top_two = set(
        sorted(
            (row.source_id for row in frozen_catalogue.records),
            key=source_priority,
            reverse=True,
        )[:2]
    )
    assert pooled_top_two == {SOURCE_IDS[SourceKind.STATUTE], SECOND_STATUTE}
    judicial = SOURCE_IDS[SourceKind.JUDICIAL_DECISION]
    assert judicial not in pooled_top_two

    receipts = instance.acquirer.acquire_host_actions([search()], research_plan())

    assert Counter(call.kind for call in instance.calls) == Counter(SourceKind)
    assert len(receipts) == 12
    judicial_receipt = next(
        row for row in receipts if row["source_kind"] == SourceKind.JUDICIAL_DECISION
    )
    numbers = judicial_receipt["citations"]
    assert isinstance(numbers, list) and len(numbers) == 1
    number = numbers[0]
    assert isinstance(number, int)
    original = instance.acquirer.ledger.get(number)
    assert original is not None and original.source_id == str(judicial)
    assert original.text == evidence(judicial).text and len(original.text) > 10_000
    assert original.question_ids == ["a"]
    assert judicial_receipt["host_arguments"] == {
        **search().arguments,
        "expand_query": False,
    }


def test_mixed_host_batch_has_exact_physical_counts_and_keeps_capabilities_unexpanded() -> (
    None
):
    instance = harness()
    actions = mixed_actions()
    expected = {
        "search_corpus": 24,
        "dependency_related_sources": 1,
        "resolve_source": 1,
        "read_provision": 1,
    }
    assert (
        instance.acquirer.pending_host_call_counts(actions, research_plan()) == expected
    )
    assert (
        ceil(sum(expected.values()) / instance.acquirer.policy.max_parallel_tools) == 3
    )
    receipts = instance.acquirer.acquire_host_actions(actions, research_plan())
    assert Counter(call.tool for call in instance.calls) == expected
    assert Counter(str(row["tool"]) for row in receipts) == expected
    for tool in ("dependency_related_sources", "resolve_source", "read_provision"):
        call = next(call for call in instance.calls if call.tool == tool)
        original = next(action for action in actions if action.tool == tool)
        assert call.kind is None and call.arguments == original.arguments
        assert call.need_ids == tuple(original.need_ids)
    for query, need in (("query one", "a"), ("query two", "b")):
        calls = [
            call for call in instance.calls if call.arguments.get("query") == query
        ]
        assert Counter(call.kind for call in calls) == Counter(SourceKind)
        assert all(call.need_ids == (need,) for call in calls)


def test_preflight_does_not_mutate_ledger_budget_receipts_or_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness()
    acquirer = instance.acquirer
    actions = mixed_actions()
    plan = research_plan()
    frozen_actions = deepcopy(actions)
    frozen_plan = deepcopy(plan)
    before_budget = acquirer.context.budget.snapshot()
    with monkeypatch.context() as guard:
        for name in ("_dispatch_action", "_scheduled_calls", "_bind_originals"):
            guard.setattr(acquirer, name, forbidden)
        for name in ("add", "get", "provision_metadata"):
            guard.setattr(acquirer.ledger, name, forbidden)
        guard.setattr(acquirer.context.budget, "snapshot", forbidden)
        assert acquirer.pending_host_call_counts(actions, plan)["search_corpus"] == 24
    assert instance.calls == []
    assert acquirer.context.budget.snapshot() == before_budget
    assert acquirer.search_calls == 0 and acquirer._completed == {}
    assert acquirer.last_receipts == [] and acquirer.ledger.citation_numbers() == ()
    assert actions == frozen_actions and plan == frozen_plan


def test_duplicate_and_completed_calls_reuse_all_lanes_without_early_rebinding() -> (
    None
):
    instance = harness(return_originals=True)
    acquirer = instance.acquirer
    first = search("same query", "a")
    first.arguments.update({"_public_update": "first", "expand_query": True})
    assert acquirer.pending_host_call_counts([first], research_plan()) == {
        "search_corpus": 12
    }
    acquirer.acquire_host_actions([first], research_plan())
    assert len(instance.calls) == 12
    originals_before = acquirer.ledger.serialize_records(
        acquirer.ledger.citation_numbers()
    )
    completed_before = deepcopy(acquirer._completed)
    budget_before = acquirer.context.budget.snapshot()
    second = search("same query", "b")
    second.arguments.update({"expand_query": False, "_public_update": "second"})
    assert acquirer.pending_host_call_counts([second], research_plan()) == {}
    assert (
        acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers())
        == originals_before
    )
    assert acquirer._completed == completed_before
    assert acquirer.context.budget.snapshot() == budget_before
    receipts = acquirer.acquire_host_actions([second], research_plan())
    assert len(instance.calls) == 12 and acquirer.search_calls == 12
    assert len(receipts) == 12 and all(row["reused"] is True for row in receipts)
    for number in acquirer.ledger.citation_numbers():
        original = acquirer.ledger.get(number)
        assert original is not None and original.question_ids == ["a", "b"]

    third = search("fresh query", "a")
    duplicate = search("fresh query", "b")
    assert acquirer.pending_host_call_counts([third, duplicate], research_plan()) == {
        "search_corpus": 12
    }
    acquirer.acquire_host_actions([third, duplicate], research_plan())
    assert len(instance.calls) == 24
    assert all(call.need_ids == ("a", "b") for call in instance.calls[12:])


def test_reranking_arguments_remain_distinct_in_counting_and_dispatch() -> None:
    instance = harness()
    base = search()
    coverage = base.model_copy(deep=True)
    coverage.arguments["coverage_item"] = "different material issue"
    evidence_target = base.model_copy(deep=True)
    evidence_target.arguments["evidence_target"] = "different decisive passage"
    actions = [base, coverage, evidence_target]
    assert instance.acquirer.pending_host_call_counts(actions, research_plan()) == {
        "search_corpus": 36
    }
    instance.acquirer.acquire_host_actions(actions, research_plan())
    assert len(instance.calls) == 36
    assert Counter(call.arguments["coverage_item"] for call in instance.calls) == {
        "observed relation": 24,
        "different material issue": 12,
    }
    assert Counter(call.arguments["evidence_target"] for call in instance.calls) == {
        "operative source": 24,
        "different decisive passage": 12,
    }


@pytest.mark.parametrize("exhausted", ["searches", "tools"])
def test_expanded_whole_batch_hits_global_bound_before_any_dispatch(
    exhausted: str,
) -> None:
    instance = harness(return_originals=True)
    acquirer = instance.acquirer
    retained = evidence(SOURCE_IDS[SourceKind.STATUTE])
    retained.question_ids = ["a"]
    acquirer.ledger.add([retained], acquirer.context)
    before = acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers())
    if exhausted == "searches":
        acquirer.search_calls = 85
    else:
        acquirer.context.budget.record_execution("tools", 181)
    before_budget = acquirer.context.budget.snapshot()
    before_searches = acquirer.search_calls
    assert acquirer.pending_host_call_counts([search()], research_plan()) == {
        "search_corpus": 12
    }
    with pytest.raises(RunStopped, match="budget exhausted"):
        acquirer.acquire_host_actions([search()], research_plan())
    assert instance.calls == [] and acquirer._completed == {}
    assert acquirer.context.budget.snapshot() == before_budget
    assert acquirer.search_calls == before_searches
    assert (
        acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers()) == before
    )


def dependency_fixture() -> CompositeDependencyExpander:
    value, _ledger, _calls, _broker, _candidate, _governing, _old_expand = (
        setup_composite_expander()
    )
    expander = cast(CompositeDependencyExpander, value)
    acquirer = expander.acquirer
    typed_registry = acquirer.registry_for_action(
        SourceAction(need_ids=["permit"], tool="search_corpus", arguments={})
    )
    router = SourceLaneRouter(catalogue(), lambda _kind: typed_registry)
    acquirer.expand_actions = router.expand
    acquirer.registry_for_action = router.registry
    acquirer.expand_host_search_lanes = True
    acquirer.policy = WorkflowPolicy(
        max_tools=192, max_search_calls=96, max_parallel_tools=12
    )
    return expander


def test_two_stage_material_discovery_keeps_query_edge_and_issue_binding() -> None:
    expander = dependency_fixture()
    observed_stages: list[tuple[HostDependencyStage, dict[str, int]]] = []

    def admit(
        actions: list[SourceAction], plan: ResearchPlan, stage: HostDependencyStage
    ) -> bool:
        observed_stages.append(
            (stage, expander.acquirer.pending_host_call_counts(actions, plan))
        )
        return True

    expander.host_stage_admission = admit
    edges = expander.expand(
        dependency_plan(), frontier={1}, material_targets=[material_target()]
    )
    assert len(edges) == 1
    edge = edges[0]
    assert observed_stages == [
        (
            "discovery",
            {"search_corpus": 24, "dependency_related_sources": 1, "resolve_source": 1},
        ),
        ("provisions", {"read_provision": 1}),
    ]
    searches = [
        receipt
        for receipt in expander.receipts
        if receipt.get("tool") == "search_corpus"
    ]
    assert len(searches) == 24
    queries = {"faaliyet kanun 27", "8917 27"}
    for query in queries:
        receipts = [
            row
            for row in searches
            if cast(dict[str, JsonValue], row["host_arguments"])["query"] == query
        ]
        assert Counter(row["source_kind"] for row in receipts) == Counter(SourceKind)
        assert all(row["need_ids"] == ["permit"] for row in receipts)
    assert edge.governing_citations and edge.candidate_citations
    assert all(
        expander._candidate_edges[source_id] == {edge.edge_id}
        for source_id in edge.candidate_source_ids
    )
    for number in edge.governing_citations + edge.candidate_citations:
        original = expander.ledger.get(number)
        assert original is not None and original.question_ids == ["permit"]
    assert material_dependency_gaps(
        edges, expander.ledger, set(expander.ledger.citation_numbers())
    ) == {edge.edge_id: []}
    assert expander._expanded[edge.edge_id] == expander._edge_state(edge)


@pytest.mark.parametrize("denied_stage", ["discovery", "provisions"])
def test_deferred_stage_never_closes_material_relation_or_passes_original_review(
    denied_stage: str,
) -> None:
    expander = dependency_fixture()
    stages: list[HostDependencyStage] = []

    def admit(
        _actions: list[SourceAction], _plan: ResearchPlan, stage: HostDependencyStage
    ) -> bool:
        stages.append(stage)
        return stage != denied_stage

    expander.host_stage_admission = admit
    edges = expander.expand(
        dependency_plan(), frontier={1}, material_targets=[material_target()]
    )
    edge = edges[0]
    assert stages == (
        ["discovery"] if denied_stage == "discovery" else ["discovery", "provisions"]
    )
    assert edge.edge_id not in expander._expanded
    assert any(denied_stage in gap for gap in edge.discovery_gaps)
    assert not edge.governing_citations
    gaps = material_dependency_gaps(
        edges, expander.ledger, set(expander.ledger.citation_numbers())
    )
    assert "discovery_gap_unresolved" in gaps[edge.edge_id]
    deferred = next(
        row
        for row in expander.receipts
        if row.get("status") == "dependency_stage_deferred"
    )
    assert deferred["stage"] == denied_stage and deferred["absence_proven"] is False
    if denied_stage == "discovery":
        assert expander.acquirer.search_calls == 0
        assert expander.context.budget.snapshot()["tools"] == 0
        assert expander.ledger.citation_numbers() == (1,)
    else:
        assert expander.acquirer.search_calls == 24 and edge.candidate_citations
    complete, safe, original_gaps = assess_dependencies(
        edges,
        dependency_plan(),
        DraftAnswer(
            answer="A complete source examination is claimed. [1]",
            unresolved_need_ids=[],
        ),
        cast(AnswerReview, review(edge)),
        expander.ledger,
        set(expander.ledger.citation_numbers()),
    )
    assert not complete and not safe and original_gaps


def engine_fixture(instance: Harness, *, enabled: bool = True) -> LegalCompositeEngine:
    acquirer = instance.acquirer
    policy = acquirer.policy.model_copy(
        update={"max_call_seconds": 75.0, "selection_reserve_seconds": 12.0}
    )
    acquirer.policy = policy
    controlled_host_registry = acquirer.host_registry
    acquirer.host_registry = CapabilityRegistry()
    expander = CompositeDependencyExpander(
        broker=cast(CorpusBroker, Mock()),
        acquirer=acquirer,
        ledger=acquirer.ledger,
        context=acquirer.context,
        source_kinds={},
    )
    acquirer.host_registry = controlled_host_registry
    engine = LegalCompositeEngine(
        gateway=cast(ModelGateway, Mock()),
        acquirer=acquirer,
        ledger=acquirer.ledger,
        policy=policy,
        check_active=acquirer.context.check_active,
        research_available=lambda: True,
        dependency_expander=expander,
        use_physical_dependency_runway=enabled,
    )
    engine._research_seconds = [8.0]
    engine.plan = research_plan()
    return engine


@pytest.mark.parametrize(
    "remaining,accepted", [(319.0, False), (320.0, False), (321.0, True)]
)
def test_discovery_admission_counts_three_physical_waves_and_subsequent_reserves(
    monkeypatch: pytest.MonkeyPatch, remaining: float, accepted: bool
) -> None:
    instance = harness()
    engine = engine_fixture(instance)
    monkeypatch.setattr(engine, "_repair_runway", lambda: remaining)
    before_budget = instance.acquirer.context.budget.snapshot()
    # 27 physical calls require three 75-second waves, plus a future provision
    # wave, eight seconds of source reading and twelve seconds of selection.
    assert (
        engine._admit_dependency_stage(mixed_actions(), research_plan(), "discovery")
        is accepted
    )
    assert instance.calls == [] and instance.acquirer._completed == {}
    assert instance.acquirer.context.budget.snapshot() == before_budget


def test_settled_tool_maximum_drives_forecast_while_failed_timing_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness()
    engine = engine_fixture(instance)
    snapshot = AcquisitionTimingSnapshot(
        completed={
            "search_corpus": TaskDurationStats(
                count=1, total_seconds=102.0, max_seconds=102.0
            ),
            "read_provision": TaskDurationStats(
                count=1, total_seconds=6.0, max_seconds=6.0
            ),
        },
        diagnostics={
            "search_corpus": {
                "cancelled": TaskDurationStats(
                    count=1, total_seconds=600.0, max_seconds=600.0
                )
            }
        },
    )
    monkeypatch.setattr(instance.acquirer, "task_timing_snapshot", lambda: snapshot)
    engine._source_seconds["search_corpus"] = 1.0
    # Three completed search waves, one exact future read wave, source inspection
    # and selection total 332 seconds. Cancelled timing cannot inflate that sum.
    monkeypatch.setattr(engine, "_repair_runway", lambda: 332.0)
    assert not engine._admit_dependency_stage(
        mixed_actions(), research_plan(), "discovery"
    )
    monkeypatch.setattr(engine, "_repair_runway", lambda: 333.0)
    assert engine._admit_dependency_stage(mixed_actions(), research_plan(), "discovery")
    assert instance.calls == []


def test_exact_provision_stage_recounts_known_reads_without_a_second_discovery_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness()
    engine = engine_fixture(instance)
    actions = [
        SourceAction(
            need_ids=["a"],
            tool="read_provision",
            arguments={
                "source_id": str(SOURCE_IDS[SourceKind.STATUTE]),
                "article": str(index),
            },
        )
        for index in range(13)
    ]
    assert instance.acquirer.pending_host_call_counts(actions, research_plan()) == {
        "read_provision": 13
    }
    monkeypatch.setattr(engine, "_repair_runway", lambda: 170.0)
    assert not engine._admit_dependency_stage(actions, research_plan(), "provisions")
    monkeypatch.setattr(engine, "_repair_runway", lambda: 171.0)
    assert engine._admit_dependency_stage(actions, research_plan(), "provisions")
    assert instance.calls == []


def test_reserved_normal_work_is_deduplicated_but_still_subject_to_physical_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness()
    engine = engine_fixture(instance)
    monkeypatch.setattr(engine, "_repair_runway", lambda: 1_000.0)
    actions = [search("same", "a")]
    instance.acquirer.search_calls = 84
    engine._dependency_reserved_actions = [search("same", "b")]
    assert engine._admit_dependency_stage(actions, research_plan(), "discovery")
    engine._dependency_reserved_actions = [search("distinct", "b")]
    assert not engine._admit_dependency_stage(actions, research_plan(), "discovery")
    assert instance.calls == [] and instance.acquirer.search_calls == 84


def test_completed_discovery_reuses_receipts_and_waits_for_exact_second_stage_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness()
    engine = engine_fixture(instance)
    actions = mixed_actions()
    instance.acquirer.acquire_host_actions(actions, research_plan())
    assert instance.acquirer.pending_host_call_counts(actions, research_plan()) == {}
    monkeypatch.setattr(engine, "_repair_runway", lambda: 21.0)
    assert engine._admit_dependency_stage(actions, research_plan(), "discovery")
    assert len(instance.calls) == 27


def test_legacy_engine_does_not_install_physical_admission_callback() -> None:
    engine = engine_fixture(harness(opt_in=False), enabled=False)
    expander = engine.dependency_expander
    assert isinstance(expander, CompositeDependencyExpander)
    assert expander.host_stage_admission is None


@pytest.mark.parametrize(
    "invalid", ["unknown need", "invalid host arguments", "unavailable host tool"]
)
def test_invalid_later_host_action_rejects_batch_before_earlier_search_dispatch(
    invalid: str,
) -> None:
    instance = harness()
    invalid_action = SourceAction(
        need_ids=["missing"] if invalid == "unknown need" else ["a"],
        tool="unavailable" if invalid == "unavailable host tool" else "resolve_source",
        arguments={"query": 7}
        if invalid == "invalid host arguments"
        else {"query": "source"},
    )
    actions = [search(), invalid_action]
    with pytest.raises(InvalidSourceAction):
        instance.acquirer.pending_host_call_counts(actions, research_plan())
    with pytest.raises(InvalidSourceAction):
        instance.acquirer.acquire_host_actions(actions, research_plan())
    assert instance.calls == [] and instance.acquirer.search_calls == 0
    assert instance.acquirer.context.budget.snapshot()["tools"] == 0
    assert instance.acquirer.ledger.citation_numbers() == ()


def test_legacy_host_actions_keep_one_pooled_search_and_unexpanded_capabilities() -> (
    None
):
    instance = harness(opt_in=False)
    actions = mixed_actions()
    expected = {
        "search_corpus": 2,
        "dependency_related_sources": 1,
        "resolve_source": 1,
        "read_provision": 1,
    }
    assert (
        instance.acquirer.pending_host_call_counts(actions, research_plan()) == expected
    )
    instance.acquirer.acquire_host_actions(actions, research_plan())
    assert Counter(call.tool for call in instance.calls) == expected
    assert all(call.kind is None for call in instance.calls)


@pytest.mark.parametrize("rejection", ["later invalid schema", "later search budget"])
def test_rejected_batch_never_rebinds_previously_completed_originals(
    rejection: str,
) -> None:
    instance = harness(return_originals=True)
    acquirer = instance.acquirer
    acquirer.acquire_host_actions([search("completed", "a")], research_plan())
    completed_originals = acquirer.ledger.serialize_records(
        acquirer.ledger.citation_numbers()
    )
    completed_receipts = deepcopy(acquirer._completed)
    actions = [search("completed", "b")]
    if rejection == "later invalid schema":
        actions.append(
            SourceAction(
                need_ids=["b"], tool="resolve_source", arguments={"query": False}
            )
        )
        error: type[RunStopped] = InvalidSourceAction
    else:
        acquirer.search_calls = 85
        actions.append(search("fresh", "b"))
        error = RunStopped
    before_budget = acquirer.context.budget.snapshot()
    before_searches = acquirer.search_calls
    with pytest.raises(error):
        acquirer.acquire_host_actions(actions, research_plan())
    assert len(instance.calls) == 12
    assert (
        acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers())
        == completed_originals
    )
    assert acquirer._completed == completed_receipts
    assert acquirer.context.budget.snapshot() == before_budget
    assert acquirer.search_calls == before_searches


@pytest.mark.parametrize(
    "bad_expansion",
    ["omitted lane", "duplicate lane", "changed query", "changed issue"],
)
def test_fanout_contract_rejects_partial_or_rewritten_lanes_before_dispatch(
    bad_expansion: str,
) -> None:
    instance = harness()

    def invalid_expand(
        actions: list[SourceAction], plan: ResearchPlan
    ) -> list[SourceAction]:
        expanded = instance.router.expand(actions, plan)
        if bad_expansion == "omitted lane":
            return expanded[:-1]
        if bad_expansion == "duplicate lane":
            return [*expanded[:-1], expanded[0]]
        if bad_expansion == "changed query":
            expanded[0] = expanded[0].model_copy(
                update={"arguments": {"query": "different"}}
            )
        else:
            expanded[0] = expanded[0].model_copy(update={"need_ids": ["b"]})
        return expanded

    instance.acquirer.expand_actions = invalid_expand
    with pytest.raises(InvalidSourceAction, match="every independent source lane"):
        instance.acquirer.pending_host_call_counts([search()], research_plan())
    with pytest.raises(InvalidSourceAction, match="every independent source lane"):
        instance.acquirer.acquire_host_actions([search()], research_plan())
    assert instance.calls == [] and instance.acquirer.search_calls == 0
    assert instance.acquirer.ledger.citation_numbers() == ()


@pytest.mark.parametrize("unsafe", ["external", "orchestrates"])
def test_material_host_preflight_rejects_noncanonical_capability_before_any_dispatch(
    unsafe: str,
) -> None:
    instance = harness()
    unsafe_spec = spec("unsafe_host", no_result).model_copy(update={unsafe: True})
    instance.acquirer.host_registry.register(unsafe_spec)
    action = SourceAction(
        need_ids=["a"],
        tool="unsafe_host",
        arguments={"source_id": str(SOURCE_IDS[SourceKind.STATUTE]), "article": "7"},
    )
    with pytest.raises(InvalidSourceAction, match="allowlist"):
        instance.acquirer.pending_host_call_counts([search(), action], research_plan())
    with pytest.raises(InvalidSourceAction, match="allowlist"):
        instance.acquirer.acquire_host_actions([search(), action], research_plan())
    assert instance.calls == [] and instance.acquirer.search_calls == 0
    assert instance.acquirer.ledger.citation_numbers() == ()


def test_opt_in_requires_a_real_lane_expander_and_boolean_flag() -> None:
    arguments = (CapabilityRegistry(), RunContext(), EvidenceLedger(), WorkflowPolicy())
    with pytest.raises(ValueError):
        CanonicalAcquirer(*arguments, expand_host_search_lanes=True)
    with pytest.raises(ValueError, match="boolean"):
        CanonicalAcquirer(*arguments, expand_host_search_lanes=cast(bool, 1))


def test_cancelled_parallel_fanout_keeps_completed_original_and_rejects_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = harness(return_originals=True)
    acquirer = instance.acquirer
    pending: list[Future[ToolOutcome]] = []
    children: list[RunContext] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    def check_research() -> None:
        if acquirer.ledger.citation_numbers():
            raise RunStopped("Synthetic cancellation after the first completed source")

    class ControlledExecutor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            assert max_workers == 12 and thread_name_prefix == "legal-composite-source"
            self.submitted = 0

        def submit(
            self, function: Callable[..., ToolOutcome], *arguments: object
        ) -> Future[ToolOutcome]:
            self.submitted += 1
            child = arguments[3]
            assert isinstance(child, RunContext)
            children.append(child)
            future: Future[ToolOutcome] = Future()
            if self.submitted == 1:
                future.set_result(function(*arguments))
            else:
                pending.append(future)
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(acquisition, "ThreadPoolExecutor", ControlledExecutor)
    monkeypatch.setattr(acquirer.context, "check_research_active", check_research)
    with pytest.raises(RunStopped, match="Synthetic cancellation"):
        acquirer.acquire_host_actions([search()], research_plan())
    assert len(instance.calls) == 1 and len(pending) == 11
    assert all(future.cancelled() for future in pending)
    assert shutdown_calls == [(False, True)]
    receipts = acquirer.last_receipts
    assert len(receipts) == 12
    assert Counter(row["status"] for row in receipts) == {"found": 1, "truncated": 11}
    assert len(acquirer.ledger.citation_numbers()) == 1
    before = acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers())
    late_stage = children[-1].services["legal_composite_original_stage"]
    assert isinstance(late_stage, CanonicalEvidenceStage)
    with pytest.raises(RunStopped, match="closed"):
        late_stage.retain([evidence(SOURCE_IDS[SourceKind.JUDICIAL_DECISION])])
    assert (
        acquirer.ledger.serialize_records(acquirer.ledger.citation_numbers()) == before
    )
