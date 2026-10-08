import math
from threading import Barrier
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import SourceKind, SourceLaneCatalogue
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import (
    LegalCompositeEngine,
    ModelGateway,
    SourceAcquirer,
    initial_discovery_actions,
)
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.legal_composite.selection import SourceSelectionResult
from tests.unit.onyx.legal_composite.test_review_assessment import original


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="n1",
                question="Which conditions apply?",
                governing_source="original",
                conditions_to_check=["exception"],
                source_kinds=[SourceKind.STATUTE, SourceKind.PRESIDENTIAL_DECREE],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def test_discovery_searches_every_kind_despite_planner_type_restrictions() -> None:
    catalogue = SourceLaneCatalogue(
        user_id=uuid4(), scope_sha256="scope", records=(), complete=True
    )
    router = SourceLaneRouter(catalogue, lambda _kind: CapabilityRegistry([]))
    actions = router.expand(
        [
            SourceAction(
                need_ids=["n1"],
                tool="search_corpus",
                arguments={},
                source_kind=SourceKind.STATUTE,
            )
        ],
        plan(),
    )
    assert [action.source_kind for action in actions] == list(SourceKind)


def test_direct_read_plan_cannot_skip_all_kind_discovery() -> None:
    frozen_plan = plan().model_copy(update={"discovery_query": "Focused legal issue"})
    actions = initial_discovery_actions(frozen_plan, "Complete user request")
    assert len(actions) == 1 and actions[0].tool == "search_corpus"
    assert actions[0].need_ids == ["n1"]
    assert actions[0].arguments["query"] == "Focused legal issue"


def test_existing_discovery_is_not_repeated() -> None:
    search = SourceAction(
        need_ids=["n1"], tool="search_corpus", arguments={"query": "issue"}
    )
    frozen_plan = plan().model_copy(update={"initial_actions": [search]})
    assert initial_discovery_actions(frozen_plan, "request") == [search]


def test_independent_kind_registries_actually_overlap_and_report_progress() -> None:
    barrier = Barrier(len(SourceKind))

    def search(_arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_research_active()
        barrier.wait(timeout=2)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original acquired")

    def registry() -> CapabilityRegistry:
        return CapabilityRegistry(
            [
                ToolSpec(
                    name="search_corpus",
                    description="search",
                    parameters={"type": "object", "properties": {}},
                    handler=search,
                )
            ]
        )

    registries = {kind: registry() for kind in SourceKind}
    progress: list[tuple[int, int]] = []

    def choose_registry(action: SourceAction) -> CapabilityRegistry:
        assert action.source_kind is not None
        return registries[action.source_kind]

    acquirer = CanonicalAcquirer(
        registry(),
        RunContext(),
        EvidenceLedger(),
        WorkflowPolicy(
            max_parallel_tools=len(SourceKind), max_search_calls=len(SourceKind)
        ),
        registry_for_action=choose_registry,
        on_batch_progress=lambda _actions, pending, completed: progress.append(
            (pending, completed)
        ),
    )
    actions = [
        SourceAction(
            need_ids=["n1"], tool="search_corpus", arguments={}, source_kind=kind
        )
        for kind in registries
    ]
    receipts = acquirer.acquire(actions, plan())
    assert {row["source_kind"] for row in receipts} == set(registries)
    assert progress[0] == (len(SourceKind), 0) and progress[-1] == (0, len(SourceKind))
    assert acquirer.search_calls == len(SourceKind)


def test_selected_uncited_exception_is_required_before_the_first_draft() -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            original("Governing rule", "rule"),
            original("Except when condition holds", "exception"),
        ],
        RunContext(),
    )
    engine = LegalCompositeEngine(
        gateway=cast(ModelGateway, MagicMock()),
        acquirer=cast(SourceAcquirer, MagicMock()),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: False,
    )
    engine.selection = SourceSelectionResult(
        protected_citations=[1, 2],
        background_citations=[],
        rejected_citations=[],
        retained_citations=[1, 2],
        selection_complete=True,
        gaps=[],
        identities=[],
        receipts=[],
        call_id="decision",
    )
    payload = engine._payload("question", "", source_phase=False)
    assert payload["required_evidence_numbers"] == [1, 2]
    records = payload["original_evidence"]
    assert isinstance(records, list)
    assert {row["citation"] for row in records if isinstance(row, dict)} == {1, 2}
    ledger.add([original("New contrary original " * 4000, "contrary")], RunContext())
    expanded = engine._payload("question", "", source_phase=False)
    originals = expanded["original_evidence"]
    assert isinstance(originals, list)
    assert any(
        isinstance(row, dict) and row["text"] == "New contrary original " * 4000
        for row in originals
    )


def test_selection_time_is_retained_without_spending_writer_review_time() -> None:
    budget = WorkflowBudget(WorkflowPolicy(), clock=lambda: 0)
    budget.retain_selection_time(12)
    assert budget.remaining_seconds() == 68
    budget.begin_selection()
    assert budget.remaining_seconds() == 80
    assert budget.remaining_seconds(finalizing=True) == 120


def test_unbounded_experiment_tracks_cost_without_duration_or_spend_admission() -> None:
    budget = WorkflowBudget(
        WorkflowPolicy(timeout_seconds=math.inf, max_cost_usd=math.inf),
        clock=lambda: 0,
        deadline=math.inf,
    )
    budget.configure_finalization(10, 10, 100)
    reservation = budget.request(100, 100, 100, 100)
    budget.settle(reservation, 100, 100)
    assert budget.remaining_seconds() == math.inf
    assert budget.snapshot()["estimated_cost_usd"] == 0.02
    assert budget.snapshot()["elapsed_seconds"] == 0
