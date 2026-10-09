"""A terminal read preserves verified evidence and every deferred obligation."""

from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.models import RunContext, RunStopped
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.budget import ResearchPhaseClosed, WorkflowBudget
from onyx.legal_composite.dependencies import material_dependency_gaps
from onyx.legal_composite.engine import LegalCompositeEngine, ModelGateway
from onyx.legal_composite.models import (
    CompositeWorkflowResult,
    IssueResearchPlan,
    IssueResearchStep,
    SourceAction,
    SourceRequirement,
    SpanSupport,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import AnswerReviewer
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.legal_composite.selection import (
    SelectionObservation,
    SourceSelectionRequest,
    SourceSelector,
)
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    original,
    search_action,
)
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    plan as base_plan,
)
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    material_target,
)
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    plan as dependency_plan,
)
from tests.unit.onyx.legal_composite.test_deferred_material_dependencies import (
    DEFERRED_GAP,
)
from tests.unit.onyx.legal_composite.test_deferred_material_dependencies import (
    fixture as dependency_fixture,
)
from tests.unit.onyx.legal_composite.test_safe_observed_reads import (
    fixture as acquisition_fixture,
)
from tests.unit.onyx.legal_composite.test_safe_observed_reads import (
    read,
)


def fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    remaining: float = 40,
) -> tuple[LegalCompositeEngine, MagicMock, CanonicalAcquirer, IssueResearchPlan]:
    monkeypatch.setattr("onyx.legal_composite.engine.time.monotonic", lambda: 100.0)
    acquirer, ledger, _built, _invoked = acquisition_fixture()
    plan = IssueResearchPlan.model_validate(base_plan().model_dump())
    for need in plan.needs:
        need.evidence_gaps = [f"Unresolved support for {need.need_id}."]
    gateway = MagicMock()
    gateway.last_delivered_citations = {1}
    gateway.budget = None
    instance = LegalCompositeEngine(
        gateway=cast(ModelGateway, gateway),
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(
            max_call_seconds=75, selection_reserve_seconds=12, max_parallel_tools=12
        ),
        check_active=lambda: None,
        research_available=lambda: True,
        evidence_context=cast(RunContext, MagicMock(research_deadline=100 + remaining)),
        reviewer=cast(AnswerReviewer, MagicMock()),
        allow_terminal_observed_reads=enabled,
    )
    instance.plan = plan
    instance._source_seconds = {"search_corpus": 95, "read_provision": 8}
    instance._research_seconds = [60]
    return instance, gateway, acquirer, plan


def step(actions: list[SourceAction]) -> IssueResearchStep:
    return IssueResearchStep(
        actions=actions, ready_to_answer=False, remaining_gaps=["Support is unread."]
    )


def test_default_off_retains_original_all_or_nothing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, acquirer, plan = fixture(monkeypatch, enabled=False)
    partition = MagicMock(side_effect=AssertionError("Default performed partition"))
    acquire = MagicMock(side_effect=AssertionError("Default dispatched sources"))
    monkeypatch.setattr(acquirer, "safe_observed_read_actions", partition)
    monkeypatch.setattr(acquirer, "acquire", acquire)
    proposal = step([search_action(), read("read_provision", "27")])
    assert not instance._try_terminal_observed_reads(proposal, plan, "Full request")
    partition.assert_not_called()
    acquire.assert_not_called()
    assert not instance.deferred_followup_steps
    assert LegalCompositeEngine.__init__.__kwdefaults__ is not None
    assert (
        LegalCompositeEngine.__init__.__kwdefaults__["allow_terminal_observed_reads"]
        is False
    )


def test_terminal_subset_preserves_exact_actions_requirements_and_open_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, acquirer, plan = fixture(monkeypatch)
    item = instance.ledger.get(1)
    assert item is not None
    requirement = SourceRequirement(
        requirement_id="observed_rule",
        need_id="rule",
        dimension="scope",
        rule="An observed rule remains available.",
        application="Its unexamined condition remains open.",
        supports=[SpanSupport(citation=1, quotation=item.text)],
    )
    instance.requirements.update([requirement], plan, {1})
    actions = [
        search_action("Generic rule question"),
        read("read_provision", "27"),
        search_action("Independent exception question"),
    ]
    proposal = step(actions)
    frozen = proposal.model_dump(mode="json")
    frozen_plan = plan.model_dump(mode="json")
    frozen_requirements = instance.requirements.export()
    frozen_evidence = instance.ledger.serialize_records((1,))
    acquire = MagicMock(return_value=[])
    monkeypatch.setattr(acquirer, "acquire", acquire)

    assert not instance._followup_has_runway(proposal)
    assert instance._try_terminal_observed_reads(proposal, plan, "Full request")
    acquired = acquire.call_args.args[0]
    assert len(acquired) == 1 and acquired[0] is actions[1]
    assert acquire.call_args.args[1] is plan
    assert instance.deferred_followup_steps == [
        {
            "actions": [
                actions[0].model_dump(mode="json"),
                actions[2].model_dump(mode="json"),
            ],
            "material_dependencies": [],
            "status": "unexecuted_navigation",
            "reason": "research_runway",
            "navigation_only": True,
        }
    ]
    assert acquirer.pending_call_counts([actions[0], actions[2]], plan) == {
        "search_corpus": 24
    }
    assert proposal.model_dump(mode="json") == frozen
    assert plan.model_dump(mode="json") == frozen_plan
    assert instance.requirements.export() == frozen_requirements
    assert instance.ledger.serialize_records((1,)) == frozen_evidence


@pytest.mark.parametrize("remaining,expected", [(20, False), (20.1, True)])
def test_terminal_runway_reserves_selection_but_not_another_reader(
    monkeypatch: pytest.MonkeyPatch, remaining: float, expected: bool
) -> None:
    instance, _gateway, _acquirer, _plan = fixture(monkeypatch, remaining=remaining)
    assert instance._reading_estimate() == 60
    assert (
        instance._terminal_reads_have_runway([read("read_provision", "27")]) is expected
    )


def test_terminal_forecast_counts_pending_physical_waves_and_duplicate_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, acquirer, plan = fixture(monkeypatch, remaining=28)
    actions = [read("read_provision", str(index)) for index in range(1, 14)]
    assert acquirer.pending_call_counts(actions, plan) == {"read_provision": 13}
    assert not instance._terminal_reads_have_runway(actions)
    # Exactly two measured eight-second waves plus the twelve-second selection reserve.
    instance.evidence_context = cast(RunContext, MagicMock(research_deadline=128.1))
    assert instance._terminal_reads_have_runway(actions)
    duplicate = actions[0].model_copy(update={"need_ids": ["exception"]})
    assert acquirer.pending_call_counts([actions[0], duplicate], plan) == {
        "read_provision": 1
    }
    instance.evidence_context = cast(RunContext, MagicMock(research_deadline=120.1))
    assert instance._terminal_reads_have_runway([actions[0], duplicate])


def test_unmeasured_exact_read_keeps_default_tool_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, _acquirer, _plan = fixture(monkeypatch, remaining=40)
    assert not instance._terminal_reads_have_runway([read("read_chunk_context")])


@pytest.mark.parametrize("delivered", [set(), {99}])
def test_undelivered_or_unknown_original_cannot_dispatch_terminal_read(
    monkeypatch: pytest.MonkeyPatch, delivered: set[int]
) -> None:
    instance, gateway, acquirer, plan = fixture(monkeypatch)
    gateway.last_delivered_citations = delivered
    acquire = MagicMock(side_effect=AssertionError("Unobserved target dispatched"))
    monkeypatch.setattr(acquirer, "acquire", acquire)
    assert not instance._try_terminal_observed_reads(
        step([read("read_provision", "27")]), plan, "request"
    )
    acquire.assert_not_called()
    assert not instance.deferred_followup_steps


def test_closed_research_budget_never_admits_an_observed_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, gateway, acquirer, plan = fixture(monkeypatch, remaining=1_000)
    budget = WorkflowBudget(instance.policy, lambda: 0.0)
    budget.close_research("host_research_deadline")
    gateway.budget = budget
    instance.research_available = budget.research_available
    acquire = MagicMock(side_effect=AssertionError("Closed research dispatched"))
    monkeypatch.setattr(acquirer, "acquire", acquire)
    assert not instance._try_terminal_observed_reads(
        step([read("read_provision", "27")]), plan, "request"
    )
    acquire.assert_not_called()
    budget.check_active(finalizing=True)


def test_cancellation_and_invalid_full_proposal_precede_any_partial_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, acquirer, plan = fixture(monkeypatch)
    acquire = MagicMock(side_effect=AssertionError("Invalid proposal dispatched"))
    monkeypatch.setattr(acquirer, "acquire", acquire)
    proposal = step([read("read_provision", "27"), search_action()])
    proposal.actions[1].need_ids = ["invented_need"]
    with pytest.raises(InvalidSourceAction):
        instance._try_terminal_observed_reads(proposal, plan, "request")
    instance.check_active = MagicMock(side_effect=RunStopped("cancelled"))
    with pytest.raises(RunStopped, match="cancelled"):
        instance._try_terminal_observed_reads(
            step([read("read_provision", "27")]), plan, "request"
        )
    acquire.assert_not_called()
    assert not instance.deferred_followup_steps


def test_terminal_read_registers_material_obligation_without_dependency_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, gateway, acquirer, _plan = fixture(monkeypatch)
    expander, ledger, calls, broker = dependency_fixture()
    plan = IssueResearchPlan.model_validate(dependency_plan().model_dump())
    plan.needs[0].evidence_gaps = ["Material governing scope remains unread."]
    instance.plan = plan
    instance.ledger = ledger
    acquirer.ledger = ledger
    instance.dependency_expander = expander
    origin = ledger.get(1)
    assert origin is not None
    router = SourceLaneRouter(
        SourceLaneCatalogue(
            user_id=uuid4(),
            scope_sha256="synthetic-observed-dependency",
            records=(
                SourceClassification(
                    source_id=UUID(origin.source_id),
                    name="Observed synthetic implementing original",
                    kind=SourceKind.COMMUNIQUE,
                    method="synthetic",
                    uncertain=False,
                    observed_document_types=(),
                ),
            ),
            complete=True,
        ),
        lambda _kind: acquirer.registry,
    )
    acquirer.expand_actions = router.expand
    acquirer.registry_for_action = router.registry
    action = SourceAction(
        need_ids=["permit"],
        tool="read_provision",
        arguments={"source_id": origin.source_id, "article": "3"},
    )
    proposal = step([action])
    proposal.material_dependencies = [material_target()]
    acquire = MagicMock(return_value=[])
    monkeypatch.setattr(acquirer, "acquire", acquire)
    frozen_plan = plan.model_dump(mode="json")
    assert instance._try_terminal_observed_reads(proposal, plan, "request")
    acquire.assert_called_once_with([action], plan)
    assert len(instance.dependencies) == 1
    edge = instance.dependencies[0]
    assert edge.discovery_gaps == [DEFERRED_GAP]
    assert material_dependency_gaps(instance.dependencies, ledger, {1})[
        edge.edge_id
    ] == ["governing_original_unread", "discovery_gap_unresolved"]
    assert calls == [] and broker.mock_calls == [] and expander._expanded == {}
    assert plan.model_dump(mode="json") == frozen_plan
    assert gateway.last_delivered_citations == {1}
    deferred = instance.deferred_followup_steps[0]
    assert isinstance(deferred, dict)
    assert deferred["actions"] == []
    assert deferred["material_dependencies"] == [
        material_target().model_dump(mode="json")
    ]
    assert deferred["status"] == "unexecuted_navigation"


def test_stopped_terminal_acquisition_retains_new_full_original_and_pending_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _gateway, acquirer, plan = fixture(monkeypatch)
    fresh = original(str(uuid4()), "Fresh canonical condition.\n" * 2_500)
    fresh.question_ids = ["condition"]
    stopped_receipt: dict[str, JsonValue] = {
        "status": "PARTIAL",
        "citations": [2],
        "truncated": True,
    }

    def acquire(
        _actions: list[SourceAction], _plan: IssueResearchPlan
    ) -> list[dict[str, JsonValue]]:
        instance.ledger.add([fresh], acquirer.context)
        acquirer.last_receipts = [stopped_receipt]
        raise RunStopped("Source phase deadline reached")

    class ClosedSelector:
        def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
            assert {row.citation for row in request.candidates} == {2}
            raise ResearchPhaseClosed("host_research_deadline")

    monkeypatch.setattr(acquirer, "acquire", acquire)
    instance.selector = SourceSelector(ClosedSelector())
    assert instance._try_terminal_observed_reads(
        step([read("read_provision", "27")]), plan, "request"
    )
    source_request = instance.source_requests[-1]
    assert isinstance(source_request, dict)
    assert source_request["status"] == "source_phase_stopped"
    assert stopped_receipt in instance.receipts
    assert instance.pending_selection_citations == {2}
    payload = instance._payload("request", "", source_phase=False)
    assert payload["required_evidence_numbers"] == [2]
    rows = payload["original_evidence"]
    assert isinstance(rows, list)
    returned = next(
        row for row in rows if isinstance(row, dict) and row["citation"] == 2
    )
    assert returned["text"] == fresh.text and returned["text_hash"] == fresh.text_hash
    assert len(str(returned["text"])) > 50_000
    assert payload["omitted_original_ids"] == []
    assert plan.needs[1].evidence_gaps == ["Unresolved support for condition."]


@pytest.mark.parametrize(
    "enabled,read_tool,expected_terminal",
    [
        (True, "read_provision", True),
        (False, "read_provision", False),
        (True, "read_chunk_context", False),
    ],
)
def test_terminal_read_breaks_to_writer_without_another_coordinator_or_search_wave(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    read_tool: str,
    expected_terminal: bool,
) -> None:
    instance, gateway, acquirer, plan = fixture(
        monkeypatch, enabled=enabled, remaining=80
    )
    proposal = step(
        [
            search_action("An independent unmet outcome"),
            read(read_tool, "27") if read_tool == "read_provision" else read(read_tool),
        ]
    )
    gateway.complete.side_effect = [plan, proposal]
    acquired: list[list[SourceAction]] = []
    initial_kinds: list[SourceKind | None] = []
    final_payload: dict[str, JsonValue] = {}

    def acquire(
        actions: list[SourceAction], selected_plan: IssueResearchPlan
    ) -> list[dict[str, JsonValue]]:
        acquired.append(actions)
        if len(acquired) == 1:
            assert acquirer.expand_actions is not None
            expanded = acquirer.expand_actions(actions, selected_plan)
            initial_kinds.extend(action.source_kind for action in expanded)
        else:
            instance.ledger.add(
                [original(str(uuid4()), "New canonical provision for final reading.")],
                acquirer.context,
            )
        return []

    def finalizing(*_arguments: object) -> CompositeWorkflowResult:
        final_payload.update(
            instance._payload("Complete requested outcomes", "", source_phase=False)
        )
        return CompositeWorkflowResult(
            answer=None, status="unavailable", gaps=["Independent outcome remains open"]
        )

    monkeypatch.setattr(acquirer, "acquire", acquire)
    monkeypatch.setattr(instance, "_finalize_semantic", finalizing)
    result = instance.run("Complete requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert gateway.complete.call_count == 2
    assert len(acquired) == (2 if expected_terminal else 1)
    if expected_terminal:
        assert acquired[1][0] is proposal.actions[1]
    assert initial_kinds == list(SourceKind)
    assert acquired[0][0].need_ids == [need.need_id for need in plan.needs]
    deferred = final_payload["deferred_followup_source_actions"]
    assert isinstance(deferred, list) and len(deferred) == 1
    assert isinstance(deferred[0], dict)
    assert deferred[0]["actions"] == [
        action.model_dump(mode="json")
        for action in ([proposal.actions[0]] if expected_terminal else proposal.actions)
    ]
    assert deferred[0]["status"] == "unexecuted_navigation"
    records = final_payload["original_evidence"]
    assert isinstance(records, list)
    assert {row["citation"] for row in records if isinstance(row, dict)} == (
        {1, 2} if expected_terminal else {1}
    )
    assert all(need.evidence_gaps for need in plan.needs)
    assert not instance.requirements.records()
