"""Navigation repair cannot replace accepted law readings or semantic review."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.budget import ResearchPhaseClosed, WorkflowBudget
from onyx.legal_composite.draft_composition import (
    CompositionSection,
    DraftComposition,
)
from onyx.legal_composite.engine import LegalCompositeEngine, ModelGateway
from onyx.legal_composite.models import (
    AuthorityDependency,
    CompositeWorkflowResult,
    DraftClaim,
    IssueResearchNeed,
    IssueResearchPlan,
    IssueResearchStep,
    ReviewCheck,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    SpanSupport,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.navigation import NavigationProposal
from onyx.legal_composite.reviewer import AnswerReviewer, ReviewQuestion, build_checks
from onyx.legal_composite.routing import SourceLaneRouter
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    CHUNK_ID,
    SOURCE_ID,
    original,
)

PRIVATE_MARKER = "SyntheticUntrustedValueMustNotEnterPublicTrace"


def search(*, valid: bool) -> SourceAction:
    return SourceAction(
        need_ids=["condition"],
        tool="search_corpus",
        arguments={
            "query": "Find the missing fictional condition's operative basis",
            "mode": "hybrid" if valid else PRIVATE_MARKER,
            "coverage_item": "condition",
            "evidence_target": "Actual fictional missing condition",
        },
    )


@dataclass
class Harness:
    engine: LegalCompositeEngine
    gateway: MagicMock
    acquirer: CanonicalAcquirer
    plan: IssueResearchPlan
    findings: list[SourceRequirement]
    context: RunContext
    acquired: list[list[SourceAction]] = field(default_factory=list)
    expanded: list[list[SourceAction]] = field(default_factory=list)
    final_payloads: list[dict[str, JsonValue]] = field(default_factory=list)
    traces: list[tuple[str, str, SimpleNamespace]] = field(default_factory=list)

    def reading(self, actions: list[SourceAction]) -> IssueResearchStep:
        return IssueResearchStep(
            actions=actions,
            ready_to_answer=False,
            remaining_gaps=["The fictional condition's own basis remains unread."],
            issue_gaps={
                "condition": ["The fictional condition's own basis remains unread."]
            },
            requirements=self.findings,
        )

    def frozen_readings(self) -> list[dict[str, JsonValue]]:
        return [finding.model_dump(mode="json") for finding in self.findings]


def fixture(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True) -> Harness:
    monkeypatch.setattr("onyx.legal_composite.engine.time.monotonic", lambda: 100.0)
    policy = WorkflowPolicy(
        timeout_seconds=420,
        finalization_reserve_seconds=140,
        selection_reserve_seconds=12,
        max_call_seconds=75,
        max_research_rounds=3,
        max_parallel_tools=12,
        max_tools=192,
        max_search_calls=96,
    )
    context = RunContext(timeout_seconds=420, research_deadline=400)
    ledger = EvidenceLedger()
    clauses = [
        f"Fictional clause {number} establishes an independent condition."
        for number in range(1, 8)
    ]
    ledger.add([original(CHUNK_ID, "\n".join(clauses))], context)
    plan = IssueResearchPlan(
        language="en",
        requires_sources=True,
        discovery_query="All fictional rule, condition and exception outcomes",
        needs=[
            IssueResearchNeed(
                need_id=need,
                question=f"What establishes the fictional {need} outcome?",
                governing_source="Unestablished",
                conditions_to_check=[],
                evidence_gaps=[f"The {need} outcome is incomplete."],
            )
            for need in ("rule", "condition", "exception")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    findings = [
        SourceRequirement(
            requirement_id=f"literal_{number}",
            need_id=plan.needs[(number - 1) % len(plan.needs)].need_id,
            dimension="scope",
            rule=clause,
            application="The fictional facts require a conditional application.",
            supports=[SpanSupport(citation=1, quotation=clause)],
        )
        for number, clause in enumerate(clauses, start=1)
    ]

    def forbidden(
        _arguments: dict[str, JsonValue], _context: RunContext
    ) -> ToolOutcome:
        raise AssertionError("An integration fixture attempted a physical source call")

    registry = CapabilityRegistry(
        spec.model_copy(update={"handler": forbidden})
        for spec in build_corpus_specs(
            cast(CorpusBroker, MagicMock()), require_search_targets=True
        )
    )
    router = SourceLaneRouter(
        SourceLaneCatalogue(
            user_id=uuid4(),
            scope_sha256="synthetic-navigation-integration",
            records=(
                SourceClassification(
                    source_id=UUID(SOURCE_ID),
                    name="Fictional immutable original",
                    kind=SourceKind.UNKNOWN,
                    method="synthetic",
                    uncertain=True,
                    observed_document_types=(),
                ),
            ),
            complete=True,
        ),
        lambda _kind: registry,
    )
    acquirer = CanonicalAcquirer(
        registry,
        context,
        ledger,
        policy,
        registry_for_action=router.registry,
        expand_actions=router.expand,
    )
    gateway = MagicMock()
    gateway.last_delivered_citations = {1}
    gateway.budget = None
    engine = LegalCompositeEngine(
        gateway=cast(ModelGateway, gateway),
        acquirer=acquirer,
        ledger=ledger,
        policy=policy,
        check_active=context.check_active,
        research_available=lambda: True,
        evidence_context=context,
        reviewer=cast(AnswerReviewer, MagicMock()),
        recover_invalid_navigation=enabled,
    )
    harness = Harness(engine, gateway, acquirer, plan, findings, context)

    def acquire(
        actions: list[SourceAction], frozen_plan: IssueResearchPlan
    ) -> list[dict[str, JsonValue]]:
        context.check_research_active()
        acquirer.pending_call_counts(actions, frozen_plan)
        harness.acquired.append(actions)
        harness.expanded.append(router.expand(actions, frozen_plan))
        return [{"status": "FOUND", "citations": [1], "navigation_only": True}]

    def finalizing(*_arguments: object) -> CompositeWorkflowResult:
        context.check_active()
        harness.final_payloads.append(
            engine._payload("Complete fictional request", "", source_phase=False)
        )
        # A navigation repair never independently permits publication.
        return CompositeWorkflowResult(
            answer=None,
            status="unavailable",
            gaps=["Independent semantic review rejected the unresolved interaction."],
            source_requirements=engine.requirements.records(),
        )

    @contextmanager
    def graph(
        operation: str, _input: dict[str, JsonValue], *, summary: str = ""
    ) -> Iterator[SimpleNamespace]:
        traced = SimpleNamespace(output_value=None)
        harness.traces.append((operation, summary, traced))
        yield traced

    monkeypatch.setattr(acquirer, "acquire", acquire)
    monkeypatch.setattr(engine, "_finalize_semantic", finalizing)
    monkeypatch.setattr("onyx.legal_composite.engine.graph_step", graph)
    return harness


def assert_readings_retained(harness: Harness) -> None:
    assert [
        row.model_dump(mode="json") for row in harness.engine.requirements.records()
    ] == harness.frozen_readings()
    for exported in harness.engine.requirements.export():
        bindings = exported["original_bindings"]
        assert isinstance(bindings, list) and len(bindings) == 1
        binding = bindings[0]
        assert isinstance(binding, dict)
        item = harness.engine.ledger.get(1)
        assert item is not None and binding["text_hash"] == item.text_hash
    assert harness.engine.ledger.citation_numbers() == (1,)
    assert harness.engine.plan is not None
    need = next(
        need for need in harness.engine.plan.needs if need.need_id == "condition"
    )
    assert isinstance(need, IssueResearchNeed)
    assert "The fictional condition's own basis remains unread." in need.evidence_gaps


def invalid_receipts(harness: Harness) -> list[dict[str, JsonValue]]:
    return [
        row
        for row in harness.engine.deferred_followup_steps
        if isinstance(row, dict) and row.get("status") == "invalid_navigation"
    ]


def test_valid_repair_preserves_seven_original_readings_and_searches_every_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    bad = search(valid=False)
    corrected = search(valid=True)
    proposal = NavigationProposal(
        actions=[corrected], reconsider_citations=[], material_dependencies=[]
    )
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([bad]),
        proposal,
        IssueResearchStep(actions=[], ready_to_answer=True, remaining_gaps=[]),
    ]
    frozen = harness.frozen_readings()
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.answer is None and result.status == "unavailable"
    assert harness.gateway.complete.call_count == 4
    assert [call.args[2] for call in harness.gateway.complete.call_args_list] == [
        IssueResearchPlan,
        IssueResearchStep,
        NavigationProposal,
        IssueResearchStep,
    ]
    assert len(harness.acquired) == 2
    assert harness.acquired[1] == [corrected]
    assert len(harness.expanded[1]) == 12
    assert [action.source_kind for action in harness.expanded[1]] == list(SourceKind)
    assert all(action.need_ids == ["condition"] for action in harness.expanded[1])
    assert all(
        action.arguments == corrected.arguments for action in harness.expanded[1]
    )
    assert_readings_retained(harness)
    assert harness.frozen_readings() == frozen
    assert len(harness.final_payloads) == 1
    retained = invalid_receipts(harness)
    assert len(retained) == 1 and retained[0]["actions"] == [
        bad.model_dump(mode="json")
    ]
    assert retained[0]["navigation_only"] is True
    assert retained[0]["failure_stage"] == "actions"


def test_second_invalid_navigation_stops_repair_loop_and_still_reaches_final_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    bad = search(valid=False)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([bad]),
        NavigationProposal(
            actions=[bad], reconsider_citations=[], material_dependencies=[]
        ),
    ]
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.answer is None and result.status == "unavailable"
    assert harness.gateway.complete.call_count == 3
    assert len(harness.acquired) == 1 and len(harness.final_payloads) == 1
    assert len(invalid_receipts(harness)) == 2
    assert_readings_retained(harness)
    assert harness.engine.protocol_defects


def test_default_off_does_not_add_a_navigation_model_call_or_change_legacy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch, enabled=False)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([search(valid=False)]),
    ]
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert harness.gateway.complete.call_count == 2
    assert len(harness.acquired) == 1 and not harness.final_payloads
    assert not invalid_receipts(harness)
    assert_readings_retained(harness)


@pytest.mark.parametrize("stop", ["closed_budget", "no_runway"])
def test_unavailable_recovery_respects_research_reserve_and_keeps_accepted_findings(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    harness = fixture(monkeypatch)
    budget = WorkflowBudget(harness.engine.policy, lambda: 100.0)
    harness.gateway.budget = budget
    harness.engine.research_available = budget.research_available

    def complete(
        _system: str,
        _payload: dict[str, JsonValue],
        response_type: type[BaseModel],
        *_arguments: object,
    ) -> BaseModel:
        if response_type is IssueResearchPlan:
            return harness.plan
        assert response_type is IssueResearchStep
        if stop == "closed_budget":
            budget.close_research("host_research_deadline")
        else:
            harness.context.research_deadline = 101.0
        return harness.reading([search(valid=False)])

    harness.gateway.complete.side_effect = complete
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.answer is None and result.status == "unavailable"
    assert harness.gateway.complete.call_count == 2
    assert len(harness.acquired) == 1 and len(harness.final_payloads) == 1
    assert len(invalid_receipts(harness)) == 1
    assert_readings_retained(harness)
    budget.check_active(finalizing=True)


def test_repair_provider_research_closure_keeps_finalization_and_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([search(valid=False)]),
        ResearchPhaseClosed("provider_timeout"),
    ]
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert harness.gateway.complete.call_count == 3
    assert len(harness.acquired) == 1 and len(harness.final_payloads) == 1
    assert_readings_retained(harness)


def test_recovery_cancellation_never_dispatches_or_admits_late_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)

    def complete(
        _system: str,
        _payload: dict[str, JsonValue],
        response_type: type[BaseModel],
        *_arguments: object,
    ) -> BaseModel:
        if response_type is IssueResearchPlan:
            return harness.plan
        if response_type is IssueResearchStep:
            return harness.reading([search(valid=False)])
        assert response_type is NavigationProposal
        harness.context.cancel()
        return NavigationProposal(
            actions=[search(valid=True)],
            reconsider_citations=[],
            material_dependencies=[],
        )

    harness.gateway.complete.side_effect = complete
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.status == "cancelled" and result.answer is None
    assert harness.gateway.complete.call_count == 3
    assert len(harness.acquired) == 1 and not harness.final_payloads
    assert_readings_retained(harness)
    with pytest.raises(RunStopped, match="cancelled"):
        harness.engine.ledger.add(
            [original(str(uuid4()), "Forbidden late body")], harness.context
        )
    assert harness.engine.ledger.citation_numbers() == (1,)


def test_navigation_admission_diagnostics_do_not_export_model_or_source_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    bad = search(valid=False)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([bad]),
        NavigationProposal(
            actions=[bad], reconsider_citations=[], material_dependencies=[]
        ),
    ]
    harness.engine.run("Complete fictional requested outcomes")
    traces = [
        (summary, span.output_value)
        for operation, summary, span in harness.traces
        if operation == "legal_composite.navigation_admission"
    ]
    assert len(traces) == 2
    assert all(
        "actions" in summary and "accepted=0" in summary for summary, _output in traces
    )
    assert PRIVATE_MARKER not in repr(traces)
    assert SOURCE_ID not in repr(traces) and CHUNK_ID not in repr(traces)
    assert all(finding.rule not in repr(traces) for finding in harness.findings)


def test_canonical_or_native_preflight_failure_is_not_reclassified_as_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([search(valid=True)]),
    ]
    preflight = MagicMock(
        side_effect=InvalidSourceAction("Material dependency origin is not canonical")
    )
    monkeypatch.setattr(harness.engine, "_preflight_navigation", preflight)
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert harness.gateway.complete.call_count == 2
    assert not invalid_receipts(harness) and not harness.final_payloads
    assert len(harness.acquired) == 1
    assert_readings_retained(harness)


def test_navigation_transport_cannot_return_new_law_readings_or_gap_closures() -> None:
    properties = NavigationProposal.model_json_schema()["properties"]
    assert set(properties) == {
        "actions",
        "reconsider_citations",
        "material_dependencies",
    }
    assert NavigationProposal.model_config.get("extra") == "forbid"


def test_recovered_navigation_still_requires_actual_complete_semantic_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = fixture(monkeypatch)
    harness.engine.policy = harness.engine.policy.model_copy(update={"max_reviews": 1})
    monkeypatch.setattr(
        harness.engine,
        "_finalize_semantic",
        LegalCompositeEngine._finalize_semantic.__get__(
            harness.engine, LegalCompositeEngine
        ),
    )
    composition = DraftComposition(
        sections=[
            CompositionSection(
                section_id=f"section_{need.need_id}", need_ids=[need.need_id]
            )
            for need in harness.plan.needs
        ],
        claims=[
            DraftClaim(
                claim_id=f"claim_{finding.requirement_id}",
                section_id=f"section_{finding.need_id}",
                need_ids=[finding.need_id],
                answer_excerpt=f"{finding.rule} [[1]]",
                requirement_ids=[finding.requirement_id],
            )
            for finding in harness.findings
        ],
        unresolved_need_ids=["condition"],
    )
    reviewer = MagicMock()

    def expected(
        *arguments: object,
        previous: SemanticReview | None = None,
        affected_sections: set[str] | None = None,
    ) -> dict[str, ReviewQuestion]:
        del previous, affected_sections
        request, plan, draft, requirements, dependencies, delivered, *_rest = arguments
        return build_checks(
            cast(str, request),
            cast(IssueResearchPlan, plan),
            cast(StructuredDraftAnswer, draft),
            cast(list[SourceRequirement], requirements),
            cast(list[AuthorityDependency], dependencies),
            cast(set[int], delivered),
            ledger=harness.engine.ledger,
        )

    def reviewed(
        *arguments: object,
        previous: SemanticReview | None = None,
        affected_sections: set[str] | None = None,
    ) -> SemanticReview:
        del previous, affected_sections
        assert arguments[5] == {1}
        assert len(cast(list[SourceRequirement], arguments[3])) == 7
        item = harness.engine.ledger.get(1)
        assert item is not None and all(
            finding.rule in item.text for finding in harness.findings
        )
        questions = expected(*arguments)
        assert isinstance(questions, dict)
        return SemanticReview(
            checks=[
                ReviewCheck(
                    check_id=identity,
                    need_ids=question.need_ids,
                    section_ids=question.section_ids,
                    status="incorrect"
                    if identity == "claim:claim_literal_1"
                    else "addressed",
                    confidence=0.99,
                )
                for identity, question in questions.items()
            ]
        )

    reviewer.expected_checks.side_effect = expected
    reviewer.review.side_effect = reviewed
    harness.engine.reviewer = cast(AnswerReviewer, reviewer)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([search(valid=False)]),
        NavigationProposal(
            actions=[search(valid=True)],
            reconsider_citations=[],
            material_dependencies=[],
        ),
        IssueResearchStep(actions=[], ready_to_answer=True, remaining_gaps=[]),
        composition,
    ]
    result = harness.engine.run("Complete fictional requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert harness.gateway.complete.call_count == 5
    reviewer.review.assert_called_once()
    writer_payload = harness.gateway.complete.call_args_list[-1].args[1]
    assert len(writer_payload["source_requirements"]) == 7
    records = writer_payload["original_evidence"]
    assert isinstance(records, list) and len(records) == 1
    item = harness.engine.ledger.get(1)
    assert isinstance(records[0], dict) and item is not None
    assert records[0]["text"] == item.text and records[0]["text_hash"] == item.text_hash
    assert_readings_retained(harness)
