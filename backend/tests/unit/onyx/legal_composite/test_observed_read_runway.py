"""Observed task durations admit bounded reads without replacing legal review."""

from dataclasses import dataclass
from threading import local
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.draft_composition import CompositionSection, DraftComposition
from onyx.legal_composite.engine import LegalCompositeEngine, ModelGateway
from onyx.legal_composite.models import (
    AuthorityDependency,
    DraftClaim,
    IssueResearchPlan,
    IssueResearchStep,
    ResearchPlan,
    ReviewCheck,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import AnswerReviewer, ReviewQuestion, build_checks
from onyx.legal_composite.routing import SourceLaneRouter
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    CHUNK_ID,
    SOURCE_ID,
    original,
    registry,
    search_action,
)
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    plan as base_plan,
)
from tests.unit.onyx.legal_composite.test_navigation_recovery import (
    assert_readings_retained,
)
from tests.unit.onyx.legal_composite.test_navigation_recovery import (
    fixture as navigation_fixture,
)
from tests.unit.onyx.legal_composite.test_safe_observed_reads import read


class WorkerClock:
    def __init__(self) -> None:
        self.thread = local()

    def monotonic(self) -> float:
        return getattr(self.thread, "elapsed", 0.0)

    def advance(self, seconds: float) -> None:
        self.thread.elapsed = self.monotonic() + seconds


@dataclass
class ControlledAcquisition:
    engine: LegalCompositeEngine
    acquirer: CanonicalAcquirer
    gateway: MagicMock
    plan: IssueResearchPlan
    now: list[float]
    searches: list[SourceKind]
    reads: list[str]

    def remaining(self, seconds: float) -> None:
        self.engine.evidence_context = cast(
            RunContext, MagicMock(research_deadline=self.now[0] + seconds)
        )


def controlled(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True
) -> ControlledAcquisition:
    clock = WorkerClock()
    now = [100.0]
    monkeypatch.setattr("onyx.legal_composite.acquisition.time", clock)
    monkeypatch.setattr(
        "onyx.legal_composite.engine.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    searches: list[SourceKind] = []
    reads: list[str] = []

    def read_original(
        arguments: dict[str, JsonValue], child: RunContext
    ) -> ToolOutcome:
        child.check_research_active()
        reads.append(str(arguments.get("article", arguments.get("chunk_id"))))
        clock.advance(6.0)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="An exact fictional condition was read.",
            evidence=[original(str(uuid4()), "An immutable fictional condition.")],
        )

    def lane(kind: SourceKind) -> CapabilityRegistry:
        def search(_arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
            child.check_research_active()
            searches.append(kind)
            clock.advance(95.0)
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="One source lane found a whole original.",
                evidence=[
                    original(str(uuid4()), "An immutable fictional lane original.")
                ],
            )

        return registry(search, read_original)

    router = SourceLaneRouter(
        SourceLaneCatalogue(
            user_id=uuid4(),
            scope_sha256="synthetic-observed-task-runway",
            records=(
                SourceClassification(
                    source_id=UUID(SOURCE_ID),
                    name="Immutable fictional source",
                    kind=SourceKind.UNKNOWN,
                    method="synthetic",
                    uncertain=True,
                    observed_document_types=(),
                ),
            ),
            complete=True,
        ),
        lane,
    )
    context = RunContext(timeout_seconds=420)
    ledger = EvidenceLedger()
    ledger.add(
        [original(CHUNK_ID, "Previously delivered whole fictional original.")], context
    )
    policy = WorkflowPolicy(
        max_parallel_tools=12,
        max_tools=192,
        max_search_calls=96,
        max_call_seconds=75,
        selection_reserve_seconds=12,
    )
    acquirer = CanonicalAcquirer(
        lane(SourceKind.UNKNOWN),
        context,
        ledger,
        policy,
        registry_for_action=router.registry,
        expand_actions=router.expand,
        capture_task_timings=True,
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
        reviewer=cast(AnswerReviewer, MagicMock()),
        allow_terminal_observed_reads=True,
        use_observed_read_runway=enabled,
    )
    frozen = IssueResearchPlan.model_validate(base_plan().model_dump())
    engine.plan = frozen
    engine._research_seconds = [8.0]
    acquired = acquirer.acquire

    def acquire(
        actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        result = acquired(actions, plan)
        now[0] += (
            95.0 if any(action.tool == "search_corpus" for action in actions) else 6.0
        )
        return result

    monkeypatch.setattr(acquirer, "acquire", acquire)
    result = ControlledAcquisition(
        engine, acquirer, gateway, frozen, now, searches, reads
    )
    result.remaining(200.0)
    return result


def proposal(actions: list[SourceAction]) -> IssueResearchStep:
    return IssueResearchStep(
        actions=actions,
        ready_to_answer=False,
        remaining_gaps=["The operative condition remains unread."],
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_mixed_acquisition_does_not_assign_search_wall_time_to_fast_read(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    value = controlled(monkeypatch, enabled=enabled)
    actions = [search_action(), read("read_provision", "17")]
    frozen_actions = [action.model_dump(mode="json") for action in actions]
    if not enabled:
        monkeypatch.setattr(
            value.acquirer,
            "task_timing_snapshot",
            MagicMock(
                side_effect=AssertionError("Default engine consumed new timings")
            ),
        )
    assert value.engine._acquire(actions, value.plan)
    assert set(value.searches) == set(SourceKind) and len(value.searches) == 12
    assert SourceKind.UNKNOWN in value.searches and value.reads == ["17"]
    assert len(value.engine.receipts) == 13
    assert [action.model_dump(mode="json") for action in actions] == frozen_actions
    assert value.engine._source_seconds["search_corpus"] == 95
    assert value.engine._source_seconds["read_provision"] == (6 if enabled else 95)
    value.remaining(60)
    assert (
        value.engine._followup_has_runway(proposal([read("read_provision", "18")]))
        is enabled
    )


def test_forecast_counts_pending_physical_waves_and_completed_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = controlled(monkeypatch)
    assert value.engine._acquire([read("read_provision", "17")], value.plan)
    actions = [read("read_provision", str(number)) for number in range(20, 33)]
    duplicate = actions[0].model_copy(deep=True, update={"need_ids": ["exception"]})
    assert value.acquirer.pending_call_counts([actions[0], duplicate], value.plan) == {
        "read_provision": 1
    }
    assert value.engine._acquisition_estimate([actions[0], duplicate]) == 6
    assert value.engine._acquisition_estimate(actions) == 12
    assert value.engine._acquisition_estimate([read("read_provision", "17")]) == 0
    value.remaining(32)
    assert not value.engine._followup_has_runway(proposal(actions))
    value.remaining(32.1)
    assert value.engine._followup_has_runway(proposal(actions))
    assert value.reads == ["17"]  # Forecasting performs no capability dispatch.


@pytest.mark.parametrize("enabled", [False, True])
def test_stopped_real_read_preserves_existing_failure_and_original_contract(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    value = controlled(monkeypatch, enabled=enabled)
    action = read("read_provision", "18")
    canonical_action = value.acquirer.expand_actions([action], value.plan)[0]
    capability_registry = value.acquirer.registry_for_action(canonical_action)
    specification = capability_registry.get(action.tool)
    assert specification is not None
    frozen = value.engine.ledger.serialize_records((1,))

    def stopped(_arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        raise RunStopped("Source phase deadline reached")

    monkeypatch.setitem(
        capability_registry._specs,
        action.tool,
        specification.model_copy(update={"handler": stopped}),
    )
    assert value.engine._acquire([action], value.plan)
    assert value.engine.receipts[-1]["status"] == OutcomeStatus.TRUNCATED.value
    assert value.engine.receipts[-1]["citations"] == []
    assert value.engine.ledger.serialize_records((1,)) == frozen
    assert value.engine._source_seconds == ({} if enabled else {"read_provision": 6})
    assert not value.engine.requirements.records() and not value.reads
    assert value.acquirer.task_timing_snapshot().completed == {}


@pytest.mark.parametrize(
    "reading_after,remaining,expected",
    [(False, 12, False), (False, 12.1, True), (True, 20, False), (True, 20.1, True)],
)
def test_unmeasured_exact_read_is_only_a_deadline_bounded_attempt(
    monkeypatch: pytest.MonkeyPatch,
    reading_after: bool,
    remaining: float,
    expected: bool,
) -> None:
    value = controlled(monkeypatch)
    value.remaining(remaining)
    actions = [read("read_provision", "18")]
    assert value.engine._acquisition_estimate(actions) == 75
    assert (
        value.engine._bounded_observed_reads_have_runway(
            actions, reading_after=reading_after
        )
        is expected
    )
    assert not value.reads and not value.searches
    assert not value.engine.requirements.records()
    assert value.engine.ledger.citation_numbers() == (1,)


@pytest.mark.parametrize("enabled,expected", [(False, False), (True, True)])
def test_postdraft_planning_reserves_only_next_reading_then_rechecks_exact_actions(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: bool
) -> None:
    value = controlled(monkeypatch, enabled=enabled)
    value.engine._source_seconds = {"search_corpus": 95.0}
    value.remaining(21)
    assert value.engine._repair_research_has_runway() is expected
    assert not value.engine._repair_actions_have_runway(proposal([search_action()]))
    assert (
        value.engine._repair_actions_have_runway(
            proposal([read("read_provision", "18")])
        )
        is expected
    )
    value.remaining(12)
    assert not value.engine._repair_actions_have_runway(
        proposal([read("read_provision", "18")])
    )
    assert not value.reads and not value.searches


@pytest.mark.parametrize("target", ["17-19", "17 ve 18", "all provisions", ""])
def test_inexact_article_never_receives_unmeasured_read_fallback(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    value = controlled(monkeypatch)
    value.remaining(25)
    assert not value.engine._repair_actions_have_runway(
        proposal([read("read_provision", target)])
    )
    assert not value.reads


def test_mixed_broad_work_and_multiple_waves_do_not_receive_read_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = controlled(monkeypatch)
    value.remaining(25)
    for actions in (
        [read("read_chunk_context"), search_action()],
        [read("read_provision", str(number)) for number in range(1, 14)],
        [search_action()],
    ):
        assert not value.engine._repair_actions_have_runway(proposal(actions))
    assert not value.reads and not value.searches and not value.engine.receipts


@pytest.mark.parametrize(
    "defect",
    [
        "undelivered",
        "hash",
        "source",
        "chunk",
        "external",
        "derived",
        "untrusted",
        "truncated",
    ],
)
def test_unmeasured_read_cannot_bypass_original_delivery_or_canonical_identity(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    value = controlled(monkeypatch)
    value.remaining(25)
    item = value.engine.ledger._items[1]
    assert item.search_doc is not None
    if defect == "undelivered":
        value.gateway.last_delivered_citations = set()
    elif defect == "hash":
        item.text_hash = "0" * 64
    elif defect == "source":
        item.search_doc.document_id = str(uuid4())
    elif defect == "chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = str(uuid4())
    else:
        item.metadata[defect] = True
    assert not value.engine._repair_actions_have_runway(
        proposal([read("read_chunk_context")])
    )
    assert not value.reads and not value.searches


def test_unknown_source_and_another_sources_chunk_cannot_receive_read_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = controlled(monkeypatch)
    value.remaining(25)
    unknown = read("read_provision", "18")
    unknown.arguments["source_id"] = str(uuid4())
    with pytest.raises(InvalidSourceAction):
        value.engine._repair_actions_have_runway(proposal([unknown]))
    other = original(str(uuid4()), "A whole original from another source.")
    other.source_id = str(uuid4())
    assert other.search_doc is not None
    other.search_doc.document_id = other.source_id
    value.engine.ledger.add([other], value.acquirer.context)
    value.gateway.last_delivered_citations = {1, 2}
    assert other.chunk_id is not None
    assert not value.engine._repair_actions_have_runway(
        proposal([read("read_chunk_context", other.chunk_id)])
    )
    assert not value.reads


@pytest.mark.parametrize(
    "stop", ["cancelled", "Workflow deadline reached", "Workflow cost limit reached"]
)
def test_terminal_attempt_never_bypasses_owner_cancellation_deadline_or_cost(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    value = controlled(monkeypatch)
    value.remaining(25)
    failure = RunStopped(stop)
    value.engine.check_active = MagicMock(side_effect=failure)
    with pytest.raises(RunStopped) as raised:
        value.engine._try_terminal_observed_reads(
            proposal([read("read_provision", "18")]), value.plan, "Full request"
        )
    assert raised.value is failure
    assert not value.reads and not value.searches and not value.engine.receipts


@pytest.mark.parametrize("remaining", [0, 12])
def test_expired_or_closed_research_cannot_dispatch_an_unmeasured_attempt(
    monkeypatch: pytest.MonkeyPatch, remaining: float
) -> None:
    value = controlled(monkeypatch)
    value.remaining(remaining)
    assert not value.engine._try_terminal_observed_reads(
        proposal([read("read_provision", "18")]), value.plan, "Full request"
    )
    value.remaining(1_000)
    value.engine.research_available = lambda: False
    assert not value.engine._try_terminal_observed_reads(
        proposal([read("read_provision", "18")]), value.plan, "Full request"
    )
    assert not value.reads and not value.searches


@pytest.mark.parametrize("enabled,source_batches", [(False, 1), (True, 2)])
def test_extra_read_admission_preserves_requirements_and_full_semantic_rejection(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, source_batches: int
) -> None:
    harness = navigation_fixture(monkeypatch)
    instance = harness.engine
    instance.use_observed_read_runway = enabled
    instance._research_seconds = [8.0]
    harness.context.research_deadline = 120.1
    instance.policy = instance.policy.model_copy(
        update={"max_research_rounds": 1, "max_reviews": 1}
    )
    monkeypatch.setattr(
        instance,
        "_finalize_semantic",
        LegalCompositeEngine._finalize_semantic.__get__(instance, LegalCompositeEngine),
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
        request, plan, draft, requirements, dependencies, delivered, *_rest = arguments
        return build_checks(
            cast(str, request),
            cast(IssueResearchPlan, plan),
            cast(StructuredDraftAnswer, draft),
            cast(list[SourceRequirement], requirements),
            cast(list[AuthorityDependency], dependencies),
            cast(set[int], delivered),
            previous=previous,
            affected_sections=affected_sections,
            ledger=instance.ledger,
        )

    def reviewed(
        *arguments: object,
        previous: SemanticReview | None = None,
        affected_sections: set[str] | None = None,
    ) -> SemanticReview:
        assert arguments[5] == {1}
        assert len(cast(list[SourceRequirement], arguments[3])) == 7
        checks = expected(
            *arguments, previous=previous, affected_sections=affected_sections
        )
        assert "claim:claim_literal_1" in checks
        return SemanticReview(
            checks=[
                ReviewCheck(
                    check_id=key,
                    need_ids=question.need_ids,
                    section_ids=question.section_ids,
                    status="incorrect"
                    if key == "claim:claim_literal_1"
                    else "addressed",
                    confidence=0.99,
                )
                for key, question in checks.items()
            ]
        )

    reviewer.expected_checks.side_effect = expected
    reviewer.review.side_effect = reviewed
    instance.reviewer = cast(AnswerReviewer, reviewer)
    harness.gateway.complete.side_effect = [
        harness.plan,
        harness.reading([read("read_provision", "18")]),
        composition,
    ]
    result = instance.run("Complete fictional requested outcomes")
    assert result.status == "unavailable" and result.answer is None
    assert len(harness.acquired) == source_batches
    assert len(harness.expanded[0]) == 12
    assert {action.source_kind for action in harness.expanded[0]} == set(SourceKind)
    if enabled:
        assert harness.acquired[1] == [read("read_provision", "18")]
    assert harness.gateway.complete.call_count == 3
    reviewer.review.assert_called_once()
    assert_readings_retained(harness)
    writer_payload = harness.gateway.complete.call_args_list[-1].args[1]
    assert len(writer_payload["source_requirements"]) == 7
    records = writer_payload["original_evidence"]
    assert (
        isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict)
    )
    item = instance.ledger.get(1)
    assert item is not None
    assert records[0]["text"] == item.text and records[0]["text_hash"] == item.text_hash
