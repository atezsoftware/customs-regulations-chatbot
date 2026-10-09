"""Defer speculative query waves without losing issues or original search targets."""

import ast
import inspect
from hashlib import sha256
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import SourceKind, SourceLaneCatalogue
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.engine import (
    LegalCompositeEngine,
    ModelGateway,
    SourceAcquirer,
    initial_discovery_actions,
    initial_issue_discovery_actions,
)
from onyx.legal_composite.models import (
    CompositeWorkflowResult,
    IssueResearchNeed,
    IssueResearchPlan,
    IssueResearchStep,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import AnswerReviewer
from onyx.legal_composite.routing import SourceLaneRouter


def plan() -> IssueResearchPlan:
    needs = [
        IssueResearchNeed(
            need_id=f"n{index}",
            question=f"Independent requested outcome {index}",
            governing_source="Not established",
            conditions_to_check=[],
            source_kinds=[SourceKind.STATUTE],
        )
        for index in range(1, 4)
    ]
    return IssueResearchPlan(
        language="tr",
        requires_sources=True,
        discovery_query="All requested eligibility, procedure and remedy outcomes",
        needs=needs,
        initial_actions=[
            SourceAction(
                need_ids=[need.need_id],
                tool="search_corpus",
                arguments={
                    "query": need.question,
                    "mode": "keyword" if index == 2 else "hybrid",
                    "coverage_item": need.need_id,
                    "evidence_target": f"Actual operative support for {need.need_id}",
                    "source_anchors": [f"Observed name {index}"],
                },
                source_kind=SourceKind.STATUTE,
            )
            for index, need in enumerate(needs, start=1)
        ],
        missing_user_facts=[],
    )


def engine() -> tuple[LegalCompositeEngine, MagicMock, MagicMock]:
    gateway = MagicMock()
    gateway.last_delivered_citations = set()
    acquirer = MagicMock()
    acquirer.acquire.return_value = []
    acquirer.definitions.return_value = []
    instance = LegalCompositeEngine(
        gateway=cast(ModelGateway, gateway),
        acquirer=cast(SourceAcquirer, acquirer),
        ledger=EvidenceLedger(),
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
        reviewer=cast(AnswerReviewer, MagicMock()),
    )
    return instance, gateway, acquirer


def test_three_planner_searches_start_one_complete_twelve_lane_frontier() -> None:
    frozen = plan()
    before = frozen.model_dump(mode="json")
    initial = initial_issue_discovery_actions(frozen, "Complete user request")
    catalogue = SourceLaneCatalogue(
        user_id=uuid4(), scope_sha256="scope", records=(), complete=True
    )
    router = SourceLaneRouter(catalogue, lambda _kind: CapabilityRegistry([]))
    expanded = router.expand(initial, frozen)
    assert len(expanded) == 12
    assert [action.source_kind for action in expanded] == list(SourceKind)
    assert SourceKind.UNKNOWN in {action.source_kind for action in expanded}
    assert all(action.need_ids == ["n1", "n2", "n3"] for action in expanded)
    assert {action.arguments["query"] for action in expanded} == {
        frozen.discovery_query
    }
    assert len(router.expand(frozen.initial_actions, frozen)) == 36
    assert initial[0].source_kind is None
    assert frozen.model_dump(mode="json") == before


def test_known_nonsearch_actions_remain_in_initial_batch() -> None:
    frozen = plan()
    read = SourceAction(
        need_ids=["n2"],
        tool="read_provision",
        arguments={"source_id": "observed-source", "article": "12"},
        source_kind=SourceKind.REGULATION,
    )
    resolve = SourceAction(
        need_ids=["n3"],
        tool="resolve_source",
        arguments={"name": "Observed implementing title"},
        source_kind=SourceKind.CIRCULAR,
    )
    frozen.initial_actions.extend([read, resolve])
    initial = initial_issue_discovery_actions(frozen, "request")
    assert initial[1:] == [read, resolve]
    assert len(frozen.initial_actions) == 5


def test_empty_discovery_query_uses_full_request_and_social_plan_is_unchanged() -> None:
    frozen = plan()
    frozen.discovery_query = "  "
    action = initial_issue_discovery_actions(frozen, " Complete original request ")[0]
    assert action.arguments["query"] == "Complete original request"
    frozen.requires_sources = False
    assert initial_issue_discovery_actions(frozen, "merhaba") == frozen.initial_actions


def test_engine_exposes_all_exact_deferred_targets_before_further_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = plan()
    instance, gateway, acquirer = engine()
    reported: list[tuple[str, str]] = []
    instance.report = lambda phase, language: reported.append((phase, language))
    gateway.complete.side_effect = [
        frozen,
        IssueResearchStep(
            actions=[],
            ready_to_answer=True,
            remaining_gaps=["Operative support remains unread"],
            issue_gaps={
                need.need_id: ["Operative support remains unread"]
                for need in frozen.needs
            },
        ),
    ]
    monkeypatch.setattr(
        instance,
        "_finalize_semantic",
        lambda *_args: CompositeWorkflowResult(
            answer=None, status="unavailable", gaps=["No original support"]
        ),
    )
    result = instance.run("Complete legal request")
    assert result.answer is None
    assert reported == [("tools", "tr"), ("reading", "tr"), ("final", "tr")]
    assert gateway.complete.call_count == 2
    acquirer.acquire.assert_called_once()
    initial = acquirer.acquire.call_args.args[0]
    assert len(initial) == 1 and initial[0].need_ids == ["n1", "n2", "n3"]
    research_payload: dict[str, JsonValue] = gateway.complete.call_args_list[1].args[1]
    deferred = research_payload["deferred_initial_source_actions"]
    assert isinstance(deferred, list) and len(deferred) == 3
    assert [row["action"] for row in deferred if isinstance(row, dict)] == [
        action.model_dump(mode="json") for action in frozen.initial_actions
    ]
    assert all(
        isinstance(row, dict)
        and row["status"] == "unexecuted_navigation"
        and row["navigation_only"] is True
        for row in deferred
    )
    assert instance.plan is not None and len(instance.plan.initial_actions) == 3
    assert all(
        need.evidence_gaps
        for need in instance.plan.needs
        if isinstance(need, IssueResearchNeed)
    )
    assert not instance.requirements.records()


@pytest.mark.parametrize(
    "error,status",
    [
        (None, "attempted"),
        (RunStopped("Research deadline"), "partially_attempted"),
        (InvalidSourceAction("Invalid source"), "invalid_attempt"),
    ],
)
def test_actual_attempt_status_preserves_proposals_and_distinct_targets(
    error: Exception | None, status: str
) -> None:
    frozen = plan()
    instance, _gateway, acquirer = engine()
    instance.plan = frozen
    same_query_other_target = frozen.initial_actions[0].model_copy(deep=True)
    same_query_other_target.arguments["evidence_target"] = "Different requested effect"
    instance.deferred_initial_actions = [
        (frozen.initial_actions[0], "unexecuted_navigation"),
        (same_query_other_target, "unexecuted_navigation"),
    ]
    if error is not None:
        acquirer.acquire.side_effect = error
    if isinstance(error, InvalidSourceAction):
        with pytest.raises(InvalidSourceAction):
            instance._acquire([frozen.initial_actions[0]], frozen)
    else:
        assert instance._acquire([frozen.initial_actions[0]], frozen) is (error is None)
    assert instance.deferred_initial_actions == [
        (frozen.initial_actions[0], status),
        (same_query_other_target, "unexecuted_navigation"),
    ]
    payload = instance._payload("request", "")
    deferred = payload["deferred_initial_source_actions"]
    assert isinstance(deferred, list) and len(deferred) == 2
    assert not instance.ledger.citation_numbers()
    assert not instance.requirements.records()


def test_base_discovery_behavior_and_source_ast_are_unchanged() -> None:
    frozen = plan()
    assert initial_discovery_actions(frozen, "request") == frozen.initial_actions
    parsed = ast.parse(inspect.getsource(initial_discovery_actions)).body[0]
    digest = sha256(ast.dump(parsed, include_attributes=False).encode()).hexdigest()
    assert (
        digest == "dc682cefc0912eed96c21d60e7d958303f21a08d4ddbf626dd3a91bda19e368b"
    )  # pragma: allowlist secret
