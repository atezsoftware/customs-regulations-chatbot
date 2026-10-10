from unittest.mock import Mock, patch

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.legal_review.acquisition import SourceAcquirer
from onyx.legal_review.models import Issue, IssuePlan, SourceAction, WorkflowPolicy
from onyx.legal_review.search import DiscoverySearchTool
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import SearchToolOverrideKwargs, ToolResponse
from onyx.tools.tool_implementations.search.search_tool import SearchTool


@pytest.mark.parametrize(
    "target",
    [
        "Which exceptions change eligibility for the refund?",
        "What event starts the application deadline and can it be extended?",
        "What conditions and judicial changes affect this licensing power?",
    ],
)
def test_relevance_uses_the_specific_evidence_question_through_acquisition(
    target: str,
) -> None:
    scope = IssuePlan(
        language="en",
        issues=[
            Issue(
                issue_id="i1",
                question="A broad transaction",
                requested_outcome="Explain the result",
            )
        ],
    )
    action = SourceAction(
        issue_ids=["i1"],
        tool="search_corpus",
        arguments={
            "query": "identified rule conditions and effects",
            "coverage_item": "Identified rule",
            "evidence_target": target,
        },
    )
    registry = Mock()
    registry.dispatch.return_value = ToolOutcome(
        status=OutcomeStatus.NOT_FOUND, summary="No evidence"
    )
    adapter = Mock()
    adapter.prepare_batch.return_value = {}
    acquirer = SourceAcquirer(
        registry=registry,
        context=RunContext(),
        ledger=EvidenceLedger(),
        policy=WorkflowPolicy(),
        search_adapter=adapter,
    )
    acquirer.acquire([action], scope)
    arguments = registry.dispatch.call_args.args[0].arguments
    assert arguments["evidence_target"] == target
    assert arguments["coverage_item"] == "Identified rule"
    overrides = SearchToolOverrideKwargs(
        starting_citation_num=1,
        rerank_context="The entire unrelated transaction and requested outcomes",
    )
    response = ToolResponse(rich_response=None, llm_facing_response="No evidence")
    tool = DiscoverySearchTool.__new__(DiscoverySearchTool)
    with patch.object(SearchTool, "run", return_value=response) as run:
        result = tool.run(
            Placement(turn_index=1),
            overrides,
            queries=[arguments["query"]],
            coverage_item=arguments["coverage_item"],
            evidence_target=arguments["evidence_target"],
        )
    assert result is response
    forwarded = run.call_args.args[1]
    assert forwarded.rerank_context == target
    assert forwarded.max_llm_chunks == 50
    assert forwarded.rerank_candidate_limit == 384
    assert forwarded.capture_candidate_audit is True
    assert forwarded.candidate_audit_run_id
    assert DiscoverySearchTool.NORMALIZED_RERANK_THRESHOLD == 0.82
    assert (
        overrides.rerank_context
        == "The entire unrelated transaction and requested outcomes"
    )


def test_initial_discovery_without_a_separate_question_keeps_its_query_scope() -> None:
    tool = DiscoverySearchTool.__new__(DiscoverySearchTool)
    overrides = SearchToolOverrideKwargs(
        starting_citation_num=1, rerank_context="Broad parent context"
    )
    with patch.object(SearchTool, "run") as run:
        tool.run(
            Placement(turn_index=0),
            overrides,
            queries=["refund application and evidence"],
        )
    assert run.call_args.args[1].rerank_context == "refund application and evidence"
