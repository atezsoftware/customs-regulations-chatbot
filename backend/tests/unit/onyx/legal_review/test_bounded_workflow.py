import time
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.search_adapter import ScopedSearchAdapter
from onyx.legal_review.acquisition import SourceAcquirer
from onyx.legal_review.drafting import (
    DraftEdits,
    GeneratedBlock,
    GeneratedDraft,
    apply_draft_edits,
)
from onyx.legal_review.models import (
    DiscoveryQuery,
    InitialDiscoveryPlan,
    PublicationFinding,
    PublicationReview,
    ReviewCheck,
    ReviewResult,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_review.research import ResearchLedger
from tests.unit.onyx.legal_review.test_engine import (
    draft,
    engine,
    original,
    plan,
    reading,
)
from tests.unit.onyx.legal_review.test_research import diagnosis
from tests.unit.onyx.legal_review.test_research import plan as research_plan


def test_current_corpus_closes_without_fabricating_verified_status() -> None:
    workflow, _, reviewer = engine([plan(), reading(), draft()], ["pass"])
    workflow.policy = WorkflowPolicy()
    result = workflow.run("Başvuru şartı nedir?", "")
    assert result.status == "verified"
    assert result.issue_closures[0].status == "closed"
    assert result.requirements[0].legal_status == "unknown"
    assert result.requirements[0].validity == "unknown"
    assert result.final_review is not None
    assert result.early_review is None and result.final_review.completed
    assert len(reviewer.states) == 1
    currency = reviewer.states[0]["corpus_currency"]
    assert isinstance(currency, dict) and currency["assume_current_versions"] is True
    assert result.source_journey[0]["requirement_ids"] == ["r1"]
    assert result.source_journey[0]["draft_claim_ids"] == ["c1"]


def test_research_deadline_still_writes_and_reviews_a_partial_answer() -> None:
    workflow, _, reviewer = engine([plan(), draft()], ["pass"])
    workflow.policy = WorkflowPolicy()

    def exhausted(*_args: object, **_kwargs: object) -> None:
        workflow.context.research_deadline = time.monotonic() - 1
        raise TimeoutError("research window exhausted")

    with patch.object(workflow, "_research", side_effect=exhausted):
        result = workflow.run("Soru", "")
    assert result.status == "partial" and result.answer
    assert result.final_review is not None
    assert result.final_review.completed and len(reviewer.states) == 1
    assert "Research time ended" in result.gaps[0]


def test_one_issue_may_discover_distinct_source_targets() -> None:
    initial = InitialDiscoveryPlan(
        **plan().model_dump(),
        discovery_queries=[
            DiscoveryQuery(query=query, issue_ids=["i1"])
            for query in ["application conditions", "application official form"]
        ],
    )
    ledger = ResearchLedger()
    actions = [
        SourceAction(
            tool="search_corpus", issue_ids=["i1"], arguments={"query": row.query}
        )
        for row in initial.discovery_queries
    ]
    assert len(ledger.admit(actions, initial)) == 2
    assert ledger.admit(actions, initial) == []


def test_one_reasoned_retry_retains_identity_and_cannot_be_repeated() -> None:
    ledger = ResearchLedger()
    need = ledger.bind_task(diagnosis(), ["parent"])
    action = SourceAction(
        tool="search_corpus",
        issue_ids=["parent"],
        research_need_ids=[need.need_id],
        arguments={"query": "condition amendment"},
    )
    admitted = ledger.admit([action], research_plan())
    ledger.record_results(
        admitted,
        [
            {
                "call_id": "attempt1",
                "tool": action.tool,
                "arguments": action.arguments,
                "evidence_ids": [],
                "status": "not_found",
            }
        ],
        set(),
    )
    action.arguments["query"] = "application condition operative judgment"
    assert ledger.admit([action], research_plan()) == []
    action.retry_reason = (
        "The first search found no operative effect; target the court disposition."
    )
    assert len(ledger.admit([action], research_plan())) == 1
    action.arguments["query"] = "another paraphrase"
    assert ledger.admit([action], research_plan()) == []
    assert len(need.attempted_queries) == 2


def test_targeted_edits_preserve_unaffected_text_and_claims() -> None:
    base = GeneratedDraft(
        blocks=[
            GeneratedBlock(block_id="a", text="Known outcome.", claims=[]),
            GeneratedBlock(block_id="b", text="Unknown effect.", claims=[]),
        ],
        unresolved_issue_ids=["i1"],
    )
    edited = apply_draft_edits(
        base,
        DraftEdits(
            replacements=[
                GeneratedBlock(
                    block_id="b",
                    text="Effect is conditional on the missing date.",
                    claims=[],
                )
            ],
            unresolved_issue_ids=["i1"],
        ),
    )
    assert edited.blocks[0].model_dump() == base.blocks[0].model_dump()
    assert edited.blocks[1].text != base.blocks[1].text
    with pytest.raises(ValueError, match="existing blocks"):
        apply_draft_edits(
            base,
            DraftEdits(
                replacements=[
                    GeneratedBlock(block_id="invented", text="text", claims=[])
                ],
                unresolved_issue_ids=["i1"],
            ),
        )


def test_used_search_hit_loads_structural_article_once_without_a_search() -> None:
    context = RunContext(timeout_seconds=600)
    ledger = EvidenceLedger()
    center = original()
    center.metadata["article_closure_complete"] = False
    ledger.add([center], context)
    sibling = center.model_copy(
        update={
            "chunk_id": "next",
            "text_hash": "",
            "text": "Except when the document has expired.",
        }
    )
    sibling.metadata = {"article_closure_complete": True}
    broker = MagicMock(spec=CorpusBroker)
    broker.hydrate_search_results.return_value = {("source-1", 1): [center, sibling]}
    acquirer = SourceAcquirer(
        registry=CapabilityRegistry([]),
        context=context,
        ledger=ledger,
        policy=WorkflowPolicy(),
        search_adapter=cast(ScopedSearchAdapter, MagicMock()),
        broker=broker,
    )
    assert acquirer.complete_articles({1}, False)
    actual = ledger.get(2)
    assert actual is not None and actual.text == sibling.text
    assert not acquirer.complete_articles({1}, False)
    assert broker.hydrate_search_results.call_count == 1
    assert acquirer.searches == 0
    assert acquirer.receipts[0]["tool"] == "read_article_context"


def test_confirmed_prose_correction_reuses_independent_diagnosis() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    examiner = MagicMock()
    workflow.diagnoser = examiner
    check = ReviewCheck(
        id="claim:c1", claim_id="c1", instructions="Check the condition"
    )
    review = ReviewResult(completed=True, flags=[check], scores={check.id: 0.9})
    workflow.draft_adjudication = PublicationReview(
        findings=[
            PublicationFinding(
                check_id=check.id,
                disposition="defect",
                target="assertion",
                reason="Omitted condition",
                required_change="Preserve the required document condition.",
                supports=draft().claims[0].supports,
                answer_quotes=[draft().answer],
                repair_kind="correction",
            )
        ]
    )
    workflow._diagnose_review("Soru", "", draft(), review)
    examiner.diagnose.assert_not_called()
    assert workflow.review_diagnoses is not None
    assert workflow.review_diagnoses.research_tasks == []
    assert workflow.review_diagnoses.diagnoses[0].kind == "correction"


def test_overlapping_confirmed_gaps_dispatch_one_search_without_rediagnosis() -> None:
    workflow, _, _ = engine([], [])
    workflow.plan = plan()
    workflow.ledger.add([original()], workflow.context)
    examiner = MagicMock()
    workflow.diagnoser = examiner
    checks = [
        ReviewCheck(
            id=identity, issue_id="i1", instructions="Check the operative effect"
        )
        for identity in ("condition", "outcome")
    ]
    review = ReviewResult(
        completed=True, flags=checks, scores={row.id: 0.9 for row in checks}
    )
    workflow.draft_adjudication = PublicationReview(
        findings=[
            PublicationFinding(
                check_id=row.id,
                disposition="defect",
                target="assertion",
                reason="The effect of the identified exception has not been read.",
                required_change="Read and apply the exception's operative conditions.",
                supports=draft().claims[0].supports,
                answer_quotes=[draft().answer],
                repair_kind="research",
                research_query="application exception operative conditions",
            )
            for row in checks
        ]
    )
    with patch.object(workflow, "_acquire") as acquire:
        workflow._diagnose_review("Soru", "", draft(), review)
    examiner.diagnose.assert_not_called()
    acquire.assert_called_once()
    actions = acquire.call_args.args[0]
    assert len(actions) == 1
    assert actions[0].arguments["query"] == "application exception operative conditions"
    assert actions[0].issue_ids == ["i1"]
    assert workflow.review_diagnoses is not None
    assert {row.check_ids[0] for row in workflow.review_diagnoses.diagnoses} == {
        row.id for row in checks
    }
