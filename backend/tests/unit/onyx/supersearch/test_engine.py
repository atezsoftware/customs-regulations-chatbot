"""Exercise canonical acquisition, actual deciding delivery and final claim guards."""

import sys
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel, JsonValue

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
from onyx.legal_composite.models import (
    AnswerReview,
    ConditionReview,
    NeedReview,
    PassageSupport,
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.supersearch.acquisition import SupersearchAcquirer, corpus_specs
from onyx.supersearch.engine import (
    SupersearchEngine,
    apply_passage_patches,
    initial_source_actions,
)
from onyx.supersearch.models import AnswerRepair, PassagePatch, WriterDecision
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
SOURCE_ID = "12328001-2f21-407e-b230-7d50a8dbf28f"
ORIGINAL = "Başvuru, bildirim tarihinden itibaren bir yıl içerisinde yapılır."
ANSWER = ORIGINAL + " [1]"


def original() -> EvidenceItem:
    return EvidenceItem(
        source_id=SOURCE_ID,
        chunk_id="atomic-clock",
        text=ORIGINAL,
        metadata={"article_closure_complete": True},
        search_doc=SearchDoc(
            document_id=SOURCE_ID,
            chunk_ind=0,
            semantic_identifier="PC özgün hüküm",
            blurb=ORIGINAL,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": "atomic-clock"},
            match_highlights=[],
        ),
    )


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="clock",
                question="Başvuru süresi nedir?",
                governing_source="PC özgün hüküm",
                conditions_to_check=["Bildirim tarihinden itibaren bir yıl"],
            )
        ],
        initial_actions=[
            SourceAction(
                need_ids=["clock"],
                tool="read_named_provision",
                arguments={"source_name": "PC özgün hüküm", "article": "10"},
            )
        ],
        missing_user_facts=[],
    )


def review(*, supported: bool = True, excerpt: str = ORIGINAL) -> AnswerReview:
    return AnswerReview(
        request_coverage_complete=True,
        material_claims_supported=supported,
        counter_authority_checked=True,
        needs=[
            NeedReview(
                need_id="clock",
                status="supported" if supported else "incorrect",
                supports=[PassageSupport(citation=1, quotation=ORIGINAL)],
                conditions_preserved=supported,
                condition_reviews=[
                    ConditionReview(
                        condition_index=0,
                        status="preserved" if supported else "incorrect",
                        answer_excerpt=excerpt,
                        support_citations=[1],
                    )
                ],
                explanation="Bildirim başlangıcı ve bir yıl birlikte korunur.",
            )
        ],
        defects=[] if supported else ["Başlangıç tarihi değiştirilmiş."],
        repair_actions=[],
    )


class FixtureGateway:
    def __init__(self, ledger: EvidenceLedger, responses: list[BaseModel]) -> None:
        self.ledger = ledger
        self.responses = responses
        self.flows: list[LLMFlow] = []
        self.payloads: list[dict[str, JsonValue]] = []
        self.last_call_id: str | None = None
        self.last_delivered_citations: set[int] = set()

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel:
        self.flows.append(flow)
        self.payloads.append(payload)
        assert "No web" in system
        assert finalizing == (flow != LLMFlow.SUPERSEARCH_PLAN)
        call_id = str(len(self.flows))
        records = payload.get("original_evidence", [])
        assert isinstance(records, list)
        self.ledger.record_delivery(
            call_id, flow.value, cast(list[dict[str, JsonValue]], records)
        )
        self.last_call_id = call_id
        self.last_delivered_citations = self.ledger.completely_delivered(call_id)
        result = self.responses.pop(0)
        assert isinstance(result, response_type)
        return result


def engine(
    responses: list[BaseModel], *, found: bool = True
) -> tuple[SupersearchEngine, FixtureGateway, list[str]]:
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    executions: list[str] = []

    def read(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executions.append("read_named_provision")
        return ToolOutcome(
            status=OutcomeStatus.FOUND if found else OutcomeStatus.NOT_FOUND,
            summary="Authorized PC originals",
            evidence=[original()] if found else [],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_named_provision",
                description="Canonical fixture read",
                parameters={"type": "object"},
                handler=read,
            )
        ]
    )
    acquirer = SupersearchAcquirer(
        registry,
        context,
        ledger,
        WorkflowPolicy(
            timeout_seconds=float("inf"),
            max_tools=sys.maxsize,
            max_search_calls=sys.maxsize,
        ),
    )
    gateway = FixtureGateway(ledger, responses)
    return (
        SupersearchEngine(
            gateway=gateway,
            acquirer=acquirer,
            ledger=ledger,
            check_active=context.check_active,
        ),
        gateway,
        executions,
    )


def test_exact_provision_path_has_three_calls_and_no_broad_discovery() -> None:
    workflow, gateway, executions = engine(
        [
            plan(),
            WriterDecision(answer=ANSWER, unresolved_need_ids=[], actions=[]),
            review(),
        ]
    )
    result = workflow.run("Başvuru süresi nedir?")
    assert result.status == "verified" and result.answer == ANSWER
    assert executions == ["read_named_provision"]
    assert gateway.flows == [
        LLMFlow.SUPERSEARCH_PLAN,
        LLMFlow.SUPERSEARCH_ANSWER,
        LLMFlow.SUPERSEARCH_REVIEW,
    ]
    assert gateway.payloads[0]["original_evidence"] == []
    assert gateway.payloads[1]["required_evidence_numbers"] == [1]
    assert (
        gateway.payloads[1]["original_evidence"]
        == gateway.payloads[2]["original_evidence"]
    )
    assert workflow.ledger.completely_delivered("2") == {1}


def test_targeted_correction_changes_the_published_body_and_is_rechecked() -> None:
    wrong = "Başvuru, ödeme tarihinden itibaren bir yıl içerisinde yapılır. [1]"
    workflow, gateway, _ = engine(
        [
            plan(),
            WriterDecision(answer=wrong, unresolved_need_ids=[], actions=[]),
            review(supported=False, excerpt=wrong),
            AnswerRepair(
                patches=[
                    PassagePatch(
                        old_text="ödeme tarihinden", new_text="bildirim tarihinden"
                    )
                ],
                unresolved_need_ids=[],
            ),
            review(),
        ]
    )
    result = workflow.run("Başvuru süresi nedir?")
    assert result.status == "verified" and result.answer == ANSWER
    assert gateway.flows.count(LLMFlow.SUPERSEARCH_ANSWER) == 1
    assert gateway.flows[-2:] == [
        LLMFlow.SUPERSEARCH_REPAIR,
        LLMFlow.SUPERSEARCH_REVIEW,
    ]
    assert gateway.payloads[-1]["draft"] == {
        "answer": ANSWER,
        "unresolved_need_ids": [],
    }


def test_failed_duplicate_source_frontier_does_not_repeat_a_read() -> None:
    workflow, gateway, executions = engine(
        [
            plan(),
            WriterDecision(
                answer=None,
                unresolved_need_ids=["clock"],
                actions=plan().initial_actions,
            ),
        ]
    )
    result = workflow.run("Başvuru süresi nedir?")
    assert result.answer is None and result.status == "unavailable"
    assert len(executions) == 1 and len(gateway.flows) == 2


def test_empty_source_result_is_a_bounded_gap_not_a_negative_legal_rule() -> None:
    workflow, gateway, executions = engine([plan()], found=False)
    result = workflow.run("Başvuru süresi nedir?")
    assert (
        result.status == "partial"
        and result.answer
        and "hükmün bulunmadığını göstermez" in result.answer
    )
    assert gateway.flows == [LLMFlow.SUPERSEARCH_PLAN]
    assert executions == ["read_named_provision"]


def test_material_need_without_conditions_fails_before_reading() -> None:
    empty = plan()
    empty.needs[0].conditions_to_check = []
    workflow, _, executions = engine([empty])
    result = workflow.run("Başvuru süresi nedir?")
    assert result.status == "unavailable" and not executions


def test_discovery_only_fills_uncovered_need() -> None:
    frozen = plan()
    frozen.needs.append(
        ResearchNeed(
            need_id="proof",
            question="Belgeyi kim düzenler?",
            governing_source="İlgili belge hükmü",
            conditions_to_check=["Yetkili belge düzenleyicisi"],
        )
    )
    actions = initial_source_actions(frozen, "Süre ve belge?")
    assert len(actions) == 2
    assert actions[0].tool == "read_named_provision"
    assert actions[1].tool == "search_corpus" and actions[1].need_ids == ["proof"]


def test_patches_reject_ambiguous_or_overlapping_witnesses() -> None:
    with pytest.raises(RunStopped, match="exact answer passage"):
        apply_passage_patches(
            "aynı aynı",
            AnswerRepair(
                patches=[PassagePatch(old_text="aynı", new_text="yeni")],
                unresolved_need_ids=[],
            ),
        )
    with pytest.raises(RunStopped, match="overlap"):
        apply_passage_patches(
            "abcde",
            AnswerRepair(
                patches=[
                    PassagePatch(old_text="abcd", new_text="a"),
                    PassagePatch(old_text="bcde", new_text="b"),
                ],
                unresolved_need_ids=[],
            ),
        )


def test_capabilities_do_not_expose_inventory_or_external_research() -> None:
    from unittest.mock import MagicMock

    specs = corpus_specs(MagicMock())
    assert {spec.name for spec in specs} <= {
        "resolve_source",
        "read_source_range",
        "read_chunk",
        "read_chunk_context",
        "read_provision",
        "read_named_provision",
        "search_source_text",
        "follow_reference",
        "compare_versions",
        "search_corpus",
    }
    assert "query_corpus" not in {spec.name for spec in specs}
    assert all(not spec.external and not spec.orchestrates for spec in specs)


def test_duplicate_source_kind_actions_coalesce_in_one_scope() -> None:
    workflow, _, executions = engine([])
    frozen = plan()
    from onyx.db.legal_composite_sources import SourceKind

    actions = [
        frozen.initial_actions[0].model_copy(update={"source_kind": kind})
        for kind in (SourceKind.JUDICIAL_DECISION, SourceKind.UNKNOWN)
    ]
    workflow.acquirer.acquire(actions, frozen)
    assert executions == ["read_named_provision"]
