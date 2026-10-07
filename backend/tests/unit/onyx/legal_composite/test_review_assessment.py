"""Fail closed on incomplete originals and contradictory legal-review judgments."""

from typing import TypeVar

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_composite.engine import LegalCompositeEngine, review_assessment
from onyx.legal_composite.models import (
    AnswerReview,
    DraftAnswer,
    NeedReview,
    PassageSupport,
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.tracing.flows import LLMFlow

# Exact decisive excerpts of archived Vaka02 originals34/81, not answer templates.
PAID_REPAIR = (
    "Ancak, izin hak sahibinin tamir masrafları dışında başka bir ödeme yapmamış "
    "olması ve bu ödemenin izin hak sahibi ile faaliyeti yapan kişi arasındaki "
    "ilişkiden etkilenmemesi gerekir."
)
REFUND_CLOCK = (
    "vergilerin yükümlüye tebliği tarihinden itibaren bir yıl içerisinde "
    "gümrük idaresine müracaat edilmesi gerekir."
)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def original(
    text: str,
    chunk: str,
    *,
    citable: bool = True,
    metadata: dict[str, JsonValue] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        source_id="12328001-2f21-407e-b230-7d50a8dbf28f",
        chunk_id=chunk,
        text=text,
        metadata=metadata or {},
        search_doc=SearchDoc(
            document_id="12328001-2f21-407e-b230-7d50a8dbf28f",
            chunk_ind=0,
            semantic_identifier="4458 SAYILI GÜMRÜK KANUNU",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={},
            match_highlights=[],
        )
        if citable
        else None,
    )


@pytest.fixture
def ledger() -> EvidenceLedger:
    result = EvidenceLedger()
    result.add(
        [original(PAID_REPAIR, "paid-repair"), original(REFUND_CLOCK, "refund-clock")],
        RunContext(),
    )
    return result


@pytest.fixture
def plan() -> ResearchPlan:
    return ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="repair",
                question="Ücretli tamir hesabının şartları nelerdir?",
                governing_source="Uygulanabilir tamir hükmü",
                conditions_to_check=["Başka ödeme ve ilişkiden etkilenme"],
            ),
            ResearchNeed(
                need_id="refund",
                question="Geri verme süresinin başlangıcı nedir?",
                governing_source="Uygulanabilir geri verme hükmü",
                conditions_to_check=["Tebliğ başlangıcı"],
            ),
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


@pytest.fixture
def draft() -> DraftAnswer:
    return DraftAnswer(
        answer=f"{PAID_REPAIR} [1]\n\n{REFUND_CLOCK} [2]",
        unresolved_need_ids=[],
    )


@pytest.fixture
def review() -> AnswerReview:
    return AnswerReview(
        request_coverage_complete=True,
        material_claims_supported=True,
        counter_authority_checked=True,
        needs=[
            NeedReview(
                need_id="repair",
                status="supported",
                supports=[PassageSupport(citation=1, quotation=PAID_REPAIR)],
                conditions_preserved=True,
                explanation="Her iki koşul korunmuştur.",
            ),
            NeedReview(
                need_id="refund",
                status="supported",
                supports=[PassageSupport(citation=2, quotation=REFUND_CLOCK)],
                conditions_preserved=True,
                explanation="Başlangıç tebliğ tarihidir.",
            ),
        ],
        defects=[],
        repair_actions=[],
    )


def test_complete_matching_witnesses_pass_structural_gate(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    assert review_assessment(plan, draft, review, ledger, {1, 2}) == (True, True, [])


@pytest.mark.parametrize("delivered", [set(), {1}])
def test_writer_must_receive_every_cited_original(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    delivered: set[int],
) -> None:
    passed, safe, gaps = review_assessment(plan, draft, review, ledger, delivered)
    assert not passed and not safe and gaps


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_review_needs_exactly_match_frozen_request(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    change: str,
) -> None:
    if change == "missing":
        review.needs.pop()
    elif change == "duplicate":
        review.needs.append(review.needs[0].model_copy(deep=True))
    else:
        review.needs[1].need_id = "invented"
    passed, safe, gaps = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and not safe and gaps


@pytest.mark.parametrize("change", ["invented_quote", "unused_support", "conditions"])
def test_delivered_condition_omission_is_not_source_completeness(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    change: str,
) -> None:
    if change == "invented_quote":
        review.needs[0].supports[0].quotation = "Başka ödeme yapılması önemli değildir."
    elif change == "unused_support":
        draft.answer = f"{PAID_REPAIR} [1]"
    else:
        review.needs[0].conditions_preserved = False
    passed, safe, gaps = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and not safe and gaps


@pytest.mark.parametrize("flag", ["truncated", "external", "derived", "untrusted"])
@pytest.mark.parametrize("nested", [False, True])
def test_navigation_or_derived_text_cannot_be_legal_original(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    flag: str,
    nested: bool,
) -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                PAID_REPAIR,
                "repair",
                metadata={"canonical_metadata": {flag: True}}
                if nested
                else {flag: True},
            ),
            original(REFUND_CLOCK, "refund"),
        ],
        RunContext(),
    )
    passed, safe, _ = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and not safe


def test_search_title_without_citable_original_is_rejected(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview
) -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(PAID_REPAIR, "repair", citable=False),
            original(REFUND_CLOCK, "refund"),
        ],
        RunContext(),
    )
    assert review_assessment(plan, draft, review, ledger, {1, 2})[:2] == (False, False)


def test_incorrect_need_is_unsafe_even_when_aggregate_judge_flag_says_supported(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    review.needs[1].status = "incorrect"
    review.needs[1].explanation = "Süre başlangıcı yanlış uygulanmıştır."
    passed, safe, gaps = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and not safe and gaps


def test_precise_gap_does_not_become_complete_acceptance(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    draft.unresolved_need_ids = ["refund"]
    draft.answer += "\nSomut uygulanma olgusu henüz bilinmiyor."
    review.needs[1].status = "unresolved"
    review.needs[1].explanation = "Somut uygulanma olgusu henüz bilinmiyor."
    review.needs[1].gap_disclosure = "Somut uygulanma olgusu henüz bilinmiyor."
    passed, safe, gaps = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and safe and gaps


@pytest.mark.parametrize("missing", ["draft_need", "disclosure", "literal_match"])
def test_unresolved_outcome_must_be_explicitly_disclosed_in_the_draft(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    missing: str,
) -> None:
    draft.unresolved_need_ids = [] if missing == "draft_need" else ["refund"]
    review.needs[1].status = "unresolved"
    review.needs[1].gap_disclosure = (
        None if missing == "disclosure" else "Somut uygulanma olgusu henüz bilinmiyor."
    )
    if missing != "literal_match":
        draft.answer += "\nSomut uygulanma olgusu henüz bilinmiyor."
    assert review_assessment(plan, draft, review, ledger, {1, 2})[:2] == (False, False)


@pytest.mark.parametrize(
    "flag", ["request_coverage_complete", "counter_authority_checked"]
)
def test_aggregate_coverage_and_counter_authority_remain_separate_gates(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    flag: str,
) -> None:
    setattr(review, flag, False)
    passed, _safe, gaps = review_assessment(plan, draft, review, ledger, {1, 2})
    assert not passed and gaps


class FakeGateway:
    last_call_id: str | None = None

    def __init__(
        self,
        values: list[BaseModel | RunStopped],
        deliveries: list[set[int]] | None = None,
    ) -> None:
        self.values = values
        self.calls: list[LLMFlow] = []
        self.requests: list[tuple[str, dict[str, JsonValue], bool]] = []
        self.last_delivered_citations = {1, 2}
        self.deliveries = deliveries

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel:
        self.calls.append(flow)
        self.requests.append((system, payload, finalizing))
        if self.deliveries is not None:
            self.last_delivered_citations = self.deliveries.pop(0)
        value = self.values.pop(0)
        if isinstance(value, RunStopped):
            raise value
        return response_type.model_validate(value.model_dump(mode="json"))


class FakeAcquirer:
    def definitions(self) -> list[dict[str, JsonValue]]:
        return []

    def acquire(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        raise AssertionError(
            f"No new acquisition expected for {len(plan.needs)} needs: {len(actions)} actions"
        )


def engine(gateway: FakeGateway, ledger: EvidenceLedger) -> LegalCompositeEngine:
    return LegalCompositeEngine(
        gateway=gateway,
        acquirer=FakeAcquirer(),
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=1),
        check_active=lambda: None,
        research_available=lambda: False,
    )


@pytest.mark.parametrize("stop_at", ["plan", "answer", "review"])
def test_budget_exhaustion_never_publishes_unreviewed_draft(
    plan: ResearchPlan, draft: DraftAnswer, ledger: EvidenceLedger, stop_at: str
) -> None:
    values: list[BaseModel | RunStopped] = []
    if stop_at != "plan":
        values.append(plan)
    if stop_at == "review":
        values.append(draft)
    values.append(RunStopped("Workflow model budget exhausted"))
    result = engine(FakeGateway(values), ledger).run("Kaynaklı hukuki sonuç gerekli.")
    assert result.status == "unavailable"
    assert result.answer is None
    assert result.gaps


def test_known_incorrect_draft_cannot_be_exposed_as_partial_answer(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    review.needs[0].status = "incorrect"
    gateway = FakeGateway([plan, draft, review])
    result = engine(gateway, ledger).run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "unavailable"
    assert result.answer is None
    assert gateway.calls == [
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        LLMFlow.LEGAL_COMPOSITE_ANSWER,
        LLMFlow.LEGAL_COMPOSITE_REVIEW,
    ]


@pytest.mark.parametrize("clipped_stage", ["answer", "review"])
def test_provider_context_fit_cannot_silently_drop_a_cited_original(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    clipped_stage: str,
) -> None:
    deliveries = [
        {1, 2},
        {1} if clipped_stage == "answer" else {1, 2},
        {1} if clipped_stage == "review" else {1, 2},
    ]
    gateway = FakeGateway([plan, draft, review], deliveries)
    result = engine(gateway, ledger).run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "unavailable" and result.answer is None


def test_safe_disclosed_partial_survives_repair_budget_exhaustion(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    disclosure = "Somut uygulanma olgusu henüz bilinmiyor."
    draft.answer += f"\n{disclosure}"
    draft.unresolved_need_ids = ["refund"]
    review.needs[1].status = "unresolved"
    review.needs[1].gap_disclosure = disclosure
    review.needs[1].explanation = disclosure
    gateway = FakeGateway([plan, draft, review, RunStopped("Repair budget exhausted")])
    instance = engine(gateway, ledger)
    instance.policy = WorkflowPolicy(max_reviews=2)
    result = instance.run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "partial" and result.answer == draft.answer
    assert result.gaps and result.review == review


def test_research_limit_retains_final_answer_and_review(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    gateway = FakeGateway(
        [plan, RunStopped("Research allocation retained"), draft, review]
    )
    instance = engine(gateway, ledger)
    instance.research_available = lambda: True
    result = instance.run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "verified"
    assert result.answer == draft.answer


def test_cancellation_returns_no_answer(ledger: EvidenceLedger) -> None:
    instance = engine(FakeGateway([]), ledger)

    def cancelled() -> None:
        raise RunStopped("Research cancelled")

    instance.check_active = cancelled
    result = instance.run("Bir soru")
    assert result.status == "cancelled" and result.answer is None


def test_wrong_planner_source_free_label_cannot_publish_legal_outcome(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    plan.requires_sources = False
    draft.answer = "Gümrük vergileri her durumda iade edilir."
    for need in review.needs:
        need.supports = []
    result = engine(FakeGateway([plan, draft, review]), ledger).run(
        "Kusurlu makinenin gümrük vergilerini geri alabilir miyim?"
    )
    assert result.status == "unavailable" and result.answer is None
    assert result.plan and result.plan.requires_sources


def test_completed_receipts_survive_research_phase_interruption(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    plan.initial_actions = [
        SourceAction(need_ids=["repair"], tool="read_original", arguments={})
    ]

    class InterruptedAcquirer(FakeAcquirer):
        def __init__(self) -> None:
            self.last_receipts: list[dict[str, JsonValue]] = [
                {"tool": "read_original", "status": "found", "citations": [1]},
                {"tool": "read_later", "status": "truncated", "citations": []},
            ]

        def acquire(
            self, actions: list[SourceAction], plan: ResearchPlan
        ) -> list[dict[str, JsonValue]]:
            assert actions == plan.initial_actions
            raise RunStopped("Research deadline; finalization retained")

    gateway = FakeGateway([plan, draft, review])
    instance = engine(gateway, ledger)
    acquirer = InterruptedAcquirer()
    instance.acquirer = acquirer
    result = instance.run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "verified"
    assert instance.receipts == acquirer.last_receipts
    assert gateway.requests[1][1]["receipts"] == acquirer.last_receipts


def test_acquisition_cancellation_does_not_spend_finalization_reserve(
    plan: ResearchPlan, draft: DraftAnswer, review: AnswerReview, ledger: EvidenceLedger
) -> None:
    plan.initial_actions = [
        SourceAction(need_ids=["repair"], tool="read_original", arguments={})
    ]
    cancelled = False

    class CancelledAcquirer(FakeAcquirer):
        def acquire(
            self, actions: list[SourceAction], plan: ResearchPlan
        ) -> list[dict[str, JsonValue]]:
            nonlocal cancelled
            assert actions == plan.initial_actions
            cancelled = True
            raise RunStopped("Research cancelled")

    def check_active() -> None:
        if cancelled:
            raise RunStopped("Research cancelled")

    gateway = FakeGateway([plan, draft, review])
    instance = engine(gateway, ledger)
    instance.acquirer = CancelledAcquirer()
    instance.check_active = check_active
    result = instance.run("Ücretli tamirin şartları ve iade süresi?")
    assert result.status == "cancelled" and result.answer is None
    assert gateway.calls == [LLMFlow.LEGAL_COMPOSITE_RESEARCH]
