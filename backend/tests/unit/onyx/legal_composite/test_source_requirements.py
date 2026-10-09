"""Requirements cannot be invented, and section repairs preserve other outcomes."""

from typing import TypeVar

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.claim_edits import ClaimEdit, ClaimRepairEdits
from onyx.legal_composite.draft_composition import DraftComposition
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AnswerSection,
    AuthorityDependency,
    CompositeWorkflowResult,
    DraftClaim,
    DraftPatch,
    PassageSupport,
    ReviewCheck,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    SpanSupport,
    WorkflowPolicy,
)
from onyx.legal_composite.models import (
    IssueResearchNeed as ResearchNeed,
)
from onyx.legal_composite.models import (
    IssueResearchPlan as ResearchPlan,
)
from onyx.legal_composite.models import (
    IssueResearchStep as ResearchStep,
)
from onyx.legal_composite.models import ResearchPlan as BaseResearchPlan
from onyx.legal_composite.models import (
    StructuredDraftAnswer as DraftAnswer,
)
from onyx.legal_composite.requirements import (
    RequirementLedger,
    apply_patch,
    draft_binding_gaps,
)
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for
from tests.unit.onyx.legal_composite.test_review_assessment import original

T = TypeVar("T", bound=BaseModel)
RULE_A = "A işlemi için belge gerekir; B işlemi bu kuralın dışındadır."
RULE_B = "Başvuru süresi bildirim tarihinde başlar."


def fixture() -> tuple[
    EvidenceLedger, ResearchPlan, list[SourceRequirement], DraftAnswer
]:
    ledger = EvidenceLedger()
    originals = [original(RULE_A, "a"), original(RULE_B, "b")]
    for item in originals:
        assert item.search_doc is not None and item.chunk_id is not None
        item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    ledger.add(originals, RunContext())
    plan = ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id=need, question=need, governing_source="", conditions_to_check=[]
            )
            for need in ("a", "b")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    requirements = [
        SourceRequirement(
            requirement_id=f"r_{need}",
            need_id=need,
            dimension="procedure_and_deadlines",
            rule=rule,
            application="Kullanıcı olgularına koşullu uygulanır.",
            supports=[SpanSupport(citation=citation, quotation=rule)],
        )
        for need, citation, rule in (("a", 1, RULE_A), ("b", 2, RULE_B))
    ]
    draft = DraftAnswer(
        unresolved_need_ids=[],
        sections=[
            AnswerSection(
                section_id=f"s_{need}", need_ids=[need], text=f"{rule} [{citation}]"
            )
            for need, citation, rule in (("a", 1, RULE_A), ("b", 2, RULE_B))
        ],
        claims=[
            DraftClaim(
                claim_id=f"c_{need}",
                section_id=f"s_{need}",
                need_ids=[need],
                answer_excerpt=rule,
                requirement_ids=[f"r_{need}"],
            )
            for need, rule in (("a", RULE_A), ("b", RULE_B))
        ],
    )
    return ledger, plan, requirements, draft


@pytest.mark.parametrize("change", ["quote", "delivery", "issue", "derived"])
def test_requirement_update_rejects_unread_or_invented_support(change: str) -> None:
    ledger, plan, requirements, _ = fixture()
    requirement = requirements[0]
    delivered = {1, 2}
    if change == "quote":
        requirement.supports[0].quotation = "Olmayan bir yıl süresi."
    elif change == "delivery":
        delivered = {2}
    elif change == "issue":
        requirement.need_id = "unknown"
    else:
        ledger = EvidenceLedger()
        ledger.add(
            [original(RULE_A, "a", metadata={"canonical_metadata": {"derived": True}})],
            RunContext(),
        )
    recorded = RequirementLedger(ledger)
    with pytest.raises(InvalidSourceAction):
        recorded.update([requirement], plan, delivered)
    assert recorded.records() == []


def test_requirement_id_cannot_change_and_receipt_binds_exact_source_span() -> None:
    ledger, plan, requirements, _ = fixture()
    recorded = RequirementLedger(ledger)
    recorded.update(requirements, plan, {1, 2})
    changed = requirements[0].model_copy(update={"rule": "Different consequence"})
    with pytest.raises(InvalidSourceAction, match="immutable"):
        recorded.update([changed], plan, {1, 2})
    receipt = recorded.export()[0]["original_bindings"]
    assert isinstance(receipt, list) and receipt[0]["start"] == 0
    assert receipt[0]["end"] == len(RULE_A)
    item = ledger.get(1)
    assert item is not None and receipt[0]["text_hash"] == item.text_hash


def test_corrected_interpretation_supersedes_obligation_without_erasing_history() -> (
    None
):
    ledger, plan, requirements, draft = fixture()
    wrong = requirements[0].model_copy(
        deep=True, update={"rule": "B işlemi için de belge gerekir."}
    )
    recorded = RequirementLedger(ledger)
    recorded.update([wrong, requirements[1]], plan, {1, 2})
    correction = requirements[0].model_copy(
        deep=True,
        update={
            "requirement_id": "r_a_v2",
            "supersedes_requirement_ids": ["r_a"],
        },
    )
    recorded.update([correction], plan, {1, 2})
    assert [row.requirement_id for row in recorded.records()] == ["r_b", "r_a_v2"]
    assert "claim:c_a:invalid_requirement_binding" in draft_binding_gaps(
        draft, plan, recorded.records(), ledger, {1, 2}
    )
    draft.claims[0].requirement_ids = ["r_a_v2"]
    assert draft_binding_gaps(draft, plan, recorded.records(), ledger, {1, 2}) == []
    checks = build_checks(
        "A ve B işlemleri?",
        plan,
        draft,
        recorded.records(),
        [],
        {1, 2},
        ledger=ledger,
    )
    assert "requirement:r_a" not in checks
    assert "requirement:r_a_v2" in checks
    history = {row["requirement_id"]: row for row in recorded.export()}
    assert history["r_a"]["rule"] == wrong.rule
    assert history["r_a"]["active"] is False
    assert history["r_a"]["superseded_by_requirement_ids"] == ["r_a_v2"]
    assert history["r_a_v2"]["active"] is True
    assert history["r_a_v2"]["superseded_by_requirement_ids"] == []
    assert history["r_a"]["original_bindings"] == history["r_a_v2"]["original_bindings"]
    recorded.update([wrong, correction], plan, {1, 2})
    assert recorded.export() == list(history.values())


def test_supersession_chain_keeps_only_active_original_citations() -> None:
    ledger, plan, requirements, _ = fixture()
    refined = original("A işlemi için belge gerekir. B işlemi için belge aranmaz.", "c")
    assert refined.search_doc is not None and refined.chunk_id is not None
    refined.search_doc.metadata["regulatory_chunk_id"] = refined.chunk_id
    ledger.add([refined], RunContext())
    recorded = RequirementLedger(ledger)
    recorded.update(requirements, plan, {1, 2})
    first = requirements[0].model_copy(
        deep=True,
        update={
            "requirement_id": "r_a_v2",
            "supersedes_requirement_ids": ["r_a"],
        },
    )
    latest = first.model_copy(
        deep=True,
        update={
            "requirement_id": "r_a_v3",
            "supersedes_requirement_ids": ["r_a_v2"],
            "rule": refined.text,
            "supports": [PassageSupport(citation=3, quotation=refined.text)],
        },
    )
    recorded.update([first, latest], plan, {1, 2, 3})
    assert [row.requirement_id for row in recorded.records({"a"})] == ["r_a_v3"]
    assert recorded.citations() == {2, 3}
    assert recorded.citations({"a"}) == {3}
    history = {row["requirement_id"]: row for row in recorded.export()}
    assert len(history) == 4
    assert history["r_a"]["active"] is False
    assert history["r_a_v2"]["active"] is False
    assert history["r_a_v2"]["superseded_by_requirement_ids"] == ["r_a_v3"]
    assert history["r_a_v3"]["active"] is True


@pytest.mark.parametrize(
    "change, message",
    [
        ("quotation", "exact delivered original"),
        ("delivery", "exact delivered original"),
        ("unknown", "Unknown requirement supersession target"),
        ("other_issue", "cannot cross issues"),
        ("self", "cannot supersede itself"),
        ("duplicate", "Duplicate requirement supersession target"),
        ("cycle", "cannot form a cycle"),
    ],
)
def test_bad_supersession_batch_preserves_history_and_active_obligations(
    change: str, message: str
) -> None:
    ledger, plan, requirements, _ = fixture()
    recorded = RequirementLedger(ledger)
    recorded.update(requirements, plan, {1, 2})
    before = recorded.export()
    correction = requirements[0].model_copy(
        deep=True,
        update={
            "requirement_id": "r_a_v2",
            "supersedes_requirement_ids": ["r_a"],
        },
    )
    extras: list[SourceRequirement] = []
    delivered = {1, 2}
    if change == "quotation":
        correction.supports[0].quotation = "Olmayan bir şart."
    elif change == "delivery":
        delivered = {2}
    elif change == "unknown":
        correction.supersedes_requirement_ids = ["unknown"]
    elif change == "other_issue":
        correction.supersedes_requirement_ids = ["r_b"]
    elif change == "self":
        correction.supersedes_requirement_ids = [correction.requirement_id]
    elif change == "duplicate":
        correction.supersedes_requirement_ids = ["r_a", "r_a"]
    else:
        correction.supersedes_requirement_ids = ["r_a_v3"]
        extras = [
            correction.model_copy(
                deep=True,
                update={
                    "requirement_id": "r_a_v3",
                    "supersedes_requirement_ids": ["r_a_v2"],
                },
            )
        ]
    valid = requirements[1].model_copy(
        deep=True, update={"requirement_id": "r_b_extra"}
    )
    with pytest.raises(InvalidSourceAction, match=message):
        recorded.update([valid, correction, *extras], plan, delivered)
    assert recorded.export() == before
    assert recorded.records() == requirements
    assert recorded.citations() == {1, 2}


def test_claim_binding_reuses_requirement_without_reinventing_quote() -> None:
    ledger, plan, requirements, draft = fixture()
    assert draft_binding_gaps(draft, plan, requirements, ledger, {1, 2}) == []
    draft.claims[0].requirement_ids = ["r_b"]
    assert "claim:c_a:invalid_requirement_binding" in draft_binding_gaps(
        draft, plan, requirements, ledger, {1, 2}
    )


def test_patch_cannot_rewrite_unaffected_section_or_claim() -> None:
    _, _, _, draft = fixture()
    patch = DraftPatch(
        sections=[draft.sections[1]], claims=[draft.claims[1]], unresolved_need_ids=[]
    )
    with pytest.raises(InvalidSourceAction, match="exactly"):
        apply_patch(draft, patch, {"s_a"})
    result = apply_patch(draft, patch, {"s_b"})
    assert (
        result.sections[0] == draft.sections[0] and result.claims[0] == draft.claims[0]
    )
    assert result.answer == "\n\n".join(section.text for section in result.sections)


@pytest.mark.parametrize("retain_additional_claim", [False, True])
def test_engine_repairs_only_missing_requirement_section_and_rechecks(
    retain_additional_claim: bool,
) -> None:
    ledger, plan, requirements, draft = fixture()
    draft.claims[0].answer_excerpt = draft.sections[0].text
    draft.claims[1].answer_excerpt = draft.sections[1].text
    if retain_additional_claim:
        draft.claims.append(
            DraftClaim(
                claim_id="c_b_retained",
                section_id="s_b",
                need_ids=["b"],
                answer_excerpt="Sürenin başlangıcı bildirim tarihidir. [2]",
                requirement_ids=["r_b"],
            )
        )
    claim_order = [
        claim.claim_id for claim in draft.claims if claim.section_id == "s_b"
    ]
    draft = DraftAnswer(
        sections=[
            AnswerSection(section_id="s_a", need_ids=["a"], claim_ids=["c_a"]),
            AnswerSection(
                section_id="s_b", need_ids=["b"], text="", claim_ids=claim_order
            ),
        ],
        claims=draft.claims,
        unresolved_need_ids=[],
    )
    initial = draft.model_copy(deep=True)
    initial.claims[1].answer_excerpt = "Başvurulabilir [2]."
    initial.sections[1].text = "\n\n".join(
        claim.answer_excerpt for claim in initial.claims if claim.section_id == "s_b"
    )
    initial.answer = "\n\n".join(s.text for s in initial.sections)
    patch = ClaimRepairEdits(
        claims=[
            ClaimEdit.model_validate(draft.claims[1].model_dump(exclude={"need_ids"}))
        ],
        unresolved_need_ids=[],
    )
    patch_inputs: list[dict[str, JsonValue]] = []

    class Gateway:
        last_call_id: str | None = "test"
        last_delivered_citations: set[int] = {1, 2}

        def complete(
            self,
            system: str,
            payload: dict[str, JsonValue],
            response_type: type[T],
            flow: LLMFlow,
            finalizing: bool = False,
        ) -> T:
            del system, flow, finalizing
            value: BaseModel
            if response_type is ResearchPlan:
                value = plan
            elif response_type is ResearchStep:
                value = ResearchStep(
                    actions=[],
                    ready_to_answer=True,
                    remaining_gaps=[],
                    requirements=requirements,
                )
            elif response_type is DraftComposition:
                value = composition_for(initial)
            else:
                assert response_type is ClaimRepairEdits
                patch_inputs.append(payload)
                value = patch
            return response_type.model_validate(value.model_dump())

    class Acquirer:
        def definitions(self) -> list[dict[str, JsonValue]]:
            return []

        def acquire(
            self, actions: list[SourceAction], plan: BaseResearchPlan
        ) -> list[dict[str, JsonValue]]:
            del actions, plan
            return []

    class Reviewer:
        calls = 0

        def __init__(self) -> None:
            self.reviewed_drafts: list[DraftAnswer] = []

        def expected_checks(
            self,
            request: str,
            plan: ResearchPlan,
            draft: DraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> dict[str, ReviewQuestion]:
            del previous, affected_sections
            return build_checks(
                request,
                plan,
                draft,
                requirements,
                dependencies,
                delivered,
                ledger=ledger,
            )

        def review(
            self,
            request: str,
            plan: ResearchPlan,
            draft: DraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> SemanticReview:
            del previous, affected_sections
            self.calls += 1
            self.reviewed_drafts.append(draft.model_copy(deep=True))
            questions = self.expected_checks(
                request, plan, draft, requirements, dependencies, delivered
            )
            return SemanticReview(
                checks=[
                    ReviewCheck(
                        check_id=identity,
                        need_ids=question.need_ids,
                        section_ids=question.section_ids,
                        status="gap"
                        if self.calls == 1 and identity == "requirement:r_b"
                        else "addressed",
                        confidence=0.99,
                    )
                    for identity, question in questions.items()
                ]
            )

    reviewer = Reviewer()
    engine = LegalCompositeEngine(
        gateway=Gateway(),
        acquirer=Acquirer(),
        ledger=ledger,
        policy=WorkflowPolicy(max_research_rounds=1),
        check_active=lambda: None,
        research_available=lambda: True,
        reviewer=reviewer,
    )
    result = engine.run("A ve B işlemleri?")
    assert isinstance(result, CompositeWorkflowResult)
    assert result.status == "verified" and result.answer == draft.answer
    assert reviewer.calls == 2 and len(patch_inputs) == 1
    assert patch_inputs[0]["affected_section_ids"] == ["s_b"]
    assert len(result.source_requirements) == 2
    assert reviewer.reviewed_drafts[-1].sections[1].text == draft.sections[1].text
    assert reviewer.reviewed_drafts[-1].sections[0] == initial.sections[0]
    if retain_additional_claim:
        retained = [
            next(claim for claim in reviewed.claims if claim.claim_id == "c_b_retained")
            for reviewed in reviewer.reviewed_drafts
        ]
        assert retained[0].model_dump_json() == retained[1].model_dump_json()
        assert retained[0] is not retained[1]
        assert len(patch.claims) == 1 and patch.claims[0].claim_id == "c_b"
