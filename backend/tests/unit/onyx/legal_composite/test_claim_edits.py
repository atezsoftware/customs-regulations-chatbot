"""Sparse repair transport preserves host topology and all existing admission fences."""

from typing import Any, TypeVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.claim_edits import (
    ClaimEdit,
    ClaimRepairEdits,
    HeadingEdit,
    claim_edits_to_delta,
)
from onyx.legal_composite.draft_composition import DraftComposition
from onyx.legal_composite.draft_repair import (
    apply_claim_delta,
    canonicalize_delta_supports,
)
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AnswerSection,
    IssueResearchPlan,
    ReviewCheck,
    SemanticReview,
    SourceRequirement,
    SpanSupport,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.requirements import draft_binding_gaps
from onyx.legal_composite.reviewer import build_checks
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for
from tests.unit.onyx.legal_composite.test_claim_delta_repair import _claim, _draft
from tests.unit.onyx.legal_composite.test_source_requirements import fixture

T = TypeVar("T", bound=BaseModel)


def edit(
    identity: str,
    section: str = "a",
    text: str | None = None,
    need_ids: list[str] | None = None,
) -> ClaimEdit:
    data = _claim(identity, section, text).model_dump()
    data.pop("need_ids")
    if need_ids is not None:
        data["need_ids"] = need_ids
    return ClaimEdit.model_validate(data, strict=True)


def repair(**changes: object) -> ClaimRepairEdits:
    data = {
        "claims": [edit("a1", text="a1: corrected full condition [1]").model_dump()],
        "unresolved_need_ids": ["b"],
        **changes,
    }
    return ClaimRepairEdits.model_validate(data, strict=True)


def test_sparse_edit_retains_existing_claim_slots_headings_and_unaffected_bytes() -> (
    None
):
    draft, edits = _draft(), repair()
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    delta = claim_edits_to_delta(draft, edits, {"a"})
    assert delta.sections[0].claim_ids == ["a1", "a2"]
    assert delta.sections[0].need_ids == ["a"]
    assert "text" not in delta.sections[0].model_fields_set
    assert [claim.claim_id for claim in delta.claims] == ["a1"]
    result = apply_claim_delta(draft, delta, {"a"})
    assert (
        result.sections[0].text
        == "A başlığı\n\na1: corrected full condition [1]\n\na2: özgün koşul [1]"
    )
    assert result.sections[1].model_dump_json() == draft.sections[1].model_dump_json()
    for identity in ("a2", "b1"):
        prior = next(row for row in draft.claims if row.claim_id == identity)
        retained = next(row for row in result.claims if row.claim_id == identity)
        assert (
            retained.model_dump_json() == prior.model_dump_json()
            and retained is not prior
        )
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before
    delta.claims[0].need_ids.append("not-aliased")
    assert draft.claims[0].need_ids == ["a"]


def test_host_order_ignores_update_order_and_appends_only_new_ids_in_returned_order() -> (
    None
):
    draft = _draft()
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id="a", need_ids=["a"], text="Heading", claim_ids=["a2", "a1"]
            ),
            draft.sections[1],
        ],
        claims=draft.claims,
        unresolved_need_ids=["b"],
    )
    edits = repair(
        claims=[
            edit("a4", need_ids=["a"]).model_dump(),
            edit("a1", text="Updated old slot [1]").model_dump(),
            edit("a3", need_ids=["a"]).model_dump(),
        ]
    )
    delta = claim_edits_to_delta(draft, edits, {"a"})
    assert delta.sections[0].claim_ids == ["a2", "a1", "a4", "a3"]
    result = apply_claim_delta(draft, delta, {"a"})
    assert result.sections[0].claim_ids == ["a2", "a1", "a4", "a3"]
    assert (
        result.answer.index("a2:")
        < result.answer.index("Updated old slot")
        < result.answer.index("a4:")
        < result.answer.index("a3:")
    )


def test_explicit_delete_and_new_claims_keep_surviving_order_and_section_scopes() -> (
    None
):
    edits = repair(
        claims=[
            edit("a3", need_ids=["a"]).model_dump(),
            edit("b2", "b", need_ids=["b"]).model_dump(),
        ],
        deleted_claim_ids=["a1"],
    )
    delta = claim_edits_to_delta(_draft(), edits, {"a", "b"})
    assert [section.section_id for section in delta.sections] == ["a", "b"]
    assert [section.claim_ids for section in delta.sections] == [
        ["a2", "a3"],
        ["b1", "b2"],
    ]
    result = apply_claim_delta(_draft(), delta, {"a", "b"})
    assert "a1:" not in result.answer
    assert [(row.claim_id, row.need_ids) for row in delta.claims] == [
        ("a3", ["a"]),
        ("b2", ["b"]),
    ]


def test_existing_claim_narrow_issue_binding_is_frozen_and_new_claim_has_explicit_subset() -> (
    None
):
    draft = _draft()
    draft.sections[0].need_ids = ["a", "a_related"]
    edits = repair(
        claims=[
            edit("a1").model_dump(),
            edit("a3", need_ids=["a"]).model_dump(),
        ]
    )
    delta = claim_edits_to_delta(draft, edits, {"a"})
    assert delta.claims[0].need_ids == ["a"]
    assert delta.claims[1].need_ids == ["a"]
    assert delta.sections[0].need_ids == ["a", "a_related"]


@pytest.mark.parametrize("explicit", [False, True], ids=["omitted", "exact"])
def test_existing_claim_accepts_only_its_exact_frozen_issue_bindings(
    explicit: bool,
) -> None:
    draft = _draft()
    draft.sections[0].need_ids = ["a", "a_related"]
    draft.claims[0].need_ids = ["a", "a_related"]
    edits = repair(
        claims=[
            edit("a1", need_ids=["a", "a_related"] if explicit else None).model_dump()
        ]
    )
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    delta = claim_edits_to_delta(draft, edits, {"a"})
    assert delta.claims[0].need_ids == ["a", "a_related"]
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before
    delta.claims[0].need_ids.reverse()
    assert draft.claims[0].need_ids == ["a", "a_related"]


@pytest.mark.parametrize(
    "scope",
    [None, [], ["a", "a"], ["b"], ["a", "foreign"], [""]],
    ids=["missing", "empty", "duplicate", "other-section", "foreign", "empty-id"],
)
def test_new_claim_requires_an_explicit_unique_target_issue_subset_atomically(
    scope: list[str] | None,
) -> None:
    draft, edits = _draft(), repair(claims=[edit("a3", need_ids=["a"]).model_dump()])
    edits.claims[0].need_ids = scope
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    with pytest.raises(InvalidSourceAction):
        claim_edits_to_delta(draft, edits, {"a"})
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before


@pytest.mark.parametrize(
    "scope",
    [
        [],
        ["a"],
        ["a_related", "a"],
        ["a", "a_related", "foreign"],
        ["a", "a_related", "a"],
        ["a", "foreign"],
    ],
    ids=["empty", "drop", "reorder", "add", "duplicate", "replace"],
)
def test_existing_claim_explicit_scope_cannot_mutate_frozen_bindings_atomically(
    scope: list[str],
) -> None:
    draft, edits = _draft(), repair()
    draft.sections[0].need_ids = ["a", "a_related", "foreign"]
    draft.claims[0].need_ids = ["a", "a_related"]
    edits.claims[0].need_ids = scope
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    with pytest.raises(InvalidSourceAction):
        claim_edits_to_delta(draft, edits, {"a"})
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before


@pytest.mark.parametrize("scope", ["a", {"a": True}, [1], [False]])
def test_claim_issue_scope_schema_rejects_malformed_values(scope: object) -> None:
    data = repair(claims=[edit("a3", need_ids=["a"]).model_dump()]).model_dump()
    data["claims"][0]["need_ids"] = scope
    with pytest.raises(ValidationError):
        ClaimRepairEdits.model_validate(data, strict=True)


def test_new_narrow_claim_does_not_expand_an_unrelated_issue_source_inventory() -> None:
    ledger, plan, requirements, original_draft = canonical_fixture()
    ledger._items[1].question_ids = ["a"]
    ledger._items[2].question_ids = ["b"]
    claims = [
        claim.model_copy(deep=True, update={"section_id": "joint"})
        for claim in original_draft.claims
    ]
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id="joint",
                need_ids=["a", "b"],
                text="Shared heading",
                claim_ids=[claim.claim_id for claim in claims],
            )
        ],
        claims=claims,
        unresolved_need_ids=[],
    )
    span = original_witness_spans(1, ledger._items[1].text)[0]
    edits = ClaimRepairEdits(
        claims=[
            ClaimEdit(
                claim_id="new_a",
                section_id="joint",
                need_ids=["a"],
                answer_excerpt=draft.claims[0].answer_excerpt,
                supports=[SpanSupport(citation=1, span_id=span["witness_id"])],
            )
        ],
        unresolved_need_ids=[],
    )
    before, originals_before = draft.model_dump_json(), ledger.export()
    initial_checks = build_checks(
        "A ve B işlemleri?", plan, draft, requirements, [], {1, 2}, ledger=ledger
    )
    delta = canonicalize_delta_supports(
        claim_edits_to_delta(draft, edits, {"joint"}), ledger, {1, 2}
    )
    result = apply_claim_delta(draft, delta, {"joint"})
    checks = build_checks(
        "A ve B işlemleri?", plan, result, requirements, [], {1, 2}, ledger=ledger
    )
    assert result.sections[0].need_ids == ["a", "b"]
    assert result.sections[0].claim_ids == ["c_a", "c_b", "new_a"]
    assert result.claims[-1].need_ids == ["a"]
    assert (
        checks["original:1"].need_ids == initial_checks["original:1"].need_ids == ["a"]
    )
    assert checks["issue:b"].citations == initial_checks["issue:b"].citations == [2]
    assert (
        checks["evidence:b"].citations == initial_checks["evidence:b"].citations == [2]
    )
    assert checks["claim:new_a"].need_ids == ["a"]
    assert draft_binding_gaps(result, plan, requirements, ledger, {1, 2}) == []
    for prior, retained in zip(draft.claims, result.claims[:2]):
        assert prior.model_dump_json() == retained.model_dump_json()
    assert draft.model_dump_json() == before and ledger.export() == originals_before


@pytest.mark.parametrize("heading", ["", "New heading"])
def test_only_explicit_heading_edits_replace_the_preserved_heading(
    heading: str,
) -> None:
    edits = repair(heading_edits=[{"section_id": "a", "text": heading}])
    delta = claim_edits_to_delta(_draft(), edits, {"a"})
    assert "text" in delta.sections[0].model_fields_set
    result = apply_claim_delta(_draft(), delta, {"a"})
    assert result.sections[0].text == "\n\n".join(
        filter(
            None, [heading, "a1: corrected full condition [1]", "a2: özgün koşul [1]"]
        )
    )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_claim",
        "duplicate_delete",
        "conflict",
        "duplicate_heading",
        "duplicate_unresolved",
        "oversized_heading",
        "sections",
        "claim_order",
        "section_need_ids",
    ],
)
def test_model_cannot_reproduce_topology_or_ambiguous_edit_inventory(
    fault: str,
) -> None:
    data = repair().model_dump(mode="json")
    if fault == "duplicate_claim":
        data["claims"] *= 2
    elif fault == "duplicate_delete":
        data["deleted_claim_ids"] = ["a2", "a2"]
    elif fault == "conflict":
        data["deleted_claim_ids"] = ["a1"]
    elif fault == "duplicate_heading":
        data["heading_edits"] = [{"section_id": "a", "text": "A"}] * 2
    elif fault == "duplicate_unresolved":
        data["unresolved_need_ids"] = ["b", "b"]
    elif fault == "oversized_heading":
        data["heading_edits"] = [{"section_id": "a", "text": "İ" * 601}]
    else:
        data[fault] = []
    with pytest.raises(ValidationError):
        ClaimRepairEdits.model_validate(data, strict=True)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_section",
        "unknown_delete",
        "unaffected_delete",
        "unaffected_edit",
        "move",
        "unknown_heading",
        "unaffected_heading",
        "mutated_duplicate",
        "frozen_cross_issue",
        "frozen_order",
    ],
)
def test_adapter_scope_faults_are_controlled_and_atomic(fault: str) -> None:
    draft, edits = _draft(), repair()
    affected = {"a"}
    if fault == "unknown_section":
        affected = {"unknown"}
    elif fault == "unknown_delete":
        edits.deleted_claim_ids = ["unknown"]
    elif fault == "unaffected_delete":
        edits.deleted_claim_ids = ["b1"]
    elif fault == "unaffected_edit":
        edits.claims = [edit("b1", "b")]
    elif fault == "move":
        edits.claims = [edit("b1", "a")]
        affected = {"a", "b"}
    elif fault in {"unknown_heading", "unaffected_heading"}:
        edits.heading_edits = [
            HeadingEdit(
                section_id="unknown" if fault == "unknown_heading" else "b",
                text="Heading",
            )
        ]
    elif fault == "mutated_duplicate":
        edits.claims.append(edits.claims[0].model_copy(deep=True))
    elif fault == "frozen_cross_issue":
        draft.claims[0].need_ids = ["b"]
    else:
        draft.sections[0].claim_ids = ["a1"]
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    with pytest.raises(InvalidSourceAction):
        claim_edits_to_delta(draft, edits, affected)
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before


def test_heading_prose_cannot_bypass_existing_rendering_guard() -> None:
    draft = _draft()
    edits = repair(
        heading_edits=[
            {"section_id": "a", "text": "Heading\n\n" + draft.claims[1].answer_excerpt}
        ]
    )
    delta = claim_edits_to_delta(draft, edits, {"a"})
    with pytest.raises(InvalidSourceAction, match="legal claim prose"):
        apply_claim_delta(draft, delta, {"a"})


def canonical_fixture() -> tuple[
    EvidenceLedger, IssueResearchPlan, list[SourceRequirement], StructuredDraftAnswer
]:
    ledger, plan, requirements, draft = fixture()
    claims = [
        claim.model_copy(
            deep=True,
            update={
                "answer_excerpt": next(
                    section.text
                    for section in draft.sections
                    if section.section_id == claim.section_id
                )
            },
        )
        for claim in draft.claims
    ]
    draft = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id=section.section_id,
                need_ids=section.need_ids,
                text="",
                claim_ids=[
                    claim.claim_id
                    for claim in claims
                    if claim.section_id == section.section_id
                ],
            )
            for section in draft.sections
        ],
        claims=claims,
        unresolved_need_ids=[],
    )
    return ledger, plan, requirements, draft


@pytest.mark.parametrize(
    "fault", ["bad_span", "bad_quote", "hash", "undelivered", "derived"]
)
def test_sparse_adapter_does_not_bypass_canonical_original_support(fault: str) -> None:
    ledger, _plan, _requirements, draft = canonical_fixture()
    original = ledger.get(1)
    assert original is not None
    span = original_witness_spans(1, original.text)[0]
    support = SpanSupport(citation=1, span_id=span["witness_id"])
    if fault == "bad_span":
        support.span_id = "not-an-original-span"
    elif fault == "bad_quote":
        support.quotation = "An invented operative condition"
    elif fault == "hash":
        ledger._items[1].text_hash = "0" * 64
    elif fault == "derived":
        ledger._items[1].metadata["derived"] = True
    edits = ClaimRepairEdits(
        claims=[
            ClaimEdit(
                claim_id="c_a",
                section_id="s_a",
                answer_excerpt=draft.claims[0].answer_excerpt,
                supports=[support],
            )
        ],
        unresolved_need_ids=[],
    )
    before, edits_before = draft.model_dump_json(), edits.model_dump_json()
    delta = claim_edits_to_delta(draft, edits, {"s_a"})
    with pytest.raises(InvalidSourceAction):
        canonicalize_delta_supports(
            delta, ledger, set() if fault == "undelivered" else {1, 2}
        )
    assert draft.model_dump_json() == before and edits.model_dump_json() == edits_before


def test_bound_sparse_support_and_cross_issue_requirement_keep_existing_guards() -> (
    None
):
    ledger, plan, requirements, draft = canonical_fixture()
    text = ledger._items[1].text
    span = original_witness_spans(1, text)[0]
    edits = ClaimRepairEdits(
        claims=[
            ClaimEdit(
                claim_id="c_a",
                section_id="s_a",
                answer_excerpt=draft.claims[0].answer_excerpt,
                supports=[SpanSupport(citation=1, span_id=span["witness_id"])],
            )
        ],
        unresolved_need_ids=[],
    )
    delta = canonicalize_delta_supports(
        claim_edits_to_delta(draft, edits, {"s_a"}), ledger, {1, 2}
    )
    assert (
        delta.claims[0].supports[0].quotation
        == text[span["start_char"] : span["end_char"]]
    )
    result = apply_claim_delta(draft, delta, {"s_a"})
    assert draft_binding_gaps(result, plan, requirements, ledger, {1, 2}) == []
    edits.claims[0].supports = []
    edits.claims[0].requirement_ids = ["r_b"]
    delta = claim_edits_to_delta(draft, edits, {"s_a"})
    invalid = apply_claim_delta(draft, delta, {"s_a"})
    assert "claim:c_a:invalid_requirement_binding" in draft_binding_gaps(
        invalid, plan, requirements, ledger, {1, 2}
    )


@pytest.mark.parametrize("kind", ["requirement", "closure", "unresolved"])
def test_sparse_output_still_obeys_existing_reading_closure_and_unresolved_scope(
    kind: str,
) -> None:
    draft, edits = _draft(), repair()
    if kind == "requirement":
        _ledger, _plan, requirements, _source_draft = fixture()
        edits.requirements = [requirements[1]]
    elif kind == "closure":
        from onyx.legal_composite.models import GapResolution

        edits.gap_resolutions = [
            GapResolution(need_id="b", gap="B gap", requirement_ids=["r_b"])
        ]
    else:
        edits.unresolved_need_ids = []
    delta = claim_edits_to_delta(draft, edits, {"a"})
    with pytest.raises(InvalidSourceAction, match="unaffected issue"):
        apply_claim_delta(draft, delta, {"a"})


@pytest.mark.parametrize(
    "bad_span", [False, True], ids=["canonical-edit", "rejected-support"]
)
def test_actual_engine_adapter_canonicalization_and_full_recheck_preserve_atomicity(
    bad_span: bool,
) -> None:
    """Assumed fake semantic judgments test the real host chain, not legal correctness."""
    ledger, plan, requirements, correct = canonical_fixture()
    text = ledger._items[1].text
    span = original_witness_spans(1, text)[0]
    edits = ClaimRepairEdits(
        claims=[
            ClaimEdit(
                claim_id="c_a",
                section_id="s_a",
                answer_excerpt=correct.claims[0].answer_excerpt,
                supports=[
                    SpanSupport(
                        citation=1,
                        span_id="unknown-span" if bad_span else span["witness_id"],
                    )
                ],
            )
        ],
        unresolved_need_ids=[],
    )
    initial = StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id=section.section_id,
                need_ids=section.need_ids,
                text="",
                claim_ids=section.claim_ids,
            )
            for section in correct.sections
        ],
        claims=[
            claim.model_copy(
                deep=True, update={"answer_excerpt": "Belge yeterlidir. [1]"}
            )
            if claim.claim_id == "c_a"
            else claim
            for claim in correct.claims
        ],
        unresolved_need_ids=[],
    )
    requests: list[type[BaseModel]] = []

    class Gateway:
        last_call_id: str | None = "synthetic-engine-chain"
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
            requests.append(response_type)
            if response_type is DraftComposition:
                return response_type.model_validate(
                    composition_for(initial).model_dump()
                )
            assert response_type is ClaimRepairEdits
            assert payload["repair_targets"] == [
                {"section_id": "s_a", "need_ids": ["a"], "existing_claim_ids": ["c_a"]}
            ]
            assert payload["active_requirement_ids"] == ["r_a", "r_b"]
            return response_type.model_validate(edits.model_dump())

    reviewed: list[StructuredDraftAnswer] = []
    judge = Mock()

    def expected(*args: Any, **_kwargs: Any) -> Any:
        return build_checks(*args[:6], ledger=ledger)

    def review(*args: Any, **kwargs: Any) -> SemanticReview:
        reviewed.append(args[2].model_copy(deep=True))
        checks = expected(*args, **kwargs)
        return SemanticReview(
            checks=[
                ReviewCheck(
                    check_id=identity,
                    need_ids=check.need_ids,
                    section_ids=check.section_ids,
                    status="gap"
                    if len(reviewed) == 1 and identity == "claim:c_a"
                    else "addressed",
                    confidence=0.99,
                )
                for identity, check in checks.items()
            ]
        )

    judge.expected_checks.side_effect = expected
    judge.review.side_effect = review
    acquirer = Mock()
    acquirer.definitions.return_value = []
    engine = LegalCompositeEngine(
        gateway=Gateway(),
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: False,
        reviewer=judge,
    )
    engine.plan = plan
    engine.requirements.update(requirements, plan, {1, 2})
    originals_before, records_before = ledger.export(), engine.requirements.export()
    result = engine._finalize_semantic("A ve B işlemlerini açıklayın.", "", None, plan)
    assert requests == [DraftComposition, ClaimRepairEdits]
    assert len(reviewed) == 2
    assert reviewed[0].model_dump_json() == initial.model_dump_json()
    assert (
        reviewed[1].sections[1].model_dump_json()
        == initial.sections[1].model_dump_json()
    )
    assert (
        ledger.export() == originals_before
        and engine.requirements.export() == records_before
    )
    if bad_span:
        assert result.status == "unavailable" and result.answer is None
        assert reviewed[1].model_dump_json() == initial.model_dump_json()
    else:
        assert result.status == "verified" and result.answer == correct.answer
        assert (
            reviewed[1].claims[0].supports[0].quotation
            == text[span["start_char"] : span["end_char"]]
        )
