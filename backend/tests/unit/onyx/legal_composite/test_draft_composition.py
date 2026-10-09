"""Host composition renders all claims without adding citations or legal judgments."""

import hashlib

import pytest
from pydantic import ValidationError

from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.draft_composition import (
    DRAFT_COMPOSITION_PROMPT,
    CompositionSection,
    DraftComposition,
    compose_draft,
)
from onyx.legal_composite.models import GapResolution, SpanSupport
from onyx.legal_composite.prompts import (
    ANSWER_CONTENT_HEAD,
    ANSWER_CONTENT_TAIL,
    ANSWER_PROMPT,
)
from onyx.legal_composite.requirements import (
    canonicalize_draft_supports,
    draft_binding_gaps,
)
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for
from tests.unit.onyx.legal_composite.test_source_requirements import fixture


def composition() -> DraftComposition:
    _ledger, _plan, requirements, draft = fixture()
    return DraftComposition(
        sections=[
            CompositionSection(section_id="s_a", need_ids=["a"], heading="A heading"),
            CompositionSection(section_id="s_b", need_ids=["b"], heading="B heading"),
        ],
        claims=[
            claim.model_copy(
                deep=True, update={"answer_excerpt": draft.sections[index].text}
            )
            for index, claim in enumerate(draft.claims)
        ],
        requirements=requirements,
        gap_resolutions=[
            GapResolution(need_id="a", gap="Recorded gap", requirement_ids=["r_a"])
        ],
        unresolved_need_ids=["b"],
    )


def test_host_derives_all_section_memberships_and_preserves_exact_prose_and_inputs() -> (
    None
):
    value = composition()
    value.sections.reverse()
    extra = value.claims[0].model_copy(
        deep=True,
        update={
            "claim_id": "extra_a",
            "answer_excerpt": "Additional exact passage [1]",
        },
    )
    value.claims.append(extra)
    before = value.model_dump_json()
    draft = compose_draft(value)
    assert [section.section_id for section in draft.sections] == ["s_b", "s_a"]
    assert [section.claim_ids for section in draft.sections] == [
        ["c_b"],
        ["c_a", "extra_a"],
    ]
    assert draft.sections[0].text == "B heading\n\n" + value.claims[1].answer_excerpt
    assert (
        draft.sections[1].text
        == "A heading\n\n"
        + value.claims[0].answer_excerpt
        + "\n\n"
        + extra.answer_excerpt
    )
    assert draft.answer == "\n\n".join(section.text for section in draft.sections)
    assert [claim.model_dump_json() for claim in draft.claims] == [
        claim.model_dump_json() for claim in value.claims
    ]
    assert (
        draft.requirements == value.requirements
        and draft.gap_resolutions == value.gap_resolutions
    )
    assert value.model_dump_json() == before
    draft.sections[0].need_ids.append("not-aliased")
    draft.claims[0].need_ids.append("not-aliased")
    draft.requirements[0].supports[0].quotation = "not-aliased"
    assert value.model_dump_json() == before


def test_seven_returned_claims_are_rendered_and_bind_instead_of_heading_only_answer() -> (
    None
):
    ledger, plan, requirements, _draft = fixture()
    value = composition()
    value.requirements = []
    value.gap_resolutions = []
    original_a, original_b = value.claims
    value.claims = [
        original_a.model_copy(
            deep=True,
            update={
                "claim_id": f"a_{index}",
                "answer_excerpt": f"{index}: {original_a.answer_excerpt}",
            },
        )
        for index in range(6)
    ] + [original_b]
    draft = canonicalize_draft_supports(compose_draft(value), ledger, {1, 2})
    assert draft.sections[0].claim_ids == [f"a_{index}" for index in range(6)]
    assert draft.sections[1].claim_ids == ["c_b"]
    for claim in value.claims:
        assert draft.answer.count(claim.answer_excerpt) == 1
    assert draft_binding_gaps(draft, plan, requirements, ledger, {1, 2}) == []


def test_composition_does_not_insert_missing_inline_citation_markers() -> None:
    ledger, plan, requirements, _draft = fixture()
    value = composition()
    value.claims[0].answer_excerpt = "A rule without a source marker."
    value.requirements = []
    value.gap_resolutions = []
    draft = canonicalize_draft_supports(compose_draft(value), ledger, {1, 2})
    assert draft.claims[0].answer_excerpt == "A rule without a source marker."
    assert "[1]" not in draft.answer
    assert "claim:c_a:invalid_original_binding" in draft_binding_gaps(
        draft, plan, requirements, ledger, {1, 2}
    )


@pytest.mark.parametrize("heading", ["", "An exact heading"])
def test_legal_section_can_have_empty_heading_but_keeps_its_complete_claim(
    heading: str,
) -> None:
    value = composition()
    value.sections[0].heading = heading
    draft = compose_draft(value)
    assert draft.sections[0].text == "\n\n".join(
        part for part in (heading, value.claims[0].answer_excerpt) if part
    )


def test_source_free_answer_uses_nonempty_heading_with_explicit_empty_claim_inventory() -> (
    None
):
    value = DraftComposition(
        sections=[
            CompositionSection(
                section_id="social", need_ids=["greeting"], heading="Merhaba."
            )
        ],
        claims=[],
        unresolved_need_ids=[],
    )
    draft = compose_draft(value)
    assert draft.answer == "Merhaba." and draft.claims == []
    assert draft.sections[0].claim_ids == []
    value.sections[0].heading = ""
    with pytest.raises(InvalidSourceAction, match="valid answer"):
        compose_draft(value)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_section",
        "duplicate_claim",
        "unknown_section",
        "foreign_issue",
        "duplicate_claim_issue",
        "duplicate_section_issue",
    ],
)
def test_composition_scope_faults_are_controlled_and_atomic(fault: str) -> None:
    value = composition()
    if fault == "duplicate_section":
        value.sections.append(value.sections[0].model_copy(deep=True))
    elif fault == "duplicate_claim":
        value.claims.append(value.claims[0].model_copy(deep=True))
    elif fault == "unknown_section":
        value.claims[0].section_id = "unknown"
    elif fault == "foreign_issue":
        value.claims[0].need_ids = ["b"]
    elif fault == "duplicate_claim_issue":
        value.claims[0].need_ids = ["a", "a"]
    else:
        value.sections[0].need_ids = ["a", "a"]
    before = value.model_dump_json()
    with pytest.raises(InvalidSourceAction):
        compose_draft(value)
    assert value.model_dump_json() == before


@pytest.mark.parametrize(
    "fault",
    [
        "answer",
        "claim_ids",
        "text",
        "missing_claims",
        "missing_unresolved",
        "oversized_heading",
        "empty_sections",
        "empty_claim_issues",
    ],
)
def test_transport_rejects_legacy_render_controls_or_incomplete_schema(
    fault: str,
) -> None:
    data = composition().model_dump()
    if fault == "answer":
        data["answer"] = "Ignored legal prose"
    elif fault in {"claim_ids", "text"}:
        data["sections"][0][fault] = (
            [] if fault == "claim_ids" else "Ignored legal prose"
        )
    elif fault == "missing_claims":
        del data["claims"]
    elif fault == "missing_unresolved":
        del data["unresolved_need_ids"]
    elif fault == "oversized_heading":
        data["sections"][0]["heading"] = "H" * 601
    elif fault == "empty_sections":
        data["sections"] = []
    else:
        data["claims"][0]["need_ids"] = []
    with pytest.raises(ValidationError):
        DraftComposition.model_validate(data, strict=True)


@pytest.mark.parametrize("fault", ["span", "hash", "undelivered", "derived"])
def test_composition_keeps_canonical_original_support_gate_unchanged(
    fault: str,
) -> None:
    ledger, _plan, _requirements, _draft = fixture()
    value = composition()
    value.requirements = []
    value.gap_resolutions = []
    span = original_witness_spans(1, ledger._items[1].text)[0]
    value.claims[0].requirement_ids = []
    value.claims[0].supports = [
        SpanSupport(
            citation=1, span_id="unknown" if fault == "span" else span["witness_id"]
        )
    ]
    if fault == "hash":
        ledger._items[1].text_hash = "0" * 64
    elif fault == "derived":
        ledger._items[1].metadata["derived"] = True
    before = value.model_dump_json()
    draft = compose_draft(value)
    with pytest.raises(InvalidSourceAction):
        canonicalize_draft_supports(
            draft, ledger, {2} if fault == "undelivered" else {1, 2}
        )
    assert value.model_dump_json() == before


def test_content_guidance_is_reused_exactly_without_changing_legacy_answer_prompt() -> (
    None
):
    assert DRAFT_COMPOSITION_PROMPT.count(ANSWER_CONTENT_HEAD) == 1
    assert DRAFT_COMPOSITION_PROMPT.count(ANSWER_CONTENT_TAIL) == 1
    assert (
        hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest()
        == "2968c023d6288552c7f132dc40bd9f82bfad75fb957c2280c13ff01a76c5dc54"
    )


def test_test_fixture_adapter_preserves_exact_draft_and_refuses_unmodeled_prose() -> (
    None
):
    draft = compose_draft(composition())
    before = draft.model_dump_json()
    transported = composition_for(draft)
    assert compose_draft(transported).model_dump_json() == before
    assert draft.model_dump_json() == before
    draft.sections[0].text += "\n\nAn unmodeled legal assertion must not disappear."
    draft.answer = "\n\n".join(section.text for section in draft.sections)
    with pytest.raises(ValueError, match="exact ordered claim body"):
        composition_for(draft)
