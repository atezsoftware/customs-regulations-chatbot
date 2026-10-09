"""Claim deltas retain supported passages and cannot cross a repair boundary."""

import json

import pytest
from pydantic import ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.draft_repair import (
    ClaimDeltaPatch,
    apply_claim_delta,
    canonicalize_delta_supports,
)
from onyx.legal_composite.models import (
    AnswerSection,
    DraftClaim,
    GapResolution,
    SourceRequirement,
    SpanSupport,
    StructuredDraftAnswer,
)
from tests.unit.onyx.legal_composite.test_review_assessment import original


def _claim(identity: str, section: str = "a", text: str | None = None) -> DraftClaim:
    return DraftClaim(
        claim_id=identity,
        section_id=section,
        need_ids=[section],
        answer_excerpt=text or f"{identity}: özgün koşul [1]",
        requirement_ids=[f"requirement_{identity}"],
    )


def _draft() -> StructuredDraftAnswer:
    return StructuredDraftAnswer(
        sections=[
            AnswerSection(
                section_id="a", need_ids=["a"], text="A başlığı", claim_ids=["a1", "a2"]
            ),
            AnswerSection(
                section_id="b", need_ids=["b"], text="B başlığı", claim_ids=["b1"]
            ),
        ],
        claims=[_claim("a1"), _claim("a2"), _claim("b1", "b")],
        unresolved_need_ids=["b"],
    )


def _patch() -> ClaimDeltaPatch:
    return ClaimDeltaPatch(
        sections=[
            AnswerSection(section_id="a", need_ids=["a"], claim_ids=["a1", "a2"])
        ],
        claims=[_claim("a1", text="a1: düzeltilen tam koşul [1]")],
        unresolved_need_ids=["b"],
    )


def _bytes(value: StructuredDraftAnswer) -> str:
    return value.model_dump_json()


def test_delta_updates_one_claim_and_retains_every_other_passage_independently() -> (
    None
):
    draft, patch = _draft(), _patch()
    before, patch_before = _bytes(draft), patch.model_dump_json()
    result = apply_claim_delta(draft, patch, {"a"})
    assert result.sections[0].text == (
        "A başlığı\n\na1: düzeltilen tam koşul [1]\n\na2: özgün koşul [1]"
    )
    assert result.sections[1].model_dump_json() == draft.sections[1].model_dump_json()
    for identity in ("a2", "b1"):
        retained = next(row for row in result.claims if row.claim_id == identity)
        prior = next(row for row in draft.claims if row.claim_id == identity)
        assert retained.model_dump_json() == prior.model_dump_json()
        assert retained is not prior and retained.need_ids is not prior.need_ids
    assert _bytes(draft) == before and patch.model_dump_json() == patch_before
    result.sections[1].need_ids.append("changed_copy")
    assert draft.sections[1].need_ids == ["b"]


def test_reapplying_an_upsert_is_idempotent_and_never_duplicates_rendered_prose() -> (
    None
):
    once = apply_claim_delta(_draft(), _patch(), {"a"})
    twice = apply_claim_delta(once, _patch(), {"a"})
    assert _bytes(twice) == _bytes(once)
    assert twice.answer.count("a1: düzeltilen tam koşul [1]") == 1


def test_explicit_deletion_and_new_claim_preserve_complete_requested_order() -> None:
    patch = ClaimDeltaPatch(
        sections=[
            AnswerSection(section_id="a", need_ids=["a"], claim_ids=["a3", "a2"])
        ],
        claims=[_claim("a3", text="a3: eklenen destekli sonraki adım [1]")],
        deleted_claim_ids=["a1"],
        unresolved_need_ids=["b"],
    )
    result = apply_claim_delta(_draft(), patch, {"a"})
    assert result.sections[0].claim_ids == ["a3", "a2"]
    assert [claim.claim_id for claim in result.claims] == ["a2", "b1", "a3"]
    assert result.answer.index("a3:") < result.answer.index("a2:")
    assert "a1:" not in result.answer


@pytest.mark.parametrize(
    "kind", ["section", "claim", "deletion", "unresolved", "upsert_delete"]
)
def test_typed_delta_rejects_duplicate_and_conflicting_identities(kind: str) -> None:
    data = _patch().model_dump(mode="json")
    if kind == "section":
        data["sections"] *= 2
    elif kind == "claim":
        data["claims"] *= 2
    elif kind == "deletion":
        data["deleted_claim_ids"] = ["a2", "a2"]
    elif kind == "unresolved":
        data["unresolved_need_ids"] = ["b", "b"]
    else:
        data["deleted_claim_ids"] = ["a1"]
    with pytest.raises(ValidationError):
        ClaimDeltaPatch.model_validate(data, strict=True)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_delete",
        "unaffected_delete",
        "unaffected_upsert",
        "move",
        "silent_loss",
        "unknown_order",
        "duplicate_order",
        "section_scope",
        "extra_section",
        "missing_section",
    ],
)
def test_invalid_delta_is_atomic(kind: str) -> None:
    draft, patch = _draft(), _patch()
    if kind == "unknown_delete":
        patch.deleted_claim_ids = ["unknown"]
    elif kind == "unaffected_delete":
        patch.deleted_claim_ids = ["b1"]
    elif kind == "unaffected_upsert":
        patch.claims = [_claim("b1", "b", "changed unaffected passage")]
    elif kind == "move":
        patch.sections.append(
            AnswerSection(section_id="b", need_ids=["b"], claim_ids=[])
        )
        patch.sections[0].claim_ids.append("b1")
        patch.claims.append(_claim("b1", "a"))
    elif kind == "silent_loss":
        patch.sections[0].claim_ids = ["a1"]
    elif kind == "unknown_order":
        patch.sections[0].claim_ids.append("unknown")
    elif kind == "duplicate_order":
        patch.sections[0].claim_ids.append("a2")
    elif kind == "section_scope":
        patch.sections[0].need_ids = ["b"]
    elif kind == "extra_section":
        patch.sections.append(
            AnswerSection(section_id="b", need_ids=["b"], claim_ids=["b1"])
        )
    else:
        patch.sections = []
    before = _bytes(draft)
    with pytest.raises(InvalidSourceAction):
        apply_claim_delta(draft, patch, {"a", "b"} if kind == "move" else {"a"})
    assert _bytes(draft) == before


def test_unchanged_claim_identity_cannot_be_reused_for_a_new_affected_claim() -> None:
    patch = _patch()
    patch.claims = [_claim("b1", "a", "new passage with colliding identity")]
    with pytest.raises(InvalidSourceAction, match="move"):
        apply_claim_delta(_draft(), patch, {"a"})


def test_explicit_empty_heading_differs_from_omitting_the_heading() -> None:
    payload = json.loads(_patch().model_dump_json(exclude_unset=True))
    omitted = ClaimDeltaPatch.model_validate(payload, strict=True)
    assert "text" not in omitted.sections[0].model_fields_set
    assert (
        apply_claim_delta(_draft(), omitted, {"a"})
        .sections[0]
        .text.startswith("A başlığı")
    )
    payload["sections"][0]["text"] = ""
    explicit = ClaimDeltaPatch.model_validate(payload, strict=True)
    assert (
        apply_claim_delta(_draft(), explicit, {"a"}).sections[0].text.startswith("a1:")
    )


def test_heading_cannot_duplicate_the_retained_legal_claim_body() -> None:
    patch = _patch()
    patch.sections[0].text = "Heading\n\na2: özgün koşul [1]"
    with pytest.raises(InvalidSourceAction, match="legal claim prose"):
        apply_claim_delta(_draft(), patch, {"a"})


def test_literal_unchanged_section_is_preserved_and_ambiguous_heading_fails() -> None:
    draft = _draft()
    draft.sections[1] = AnswerSection(
        section_id="b", need_ids=["b"], text="Literal B [2]"
    )
    draft.answer = "\n\n".join(section.text for section in draft.sections)
    assert apply_claim_delta(draft, _patch(), {"a"}).sections[1].text == "Literal B [2]"
    patch = ClaimDeltaPatch(
        sections=[AnswerSection(section_id="b", need_ids=["b"], claim_ids=["b1"])],
        claims=[],
        unresolved_need_ids=["b"],
    )
    with pytest.raises(InvalidSourceAction, match="explicit repair heading"):
        apply_claim_delta(draft, patch, {"b"})


def test_delta_canonicalization_resolves_only_delivered_original_selectors() -> None:
    text = "Başvuru süresi bildirim tarihinde başlar. İzin ayrıca gerekir."
    item = original(text, "a")
    assert item.search_doc is not None and item.chunk_id is not None
    item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    ledger = EvidenceLedger()
    ledger.add([item], RunContext())
    span = original_witness_spans(1, text)[0]
    selector = SpanSupport(citation=1, span_id=span["witness_id"])
    patch = _patch()
    patch.claims[0].supports = [selector]
    patch.requirements = [
        SourceRequirement(
            requirement_id="new_requirement",
            need_id="a",
            dimension="procedure",
            rule="Başlangıç bildirimdir.",
            application="Olgulara koşullu uygulanır.",
            supports=[selector],
        )
    ]
    before = patch.model_dump_json()
    bound = canonicalize_delta_supports(patch, ledger, {1})
    expected = text[span["start_char"] : span["end_char"]]
    assert bound.claims[0].supports[0].quotation == expected
    assert bound.requirements[0].supports[0].quotation == expected
    assert bound.claims[0].supports[0].span_id == span["witness_id"]
    assert bound.claims[0].supports[0].citation == 1
    assert "text" not in bound.sections[0].model_fields_set
    assert patch.model_dump_json() == before
    with pytest.raises(InvalidSourceAction, match="delivered original"):
        canonicalize_delta_supports(patch, ledger, set())
    patch.claims[0].supports[0].span_id = "unknown_span"
    with pytest.raises(InvalidSourceAction, match="span ID"):
        canonicalize_delta_supports(patch, ledger, {1})


def test_new_requirements_and_closures_preserve_existing_history() -> None:
    prior = SourceRequirement(
        requirement_id="r1",
        need_id="a",
        dimension="scope",
        rule="Rule",
        application="Conditional",
        supports=[SpanSupport(citation=1, quotation="Rule")],
    )
    draft = _draft()
    draft.requirements = [prior]
    draft.gap_resolutions = [
        GapResolution(need_id="a", gap="Exact prior gap", requirement_ids=["r1"])
    ]
    patch = _patch()
    patch.requirements = [
        prior.model_copy(deep=True),
        prior.model_copy(
            deep=True,
            update={"requirement_id": "r2", "supersedes_requirement_ids": ["r1"]},
        ),
    ]
    patch.gap_resolutions = [
        GapResolution(need_id="a", gap="Exact prior gap", requirement_ids=["r2"])
    ]
    result = apply_claim_delta(draft, patch, {"a"})
    assert [row.requirement_id for row in result.requirements] == ["r1", "r2"]
    assert [row.requirement_ids for row in result.gap_resolutions] == [["r1"], ["r2"]]
    patch.requirements[0].rule = "Rewrite immutable ID"
    with pytest.raises(InvalidSourceAction, match="rewrite an existing requirement"):
        apply_claim_delta(draft, patch, {"a"})


def test_delta_cannot_drop_an_unaffected_unresolved_issue() -> None:
    patch = _patch()
    patch.unresolved_need_ids = []
    with pytest.raises(InvalidSourceAction, match="unaffected issue"):
        apply_claim_delta(_draft(), patch, {"a"})


@pytest.mark.parametrize("need_ids", [["b"], ["unknown"], ["a", "b"]])
def test_changed_claim_cannot_cross_section_issue_scope_and_leaves_inputs_unchanged(
    need_ids: list[str],
) -> None:
    draft, patch = _draft(), _patch()
    patch.claims[0].need_ids = need_ids
    before, patch_before = _bytes(draft), patch.model_dump_json()
    with pytest.raises(InvalidSourceAction, match="another section's issue"):
        apply_claim_delta(draft, patch, {"a"})
    assert _bytes(draft) == before
    assert patch.model_dump_json() == patch_before


def test_delete_all_claims_and_empty_heading_is_a_controlled_recomposition_failure() -> (
    None
):
    draft = _draft()
    patch = ClaimDeltaPatch(
        sections=[AnswerSection(section_id="a", need_ids=["a"], text="", claim_ids=[])],
        claims=[],
        deleted_claim_ids=["a1", "a2"],
        unresolved_need_ids=["b"],
    )
    before, patch_before = _bytes(draft), patch.model_dump_json()
    with pytest.raises(InvalidSourceAction, match="valid complete structured draft"):
        apply_claim_delta(draft, patch, {"a"})
    assert _bytes(draft) == before
    assert patch.model_dump_json() == patch_before


def test_empty_mutated_claim_issue_list_is_a_controlled_final_schema_failure() -> None:
    draft, patch = _draft(), _patch()
    patch.claims[0].need_ids = []
    before, patch_before = _bytes(draft), patch.model_dump_json()
    with pytest.raises(InvalidSourceAction, match="valid complete structured draft"):
        apply_claim_delta(draft, patch, {"a"})
    assert _bytes(draft) == before
    assert patch.model_dump_json() == patch_before


def test_mutated_duplicate_delta_is_still_rejected_atomically() -> None:
    draft, patch = _draft(), _patch()
    patch.claims.append(patch.claims[0].model_copy(deep=True))
    before = _bytes(draft)
    with pytest.raises(InvalidSourceAction, match="unique and disjoint"):
        apply_claim_delta(draft, patch, {"a"})
    assert _bytes(draft) == before


@pytest.mark.parametrize("kind", ["requirement", "gap_resolution"])
def test_delta_cannot_register_reading_or_closure_for_unaffected_issue(
    kind: str,
) -> None:
    draft, patch = _draft(), _patch()
    if kind == "requirement":
        patch.requirements = [
            SourceRequirement(
                requirement_id="new_b",
                need_id="b",
                dimension="scope",
                rule="Rule",
                application="Conditional",
                supports=[SpanSupport(citation=1, quotation="Rule")],
            )
        ]
    else:
        patch.gap_resolutions = [
            GapResolution(need_id="b", gap="B gap", requirement_ids=["r_b"])
        ]
    before = _bytes(draft)
    with pytest.raises(InvalidSourceAction, match="unaffected issue"):
        apply_claim_delta(draft, patch, {"a"})
    assert _bytes(draft) == before
