"""Canonical selectors avoid quotation transcription without weakening source binding."""

import json

import pytest
from pydantic import ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    AnswerSection,
    DraftAnswer,
    DraftClaim,
    DraftPatch,
    IssueResearchNeed,
    IssueResearchPlan,
    PassageSupport,
    SourceRequirement,
    SpanSupport,
    StructuredDraftAnswer,
)
from onyx.legal_composite.requirements import (
    RequirementLedger,
    apply_patch,
    canonicalize_draft_supports,
    canonicalize_source_support,
    draft_binding_gaps,
    support_is_original,
)
from tests.unit.onyx.legal_composite.test_review_assessment import original

RULE = "Başvuru onaylanırsa işlem yapılır; ret halinde işlem yapılmaz."
OTHER_RULE = "Başvuru süresi, bildirimin yapıldığı tarihte başlar."


def originals() -> EvidenceLedger:
    ledger = EvidenceLedger()
    items = [original(RULE, "one"), original(OTHER_RULE, "two")]
    for item in items:
        assert item.search_doc is not None and item.chunk_id is not None
        item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    ledger.add(items, RunContext())
    return ledger


def plan() -> IssueResearchPlan:
    return IssueResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            IssueResearchNeed(
                need_id="approval",
                question="Başvuru sonucu ne olur?",
                governing_source="",
                conditions_to_check=[],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def selected_support(citation: int = 1, text: str = RULE) -> SpanSupport:
    return SpanSupport(
        citation=citation,
        span_id=original_witness_spans(citation, text)[0]["witness_id"],
    )


def requirement(support: SpanSupport | PassageSupport) -> SourceRequirement:
    return SourceRequirement(
        requirement_id="approval_rule",
        need_id="approval",
        dimension="procedure",
        rule="İşlem onaya bağlıdır; ret halinde yapılamaz.",
        application="Başvurunun sonucu verilmediğinden iki durum ayrılır.",
        supports=[SpanSupport.model_validate(support)],
    )


def test_selector_is_resolved_into_full_original_and_preserves_input() -> None:
    ledger = originals()
    submitted = requirement(selected_support())
    recorded = RequirementLedger(ledger)
    recorded.update([submitted], plan(), {1, 2})
    resolved = recorded.records()[0].supports[0]
    assert submitted.supports[0].quotation == ""
    assert resolved.quotation == RULE
    assert resolved.span_id == submitted.supports[0].span_id
    assert support_is_original(resolved, ledger, {1})
    recorded.update([submitted], plan(), {1, 2})
    assert len(recorded.records()) == 1
    receipt = recorded.export()[0]["original_bindings"]
    assert isinstance(receipt, list)
    assert receipt[0]["start"] == 0
    assert receipt[0]["end"] == len(RULE)
    assert receipt[0]["span_id"] == resolved.span_id


def test_literal_support_remains_compatible_without_changing_legacy_schema() -> None:
    legacy = PassageSupport(citation=1, quotation=RULE)
    assert set(PassageSupport.model_json_schema()["properties"]) == {
        "citation",
        "quotation",
    }
    assert PassageSupport.model_json_schema()["required"] == [
        "citation",
        "quotation",
    ]
    recorded = RequirementLedger(originals())
    recorded.update([requirement(legacy)], plan(), {1})
    assert recorded.records()[0].supports[0].quotation == RULE
    assert recorded.records()[0].supports[0].span_id is None
    assert DraftAnswer(answer="Merhaba", unresolved_need_ids=[]).answer == "Merhaba"


@pytest.mark.parametrize(
    "defect", ["citation", "hash", "range", "quotation", "delivery"]
)
def test_bad_selector_rejects_entire_batch_without_promoting_valid_neighbor(
    defect: str,
) -> None:
    ledger = originals()
    support = selected_support()
    delivered = {1, 2}
    if defect == "citation":
        support.citation = 2
    elif defect == "hash":
        assert support.span_id is not None
        support.span_id = support.span_id.replace(
            support.span_id.split("-")[1], "0" * 16
        )
    elif defect == "range":
        assert support.span_id is not None
        support.span_id += "-0-1"
    elif defect == "quotation":
        support.quotation = "Başvuru yapılırsa işlem yapılır."
    else:
        delivered = {2}
    valid = requirement(selected_support(2, OTHER_RULE)).model_copy(
        update={"requirement_id": "valid_neighbor"}
    )
    recorded = RequirementLedger(ledger)
    with pytest.raises(InvalidSourceAction):
        recorded.update([valid, requirement(support)], plan(), delivered)
    assert recorded.records() == []


@pytest.mark.parametrize("defect", ["hash", "source", "chunk", "derived", "truncated"])
def test_selector_never_bypasses_canonical_provenance(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = originals()
    unchanged_get = ledger.get

    def changed_get(citation: int) -> EvidenceItem | None:
        item = unchanged_get(citation)
        if item is None or citation != 1:
            return item
        assert item.search_doc is not None
        if defect == "hash":
            item.text_hash = "0" * 64
        elif defect == "source":
            item.search_doc.document_id = "another-source"
        elif defect == "chunk":
            item.search_doc.metadata["regulatory_chunk_id"] = "another-chunk"
        else:
            item.metadata[defect] = True
        return item

    monkeypatch.setattr(ledger, "get", changed_get)
    with pytest.raises(InvalidSourceAction, match="canonical original"):
        canonicalize_source_support(selected_support(), ledger, {1})


def test_full_delivered_original_catalogue_has_the_same_resolvable_selector() -> None:
    ledger = originals()
    records = json.loads(
        ledger.serialize_records([1], required=[1], include_witness_spans=True)
    )
    span = records[0]["witness_spans"][0]
    normalized = canonicalize_source_support(
        SpanSupport(citation=1, span_id=span["witness_id"]), ledger, {1}
    )
    assert (
        normalized.quotation
        == records[0]["text"][span["start_char"] : span["end_char"]]
    )


def test_writer_and_patch_resolve_same_canonical_support_without_changing_claim() -> (
    None
):
    ledger = originals()
    section = AnswerSection(
        section_id="approval_section", need_ids=["approval"], text=f"{RULE} [1]"
    )
    claim = DraftClaim(
        claim_id="approval_claim",
        section_id=section.section_id,
        need_ids=["approval"],
        answer_excerpt=RULE,
        supports=[selected_support()],
    )
    draft = StructuredDraftAnswer(
        unresolved_need_ids=[],
        sections=[section],
        claims=[claim],
        requirements=[requirement(selected_support())],
    )
    normalized = canonicalize_draft_supports(draft, ledger, {1})
    assert isinstance(normalized, StructuredDraftAnswer)
    assert normalized.answer == draft.answer
    assert normalized.claims[0].answer_excerpt == draft.claims[0].answer_excerpt
    assert normalized.claims[0].supports[0].quotation == RULE
    assert normalized.requirements[0].supports[0].quotation == RULE
    assert draft.claims[0].supports[0].quotation == ""
    assert (
        draft_binding_gaps(normalized, plan(), normalized.requirements, ledger, {1})
        == []
    )
    patch = DraftPatch(sections=[section], claims=[claim], unresolved_need_ids=[])
    resolved_patch = canonicalize_draft_supports(patch, ledger, {1})
    assert isinstance(resolved_patch, DraftPatch)
    assert resolved_patch.claims[0].supports[0].quotation == RULE
    assert patch.claims[0].supports[0].quotation == ""


def test_binding_rejects_real_quotation_attached_to_a_false_span() -> None:
    ledger = originals()
    assert not support_is_original(
        SpanSupport(citation=1, span_id="invented", quotation=RULE), ledger, {1}
    )


def test_repeated_text_receipt_uses_selected_occurrence_not_first_match() -> None:
    repeated = "x" * 800
    text = repeated * 2
    ledger = EvidenceLedger()
    item = original(text, "repeated")
    assert item.search_doc is not None and item.chunk_id is not None
    item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    ledger.add([item], RunContext())
    span = original_witness_spans(1, text)[1]
    recorded = RequirementLedger(ledger)
    recorded.update(
        [requirement(SpanSupport(citation=1, span_id=span["witness_id"]))],
        plan(),
        {1},
    )
    receipt = recorded.export()[0]["original_bindings"]
    assert isinstance(receipt, list)
    assert receipt[0]["start"] == 800
    assert receipt[0]["end"] == 1600


@pytest.mark.parametrize("missing", ["sections", "claims"])
def test_structured_writer_must_supply_both_inventories(missing: str) -> None:
    value = {
        "answer": "Merhaba",
        "sections": [
            {"section_id": "greeting", "need_ids": ["social"], "text": "Merhaba"}
        ],
        "claims": [],
        "unresolved_need_ids": [],
    }
    del value[missing]
    with pytest.raises(ValidationError):
        StructuredDraftAnswer.model_validate(value)


def claim_rendered_draft() -> StructuredDraftAnswer:
    return StructuredDraftAnswer(
        unresolved_need_ids=[],
        sections=[
            AnswerSection(
                section_id="approval_section",
                need_ids=["approval"],
                text="### Başvuru",
                claim_ids=["approval_claim", "approval_application"],
            ),
            AnswerSection(
                section_id="clock_section",
                need_ids=["clock"],
                claim_ids=["clock_claim"],
            ),
        ],
        claims=[
            DraftClaim(
                claim_id="approval_claim",
                section_id="approval_section",
                need_ids=["approval"],
                answer_excerpt=f"{RULE} [1]",
                supports=[selected_support()],
            ),
            DraftClaim(
                claim_id="approval_application",
                section_id="approval_section",
                need_ids=["approval"],
                answer_excerpt="Onay yoksa işlemin yapılabildiği söylenemez [1].",
                supports=[selected_support()],
            ),
            DraftClaim(
                claim_id="clock_claim",
                section_id="clock_section",
                need_ids=["clock"],
                answer_excerpt=f"{OTHER_RULE} [2]",
                supports=[selected_support(2, OTHER_RULE)],
            ),
        ],
    )


def test_claim_rendering_uses_exact_ordered_prose_and_empty_heading() -> None:
    draft = claim_rendered_draft()
    assert draft.sections[0].text == "\n\n".join(
        ["### Başvuru", draft.claims[0].answer_excerpt, draft.claims[1].answer_excerpt]
    )
    assert draft.sections[1].text == draft.claims[2].answer_excerpt
    assert draft.answer == "\n\n".join(section.text for section in draft.sections)
    assert all(claim.answer_excerpt in draft.answer for claim in draft.claims)


def test_claim_rendering_is_idempotent_across_export_and_normalization() -> None:
    draft = claim_rendered_draft()
    replay = StructuredDraftAnswer.model_validate(draft.model_dump(mode="json"))
    replay = StructuredDraftAnswer.model_validate(replay.model_dump(mode="json"))
    normalized = canonicalize_draft_supports(replay, originals(), {1, 2})
    assert replay.answer == normalized.answer == draft.answer
    assert normalized.answer.count(draft.claims[0].answer_excerpt) == 1
    assert normalized.answer.count(draft.claims[2].answer_excerpt) == 1


@pytest.mark.parametrize(
    "defect", ["unknown", "duplicate", "cross_section", "unlisted", "issue"]
)
def test_claim_rendering_rejects_invalid_references(defect: str) -> None:
    draft = claim_rendered_draft()
    section = draft.sections[0].model_copy(deep=True, update={"text": "### Başvuru"})
    if defect == "unknown":
        section.claim_ids = ["unknown"]
    elif defect == "duplicate":
        section.claim_ids = ["approval_claim", "approval_claim", "approval_application"]
    elif defect == "cross_section":
        section.claim_ids = ["clock_claim"]
    elif defect == "unlisted":
        section.claim_ids = ["approval_claim"]
    else:
        section.need_ids = ["another_issue"]
    with pytest.raises(ValidationError):
        StructuredDraftAnswer(
            sections=[section, draft.sections[1]],
            claims=draft.claims,
            unresolved_need_ids=[],
        )


def test_claim_rendered_patch_preserves_unchanged_section_and_renders_once() -> None:
    draft = claim_rendered_draft()
    replacement = DraftClaim(
        claim_id="replacement",
        section_id="approval_section",
        need_ids=["approval"],
        answer_excerpt=f"İşlem onaya bağlıdır: {RULE} [1]",
        supports=[selected_support()],
    )
    patch = DraftPatch(
        sections=[
            AnswerSection(
                section_id="approval_section",
                need_ids=["approval"],
                text="### Onay koşulu",
                claim_ids=["replacement"],
            )
        ],
        claims=[replacement],
        unresolved_need_ids=[],
    )
    normalized = canonicalize_draft_supports(patch, originals(), {1, 2})
    revised = apply_patch(draft, normalized, {"approval_section"})
    assert revised.sections[1] == draft.sections[1]
    assert revised.answer.count(replacement.answer_excerpt) == 1
    assert draft.claims[0].claim_id not in {claim.claim_id for claim in revised.claims}
    replay = apply_patch(revised, normalized, {"approval_section"})
    assert replay.answer == revised.answer


@pytest.mark.parametrize("mode", ["literal", "span"])
@pytest.mark.parametrize(
    "defect", ["text", "hash", "derived", "truncated", "source", "chunk"]
)
def test_export_retains_history_but_marks_invalid_canonical_bindings(
    mode: str, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = originals()
    recorded = RequirementLedger(ledger)
    support = (
        selected_support()
        if mode == "span"
        else PassageSupport(citation=1, quotation=RULE)
    )
    recorded.update([requirement(support)], plan(), {1})
    original_history = recorded.records()
    unchanged_get = ledger.get

    def changed_get(citation: int) -> EvidenceItem | None:
        item = unchanged_get(citation)
        if item is None or citation != 1:
            return item
        assert item.search_doc is not None
        if defect == "text":
            item.text = "Metin artık aynı asli hüküm değildir."
        elif defect == "hash":
            item.text_hash = "0" * 64
        elif defect == "source":
            item.search_doc.document_id = "changed-source"
        elif defect == "chunk":
            item.search_doc.metadata["regulatory_chunk_id"] = "changed-chunk"
        else:
            item.metadata[defect] = True
        return item

    monkeypatch.setattr(ledger, "get", changed_get)
    exported = recorded.export()[0]
    assert recorded.records() == original_history
    assert exported["original_bindings"] == []
    assert exported["binding_status"] == "invalid"
    assert exported["original_binding_errors"]
    assert not support_is_original(recorded.records()[0].supports[0], ledger, {1})


def test_intact_binding_export_keeps_existing_receipt_shape() -> None:
    ledger = originals()
    recorded = RequirementLedger(ledger)
    recorded.update([requirement(selected_support())], plan(), {1})
    exported = recorded.export()[0]
    assert "binding_status" not in exported
    assert "original_binding_errors" not in exported
    assert exported["original_bindings"]


def test_repair_claim_id_cannot_collide_with_unchanged_section() -> None:
    draft = claim_rendered_draft()
    before = draft.model_dump(mode="json")
    patch = DraftPatch(
        sections=[
            AnswerSection(
                section_id="approval_section",
                need_ids=["approval"],
                claim_ids=["clock_claim"],
            )
        ],
        claims=[
            DraftClaim(
                claim_id="clock_claim",
                section_id="approval_section",
                need_ids=["approval"],
                answer_excerpt=f"{RULE} [1]",
                supports=[selected_support()],
            )
        ],
        unresolved_need_ids=[],
    )
    with pytest.raises(InvalidSourceAction, match="unchanged section"):
        apply_patch(draft, patch, {"approval_section"})
    assert draft.model_dump(mode="json") == before
