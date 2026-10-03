"""Original-bound review selectors and material details across unrelated domains."""

import json

import pytest
from pydantic import ValidationError

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
    assertion_support_defect,
    assertion_witness_valid,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import MaterialSourceOmission, ResearchModel
from onyx.asv3.models import EvidenceItem, RunStopped
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record
from onyx.asv3.witnesses import original_witness_spans, original_witness_text
from tests.unit.onyx.asv3.test_assessment_contract_repair import review_fixture
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response


def test_ranges_cover_large_unicode_original_without_duplication_or_clipping() -> None:
    text = ("İşlem şartları ve istisnalar.\n" * 200) + "最終条件。"
    spans = original_witness_spans(7, text)
    assert spans[0]["start_char"] == 0 and spans[-1]["end_char"] == len(text)
    assert all(0 < span["end_char"] - span["start_char"] <= 800 for span in spans)
    assert all(set(span) == {"witness_id", "start_char", "end_char"} for span in spans)
    assert (
        "".join(
            original_witness_text(7, text, span["witness_id"]) or "" for span in spans
        )
        == text
    )


@pytest.mark.parametrize("defect", ["foreign", "changed_text", "invented_range"])
def test_foreign_stale_or_non_catalogued_selectors_are_rejected(defect: str) -> None:
    text = "A signed certificate is required before release."
    selector = original_witness_spans(1, text)[0]["witness_id"]
    citation = 2 if defect == "foreign" else 1
    if defect == "changed_text":
        text += " Different operative exception."
    elif defect == "invented_range":
        selector = selector.rsplit("-", 1)[0] + "-1"
    assert not assertion_witness_valid(
        AssertionWitness(citation=citation, witness_id=selector), {citation: text}
    )


@pytest.mark.parametrize(
    "fields", [{}, {"source_quote": " "}, {"source_quote": "rule", "witness_id": "id"}]
)
def test_selector_schema_requires_exactly_one_witness(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AssertionWitness(citation=1, **fields)


def test_literal_checkpoints_and_catalogued_passages_share_publication_contract() -> (
    None
):
    text = "The condition and exception are cumulative."
    unit = assertion_inventory("Both conditions are required [1].")[0]
    for witness in (
        AssertionWitness(citation=1, source_quote="condition and exception"),
        AssertionWitness(
            citation=1, witness_id=original_witness_spans(1, text)[0]["witness_id"]
        ),
    ):
        review = AssertionVerification(
            unit_id=unit["unit_id"], status="supported", witnesses=[witness]
        )
        assert assertion_support_defect(unit, review, {1: text}, "facts") is None
        restored = AssertionVerification.model_validate_json(review.model_dump_json())
        assert restored == review


def test_catalogue_preserves_original_provenance_and_counts_in_delivery_budget() -> (
    None
):
    ledger, context = original_ledger()
    evidence = _evidence_record(ledger, "Rule [1].", include_witness_spans=True)
    rows = json.loads(evidence)
    item = ledger.get(1)
    assert item is not None
    assert rows[0]["text"] == item.text and rows[0]["text_hash"] == item.text_hash
    assert rows[0]["witness_spans"] == original_witness_spans(1, item.text)
    assert rows[0]["truncated"] is False
    assert "witness_spans" not in json.loads(_evidence_record(ledger, "Rule [1]."))[0]
    with pytest.raises(RunStopped):
        ledger.serialize_records(
            [1], required=[1], max_chars=len(evidence) - 1, include_witness_spans=True
        )
    _old_ledger, _old_context, _answer, data, review = review_fixture(bad_quote=False)
    payload = json.loads(data)
    payload["evidence"] = evidence
    review.assertion_results[0].witnesses = [
        AssertionWitness(
            citation=1, witness_id=rows[0]["witness_spans"][0]["witness_id"]
        )
    ]
    llm = scripted_model()
    llm.invoke.return_value = text_response(review.model_dump(mode="json"))
    model = ResearchModel(llm, context)
    result = model.invoke_verification("Assess the originals", json.dumps(payload))
    assert result.safe_to_publish and llm.invoke.call_count == 1
    assert model.last_call_id is not None
    assert ledger.completely_delivered(model.last_call_id) == {1}


@pytest.mark.parametrize(
    ("original", "answer", "detail"),
    [
        (
            "Operation may restart after inspection. A signed inspector certificate must be filed before restart.",
            "Operation may restart after inspection [1].",
            "A signed inspector certificate must be filed before restart.",
        ),
        (
            "Ödeme koşullar sağlanınca iade edilir. Başvuru bildirimden itibaren otuz gün içinde yapılır.",
            "Koşullar sağlanınca ödeme iade edilir [1].",
            "Başvuru süresi bildirimden itibaren otuz gündür.",
        ),
    ],
)
def test_original_supported_material_condition_blocks_a_broad_positive_review(
    original: str, answer: str, detail: str
) -> None:
    old_ledger, context = original_ledger()
    item = old_ledger.get(1)
    assert item is not None
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id=item.source_id, text=original, search_doc=item.search_doc
            )
        ],
        context,
    )
    review = supported([1])
    review.omitted_material_source_details = [
        MaterialSourceOmission(
            witness=AssertionWitness(
                citation=1,
                witness_id=original_witness_spans(1, original)[0]["witness_id"],
            ),
            determination_ids=["q0:d0"],
            detail=detail,
            applicability="This is an operative condition of the requested permission/refund.",
        )
    ]
    for partial in (False, True):
        gap = publication_gap(
            answer,
            review,
            ["What procedure applies?"],
            ledger,
            allow_explicit_gaps=partial,
        )
        assert gap is not None
        assert detail in str(gap.data["gaps"])
        assert "omitted_material_source_details" in gap.data
    review.omitted_material_source_details = []
    assert publication_gap(answer, review, ["What procedure applies?"], ledger) is None


@pytest.mark.parametrize("defect", ["foreign_part", "foreign_source", "undelivered"])
def test_material_omissions_need_current_original_and_requested_identity(
    defect: str,
) -> None:
    ledger, _context = original_ledger()
    item = ledger.get(1)
    assert item is not None
    review = supported([1])
    review.omitted_material_source_details = [
        MaterialSourceOmission(
            witness=AssertionWitness(
                citation=999 if defect == "foreign_source" else 1,
                source_quote="original tail 1",
            ),
            determination_ids=["q99:d0" if defect == "foreign_part" else "q0:d0"],
            detail="The operative tail was omitted.",
            applicability="It affects the requested procedure.",
        )
    ]
    gap = publication_gap(
        "Apply the condition [1].",
        review,
        ["What procedure applies?"],
        ledger,
        verification_call_id="no-original-delivery"
        if defect == "undelivered"
        else None,
    )
    assert gap is not None
    assert (
        "not completely delivered" in str(gap.data)
        if defect == "undelivered"
        else "invalid original witness" in str(gap.data)
    )
