import json
from typing import Any

import pytest
from pydantic import ValidationError

from onyx.legal_review.drafting import GeneratedDraft, compile_draft
from onyx.legal_review.passages import PassageReference


def draft_payload() -> dict[str, Any]:
    return {
        "blocks": [
            {"block_id": "heading", "text": "### Başvuru ve mali sonuç", "claims": []},
            {
                "block_id": "procedure",
                "text": "Başvuru ilgili idareye yapılır. [1]\nBelge koşulu ayrıca değerlendirilir. [2]",
                "claims": [
                    {
                        "claim_id": "procedure-and-evidence",
                        "issue_ids": ["I1", "I2"],
                        "supports": [
                            {"citation": 1, "span_number": 2},
                            {"citation": 2, "span_number": 1},
                        ],
                    }
                ],
            },
            {
                "block_id": "conditional-result",
                "text": "İade sonucu, eşyanın koşullarına bağlıdır. [2]",
                "claims": [
                    {
                        "claim_id": "refund",
                        "issue_ids": ["I1"],
                        "supports": [{"citation": 2, "span_number": 3}],
                    }
                ],
            },
        ],
        "unresolved_issue_ids": ["I2"],
    }


def test_integrated_draft_compiles_exact_blocks_and_multiple_source_bindings() -> None:
    generated = GeneratedDraft.model_validate_json(
        json.dumps(draft_payload(), ensure_ascii=False)
    )
    compiled = compile_draft(generated)

    assert compiled.answer == "\n\n".join(block.text for block in generated.blocks)
    assert "[1]\nBelge" in compiled.answer
    assert [claim.claim_id for claim in compiled.claims] == [
        "procedure-and-evidence",
        "refund",
    ]
    assert compiled.claims[0].issue_ids == ["I1", "I2"]
    assert compiled.claims[0].answer_excerpt == generated.blocks[1].text
    assert compiled.claims[1].answer_excerpt == generated.blocks[2].text
    assert all(claim.answer_excerpt in compiled.answer for claim in compiled.claims)
    assert compiled.claims[0].supports == [
        PassageReference(citation=1, span_number=2),
        PassageReference(citation=2, span_number=1),
    ]
    assert compiled.unresolved_issue_ids == ["I2"]
    assert "quotation" not in str(generated.model_dump())
    assert "answer_excerpt" not in str(generated.model_dump())


def test_claims_field_is_required_even_for_heading() -> None:
    payload = draft_payload()
    del payload["blocks"][0]["claims"]
    with pytest.raises(ValidationError):
        GeneratedDraft.model_validate(payload)
    schema = GeneratedDraft.model_json_schema()
    assert "claims" in schema["$defs"]["GeneratedBlock"]["required"]
    assert "answer_excerpt" not in schema["$defs"]["GeneratedClaim"]["properties"]


@pytest.mark.parametrize("identity", ["block_id", "claim_id"])
def test_duplicate_identities_cannot_bind_ambiguous_text(identity: str) -> None:
    payload = draft_payload()
    if identity == "block_id":
        payload["blocks"][2]["block_id"] = payload["blocks"][1]["block_id"]
    else:
        payload["blocks"][2]["claims"][0]["claim_id"] = payload["blocks"][1]["claims"][
            0
        ]["claim_id"]
    with pytest.raises(ValidationError, match="unique"):
        GeneratedDraft.model_validate(payload)


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_block_never_compiles(text: str) -> None:
    payload = draft_payload()
    payload["blocks"][0]["text"] = text
    with pytest.raises(ValidationError):
        GeneratedDraft.model_validate(payload)


@pytest.mark.parametrize("field", ["quotation", "answer_excerpt"])
def test_model_cannot_supply_copied_proof_or_answer_excerpt(field: str) -> None:
    payload = draft_payload()
    claim = payload["blocks"][1]["claims"][0]
    if field == "quotation":
        claim["supports"][0][field] = "Kaynak metninin yeniden yazılması"
    else:
        claim[field] = "Yeniden yazılmış cevap parçası"
    with pytest.raises(ValidationError, match="Extra inputs"):
        GeneratedDraft.model_validate(payload)


@pytest.mark.parametrize("field", ["citation", "span_number"])
@pytest.mark.parametrize("value", [0, -1, "1", True])
def test_invalid_canonical_selector_cannot_enter_claim(
    field: str, value: object
) -> None:
    payload = draft_payload()
    payload["blocks"][1]["claims"][0]["supports"][0][field] = value
    with pytest.raises(ValidationError):
        GeneratedDraft.model_validate(payload)


def test_claimless_answer_and_mutated_duplicate_ids_are_rejected() -> None:
    payload = draft_payload()
    for block in payload["blocks"]:
        block["claims"] = []
    with pytest.raises(ValidationError, match="at least one supported claim"):
        GeneratedDraft.model_validate(payload)

    generated = GeneratedDraft.model_validate(draft_payload())
    generated.blocks[2].block_id = generated.blocks[1].block_id
    with pytest.raises(ValidationError, match="unique"):
        compile_draft(generated)
