import json
from typing import Any

import mistune
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


def test_claimless_draft_needs_explicit_unresolved_issues() -> None:
    payload = {
        "blocks": [
            {
                "block_id": "limitation",
                "text": "Özgün dayanak belirlenemediği için kesin sonuç verilemiyor.",
                "claims": [],
            }
        ],
        "unresolved_issue_ids": ["I1"],
    }
    compiled = compile_draft(GeneratedDraft.model_validate(payload))
    assert compiled.claims == []
    assert compiled.unresolved_issue_ids == ["I1"]
    assert compiled.answer == payload["blocks"][0]["text"]

    payload["unresolved_issue_ids"] = []
    with pytest.raises(ValidationError, match="identify unresolved issues"):
        GeneratedDraft.model_validate(payload)


def test_mutated_duplicate_ids_are_rejected_at_compilation() -> None:
    generated = GeneratedDraft.model_validate(draft_payload())
    generated.blocks[2].block_id = generated.blocks[1].block_id
    with pytest.raises(ValidationError, match="unique"):
        compile_draft(generated)


def test_host_renders_selected_citations_without_model_marker_bookkeeping() -> None:
    payload = draft_payload()
    payload["blocks"][1]["text"] = "İlgili koşul.\nBelge ibrazı değerlendirilir. [1]"
    payload["blocks"][1]["claims"][0]["supports"].extend(
        [
            {"citation": 2, "span_number": 4},
            {"citation": 3, "span_number": 1},
        ]
    )
    compiled = compile_draft(GeneratedDraft.model_validate(payload))
    rendered = payload["blocks"][1]["text"] + "\n\n[2] [3]"
    assert compiled.claims[0].answer_excerpt == rendered
    assert rendered in compiled.answer
    assert compiled.claims[0].answer_excerpt.count("[2]") == 1
    assert len(compiled.claims[0].supports) == 4


def test_marker_rendering_preserves_foreign_model_citations_for_engine_rejection() -> (
    None
):
    payload = draft_payload()
    payload["blocks"][1]["text"] = "İlgili koşul. [999]"
    compiled = compile_draft(GeneratedDraft.model_validate(payload))
    assert compiled.claims[0].answer_excerpt == "İlgili koşul. [999]\n\n[1] [2]"
    assert "[999]" in compiled.answer
    assert all(support.citation != 999 for support in compiled.claims[0].supports)


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_host_markers_preserve_closed_fences_and_following_prose(fence: str) -> None:
    text = f"{fence}text\nUsul talimatı\n{fence}"
    generated = GeneratedDraft.model_validate(
        {
            "blocks": [
                {
                    "block_id": "instructions",
                    "text": text,
                    "claims": [
                        {
                            "claim_id": "procedure",
                            "issue_ids": ["I1"],
                            "supports": [{"citation": 1, "span_number": 1}],
                        }
                    ],
                },
                {"block_id": "continuation", "text": "Sonraki paragraf.", "claims": []},
            ]
        }
    )
    compiled = compile_draft(generated)
    html = mistune.create_markdown()(compiled.answer)
    assert isinstance(html, str)
    assert "Usul talimatı\n</code></pre>" in html
    assert "<p>[1]</p>" in html
    assert "<p>Sonraki paragraf.</p>" in html
    assert compiled.claims[0].answer_excerpt == text + "\n\n[1]"


def test_host_markers_preserve_markdown_table_body() -> None:
    text = "| Adım | Koşul |\n| --- | --- |\n| Başvuru | Belge |"
    generated = GeneratedDraft.model_validate(
        {
            "blocks": [
                {
                    "block_id": "steps",
                    "text": text,
                    "claims": [
                        {
                            "claim_id": "procedure",
                            "issue_ids": ["I1"],
                            "supports": [{"citation": 1, "span_number": 1}],
                        }
                    ],
                }
            ]
        }
    )
    compiled = compile_draft(generated)
    html = mistune.create_markdown(plugins=["table"])(compiled.answer)
    assert isinstance(html, str)
    assert "<td>Başvuru</td>" in html and "<td>Belge</td>" in html
    assert "</table>\n<p>[1]</p>" in html
    assert compiled.claims[0].answer_excerpt == text + "\n\n[1]"
