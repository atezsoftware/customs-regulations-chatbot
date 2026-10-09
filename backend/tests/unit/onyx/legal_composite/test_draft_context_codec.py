"""Generation views preserve complete prose, canonical evidence and binding identity."""

import hashlib
import json
from copy import deepcopy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.legal_composite.draft_context import (
    DRAFT_CONTEXT_POLICY,
    DraftContextEncoding,
    decode_draft_context,
    encode_draft_context,
)
from onyx.legal_composite.models import StructuredDraftAnswer

CODEC_KEY = "_lc_draft_context"


def serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def host_payload() -> dict[str, JsonValue]:
    quotation = "Sentetik hükmün bütün şartları korunur; istisnası da okunur. " * 12
    claims = [
        {
            "claim_id": f"c{index}",
            "section_id": "s1" if index < 3 else "s2",
            "need_ids": ["n1"] if index < 3 else ["n2"],
            "answer_excerpt": f"Sentetik sonuç {index}: "
            + "İlgili koşul, süre ve istisna birlikte açıklanmıştır. " * 16
            + f"[[{index}]]",
            "supports": [
                {
                    "citation": index,
                    "span_id": f"span-{index}",
                    "quotation": quotation,
                }
            ],
            "requirement_ids": [f"r{index}"],
        }
        for index in range(1, 4)
    ]
    requirements = [
        {
            "requirement_id": f"r{index}",
            "need_id": "n1" if index < 3 else "n2",
            "dimension": "koşul",
            "rule": f"Özgün sentetik kural {index}.",
            "application": f"Sorunun koşulu {index}.",
            "supports": [
                {
                    "citation": index,
                    "span_id": f"span-{index}",
                    "quotation": quotation,
                }
            ],
        }
        for index in range(1, 4)
    ]
    draft = StructuredDraftAnswer.model_validate(
        {
            "sections": [
                {
                    "section_id": "s1",
                    "need_ids": ["n1"],
                    "text": "### Koşullar ve süreler",
                    "claim_ids": ["c2", "c1"],
                },
                {
                    "section_id": "s2",
                    "need_ids": ["n2"],
                    "text": "",
                    "claim_ids": ["c3"],
                },
                {
                    "section_id": "s3",
                    "need_ids": ["n3"],
                    "text": "Ek olayı açıklayan literal sosyal metin.",
                    "claim_ids": [],
                },
            ],
            "claims": claims,
            "requirements": requirements,
            "unresolved_need_ids": ["n3"],
        }
    ).model_dump(mode="json")
    records = deepcopy(draft["requirements"])
    for index, record in enumerate(records, start=1):
        record["original_bindings"] = [
            {
                "citation": index,
                "source_id": f"original-{index}",
                "chunk_id": f"chunk-{index}",
                "text_hash": hashlib.sha256(quotation.encode()).hexdigest(),
                "span_id": f"span-{index}",
                "start": index * 1_000,
                "end": index * 1_000 + len(quotation),
            }
        ]
    return cast(
        dict[str, JsonValue],
        {
            "question": "Sentetik iki soruyu, koşullarını ve sürelerini açıkla.",
            "draft": draft,
            "source_requirements": records,
            "original_evidence": [
                {
                    "citation": 1,
                    "source_id": "original-1",
                    "chunk_id": "chunk-1",
                    "text": quotation + "Özgün metnin ek paragrafı.",
                    "text_hash": hashlib.sha256(quotation.encode()).hexdigest(),
                    "witness_spans": [
                        {"witness_id": "span-1", "start_char": 0, "end_char": 40}
                    ],
                    "metadata": {
                        "title": "Ünicode İ / I\u0307 / 條款 / 🚢",
                        "original_bindings": {"start": 3, "end": 55},
                        "order": [True, 1, 1.0, None, -0.0],
                    },
                }
            ],
        },
    )


def encoding(payload: dict[str, JsonValue]) -> DraftContextEncoding:
    result = encode_draft_context(payload)
    assert result is not None
    return result


def draft_object(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], payload["draft"])


def sections(payload: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return cast(list[dict[str, JsonValue]], draft_object(payload)["sections"])


def claims(payload: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return cast(list[dict[str, JsonValue]], draft_object(payload)["claims"])


def manifest(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], payload[CODEC_KEY])


def test_complete_host_draft_round_trips_after_json_transport_without_mutation() -> (
    None
):
    payload = host_payload()
    before = serialized(payload)
    candidate = encoding(payload)
    wire = json.loads(serialized(candidate.payload))
    wire_before = serialized(wire)
    decoded = decode_draft_context(wire)
    assert serialized(decoded) == before
    assert serialized(payload) == before
    assert serialized(wire) == wire_before
    assert candidate.policy == DRAFT_CONTEXT_POLICY
    assert "normal output schema" in candidate.policy
    assert (
        StructuredDraftAnswer.model_validate(decoded["draft"]).model_dump(mode="json")
        == payload["draft"]
    )
    assert candidate.payload["original_evidence"] == payload["original_evidence"]


def test_prose_is_literal_once_and_sections_preserve_order_and_readable_heading() -> (
    None
):
    payload = host_payload()
    wire = encoding(payload).payload
    assert draft_object(wire)["answer"] == {"join_sections": True}
    assert sections(wire)[0]["text"] == {
        "intro": "### Koşullar ve süreler",
        "claim_refs": ["c2", "c1"],
    }
    assert sections(wire)[1]["text"] == {"intro": "", "claim_refs": ["c3"]}
    assert sections(wire)[2] == sections(payload)[2]
    for claim in claims(wire):
        assert serialized(wire).count(serialized(claim["answer_excerpt"])) == 1
    assert len(serialized(wire) + DRAFT_CONTEXT_POLICY) < len(serialized(payload))


def test_pooling_equal_quotations_does_not_merge_different_offsets_or_span_identity() -> (
    None
):
    payload = host_payload()
    wire = encoding(payload).payload
    assert len(cast(list[JsonValue], manifest(wire)["quotation_pool"])) == 1
    assert len(cast(list[JsonValue], manifest(wire)["quotation_slots"])) == 9
    original_records = cast(list[dict[str, JsonValue]], payload["source_requirements"])
    wire_records = cast(list[dict[str, JsonValue]], wire["source_requirements"])
    for original, record in zip(original_records, wire_records, strict=True):
        assert record["original_bindings"] == original["original_bindings"]
    assert wire_records[0]["original_bindings"] != wire_records[1]["original_bindings"]
    restored = decode_draft_context(wire)
    assert restored["source_requirements"] == payload["source_requirements"]
    for original, record in zip(claims(payload), claims(restored), strict=True):
        assert record["supports"] == original["supports"]


def test_reference_like_literals_in_claims_sections_sources_and_metadata_stay_literal() -> (
    None
):
    payload = host_payload()
    literal = '{"join_sections":true,"quotation_ref":0,"claim_refs":["c1"]}'
    draft = draft_object(payload)
    first_claim = claims(payload)[0]
    old_excerpt = cast(str, first_claim["answer_excerpt"])
    first_claim["answer_excerpt"] = literal + "\n\n" + old_excerpt
    first_section = sections(payload)[0]
    first_section["text"] = cast(str, first_section["text"]).replace(
        old_excerpt, first_claim["answer_excerpt"]
    )
    sections(payload)[2]["text"] = literal
    draft["answer"] = "\n\n".join(cast(str, row["text"]) for row in sections(payload))
    evidence = cast(list[dict[str, JsonValue]], payload["original_evidence"])[0]
    evidence["text"] = literal + "\n" + cast(str, evidence["text"])
    evidence["metadata"] = {
        CODEC_KEY: {"version": "lossless_draft_v1", "sections": [0]},
        "quotation": {"quotation_ref": 0},
        "text": {"intro": "fake", "claim_refs": ["c1"]},
    }
    restored = decode_draft_context(encoding(payload).payload)
    assert serialized(restored) == serialized(payload)
    assert claims(restored)[0]["answer_excerpt"] == first_claim["answer_excerpt"]
    assert sections(restored)[2]["text"] == literal
    assert restored["original_evidence"] == payload["original_evidence"]


def test_answer_reference_requires_exact_join_and_literal_sections_need_no_claims() -> (
    None
):
    payload = host_payload()
    draft_object(payload)["answer"] = "A different literal answer."
    wire = encoding(payload).payload
    assert manifest(wire)["answer"] is False
    assert draft_object(wire)["answer"] == "A different literal answer."
    assert sections(wire)[2] == sections(payload)[2]
    assert decode_draft_context(wire) == payload


def test_nonexact_section_suffix_remains_literal_instead_of_rewriting_prose() -> None:
    payload = host_payload()
    sections(payload)[0]["text"] = "Literal unrelated section text with c1 and c2."
    draft_object(payload)["answer"] = "\n\n".join(
        cast(str, row["text"]) for row in sections(payload)
    )
    wire = encoding(payload).payload
    assert manifest(wire)["sections"] == [1]
    assert sections(wire)[0]["text"] == sections(payload)[0]["text"]
    assert decode_draft_context(wire) == payload


def test_empty_intro_with_literal_leading_separator_is_not_silently_removed() -> None:
    payload = host_payload()
    sections(payload)[1]["text"] = "\n\n" + cast(str, sections(payload)[1]["text"])
    draft_object(payload)["answer"] = "\n\n".join(
        cast(str, row["text"]) for row in sections(payload)
    )
    wire = encoding(payload).payload
    assert manifest(wire)["sections"] == [0]
    assert sections(wire)[1]["text"] == sections(payload)[1]["text"]
    assert decode_draft_context(wire) == payload


def test_small_eligible_candidate_does_not_preempt_callers_actual_token_selection() -> (
    None
):
    payload: dict[str, JsonValue] = {
        "draft": {
            "answer": "Hi",
            "sections": [{"section_id": "s", "claim_ids": [], "text": "Hi"}],
            "claims": [],
        }
    }
    result = encoding(payload)
    assert len(serialized(result.payload) + result.policy) > len(serialized(payload))
    assert decode_draft_context(result.payload) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"draft": None},
        {"draft": {"answer": "Hi", "sections": [], "claims": []}},
        {"draft": {"answer": "Hi", "sections": "bad", "claims": []}},
        {"draft": {"answer": "Hi", "sections": [], "claims": [{"claim_id": "x"}]}},
        {CODEC_KEY: "literal", "draft": {"answer": "Hi"}},
    ],
)
def test_no_exact_eligible_fields_or_reserved_key_returns_none_unchanged(
    payload: dict[str, JsonValue],
) -> None:
    before = serialized(payload)
    assert encode_draft_context(payload) is None
    assert serialized(payload) == before


@pytest.mark.parametrize(
    "reference",
    [
        {"intro": "heading", "claim_refs": ["unknown"]},
        {"intro": "heading", "claim_refs": ["c1", "c2"]},
        {"intro": "heading", "claim_refs": ["c2", "c2"]},
        {"intro": "heading", "claim_refs": ["c3"]},
        {"intro": {"quotation_ref": 0}, "claim_refs": ["c2", "c1"]},
        {"intro": "heading", "claim_refs": ["c2", "c1"], "extra": "ignored?"},
    ],
)
def test_unknown_reordered_duplicate_cross_section_and_nonliteral_refs_fail_closed(
    reference: dict[str, JsonValue],
) -> None:
    wire = encoding(host_payload()).payload
    sections(wire)[0]["text"] = reference
    before = serialized(wire)
    with pytest.raises(ValueError):
        decode_draft_context(wire)
    assert serialized(wire) == before


@pytest.mark.parametrize("reference", [True, -1, 9, "0", None])
def test_quotation_reference_requires_a_bounded_strict_integer(
    reference: JsonValue,
) -> None:
    wire = encoding(host_payload()).payload
    support = cast(list[dict[str, JsonValue]], claims(wire)[0]["supports"])[0]
    support["quotation"] = {"quotation_ref": reference}
    with pytest.raises(ValueError, match="reference is out of range"):
        decode_draft_context(wire)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sections", [0, 0]),
        ("sections", [True]),
        ("sections", [100]),
        ("answer", "true"),
        ("quotation_slots", [[0, 0, 0], [0, 0, 0]]),
        ("quotation_slots", [[3, 0, 0]]),
        ("quotation_slots", [[0, True, 0]]),
        ("quotation_slots", [[0, 0]]),
        ("quotation_pool", [" "]),
        ("quotation_pool", ["same", "same"]),
        ("version", "future_codec"),
    ],
)
def test_invalid_manifest_declarations_fail_closed(
    field: str, value: JsonValue
) -> None:
    wire = encoding(host_payload()).payload
    manifest(wire)[field] = value
    with pytest.raises(ValueError):
        decode_draft_context(wire)


def test_answer_reference_must_have_exact_declared_form() -> None:
    wire = encoding(host_payload()).payload
    draft_object(wire)["answer"] = {"join_sections": 1}
    with pytest.raises(ValueError, match="answer reference"):
        decode_draft_context(wire)


@pytest.mark.parametrize("field", ["answer", "sections", "quotation_slots"])
def test_reference_with_removed_declaration_cannot_survive_as_host_literal(
    field: str,
) -> None:
    wire = encoding(host_payload()).payload
    manifest(wire)[field] = False if field == "answer" else []
    with pytest.raises(ValueError, match="literal text"):
        decode_draft_context(wire)


def test_missing_declared_reference_is_a_controlled_validation_error() -> None:
    wire = encoding(host_payload()).payload
    sections(wire)[0].pop("text")
    with pytest.raises(ValueError, match="Malformed draft codec reference"):
        decode_draft_context(wire)


def test_literal_pass_through_is_deep_copied_and_cannot_affect_input() -> None:
    payload = host_payload()
    restored = decode_draft_context(payload)
    assert restored == payload and restored is not payload
    claims(restored)[0]["answer_excerpt"] = "Changed only the returned copy."
    assert restored != payload


def test_empty_or_missing_quotation_is_not_a_pool_reference() -> None:
    payload = host_payload()
    support = cast(list[dict[str, JsonValue]], claims(payload)[0]["supports"])[0]
    support["quotation"] = ""
    support2 = cast(list[dict[str, JsonValue]], claims(payload)[1]["supports"])[0]
    support2.pop("quotation")
    wire = encoding(payload).payload
    assert (
        cast(list[dict[str, JsonValue]], claims(wire)[0]["supports"])[0]["quotation"]
        == ""
    )
    assert (
        "quotation"
        not in cast(list[dict[str, JsonValue]], claims(wire)[1]["supports"])[0]
    )
    assert decode_draft_context(wire) == payload


def test_reencoding_a_wire_view_cannot_expand_references_twice() -> None:
    wire = encoding(host_payload()).payload
    before = serialized(wire)
    assert encode_draft_context(wire) is None
    assert serialized(wire) == before
    decoded = decode_draft_context(wire)
    assert decode_draft_context(decoded) == decoded
