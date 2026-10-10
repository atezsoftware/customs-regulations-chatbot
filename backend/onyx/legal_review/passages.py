"""Select exact canonical passages without model-authored quotes or offsets."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem
from onyx.asv3.witnesses import original_witness_spans


class PassageReference(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    citation: int = Field(gt=0, strict=True)
    span_number: int = Field(gt=0, strict=True)


def source_passage_references(
    citation: int, ledger: EvidenceLedger
) -> list[PassageReference]:
    """An inline source selection refers to its complete authorized original."""
    item = ledger.get(citation)
    validate_canonical_original(item)
    assert item is not None
    return [
        PassageReference(citation=citation, span_number=index)
        for index, _ in enumerate(original_witness_spans(citation, item.text), 1)
    ]


class CanonicalPassage(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    citation: int = Field(gt=0, strict=True)
    span_number: int = Field(gt=0, strict=True)
    source_id: str
    chunk_id: str
    text_hash: str
    witness_id: str
    start_char: int = Field(ge=0, strict=True)
    end_char: int = Field(gt=0, strict=True)
    quotation: str = Field(min_length=1)


_EVIDENCE_RECORDS = TypeAdapter(list[dict[str, JsonValue]])
_UNSUPPORTED_FLAGS = ("external", "derived", "untrusted", "truncated")


def validate_canonical_original(item: EvidenceItem | None) -> None:
    """Verify recorded provenance, not semantic support or legal validity."""
    if (
        item is None
        or item.search_doc is None
        or not item.source_id
        or not item.chunk_id
    ):
        raise ValueError("Support has no canonical original citation")
    if (
        not item.text.strip()
        or hashlib.sha256(item.text.encode("utf-8")).hexdigest() != item.text_hash
        or item.search_doc.document_id != item.source_id
        or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
    ):
        raise ValueError("Support is not an exact authorized original passage")
    for metadata in (item.metadata, item.search_doc.metadata):
        canonical = metadata.get("canonical_metadata")
        layers = (metadata, canonical if isinstance(canonical, dict) else {})
        if any(
            layer.get(flag) is True for layer in layers for flag in _UNSUPPORTED_FLAGS
        ):
            raise ValueError("Support is not an exact authorized original passage")


def canonical_evidence_view(ledger: EvidenceLedger) -> list[dict[str, JsonValue]]:
    """Retain every original character once, adjacent to its numbered selector."""
    numbers = ledger.citation_numbers()
    records = _EVIDENCE_RECORDS.validate_json(
        ledger.serialize_records(numbers, required=numbers, max_chars=None)
    )
    for record in records:
        citation = record.get("citation")
        if type(citation) is not int:
            raise ValueError("Canonical original has no citation number")
        item = ledger.get(citation)
        validate_canonical_original(item)
        assert item is not None
        if any(
            record.get(key) != expected
            for key, expected in (
                ("text", item.text),
                ("text_hash", item.text_hash),
                ("source_id", item.source_id),
                ("chunk_id", item.chunk_id),
            )
        ):
            raise ValueError("Canonical original changed during passage projection")
        passages: list[JsonValue] = [
            {
                "span_number": number,
                "text": item.text[span["start_char"] : span["end_char"]],
            }
            for number, span in enumerate(
                original_witness_spans(citation, item.text), start=1
            )
        ]
        record.pop("text")
        record["passages"] = passages
    return records


def resolve_passage(
    reference: PassageReference, ledger: EvidenceLedger
) -> CanonicalPassage:
    """Materialize an original selector; JEV still judges relevance and entailment."""
    reference = PassageReference.model_validate(
        reference.model_dump(mode="python"), strict=True
    )
    item = ledger.get(reference.citation)
    validate_canonical_original(item)
    assert item is not None and item.chunk_id is not None
    spans = original_witness_spans(reference.citation, item.text)
    if reference.span_number > len(spans):
        raise ValueError("Support selects an unknown canonical passage")
    span = spans[reference.span_number - 1]
    quotation = item.text[span["start_char"] : span["end_char"]]
    if not quotation.strip():
        raise ValueError("Support selects an empty canonical passage")
    return CanonicalPassage(
        citation=reference.citation,
        span_number=reference.span_number,
        source_id=item.source_id,
        chunk_id=item.chunk_id,
        text_hash=item.text_hash,
        witness_id=span["witness_id"],
        start_char=span["start_char"],
        end_char=span["end_char"],
        quotation=quotation,
    )
