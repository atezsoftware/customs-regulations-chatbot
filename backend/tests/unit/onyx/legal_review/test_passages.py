import hashlib
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_review.passages import (
    PassageReference,
    canonical_evidence_view,
    resolve_passage,
)


def original(text: str, identity: str = "1") -> EvidenceItem:
    return EvidenceItem(
        source_id=f"source-{identity}",
        chunk_id=f"chunk-{identity}",
        text=text,
        metadata={"title": "Özgün düzenleme", "article_no": "2"},
        search_doc=SearchDoc(
            document_id=f"source-{identity}",
            chunk_ind=1,
            semantic_identifier="Özgün düzenleme",
            source_type=DocumentSource.FILE,
            blurb=text[:100],
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": f"chunk-{identity}"},
            match_highlights=[],
        ),
    )


def ledger_with(*items: EvidenceItem) -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(items, RunContext())
    return ledger


def test_complete_numbered_view_preserves_unicode_paragraphs_and_headers() -> None:
    text = "\ufeffİade şartı: ğ, İ, ı ve € — e\u0301.\r\n\r\n" + (
        "Belge ibraz edilir; ancak istisna saklıdır.\n" * 60
    )
    ledger = ledger_with(original(text), original("İkinci kaynak.\n", "2"))
    records = canonical_evidence_view(ledger)
    assert [record["citation"] for record in records] == [1, 2]
    record = records[0]
    assert "text" not in record
    assert record["citable"] is True and record["truncated"] is False
    assert record["metadata"] == {"title": "Özgün düzenleme", "article_no": "2"}
    assert record["source_id"] == "source-1" and record["chunk_id"] == "chunk-1"
    assert record["text_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    passages = record["passages"]
    assert isinstance(passages, list) and len(passages) > 1
    rebuilt = ""
    for number, passage in enumerate(passages, start=1):
        assert isinstance(passage, dict)
        assert passage["span_number"] == number
        selected = resolve_passage(
            PassageReference(citation=1, span_number=number), ledger
        )
        assert passage["text"] == selected.quotation
        assert selected.start_char == len(rebuilt)
        assert selected.end_char == selected.start_char + len(selected.quotation)
        assert selected.quotation == text[selected.start_char : selected.end_char]
        assert selected.text_hash == record["text_hash"]
        rebuilt += selected.quotation
    assert rebuilt.encode("utf-8") == text.encode("utf-8")


def test_view_keeps_complete_originals_beyond_default_serialization_limit() -> None:
    text = "Kural ve istisna.\n" * 12_000
    ledger = ledger_with(original(text))
    record = canonical_evidence_view(ledger)[0]
    passages = record["passages"]
    assert isinstance(passages, list)
    assert len(text) > 180_000
    assert (
        "".join(
            str(passage["text"]) for passage in passages if isinstance(passage, dict)
        )
        == text
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"citation": True, "span_number": 1},
        {"citation": "1", "span_number": 1},
        {"citation": 0, "span_number": 1},
        {"citation": 1, "span_number": False},
        {"citation": 1, "span_number": 1.0},
        {"citation": 1, "span_number": -1},
        {"citation": 1, "span_number": 1, "quotation": "Invented text"},
        {"citation": 1, "span_number": 1, "start_char": 0},
        {"citation": 1, "span_number": 1, "witness_id": "foreign-hash"},
    ],
)
def test_model_selector_cannot_supply_coordinates_quotes_or_coerced_ids(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PassageReference.model_validate(payload)


def test_unknown_citation_and_foreign_span_number_are_rejected() -> None:
    ledger = ledger_with(original("Kural.\n" * 400), original("Başka kural.", "2"))
    first = canonical_evidence_view(ledger)[0]["passages"]
    assert isinstance(first, list) and len(first) > 1
    with pytest.raises(ValueError, match="no canonical original citation"):
        resolve_passage(PassageReference(citation=3, span_number=1), ledger)
    with pytest.raises(ValueError, match="unknown canonical passage"):
        resolve_passage(PassageReference(citation=2, span_number=len(first)), ledger)


@pytest.mark.parametrize("mutation", ["text", "hash", "document", "chunk", "citable"])
def test_current_canonical_integrity_is_checked_before_selector_resolution(
    mutation: str,
) -> None:
    item = original("Başvuru şartı.")
    assert item.search_doc is not None
    if mutation == "text":
        item.text = "Değiştirilmiş kaynak."
    elif mutation == "hash":
        item.text_hash = "0" * 64
    elif mutation == "document":
        item.search_doc.document_id = "foreign-source"
    elif mutation == "chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = "foreign-chunk"
    else:
        item.search_doc = None
    ledger = Mock(spec=EvidenceLedger)
    ledger.get.return_value = item
    with pytest.raises(
        ValueError, match="canonical original|exact authorized original"
    ):
        resolve_passage(PassageReference(citation=1, span_number=1), ledger)


@pytest.mark.parametrize("flag", ["derived", "external", "untrusted", "truncated"])
@pytest.mark.parametrize(
    "layer", ["item", "canonical", "search_doc", "search_doc_canonical"]
)
def test_untrusted_original_flags_cannot_be_hidden_behind_a_valid_selector(
    flag: str,
    layer: str,
) -> None:
    item = original("Başvuru şartı.")
    assert item.search_doc is not None
    if layer == "item":
        item.metadata[flag] = True
    elif layer == "canonical":
        item.metadata["canonical_metadata"] = {flag: True}
    else:
        item.search_doc = item.search_doc.model_copy(
            update={
                "metadata": {
                    **item.search_doc.metadata,
                    **(
                        {flag: True}
                        if layer == "search_doc"
                        else {"canonical_metadata": {flag: True}}
                    ),
                }
            }
        )
    ledger = ledger_with(original("Başvuru şartı."))
    with patch.object(ledger, "get", return_value=item):
        with pytest.raises(ValueError, match="exact authorized original"):
            resolve_passage(PassageReference(citation=1, span_number=1), ledger)
        with pytest.raises(ValueError, match="exact authorized original"):
            canonical_evidence_view(ledger)


def test_materialized_reference_is_immutable_and_only_proves_source_binding() -> None:
    ledger = ledger_with(original("Başka bir rejimin cezası düzenlenir."))
    selected = resolve_passage(PassageReference(citation=1, span_number=1), ledger)
    assert selected.quotation == "Başka bir rejimin cezası düzenlenir."
    assert selected.source_id == "source-1" and selected.chunk_id == "chunk-1"
    assert selected.witness_id.startswith("w1-")
    assert "legal_status" not in type(selected).model_fields
    with pytest.raises(ValidationError, match="frozen"):
        selected.quotation = "Sorulan rejimde ceza uygulanmaz."


def test_mutating_a_view_does_not_change_ledger_text_or_future_resolution() -> None:
    ledger = ledger_with(original("Başvuru şartı.\r\n"))
    view = canonical_evidence_view(ledger)
    passages = view[0]["passages"]
    assert isinstance(passages, list) and isinstance(passages[0], dict)
    passages[0]["text"] = "Forged model view"
    resolved = resolve_passage(PassageReference(citation=1, span_number=1), ledger)
    assert resolved.quotation == "Başvuru şartı.\r\n"


def test_initial_discovery_has_an_empty_evidence_view() -> None:
    assert canonical_evidence_view(EvidenceLedger()) == []


def test_whitespace_is_preserved_but_cannot_be_selected_as_rule_support() -> None:
    text = "Başvuru şartı." + " " * 1600
    ledger = ledger_with(original(text))
    passages = canonical_evidence_view(ledger)[0]["passages"]
    assert isinstance(passages, list)
    assert (
        "".join(
            str(passage["text"]) for passage in passages if isinstance(passage, dict)
        )
        == text
    )
    with pytest.raises(ValueError, match="empty canonical passage"):
        resolve_passage(PassageReference(citation=1, span_number=2), ledger)
