"""Paragraph locators retain raw formatting and require actual source context."""

import pytest

from onyx.regulatory.document_intake import metadata


def _chunk(identifier: str, text: str, *headings: str) -> metadata.MetadataChunk:
    return metadata.MetadataChunk(
        chunk_id=identifier,
        text=text,
        heading_path=["MADDE 12 - Usul", *headings],
        chunk_type="paragraph",
    )


def _validate(
    source: metadata.MetadataChunk,
    target: metadata.MetadataChunk,
    quote: str,
    value: str,
) -> metadata.LegalMetadataResult:
    response = metadata.LegalMetadataResult(
        chunks=[
            metadata.ChunkLegalMetadata(
                chunk_id=target.chunk_id,
                fields=[
                    metadata.LegalMetadataFact(
                        field="paragraph_no",
                        value=value,
                        source_chunk_id=source.chunk_id,
                        evidence_quote=quote,
                    )
                ],
            )
        ]
    )
    return metadata.validate_metadata_response(
        response.model_dump_json(),
        chunks=[source, target] if source != target else [source],
        target_ids=[target.chunk_id],
    )


@pytest.mark.parametrize(
    "value,source_number,quote_has_marker",
    [
        ("1", "4", True),
        ("4", "1", False),
        ("4", "1", False),
        ("5", "2", True),
        ("5", "2", True),
        ("6", "1", True),
        ("2", "1", False),
    ],
)
def test_observed_seven_wrong_paragraph_sources_are_rejected(
    value: str, source_number: str, quote_has_marker: bool
) -> None:
    source = _chunk("source", f"({source_number}) Kaynak hüküm.")
    target = _chunk("target", f"({value}) Hedef hüküm.")
    quote = source.text if quote_has_marker else "Kaynak hüküm."
    with pytest.raises(ValueError, match="paragraph locator contradicts"):
        _validate(source, target, quote, value)


@pytest.mark.parametrize("value", ["3", "(3)"])
@pytest.mark.parametrize(
    "text,headings,quote",
    [
        ("(3) Kaynak hüküm.", (), "(3) Kaynak hüküm."),
        ("MADDE 12 – (3) Kaynak hüküm.", (), "(3) Kaynak hüküm."),
        ("MADDE12 – Usul\n(3) Kaynak hüküm.", (), "Kaynak hüküm."),
        ("Devam hükmü uygulanır.", ("(3) Kaynak hüküm.",), "Devam hükmü"),
        ("(3) Kaynak hüküm.", ("(1) Eski başlık.",), "Kaynak hüküm."),
        ("3) Kaynak hüküm.", (), "3) Kaynak hüküm."),
        ("1) Alt hüküm.", ("(3) Üst fıkra.",), "1) Alt hüküm."),
        ("b) Alt hüküm.", ("3) Üst hüküm.",), "3) Üst hüküm."),
    ],
)
def test_supported_paragraph_context_preserves_exact_raw_value(
    value: str, text: str, headings: tuple[str, ...], quote: str
) -> None:
    source = _chunk("source", text, *headings)
    result = _validate(source, source, quote, value)
    assert result.chunks[0].fields[0].value == value
    assert result.chunks[0].fields[0].evidence_quote == quote


def test_selected_quote_and_target_context_cannot_override_each_other() -> None:
    source = _chunk("source", "(4) Kaynak hüküm.")
    target = _chunk("target", "(1) Hedef hüküm.")
    with pytest.raises(ValueError, match="paragraph locator contradicts"):
        _validate(source, target, source.text, "4")


@pytest.mark.parametrize(
    "text,quote,error",
    [
        (
            "(1) İlk hüküm.\n(2) İkinci hüküm.",
            "(1) İlk hüküm.\n(2) İkinci hüküm.",
            "ambiguous locators",
        ),
        (
            "Önceki (1) numaralı fıkra uygulanır.",
            "(1) numaralı fıkra",
            "requires explicit source context",
        ),
        ("Kaynak hüküm.", "Kaynak hüküm.", "requires explicit source context"),
    ],
)
def test_ambiguous_or_incidental_references_do_not_prove_paragraph_context(
    text: str, quote: str, error: str
) -> None:
    source = _chunk("source", text)
    with pytest.raises(ValueError, match=error):
        _validate(source, source, quote, "1")


def test_exact_selected_paragraph_remains_valid_in_a_multi_paragraph_source() -> None:
    source = _chunk("source", "(1) İlk hüküm.\n(2) İkinci hüküm.")
    target = _chunk("target", "(2) Hedef hüküm.")
    assert _validate(source, target, "(2) İkinci hüküm.", "2")


def test_nested_numbered_parent_is_not_a_conflicting_paragraph_identity() -> None:
    source = _chunk("source", "(1) Teklif bildirim şartlarına göre uygulanır.")
    target = _chunk("target", "b) Eşleşme gerçekleştirilir.", "3) Alış teklif miktarı.")
    assert _validate(source, target, source.text, "1")
