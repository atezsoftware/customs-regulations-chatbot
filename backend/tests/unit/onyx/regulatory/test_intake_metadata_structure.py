"""Structural hints become facts only with exact, target-valid source evidence."""

from onyx.regulatory.document_intake.metadata import MetadataChunk


def test_saved_structural_receipt_requires_positive_source_marker() -> None:
    from types import SimpleNamespace
    from typing import cast

    import pytest

    from onyx.db.document_intake_structure_metadata import (
        validate_structural_verification,
    )
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.document_intake import metadata
    from onyx.regulatory.document_intake.metadata_checkpoint import metadata_digest

    chunks = [
        MetadataChunk(
            chunk_id="a",
            text="Belge ibraz edilir.",
            heading_path=[],
            chunk_type="paragraph",
        )
    ]
    source_hash = metadata.metadata_input_hash(chunks)
    fact = metadata.LegalMetadataFact(
        field="article_no",
        value="999",
        source_chunk_id="a",
        evidence_quote=chunks[0].text,
    )
    witness = {
        "target_chunk_id": "a",
        **fact.model_dump(mode="json"),
        "input_sha256": source_hash,
    }
    row = SimpleNamespace(
        id="a",
        chunk_metadata={
            "article_no": "999",
            metadata.METADATA_AUDIT_KEY: {
                "fields": {"article_no": {"source": "structural"}},
                "structural_verification": {
                    "version": 1,
                    "input_sha256": source_hash,
                    "fields": {
                        "article_no": {
                            "witness": witness,
                            "sha256": metadata_digest(witness),
                        }
                    },
                },
            },
        },
    )
    with pytest.raises(ValueError, match="source marker"):
        validate_structural_verification([cast(RegulatoryChunk, row)], chunks)


def test_unmarked_structural_article_guess_is_not_a_verified_fact() -> None:
    from onyx.regulatory.document_intake.metadata_structure import (
        verified_structural_metadata,
    )

    chunks = [
        MetadataChunk(
            chunk_id="unmarked",
            text="Belge ibraz edilir.",
            heading_path=["Amaç"],
            chunk_type="paragraph",
            structural_metadata={"article_no": "999"},
        )
    ]
    assert "article_no" not in verified_structural_metadata(chunks)["unmarked"]


def test_proves_distinct_article_and_paragraph_locators_without_accepting_bad_clause() -> (
    None
):
    from onyx.regulatory.document_intake.metadata_structure import (
        verified_structural_metadata,
    )

    chunks = [
        MetadataChunk(
            chunk_id="a",
            text="MADDE 1-\n(1) İlk hüküm.",
            heading_path=["MADDE 1"],
            chunk_type="paragraph",
            structural_metadata={
                "article_no": "1",
                "paragraph_no": "1",
                "clause_label": "a",
            },
        ),
        MetadataChunk(
            chunk_id="b",
            text="MADDE 2-\n(2) İkinci hüküm.",
            heading_path=["MADDE 2"],
            chunk_type="paragraph",
            structural_metadata={"article_no": "2", "paragraph_no": "2"},
        ),
    ]
    facts = verified_structural_metadata(chunks)
    assert facts["a"]["article_no"].value == "1"
    assert facts["a"]["paragraph_no"].value == "1"
    assert "clause_label" not in facts["a"]
    assert facts["b"]["article_no"].value == "2"
    assert facts["b"]["paragraph_no"].value == "2"


def test_scoped_output_retains_locally_proved_locators_with_server_provenance() -> None:
    import json

    from onyx.regulatory.document_intake.metadata_scopes import (
        expand_scoped_metadata_response,
    )
    from onyx.regulatory.document_intake.metadata_structure import (
        VerifiedStructuralFact,
    )

    chunks = [
        MetadataChunk(
            chunk_id="a",
            text="MADDE 1-\n(1) İlk hüküm.",
            heading_path=["MADDE 1"],
            chunk_type="paragraph",
            structural_metadata={"article_no": "1", "paragraph_no": "1"},
        )
    ]
    result = expand_scoped_metadata_response(
        json.dumps(
            {
                "covered_targets": [0],
                "shared_fields": [],
                "assignments": [],
                "warnings": [],
            }
        ),
        chunks=chunks,
    )
    facts = {fact.field: fact for fact in result[0].chunks[0].fields}
    assert facts["article_no"].value == "1"
    assert facts["paragraph_no"].value == "1"
    assert all(isinstance(fact, VerifiedStructuralFact) for fact in facts.values())
