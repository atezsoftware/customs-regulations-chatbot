"""Structural hints become facts only with exact, target-valid source evidence."""

from onyx.regulatory.document_intake.metadata import MetadataChunk


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
