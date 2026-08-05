from datetime import date, datetime, timezone

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned,
)
from onyx.document_index.elasticsearch.schema import DocumentChunkWithoutVectors


def _make_elasticsearch_chunk(
    *,
    regulatory_chunk_id: str | None = None,
    heading_path: list[str] | None = None,
    validity_start_date: datetime | None = None,
    validity_end_date: datetime | None = None,
) -> DocumentChunkWithoutVectors:
    return DocumentChunkWithoutVectors(
        document_id="document-1",
        chunk_index=3,
        content="Madde metni",
        source_type=DocumentSource.FILE.value,
        public=True,
        access_control_list=[],
        global_boost=0,
        semantic_identifier="Mevzuat — BÖLÜM I > MADDE 3",
        blurb="Madde metni",
        doc_summary="",
        chunk_context="",
        regulatory_chunk_id=regulatory_chunk_id,
        heading_path=heading_path,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
    )


def test_regulatory_fields_survive_retrieval_and_search_doc_conversion() -> None:
    source = _make_elasticsearch_chunk(
        regulatory_chunk_id="regulatory-row-3",
        heading_path=["BÖLÜM I", "MADDE 3"],
        validity_start_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        validity_end_date=datetime(2025, 6, 7, tzinfo=timezone.utc),
    )

    uncleaned = convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
        source, score=0.91, highlights={}
    )
    chunk = uncleaned.to_inference_chunk()

    assert chunk.regulatory_chunk_id == "regulatory-row-3"
    assert chunk.heading_path == ["BÖLÜM I", "MADDE 3"]
    assert chunk.validity_start_date == date(2024, 1, 2)
    assert chunk.validity_end_date == date(2025, 6, 7)

    chunk.metadata["department"] = "customs"
    search_doc = SearchDoc.from_chunks_or_sections([chunk])[0]
    assert search_doc.metadata == {
        "department": "customs",
        "regulatory_chunk_id": "regulatory-row-3",
        "regulatory_heading_path": ["BÖLÜM I", "MADDE 3"],
        "regulatory_validity_start_date": "2024-01-02",
        "regulatory_validity_end_date": "2025-06-07",
    }
    assert chunk.metadata == {"department": "customs"}
    assert SearchDoc.model_validate(search_doc.model_dump()).metadata == (
        search_doc.metadata
    )


def test_generic_chunk_remains_compatible_without_regulatory_fields() -> None:
    uncleaned = convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
        _make_elasticsearch_chunk(), score=None, highlights={}
    )
    chunk = uncleaned.to_inference_chunk()

    assert chunk.regulatory_chunk_id is None
    assert chunk.heading_path is None
    assert chunk.validity_start_date is None
    assert chunk.validity_end_date is None
    assert SearchDoc.from_chunks_or_sections([chunk])[0].metadata == {}
