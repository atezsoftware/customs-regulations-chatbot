"""Contextual retrieval text must never become citeable document evidence."""

import json

import pytest

from onyx.configs.constants import RETURN_SEPARATOR, DocumentSource
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.document_index.chunk_content_enrichment import cleanup_content_for_chunks
from onyx.document_index.opensearch.opensearch_document_index import (
    convert_retrieved_opensearch_chunk_to_inference_chunk_uncleaned,
)
from onyx.document_index.opensearch.schema import DocumentChunkWithoutVectors
from onyx.tools.tool_implementations.utils import (
    convert_inference_sections_to_llm_string,
)
from onyx.utils.text_processing import remove_invalid_unicode_chars


def _stored_contextual_chunk(*, invalid_unicode: bool) -> DocumentChunkWithoutVectors:
    summary_text = "Generated document summary"
    context_text = "Generated chunk context"
    if invalid_unicode:
        summary_text = "Generated\x00 document summary"
        context_text = "Generated\x0b chunk context"

    doc_summary = f"{summary_text}{RETURN_SEPARATOR}"
    chunk_context = f"{RETURN_SEPARATOR}{context_text}"
    legal_text = "MADDE 7 — Yalnızca kaynak mevzuat metni."

    # OpenSearch stores a sanitized concatenation in `content`, while the
    # separately stored inverse fields may retain the original characters.
    stored_content = remove_invalid_unicode_chars(
        f"{doc_summary}{legal_text}{chunk_context}"
    )
    return DocumentChunkWithoutVectors(
        document_id="regulation-1",
        chunk_index=7,
        content=stored_content,
        source_type=DocumentSource.FILE.value,
        public=True,
        access_control_list=[],
        global_boost=0,
        semantic_identifier="Kanun — MADDE 7",
        blurb=legal_text,
        doc_summary=doc_summary,
        chunk_context=chunk_context,
        regulatory_chunk_id="regulatory-row-7",
        heading_path=["MADDE 7"],
    )


@pytest.mark.parametrize("invalid_unicode", [False, True])
def test_contextual_fields_are_not_exposed_in_llm_result_content(
    invalid_unicode: bool,
) -> None:
    uncleaned = convert_retrieved_opensearch_chunk_to_inference_chunk_uncleaned(
        _stored_contextual_chunk(invalid_unicode=invalid_unicode),
        score=0.9,
        highlights={},
    )
    cleaned = cleanup_content_for_chunks([uncleaned])[0]

    llm_json, citation_mapping, citation_chunk_mapping = (
        convert_inference_sections_to_llm_string(
            [inference_section_from_single_chunk(cleaned)]
        )
    )
    result = json.loads(llm_json)["results"][0]

    assert result["content"] == "MADDE 7 — Yalnızca kaynak mevzuat metni."
    assert "Generated" not in result["content"]
    assert "doc_summary" not in result
    assert "chunk_context" not in result
    assert citation_mapping == {1: "regulation-1"}
    assert citation_chunk_mapping == {1: 7}
