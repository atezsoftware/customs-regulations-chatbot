from unittest.mock import MagicMock, patch

import pytest

from onyx.access.models import DocumentAccess
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationError,
    IndexingMetadata,
    TenantState,
)
from onyx.indexing.models import ChunkEmbedding, DocMetadataAwareIndexChunk


def _make_chunk(
    doc_id: str,
    chunk_id: int,
    *,
    content: str = "test content",
    heading_path: list[str] | None = None,
    metadata: dict[str, str | list[str]] | None = None,
) -> DocMetadataAwareIndexChunk:
    """Creates a minimal DocMetadataAwareIndexChunk for testing."""
    doc = Document(
        id=doc_id,
        sections=[TextSection(text="test", link="http://test.com")],
        source=DocumentSource.FILE,
        semantic_identifier="test_doc",
        metadata=metadata or {},
    )
    access = DocumentAccess.build(
        user_emails=[],
        user_groups=[],
        external_user_emails=[],
        external_user_group_ids=[],
        is_public=True,
    )
    return DocMetadataAwareIndexChunk(
        chunk_id=chunk_id,
        blurb="test",
        content=content,
        source_links={0: "http://test.com"},
        image_file_id=None,
        section_continuation=False,
        source_document=doc,
        title_prefix="",
        metadata_suffix_semantic="",
        metadata_suffix_keyword="",
        mini_chunk_texts=None,
        large_chunk_id=None,
        doc_summary="",
        chunk_context="",
        contextual_rag_reserved_tokens=0,
        embeddings=ChunkEmbedding(full_embedding=[0.1] * 10, mini_chunk_embeddings=[]),
        title_embedding=[0.1] * 10,
        tenant_id="test_tenant",
        access=access,
        document_sets=set(),
        user_project=[],
        personas=[],
        boost=0,
        aggregated_chunk_boost_factor=1.0,
        ancestor_hierarchy_node_ids=[],
        heading_path=heading_path,
    )


def _make_index() -> tuple[ElasticsearchDocumentIndex, MagicMock]:
    """Creates an ElasticsearchDocumentIndex with a mocked client.
    Returns the index and the mock for bulk_index_documents."""
    mock_client = MagicMock()
    mock_bulk = MagicMock()
    mock_client.bulk_index_documents = mock_bulk

    tenant_state = TenantState(tenant_id="test_tenant", multitenant=False)

    index = ElasticsearchDocumentIndex.__new__(ElasticsearchDocumentIndex)
    index._index_name = "test_index"
    index._client = mock_client
    index._tenant_state = tenant_state

    return index, mock_bulk


def _make_metadata(doc_id: str, chunk_count: int) -> IndexingMetadata:
    return IndexingMetadata(
        doc_id_to_chunk_cnt_diff={
            doc_id: IndexingMetadata.ChunkCounts(
                old_chunk_cnt=0,
                new_chunk_cnt=chunk_count,
            ),
        },
    )


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_single_doc_under_batch_limit_flushes_once() -> None:
    """A document with fewer chunks than MAX_CHUNKS_PER_DOC_BATCH should flush once."""
    index, mock_bulk = _make_index()
    doc_id = "doc_1"
    num_chunks = 50
    chunks = [_make_chunk(doc_id, i) for i in range(num_chunks)]
    metadata = _make_metadata(doc_id, num_chunks)

    with patch.object(index, "delete", return_value=0):
        index.index(chunks, metadata)

    assert mock_bulk.call_count == 1
    batch_arg = mock_bulk.call_args_list[0]
    assert len(batch_arg.kwargs["documents"]) == num_chunks


def test_bulk_payload_contains_legal_exact_fields_from_text_and_metadata() -> None:
    index, mock_bulk = _make_index()
    chunk = _make_chunk(
        "legal-doc",
        0,
        content="2024/17 sayılı karar 06.08.2026 tarihinde verildi.",
        heading_path=["Geçici Madde 2"],
        metadata={"karar_no": "2024/17"},
    )

    with patch.object(index, "delete", return_value=0):
        index.index([chunk], _make_metadata("legal-doc", 1))

    indexed_document = mock_bulk.call_args.kwargs["documents"][0]
    assert indexed_document.provision_identifiers == ["geçici madde 2"]
    assert indexed_document.decision_numbers == ["2024/17"]
    assert indexed_document.legal_dates == ["2026-08-06"]


def test_partial_upsert_preserves_sibling_chunks() -> None:
    index, mock_bulk = _make_index()
    chunks = [_make_chunk("legal-doc", 90), _make_chunk("legal-doc", 461)]

    with patch.object(index, "delete") as delete:
        index.upsert_chunks(chunks)

    delete.assert_not_called()
    assert len(mock_bulk.call_args.kwargs["documents"]) == 2
    assert mock_bulk.call_args.kwargs["update_if_exists"] is True


def test_partial_identity_preflight_rejects_shifted_chunk() -> None:
    index, _ = _make_index()
    chunk_id = get_elasticsearch_doc_chunk_id(
        tenant_state=index._tenant_state,  # noqa: SLF001
        document_id="legal-doc",
        chunk_index=90,
    )
    with (
        patch.object(
            index._client,  # noqa: SLF001
            "get_document_chunk_identities",
            return_value={chunk_id: (90, "wrong-row")},
        ),
        pytest.raises(
            DocumentChunkVerificationError,
            match="canonical regulatory chunk identity mismatch",
        ),
    ):
        index.verify_chunk_identities(
            document_id="legal-doc",
            expected_by_ordinal={90: "expected-row"},
            required_ordinals={90},
        )


def test_partial_identity_preflight_allows_missing_unpublished_amendment() -> None:
    index, _ = _make_index()
    imported_id = get_elasticsearch_doc_chunk_id(
        tenant_state=index._tenant_state,  # noqa: SLF001
        document_id="legal-doc",
        chunk_index=0,
    )

    with patch.object(
        index._client,  # noqa: SLF001
        "get_document_chunk_identities",
        return_value={imported_id: (0, "imported-row")},
    ):
        index.verify_chunk_identities(
            document_id="legal-doc",
            expected_by_ordinal={
                0: "imported-row",
                1_000_000_066: "amendment-row",
            },
            required_ordinals={0},
        )


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_single_doc_over_batch_limit_flushes_multiple_times() -> None:
    """A document with more chunks than MAX_CHUNKS_PER_DOC_BATCH should flush multiple times."""
    index, mock_bulk = _make_index()
    doc_id = "doc_1"
    num_chunks = 250
    chunks = [_make_chunk(doc_id, i) for i in range(num_chunks)]
    metadata = _make_metadata(doc_id, num_chunks)

    with patch.object(index, "delete", return_value=0):
        index.index(chunks, metadata)

    # 250 chunks / 100 per batch = 3 flushes (100 + 100 + 50)
    assert mock_bulk.call_count == 3
    batch_sizes = [len(call.kwargs["documents"]) for call in mock_bulk.call_args_list]
    assert batch_sizes == [100, 100, 50]


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_single_doc_exactly_at_batch_limit() -> None:
    """A document with exactly MAX_CHUNKS_PER_DOC_BATCH chunks should flush once
    (the flush happens on the next chunk, not at the boundary)."""
    index, mock_bulk = _make_index()
    doc_id = "doc_1"
    num_chunks = 100
    chunks = [_make_chunk(doc_id, i) for i in range(num_chunks)]
    metadata = _make_metadata(doc_id, num_chunks)

    with patch.object(index, "delete", return_value=0):
        index.index(chunks, metadata)

    # 100 chunks hit the >= check on chunk 101 which doesn't exist,
    # so final flush handles all 100
    # Actually: the elif fires when len(current_chunks) >= 100, which happens
    # when current_chunks has 100 items and the 101st chunk arrives.
    # With exactly 100 chunks, the 100th chunk makes len == 99, then appended -> 100.
    # No 101st chunk arrives, so the final flush handles all 100.
    assert mock_bulk.call_count == 1


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_single_doc_one_over_batch_limit() -> None:
    """101 chunks for one doc: first 100 flushed when the 101st arrives, then
    the 101st is flushed at the end."""
    index, mock_bulk = _make_index()
    doc_id = "doc_1"
    num_chunks = 101
    chunks = [_make_chunk(doc_id, i) for i in range(num_chunks)]
    metadata = _make_metadata(doc_id, num_chunks)

    with patch.object(index, "delete", return_value=0):
        index.index(chunks, metadata)

    assert mock_bulk.call_count == 2
    batch_sizes = [len(call.kwargs["documents"]) for call in mock_bulk.call_args_list]
    assert batch_sizes == [100, 1]


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_multiple_docs_each_under_limit_flush_per_doc() -> None:
    """Multiple documents each under the batch limit should flush once per document."""
    index, mock_bulk = _make_index()
    chunks = []
    for doc_idx in range(3):
        doc_id = f"doc_{doc_idx}"
        for chunk_idx in range(50):
            chunks.append(_make_chunk(doc_id, chunk_idx))

    metadata = IndexingMetadata(
        doc_id_to_chunk_cnt_diff={
            f"doc_{i}": IndexingMetadata.ChunkCounts(old_chunk_cnt=0, new_chunk_cnt=50)
            for i in range(3)
        },
    )

    with patch.object(index, "delete", return_value=0):
        index.index(chunks, metadata)

    # 3 documents = 3 flushes (one per doc boundary + final)
    assert mock_bulk.call_count == 3


@patch(
    "onyx.document_index.elasticsearch.elasticsearch_document_index.MAX_CHUNKS_PER_DOC_BATCH",
    100,
)
def test_delete_called_once_per_document() -> None:
    """Even with multiple flushes for a single document, delete should only be
    called once per document."""
    index, _mock_bulk = _make_index()
    doc_id = "doc_1"
    num_chunks = 250
    chunks = [_make_chunk(doc_id, i) for i in range(num_chunks)]
    metadata = _make_metadata(doc_id, num_chunks)

    with patch.object(index, "delete", return_value=0) as mock_delete:
        index.index(chunks, metadata)

    mock_delete.assert_called_once_with(doc_id, None)
