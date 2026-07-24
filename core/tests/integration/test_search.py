"""Tests for search filtering and merged retrieval ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import fs_explorer_indexer.indexing.pipeline as pipeline_module
import pytest

from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_indexer.indexing.pipeline import IndexingPipeline
from fs_explorer_api.search import (
    IndexedQueryEngine,
    MetadataFilterParseError,
    parse_metadata_filters,
)
from fs_explorer_shared.storage import PostgresStorage


def test_parse_metadata_filters_supports_scalar_and_list_values() -> None:
    parsed = parse_metadata_filters(
        "document_type=agreement and mentions_currency=true, file_size_bytes>=100, "
        "document_type in (agreement, report)"
    )

    assert len(parsed) == 4
    assert parsed[0].field == "document_type"
    assert parsed[0].operator == "eq"
    assert parsed[0].value == "agreement"
    assert parsed[1].field == "mentions_currency"
    assert parsed[1].value is True
    assert parsed[2].operator == "gte"
    assert parsed[2].value == 100
    assert parsed[3].operator == "in"
    assert parsed[3].value == ["agreement", "report"]


def test_parse_metadata_filters_rejects_unknown_schema_fields() -> None:
    with pytest.raises(MetadataFilterParseError):
        parse_metadata_filters(
            "owner=finance",
            allowed_fields={"document_type", "mentions_currency"},
        )


def test_indexed_query_engine_unions_semantic_and_metadata_results(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a_agreement.md").write_text("Purchase price is $45,000,000.")
    (corpus / "b_report.md").write_text(
        "Risk register and litigation exposure summary."
    )

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    result = IndexingPipeline(storage=storage).index_folder(
        str(corpus), discover_schema=True
    )
    engine = IndexedQueryEngine(storage)

    hits = engine.search(
        corpus_id=result.corpus_id,
        query="purchase price",
        filters="document_type=report",
        limit=5,
    )

    by_path = {hit.relative_path: hit for hit in hits}
    assert "a_agreement.md" in by_path
    assert "b_report.md" in by_path
    assert by_path["a_agreement.md"].semantic_score > 0
    assert by_path["b_report.md"].metadata_score > 0


class _SlowStorage:
    def search_chunks(
        self,
        *,
        corpus_id: str,
        query: str,
        limit: int = 5,
        as_of_date=None,
    ):  # noqa: ARG002
        time.sleep(0.3)
        return [
            {
                "doc_id": "doc_semantic",
                "chunk_id": "chunk_semantic",
                "relative_path": "a.md",
                "absolute_path": "/tmp/a.md",
                "position": 0,
                "text": "semantic hit",
                "score": 3,
                "chunk_type": "text",
                "metadata": {},
            }
        ]

    def search_documents_by_metadata(self, *, corpus_id: str, filters, limit: int = 20):  # noqa: ARG002
        time.sleep(0.3)
        return [
            {
                "doc_id": "doc_metadata",
                "relative_path": "b.md",
                "absolute_path": "/tmp/b.md",
                "preview_text": "metadata hit",
                "metadata_score": 1,
            }
        ]

    def get_active_schema(self, *, corpus_id: str):  # noqa: ARG002
        return None

    def list_document_chunks(self, *, doc_id: str):
        assert doc_id == "doc_metadata"
        return [
            {
                "id": "chunk_metadata",
                "doc_id": doc_id,
                "position": 7,
                "text": "metadata full chunk text",
                "chunk_type": "text",
                "metadata": {"article_no": "12"},
            }
        ]


def test_indexed_query_engine_executes_semantic_and_metadata_in_parallel() -> None:
    engine = IndexedQueryEngine(_SlowStorage())

    start = time.perf_counter()
    hits = engine.search(
        corpus_id="corpus_test",
        query="test",
        filters="document_type=agreement",
        limit=5,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.58
    assert {hit.doc_id for hit in hits} == {"doc_semantic", "doc_metadata"}


def test_search_enable_semantic_false_returns_only_metadata() -> None:
    """When enable_semantic=False, only metadata results are returned."""
    engine = IndexedQueryEngine(_SlowStorage())

    hits = engine.search(
        corpus_id="corpus_test",
        query="test",
        filters="document_type=agreement",
        limit=5,
        enable_semantic=False,
    )

    assert len(hits) == 1
    assert hits[0].doc_id == "doc_metadata"
    assert hits[0].chunk_id == "chunk_metadata"
    assert hits[0].text == "metadata full chunk text"


def test_search_enable_metadata_false_returns_only_semantic() -> None:
    """When enable_metadata=False, only semantic results are returned."""
    engine = IndexedQueryEngine(_SlowStorage())

    hits = engine.search(
        corpus_id="corpus_test",
        query="test",
        filters="document_type=agreement",
        limit=5,
        enable_metadata=False,
    )

    assert len(hits) == 1
    assert hits[0].doc_id == "doc_semantic"


def test_search_both_disabled_returns_empty() -> None:
    """When both enable_semantic and enable_metadata are False, no results."""
    engine = IndexedQueryEngine(_SlowStorage())

    hits = engine.search(
        corpus_id="corpus_test",
        query="test",
        filters="document_type=agreement",
        limit=5,
        enable_semantic=False,
        enable_metadata=False,
    )

    assert hits == []


# ---------------------------------------------------------------------------
# Mock embedding helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeEmbedding:
    values: list[float]


@dataclass
class _FakeEmbedResult:
    embeddings: list[_FakeEmbedding]


class _FakeEmbedModels:
    def embed_content(
        self, *, model: str, contents: list[str], config: dict
    ) -> _FakeEmbedResult:
        dim = config.get("output_dimensionality", 768)
        return _FakeEmbedResult(
            embeddings=[
                _FakeEmbedding(values=[0.1 * (i + 1)] * dim)
                for i in range(len(contents))
            ]
        )


class _FakeEmbedClient:
    def __init__(self) -> None:
        self.models = _FakeEmbedModels()


# ---------------------------------------------------------------------------
# Vector search tests
# ---------------------------------------------------------------------------


def test_vector_search_with_pre_stored_embeddings(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "agreement.md").write_text("Purchase price is $45,000,000.")
    (corpus / "report.md").write_text("Risk register and litigation exposure summary.")

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    provider = EmbeddingProvider(client=_FakeEmbedClient(), dim=768)
    pipeline = IndexingPipeline(storage=storage, embedding_provider=provider)

    result = pipeline.index_folder(str(corpus), discover_schema=True)
    assert result.embeddings_written > 0

    engine = IndexedQueryEngine(storage, embedding_provider=provider)
    hits = engine.search(
        corpus_id=result.corpus_id,
        query="purchase price",
        limit=5,
    )

    assert len(hits) > 0
    # All hits should have float semantic scores from cosine similarity
    for hit in hits:
        assert isinstance(hit.semantic_score, float)


def test_keyword_fallback_when_no_embeddings(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "agreement.md").write_text("Purchase price is $45,000,000.")

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    IndexingPipeline(storage=storage).index_folder(str(corpus), discover_schema=True)

    # Create engine with embedding provider but no embeddings stored
    provider = EmbeddingProvider(client=_FakeEmbedClient(), dim=768)
    engine = IndexedQueryEngine(storage, embedding_provider=provider)
    result_corpus_id = storage.get_corpus_id(str(Path(corpus).resolve()))
    assert result_corpus_id is not None

    hits = engine.search(
        corpus_id=result_corpus_id,
        query="purchase price",
        limit=5,
    )
    # Should still return results via keyword fallback
    assert len(hits) > 0


def test_get_metadata_field_values_returns_distinct_values(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a_agreement.md").write_text("Purchase price is $45,000,000.")
    (corpus / "b_report.md").write_text("Risk register summary.")
    (corpus / "c_agreement.md").write_text("Escrow details for the deal.")

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    result = IndexingPipeline(storage=storage).index_folder(
        str(corpus), discover_schema=True
    )

    values = storage.get_metadata_field_values(
        corpus_id=result.corpus_id,
        field_names=["document_type", "mentions_currency"],
    )
    assert "document_type" in values
    assert "agreement" in values["document_type"]
    assert "report" in values["document_type"]
    assert "mentions_currency" in values


def test_get_metadata_field_values_empty_corpus(
    tmp_path: Path, database_url: str
) -> None:
    storage = PostgresStorage(database_url)
    corpus_id = storage.get_or_create_corpus(str(tmp_path / "empty"))
    values = storage.get_metadata_field_values(
        corpus_id=corpus_id,
        field_names=["document_type"],
    )
    assert values == {"document_type": []}


def test_get_metadata_field_values_respects_max_distinct(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    for i in range(5):
        (corpus / f"doc_{i:02d}_type{i}.md").write_text(f"Content {i}")

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    result = IndexingPipeline(storage=storage).index_folder(
        str(corpus), discover_schema=True
    )

    values = storage.get_metadata_field_values(
        corpus_id=result.corpus_id,
        field_names=["document_type"],
        max_distinct=2,
    )
    assert len(values["document_type"]) <= 2


def test_semantic_search_includes_field_catalog_on_first_call(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
) -> None:
    import fs_explorer_api.agent as agent_module

    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a_agreement.md").write_text("Purchase price is $45,000,000.")

    monkeypatch.setattr(
        pipeline_module,
        "parse_file",
        lambda file_path: Path(file_path).read_text(),
    )

    storage = PostgresStorage(database_url)
    IndexingPipeline(storage=storage).index_folder(str(corpus), discover_schema=True)

    agent_module.set_index_context(str(corpus), database_url)
    agent_module.set_search_flags(enable_semantic=True, enable_metadata=True)
    try:
        first = agent_module.semantic_search("purchase price")
        assert "Available filter fields" in first
        assert "document_type" in first

        second = agent_module.semantic_search("purchase price")
        assert "Available filter fields" not in second
    finally:
        agent_module.clear_index_context()


# ---------------------------------------------------------------------------
# Cross-encoder reranker integration
# ---------------------------------------------------------------------------


class _FakeMultiHitStorage:
    def search_chunks(
        self, *, corpus_id: str, query: str, limit: int = 5, as_of_date=None
    ):  # noqa: ARG002
        return [
            {
                "doc_id": "doc_a",
                "chunk_id": "chunk_a",
                "relative_path": "a.md",
                "absolute_path": "/a.md",
                "position": 0,
                "text": "alpha content",
                "score": 1,
            },
            {
                "doc_id": "doc_b",
                "chunk_id": "chunk_b",
                "relative_path": "b.md",
                "absolute_path": "/b.md",
                "position": 0,
                "text": "beta content",
                "score": 2,
            },
            {
                "doc_id": "doc_c",
                "chunk_id": "chunk_c",
                "relative_path": "c.md",
                "absolute_path": "/c.md",
                "position": 0,
                "text": "gamma content",
                "score": 3,
            },
        ]


def test_indexed_query_engine_uses_reranker_ordering_when_available(
    monkeypatch,
) -> None:
    fake_reranker = Mock(rerank=Mock(return_value=[(0, 0.95), (2, 0.5), (1, 0.2)]))
    monkeypatch.setattr(
        "fs_explorer_api.search.query.get_reranker", lambda: fake_reranker
    )

    engine = IndexedQueryEngine(_FakeMultiHitStorage())
    hits = engine.search(corpus_id="c", query="q", limit=3, enable_metadata=False)

    # Reranker's order wins over the heuristic semantic_score ordering
    # (which would have put doc_c first, per its higher raw `score`).
    assert [hit.doc_id for hit in hits] == ["doc_a", "doc_c", "doc_b"]
    assert hits[0].score == 0.95
    fake_reranker.rerank.assert_called_once_with(
        "q",
        ["alpha content", "beta content", "gamma content"],
        top_n=3,
    )


class _MetadataChunkStorage:
    def get_active_schema(self, *, corpus_id: str):  # noqa: ARG002
        return None

    def search_documents_by_metadata(
        self, *, corpus_id: str, filters, limit: int = 20
    ):  # noqa: ARG002
        return [
            {
                "doc_id": "doc_metadata",
                "relative_path": "metadata.md",
                "absolute_path": "/metadata.md",
                "preview_text": "preview must not reach reranker",
                "metadata_score": 1,
            }
        ]

    def list_document_chunks(self, *, doc_id: str):
        return [
            {
                "id": "chunk_1",
                "doc_id": doc_id,
                "position": 0,
                "text": "first complete metadata chunk",
                "chunk_type": "text",
                "metadata": {},
            },
            {
                "id": "chunk_2",
                "doc_id": doc_id,
                "position": 1,
                "text": "second complete metadata chunk",
                "chunk_type": "text",
                "metadata": {},
            },
            {
                "id": "chunk_3",
                "doc_id": doc_id,
                "position": 2,
                "text": "third complete metadata chunk",
                "chunk_type": "text",
                "metadata": {},
            },
        ]


def test_metadata_only_search_hydrates_full_chunks_before_rerank(monkeypatch) -> None:
    fake_reranker = Mock(
        rerank=Mock(return_value=[(1, 0.9), (2, 0.6), (0, 0.4)])
    )
    monkeypatch.setattr(
        "fs_explorer_api.search.query.get_reranker", lambda: fake_reranker
    )

    engine = IndexedQueryEngine(_MetadataChunkStorage())
    hits = engine.search(
        corpus_id="c",
        query="metadata query",
        filters="document_type=agreement",
        limit=2,
        enable_semantic=False,
    )

    fake_reranker.rerank.assert_called_once_with(
        "metadata query",
        [
            "first complete metadata chunk",
            "second complete metadata chunk",
            "third complete metadata chunk",
        ],
        top_n=3,
    )
    assert [hit.chunk_id for hit in hits] == ["chunk_2", "chunk_3"]
    assert all("complete metadata chunk" in hit.text for hit in hits)


def test_indexed_query_engine_falls_back_when_reranker_returns_none(
    monkeypatch,
) -> None:
    fake_reranker = Mock(rerank=Mock(return_value=None))
    monkeypatch.setattr(
        "fs_explorer_api.search.query.get_reranker", lambda: fake_reranker
    )

    engine = IndexedQueryEngine(_FakeMultiHitStorage())
    hits = engine.search(corpus_id="c", query="q", limit=3, enable_metadata=False)

    # Falls back to the linear heuristic: highest raw semantic score first.
    assert [hit.doc_id for hit in hits] == ["doc_c", "doc_b", "doc_a"]


def test_rank_candidates_can_skip_diversification_for_adaptive_facets(
    monkeypatch,
) -> None:
    from fs_explorer_api.search.ranker import RankedDocument

    documents = [
        RankedDocument(
            doc_id="same_doc",
            relative_path="rule.md",
            absolute_path="/rule.md",
            position=index,
            text=f"facet {index}",
            semantic_score=0.8,
            metadata_score=0,
            chunk_id=f"chunk_{index}",
        )
        for index in range(3)
    ]
    fake_reranker = Mock(
        rerank=Mock(return_value=[(0, 0.9), (1, 0.8), (2, 0.7)])
    )
    monkeypatch.setattr(
        "fs_explorer_api.search.query.get_reranker", lambda: fake_reranker
    )

    diversified = IndexedQueryEngine.rank_candidates(
        query="q", documents=documents, limit=3
    )
    full_scored = IndexedQueryEngine.rank_candidates(
        query="q", documents=documents, limit=3, diversify=False
    )

    assert len(diversified) == 2
    assert [document.chunk_id for document, _score in full_scored] == [
        "chunk_0",
        "chunk_1",
        "chunk_2",
    ]


def test_float_scoring_in_ranked_documents() -> None:
    from fs_explorer_api.search.ranker import RankedDocument, rank_documents

    docs = [
        RankedDocument(
            doc_id="d1",
            relative_path="a.md",
            absolute_path="/a.md",
            position=0,
            text="doc 1",
            semantic_score=0.95,
            metadata_score=1,
        ),
        RankedDocument(
            doc_id="d2",
            relative_path="b.md",
            absolute_path="/b.md",
            position=0,
            text="doc 2",
            semantic_score=0.5,
            metadata_score=2,
        ),
    ]
    ranked = rank_documents(docs, limit=2)
    assert ranked[0].doc_id == "d1"
    assert ranked[0].combined_score > ranked[1].combined_score
