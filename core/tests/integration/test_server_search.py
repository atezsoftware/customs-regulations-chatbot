"""Tests for the api service's /api/search endpoint, seeded via the indexer's pipeline."""

from __future__ import annotations

from pathlib import Path

import fs_explorer_indexer.indexing.pipeline as pipeline_module
import pytest
from fastapi.testclient import TestClient

from fs_explorer_indexer.indexing.pipeline import IndexingPipeline
from fs_explorer_api.server import app
from fs_explorer_shared.storage import PostgresStorage


@pytest.fixture()
def indexed_corpus(tmp_path: Path, monkeypatch, database_url: str):
    """Create a small indexed corpus and return (folder, database_url)."""
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
    IndexingPipeline(storage=storage).index_folder(str(corpus), discover_schema=True)
    storage.close()
    return str(corpus), database_url


def test_search_endpoint_returns_hits(indexed_corpus) -> None:
    corpus_folder, database_url = indexed_corpus
    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "corpus_folder": corpus_folder,
            "query": "purchase price",
            "database_url": database_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert len(data["hits"]) > 0
    assert data["hits"][0]["semantic_score"] > 0


def test_search_endpoint_with_filters(indexed_corpus) -> None:
    corpus_folder, database_url = indexed_corpus
    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "corpus_folder": corpus_folder,
            "query": "litigation",
            "filters": "document_type=report",
            "database_url": database_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "hits" in data


def test_search_endpoint_missing_index(tmp_path: Path, database_url: str) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()

    client = TestClient(app)
    response = client.post(
        "/api/search",
        json={
            "corpus_folder": str(corpus),
            "query": "test",
            "database_url": database_url,
        },
    )

    assert response.status_code in (404, 500)


def test_search_endpoint_invalid_folder() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/search",
        json={
            "corpus_folder": "/nonexistent/path/abc123",
            "query": "test",
        },
    )

    assert response.status_code == 400
