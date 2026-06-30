"""Tests for the indexer service's /api/index/status and /api/index/auto-profile endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fs_explorer_indexer.indexing.pipeline as pipeline_module
import pytest
from fastapi.testclient import TestClient

from fs_explorer_indexer.indexing.pipeline import IndexingPipeline
from fs_explorer_indexer.indexer_server import app
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


# ---------------------------------------------------------------------------
# /api/index/status tests
# ---------------------------------------------------------------------------


def test_index_status_not_indexed(tmp_path: Path, database_url: str) -> None:
    corpus = tmp_path / "empty_folder"
    corpus.mkdir()

    client = TestClient(app)
    response = client.get(
        "/api/index/status",
        params={"folder": str(corpus), "database_url": database_url},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is False


def test_index_status_after_indexing(indexed_corpus) -> None:
    corpus_folder, database_url = indexed_corpus
    client = TestClient(app)

    response = client.get(
        "/api/index/status",
        params={"folder": corpus_folder, "database_url": database_url},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is True
    assert data["document_count"] == 2
    assert data["schema_name"] is not None
    assert isinstance(data["has_metadata"], bool)
    assert isinstance(data["has_embeddings"], bool)


def test_index_status_includes_schema_fields(indexed_corpus) -> None:
    corpus_folder, database_url = indexed_corpus
    client = TestClient(app)

    response = client.get(
        "/api/index/status",
        params={"folder": corpus_folder, "database_url": database_url},
    )

    assert response.status_code == 200
    data = response.json()
    assert "schema_fields" in data
    assert isinstance(data["schema_fields"], list)
    assert len(data["schema_fields"]) > 0
    assert "document_type" in data["schema_fields"]


# ---------------------------------------------------------------------------
# /api/index/auto-profile tests
# ---------------------------------------------------------------------------


def test_auto_profile_endpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "contract.md").write_text("TechCorp acquires StartupXYZ for $10M.")

    fake_profile = {
        "name": "test_auto",
        "description": "Auto-generated.",
        "prompt_description": "Extract metadata.",
        "fields": [
            {
                "name": "lx_organizations",
                "type": "string",
                "description": "Org names.",
                "source": "entities",
                "source_classes": ["organization"],
                "mode": "values",
            }
        ],
    }

    client = TestClient(app)
    with patch(
        "fs_explorer_indexer.indexer_server.auto_discover_profile",
        return_value=fake_profile,
    ):
        response = client.post(
            "/api/index/auto-profile",
            json={"folder": str(corpus)},
        )

    assert response.status_code == 200
    data = response.json()
    assert "profile" in data
    assert data["profile"]["name"] == "test_auto"
    field_names = {f["name"] for f in data["profile"]["fields"]}
    assert "lx_organizations" in field_names


def test_auto_profile_invalid_folder() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/index/auto-profile",
        json={"folder": "/nonexistent/path/abc123"},
    )

    assert response.status_code == 400
