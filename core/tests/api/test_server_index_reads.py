from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from fs_explorer_api.server import app

DUMMY_DATABASE_URL = "postgresql://user:pass@localhost:5432/app"


class FakeStorage:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def get_corpus_id(self, corpus_root: str):
        if corpus_root.endswith("/missing"):
            return None
        return "corpus-1"

    def list_documents(self, *, corpus_id: str, include_deleted: bool = False):
        return [
            {
                "relative_path": "1-regulation.pdf",
                "file_size": 100,
                "file_mtime": 123,
            }
        ]

    def get_active_schema(self, *, corpus_id: str):
        return SimpleNamespace(
            name="regulatory",
            schema_def={
                "metadata_profile": {"name": "regulatory"},
                "fields": [{"name": "article_no"}, {"name": "document_type"}],
            },
        )

    def has_embeddings(self, *, corpus_id: str):
        return True

    def get_document_chunks_by_prefix(self, *, corpus_root: str, relative_path_prefix: str):
        if relative_path_prefix == "missing":
            return None
        return {
            "document": {
                "id": "doc-1",
                "relative_path": "1-regulation.pdf",
                "absolute_path": "/tmp/regulation.pdf",
            },
            "chunks": [
                {
                    "id": "chunk-1",
                    "document_id": "doc-1",
                    "relative_path": "1-regulation.pdf",
                    "absolute_path": "/tmp/regulation.pdf",
                    "text": "MADDE 1",
                    "position": 0,
                    "start_char": 0,
                    "end_char": 7,
                    "chunk_type": "article",
                    "metadata": {"article_no": "1"},
                    "has_embedding": True,
                }
            ],
        }

    def close(self):
        self.closed = True


def test_core_api_index_status_reads_existing_corpus() -> None:
    client = TestClient(app)
    with patch("fs_explorer_api.server.PostgresStorage", FakeStorage):
        response = client.get(
            "/api/index/status",
            params={"folder": "/indexed", "database_url": DUMMY_DATABASE_URL},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is True
    assert data["document_count"] == 1
    assert data["has_embeddings"] is True
    assert data["schema_fields"] == ["article_no", "document_type"]


def test_core_api_index_status_returns_not_indexed() -> None:
    client = TestClient(app)
    with patch("fs_explorer_api.server.PostgresStorage", FakeStorage):
        response = client.get(
            "/api/index/status",
            params={"folder": "/missing", "database_url": DUMMY_DATABASE_URL},
        )

    assert response.status_code == 200
    assert response.json() == {"indexed": False}


def test_core_api_document_chunks_reads_by_prefix() -> None:
    client = TestClient(app)
    with patch("fs_explorer_api.server.PostgresStorage", FakeStorage):
        response = client.get(
            "/api/index/document-chunks",
            params={
                "corpus_key": "/indexed",
                "relative_path_prefix": "1-",
                "database_url": DUMMY_DATABASE_URL,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["id"] == "doc-1"
    assert data["chunks"][0]["id"] == "chunk-1"


def test_core_api_document_chunks_returns_empty_result() -> None:
    client = TestClient(app)
    with patch("fs_explorer_api.server.PostgresStorage", FakeStorage):
        response = client.get(
            "/api/index/document-chunks",
            params={
                "corpus_key": "/indexed",
                "relative_path_prefix": "missing",
                "database_url": DUMMY_DATABASE_URL,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"document": None, "chunks": []}
