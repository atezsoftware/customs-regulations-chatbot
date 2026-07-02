"""
FastAPI server for the FsExplorer indexer.

Owns the index lifecycle (build/refresh, embed, schema auto-profile, status,
chunk lookups) backed by Docling parsing and the regulatory chunking
pipeline. The chat-facing `fs-explorer-api` service calls this over HTTP and
never imports Docling/langextract itself.
"""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fs_explorer_shared.auth import require_internal_token
from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.fs import SUPPORTED_EXTENSIONS
from fs_explorer_shared.index_config import corpus_root as resolve_corpus_root
from fs_explorer_shared.index_config import resolve_database_url
from fs_explorer_shared.storage import PostgresStorage
from .indexing import IndexingPipeline
from .indexing.pipeline import SourceDocument
from .indexing.metadata import auto_discover_profile

app = FastAPI(title="FsExplorer Indexer", description="Document indexing service")

_corpus_locks: dict[str, asyncio.Lock] = {}


def _get_corpus_lock(folder: str) -> asyncio.Lock:
    """Return a per-folder asyncio lock, creating one if needed."""
    normalized = str(Path(folder).resolve())
    if normalized not in _corpus_locks:
        _corpus_locks[normalized] = asyncio.Lock()
    return _corpus_locks[normalized]


class IndexDocument(BaseModel):
    """A backend-supplied source document for DB-backed indexing."""

    file_path: str
    relative_path: str
    display_name: str
    logical_path: str | None = None


class IndexRequest(BaseModel):
    """Request model for index build/refresh."""

    folder: str = "."
    corpus_key: str | None = None
    documents: list[IndexDocument] | None = None
    database_url: str | None = None
    discover_schema: bool = False
    schema_name: str | None = None
    with_metadata: bool = False
    metadata_profile: dict[str, Any] | None = None
    with_embeddings: bool = False


class EmbedIndexRequest(BaseModel):
    """Request model for embedding an already-chunked corpus."""

    folder: str = "."
    corpus_key: str | None = None
    database_url: str | None = None


class AutoProfileRequest(BaseModel):
    """Request model for auto-profile generation."""

    folder: str = "."


def _iter_supported_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.name.startswith("~$"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.resolve())
    return sorted(files)


def _index_freshness(folder_path: Path, docs: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {str(doc["relative_path"]): doc for doc in docs}
    current: dict[str, Path] = {}
    for path in _iter_supported_files(folder_path):
        current[str(path.relative_to(folder_path))] = path

    added = sorted(set(current) - set(indexed))
    removed = sorted(set(indexed) - set(current))
    changed: list[str] = []
    for relative_path in sorted(set(current) & set(indexed)):
        path = current[relative_path]
        stat = path.stat()
        doc = indexed[relative_path]
        if (
            int(stat.st_size) != int(doc["file_size"])
            or abs(float(stat.st_mtime) - float(doc["file_mtime"])) > 0.001
        ):
            changed.append(relative_path)

    return {
        "fresh": not added and not changed and not removed,
        "added_files": added,
        "changed_files": changed,
        "removed_files": removed,
        "current_file_count": len(current),
    }


@app.get("/api/index/status", dependencies=[Depends(require_internal_token)])
async def index_status(folder: str, database_url: str | None = None):
    """Check whether a folder has been indexed and return status details."""
    try:
        corpus_root = resolve_corpus_root(folder)
        folder_path = Path(corpus_root)

        try:
            resolved_database_url = resolve_database_url(database_url)
            storage = PostgresStorage(
                resolved_database_url, read_only=True, initialize=False
            )
        except Exception:
            return {"indexed": False}

        try:
            corpus_id = storage.get_corpus_id(corpus_root)
            if corpus_id is None:
                storage.close()
                return {"indexed": False}

            docs = storage.list_documents(corpus_id=corpus_id, include_deleted=False)
            freshness = (
                _index_freshness(folder_path, docs)
                if folder_path.exists() and folder_path.is_dir()
                else {
                    "fresh": True,
                    "added_files": [],
                    "changed_files": [],
                    "removed_files": [],
                    "current_file_count": len(docs),
                }
            )
            active_schema = storage.get_active_schema(corpus_id=corpus_id)
            has_embeddings = storage.has_embeddings(corpus_id=corpus_id)

            schema_name: str | None = None
            has_metadata = False
            schema_fields: list[str] = []
            if active_schema is not None:
                schema_name = active_schema.name
                has_metadata = (
                    active_schema.schema_def.get("metadata_profile") is not None
                )
                fields_def = active_schema.schema_def.get("fields")
                if isinstance(fields_def, list):
                    for f in fields_def:
                        if isinstance(f, dict) and isinstance(f.get("name"), str):
                            schema_fields.append(f["name"])

            storage.close()
            return {
                "indexed": True,
                "corpus_id": corpus_id,
                "document_count": len(docs),
                "schema_name": schema_name,
                "has_metadata": has_metadata,
                "has_embeddings": has_embeddings,
                "schema_fields": schema_fields,
                **freshness,
            }
        except Exception:
            storage.close()
            return {"indexed": False}
    except Exception:
        return {"indexed": False}


@app.get("/api/index/document-chunks", dependencies=[Depends(require_internal_token)])
async def document_chunks(
    corpus_key: str, relative_path_prefix: str, database_url: str | None = None
):
    """Look up a document and its chunks by corpus + relative-path prefix.

    The single read path for chunks of a specific uploaded file — callers
    (e.g. `backend`) should use this instead of querying `core_*` tables
    directly, so reads always agree with how `/api/index` wrote the data.
    """
    try:
        corpus_root = resolve_corpus_root(corpus_key)
        resolved_database_url = resolve_database_url(database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        try:
            result = storage.get_document_chunks_by_prefix(
                corpus_root=corpus_root, relative_path_prefix=relative_path_prefix
            )
        finally:
            storage.close()

        if result is None:
            return {"document": None, "chunks": []}
        return result
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/index/auto-profile", dependencies=[Depends(require_internal_token)])
async def generate_auto_profile(request: AutoProfileRequest):
    """Generate an auto-discovered metadata profile for preview/editing."""
    try:
        folder_path = Path(request.folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse(
                {"error": f"Invalid folder: {request.folder}"}, status_code=400
            )

        profile = await asyncio.to_thread(auto_discover_profile, str(folder_path))
        return {"profile": profile}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/index", dependencies=[Depends(require_internal_token)])
async def build_index(request: IndexRequest):
    """Build or refresh the index for a selected folder."""
    try:
        has_manifest = bool(request.documents)
        folder_path = Path(request.folder).resolve()
        corpus_root = resolve_corpus_root(request.corpus_key or request.folder)
        if not has_manifest:
            if not folder_path.exists():
                return JSONResponse({"error": "Path not found"}, status_code=404)
            if not folder_path.is_dir():
                return JSONResponse({"error": "Not a directory"}, status_code=400)

        lock = _get_corpus_lock(corpus_root)
        async with lock:
            resolved_database_url = resolve_database_url(request.database_url)
            embedding_provider: EmbeddingProvider | None = None
            if request.with_embeddings:
                try:
                    embedding_provider = EmbeddingProvider()
                except ValueError:
                    embedding_provider = None
            pipeline = IndexingPipeline(
                storage=PostgresStorage(resolved_database_url),
                embedding_provider=embedding_provider,
            )
            effective_with_metadata = (
                request.with_metadata or request.metadata_profile is not None
            )
            discover_schema = request.discover_schema or effective_with_metadata
            if has_manifest:
                result = pipeline.index_documents(
                    corpus_root=corpus_root,
                    documents=[
                        SourceDocument(
                            file_path=document.file_path,
                            relative_path=document.relative_path,
                            display_name=document.display_name,
                            logical_path=document.logical_path,
                        )
                        for document in request.documents or []
                    ],
                    discover_schema=discover_schema,
                    schema_name=request.schema_name,
                    with_metadata=effective_with_metadata,
                    metadata_profile=request.metadata_profile,
                    preserve_existing=True,
                )
            else:
                result = pipeline.index_folder(
                    str(folder_path),
                    discover_schema=discover_schema,
                    schema_name=request.schema_name,
                    with_metadata=effective_with_metadata,
                    metadata_profile=request.metadata_profile,
                )

        return {
            "database_url": resolved_database_url,
            "folder": corpus_root,
            "corpus_id": result.corpus_id,
            "indexed_files": result.indexed_files,
            "skipped_files": result.skipped_files,
            "deleted_files": result.deleted_files,
            "chunks_written": result.chunks_written,
            "active_documents": result.active_documents,
            "schema_used": result.schema_used,
            "embeddings_written": result.embeddings_written,
            "metadata_mode": "langextract" if effective_with_metadata else "heuristic",
            "indexed_paths": result.indexed_paths,
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/index/embed", dependencies=[Depends(require_internal_token)])
async def embed_index(request: EmbedIndexRequest):
    """Generate embeddings for an already-chunked corpus.

    Reads chunk text straight from `core_chunks` — does not re-parse or
    re-chunk source files. Intended for the second ("Index") step once
    "Generate Chunks" (`/api/index` with `with_embeddings=false`) has
    already populated the corpus.
    """
    try:
        corpus_root = resolve_corpus_root(request.corpus_key or request.folder)
        lock = _get_corpus_lock(corpus_root)
        async with lock:
            resolved_database_url = resolve_database_url(request.database_url)
            storage = PostgresStorage(resolved_database_url)
            corpus_id = storage.get_corpus_id(corpus_root)
            if corpus_id is None:
                storage.close()
                return JSONResponse(
                    {
                        "error": "No chunks found for this folder. Generate chunks first."
                    },
                    status_code=404,
                )

            try:
                embedding_provider = EmbeddingProvider()
            except ValueError as exc:
                storage.close()
                return JSONResponse({"error": str(exc)}, status_code=400)

            pipeline = IndexingPipeline(
                storage=storage,
                embedding_provider=embedding_provider,
            )
            result = pipeline.embed_corpus(corpus_id)
            storage.close()

        return {
            "database_url": resolved_database_url,
            "folder": corpus_root,
            "corpus_id": result.corpus_id,
            "chunks_embedded": result.chunks_embedded,
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def run_server(host: str = "127.0.0.1", port: int = 8001):
    """Run the indexer FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
