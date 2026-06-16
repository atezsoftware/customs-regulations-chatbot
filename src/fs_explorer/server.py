"""
FastAPI server for FsExplorer web UI.

Provides a WebSocket endpoint for real-time workflow streaming
and serves the single-page HTML interface.
"""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent import clear_index_context, set_index_context, set_search_flags
from .embeddings import EmbeddingProvider
from .exploration_trace import ExplorationTrace, extract_cited_sources
from .index_config import resolve_db_path
from .indexing import IndexingPipeline
from .fs import SUPPORTED_EXTENSIONS
from .indexing.metadata import auto_discover_profile
from .search import IndexedQueryEngine
from .storage import DuckDBStorage
from .workflow import (
    AskHumanEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ToolCallEvent,
    get_agent,
    reset_agent,
    workflow,
)

app = FastAPI(title="FsExplorer", description="AI-powered filesystem exploration")

_corpus_locks: dict[str, asyncio.Lock] = {}


def _get_corpus_lock(folder: str) -> asyncio.Lock:
    """Return a per-folder asyncio lock, creating one if needed."""
    normalized = str(Path(folder).resolve())
    if normalized not in _corpus_locks:
        _corpus_locks[normalized] = asyncio.Lock()
    return _corpus_locks[normalized]


class TaskRequest(BaseModel):
    """Request model for task submission."""

    task: str
    folder: str = "."
    use_index: bool = False
    db_path: str | None = None


class IndexRequest(BaseModel):
    """Request model for index build/refresh."""

    folder: str = "."
    db_path: str | None = None
    discover_schema: bool = False
    schema_name: str | None = None
    with_metadata: bool = False
    metadata_profile: dict[str, Any] | None = None
    with_embeddings: bool = False


class AutoProfileRequest(BaseModel):
    """Request model for auto-profile generation."""

    folder: str = "."


class SearchRequest(BaseModel):
    """Request model for search queries."""

    corpus_folder: str
    query: str
    filters: str | None = None
    limit: int = 5
    db_path: str | None = None


def _format_conversation_context(raw_context: Any) -> str:
    """Format short-lived frontend memory into prompt context."""
    if not isinstance(raw_context, list):
        return ""

    lines: list[str] = []
    for item in raw_context[-10:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        cleaned = " ".join(content.split())
        if not cleaned:
            continue
        if len(cleaned) > 1200:
            cleaned = f"{cleaned[:1200]}..."
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {cleaned}")

    if not lines:
        return ""

    return (
        "Previous conversation context from this browser session:\n"
        + "\n".join(lines)
        + "\n\nUse this context when it is relevant, but answer the current "
        "question using the selected folder and available tools."
    )


def _task_with_context(task: str, raw_context: Any) -> str:
    context = _format_conversation_context(raw_context)
    if not context:
        return task
    return f"{context}\n\nCurrent question:\n{task}"


def _status_for_tool(tool_name: str, tool_input: dict[str, Any]) -> dict[str, str]:
    """Return compact UI status copy for a tool call."""
    if tool_name == "semantic_search":
        query = tool_input.get("query")
        return {
            "label": "Searching documents",
            "detail": str(query) if query else "Running indexed retrieval",
        }
    if tool_name in {"parse_file", "preview_file", "get_document", "read"}:
        target = (
            tool_input.get("file_path")
            or tool_input.get("doc_id")
            or tool_input.get("directory")
            or "document"
        )
        return {"label": "Reading source", "detail": str(target)}
    if tool_name == "scan_folder":
        return {
            "label": "Scanning folder",
            "detail": str(tool_input.get("directory", "")),
        }
    if tool_name in {"grep", "glob"}:
        return {"label": "Searching files", "detail": str(tool_input)}
    return {"label": "Using tool", "detail": tool_name}


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
        if int(stat.st_size) != int(doc["file_size"]) or abs(
            float(stat.st_mtime) - float(doc["file_mtime"])
        ) > 0.001:
            changed.append(relative_path)

    return {
        "fresh": not added and not changed and not removed,
        "added_files": added,
        "changed_files": changed,
        "removed_files": removed,
        "current_file_count": len(current),
    }


def _source_links(
    *,
    cited_sources: list[str],
    referenced_documents: list[str],
    root_directory: str,
) -> dict[str, str]:
    candidates: list[Path] = [Path(path).resolve() for path in referenced_documents]
    root = Path(root_directory).resolve()

    # If the final answer cites a document that was not in a direct tool call,
    # fall back to a quick filename search under the selected folder.
    known_names = {path.name for path in candidates}
    for source in cited_sources:
        source_name = Path(source).name
        if source_name in known_names:
            continue
        for path in root.rglob(source_name):
            if path.is_file():
                candidates.append(path.resolve())
                known_names.add(path.name)
                break

    links: dict[str, str] = {}
    for source in cited_sources:
        source_name = Path(source).name
        for path in candidates:
            if path.name == source or path.name == source_name or source in str(path):
                links[source] = f"/api/document?path={str(path)}"
                break
    return links


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI HTML file."""
    html_path = Path(__file__).parent / "ui.html"
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"), status_code=200
        )
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)


@app.get("/api/document")
async def open_document(path: str):
    """Serve a local source document for citation links."""
    document_path = Path(path).expanduser().resolve()
    if not document_path.exists() or not document_path.is_file():
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return FileResponse(str(document_path), filename=document_path.name)


@app.get("/api/folders")
async def list_folders(path: str = "."):
    """
    List folders in the given path.
    Returns list of folder names and current path info.
    """
    try:
        base_path = Path(path).resolve()
        if not base_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not base_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        # Get folders (non-hidden)
        folders = sorted(
            [
                f.name
                for f in base_path.iterdir()
                if f.is_dir() and not f.name.startswith(".")
            ]
        )

        # Get parent path (if not at root)
        parent = str(base_path.parent) if base_path != base_path.parent else None

        return {
            "current": str(base_path),
            "parent": parent,
            "folders": folders,
            "files_count": len([f for f in base_path.iterdir() if f.is_file()]),
        }
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/index/status")
async def index_status(folder: str, db_path: str | None = None):
    """Check whether a folder has been indexed and return status details."""
    try:
        folder_path = Path(folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return {"indexed": False}

        resolved_db_path = resolve_db_path(db_path)
        if not Path(resolved_db_path).exists():
            return {"indexed": False}

        try:
            storage = DuckDBStorage(resolved_db_path, read_only=True, initialize=False)
        except Exception:
            return {"indexed": False}

        try:
            corpus_id = storage.get_corpus_id(str(folder_path))
            if corpus_id is None:
                storage.close()
                return {"indexed": False}

            docs = storage.list_documents(corpus_id=corpus_id, include_deleted=False)
            freshness = _index_freshness(folder_path, docs)
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


@app.post("/api/index/auto-profile")
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


@app.post("/api/index")
async def build_index(request: IndexRequest):
    """Build or refresh the index for a selected folder."""
    try:
        folder_path = Path(request.folder).resolve()
        if not folder_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not folder_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)

        lock = _get_corpus_lock(str(folder_path))
        async with lock:
            resolved_db_path = resolve_db_path(request.db_path)
            embedding_provider: EmbeddingProvider | None = None
            if request.with_embeddings:
                try:
                    embedding_provider = EmbeddingProvider()
                except ValueError:
                    embedding_provider = None
            pipeline = IndexingPipeline(
                storage=DuckDBStorage(resolved_db_path),
                embedding_provider=embedding_provider,
            )
            effective_with_metadata = (
                request.with_metadata or request.metadata_profile is not None
            )
            discover_schema = request.discover_schema or effective_with_metadata
            result = pipeline.index_folder(
                str(folder_path),
                discover_schema=discover_schema,
                schema_name=request.schema_name,
                with_metadata=effective_with_metadata,
                metadata_profile=request.metadata_profile,
            )

        return {
            "db_path": resolved_db_path,
            "folder": str(folder_path),
            "corpus_id": result.corpus_id,
            "indexed_files": result.indexed_files,
            "skipped_files": result.skipped_files,
            "deleted_files": result.deleted_files,
            "chunks_written": result.chunks_written,
            "active_documents": result.active_documents,
            "schema_used": result.schema_used,
            "embeddings_written": result.embeddings_written,
            "metadata_mode": "langextract" if effective_with_metadata else "heuristic",
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/search")
async def search_index(request: SearchRequest):
    """Search an indexed corpus and return ranked hits."""
    try:
        folder_path = Path(request.corpus_folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse(
                {"error": f"Invalid folder: {request.corpus_folder}"}, status_code=400
            )

        resolved_db_path = resolve_db_path(request.db_path)
        storage = DuckDBStorage(resolved_db_path, read_only=True, initialize=False)
        corpus_id = storage.get_corpus_id(str(folder_path))
        if corpus_id is None:
            storage.close()
            return JSONResponse(
                {"error": "No index found for this folder."}, status_code=404
            )

        embedding_provider: EmbeddingProvider | None = None
        if storage.has_embeddings(corpus_id=corpus_id):
            try:
                embedding_provider = EmbeddingProvider()
            except ValueError:
                pass

        engine = IndexedQueryEngine(storage, embedding_provider=embedding_provider)
        hits = engine.search(
            corpus_id=corpus_id,
            query=request.query,
            filters=request.filters,
            limit=request.limit,
        )
        storage.close()

        return {
            "corpus_folder": str(folder_path),
            "query": request.query,
            "hits": [
                {
                    "doc_id": hit.doc_id,
                    "relative_path": hit.relative_path,
                    "absolute_path": hit.absolute_path,
                    "position": hit.position,
                    "text": hit.text,
                    "semantic_score": hit.semantic_score,
                    "metadata_score": hit.metadata_score,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                }
                for hit in hits
            ],
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    """
    WebSocket endpoint for real-time exploration streaming.

    Protocol:
    1. Client sends: {"task": "user question"}
    2. Server streams events: {"type": "...", "data": {...}}
    3. Final event: {"type": "complete", "data": {...}}
    """
    await websocket.accept()
    index_storage: DuckDBStorage | None = None

    try:
        # Receive the task
        data = await websocket.receive_json()
        task = data.get("task", "")
        original_task = task
        folder = data.get("folder", ".")
        use_index = bool(data.get("use_index", False))
        db_path = data.get("db_path")
        enable_semantic = bool(data.get("enable_semantic", False))
        enable_metadata = bool(data.get("enable_metadata", False))
        conversation_context = data.get("conversation_context")

        if not task:
            await websocket.send_json(
                {"type": "error", "data": {"message": "No task provided"}}
            )
            return

        # Validate folder
        folder_path = Path(folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            await websocket.send_json(
                {"type": "error", "data": {"message": f"Invalid folder: {folder}"}}
            )
            return

        clear_index_context()
        if use_index:
            resolved_db_path = resolve_db_path(
                db_path if isinstance(db_path, str) else None
            )
            storage = DuckDBStorage(resolved_db_path)
            corpus_id = storage.get_corpus_id(str(folder_path))
            if corpus_id is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": (
                                "No index found for the selected folder. "
                                "Run `explore index <folder>` first."
                            )
                        },
                    }
                )
                return
            index_storage = storage
            set_index_context(str(folder_path), resolved_db_path)

        set_search_flags(
            enable_semantic=enable_semantic and use_index,
            enable_metadata=enable_metadata and use_index,
        )

        task = _task_with_context(str(task), conversation_context)
        trace = ExplorationTrace(root_directory=str(folder_path))

        # Reset agent for fresh state
        reset_agent()

        # Send start event
        await websocket.send_json(
            {
                "type": "start",
                "data": {
                    "task": original_task,
                    "folder": str(folder_path),
                    "use_index": use_index,
                },
            }
        )
        await websocket.send_json(
            {
                "type": "status",
                "data": {
                    "label": "Thinking",
                    "detail": "Planning the first step",
                },
            }
        )

        # Run the workflow
        step_number = 0
        handler = workflow.run(
            start_event=InputEvent(
                task=task,
                folder=str(folder_path),
                use_index=use_index,
                enable_semantic=enable_semantic and use_index,
                enable_metadata=enable_metadata and use_index,
            )
        )

        async for event in handler.stream_events():
            if isinstance(event, ToolCallEvent):
                step_number += 1
                resolved_document_path: str | None = None
                if event.tool_name == "get_document":
                    doc_id = event.tool_input.get("doc_id")
                    if index_storage is not None and isinstance(doc_id, str) and doc_id:
                        document = index_storage.get_document(doc_id=doc_id)
                        if document and not document["is_deleted"]:
                            resolved_document_path = str(document["absolute_path"])
                trace.record_tool_call(
                    step_number=step_number,
                    tool_name=event.tool_name,
                    tool_input=event.tool_input,
                    resolved_document_path=resolved_document_path,
                )
                tool_status = _status_for_tool(event.tool_name, event.tool_input)
                await websocket.send_json({"type": "status", "data": tool_status})
                await websocket.send_json(
                    {
                        "type": "tool_call",
                        "data": {
                            "step": step_number,
                            "tool_name": event.tool_name,
                            "tool_input": event.tool_input,
                            "reason": event.reason,
                            "status_label": tool_status["label"],
                            "status_detail": tool_status["detail"],
                        },
                    }
                )

            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                trace.record_go_deeper(
                    step_number=step_number, directory=event.directory
                )
                await websocket.send_json(
                    {
                        "type": "go_deeper",
                        "data": {
                            "step": step_number,
                            "directory": event.directory,
                            "reason": event.reason,
                        },
                    }
                )
                await websocket.send_json(
                    {
                        "type": "status",
                        "data": {
                            "label": "Thinking",
                            "detail": "Inspecting the selected folder",
                        },
                    }
                )

            elif isinstance(event, AskHumanEvent):
                step_number += 1
                await websocket.send_json(
                    {
                        "type": "ask_human",
                        "data": {
                            "step": step_number,
                            "question": event.question,
                            "reason": event.reason,
                        },
                    }
                )

                # Wait for human response
                response_data = await websocket.receive_json()
                if response_data.get("type") == "human_response":
                    handler.ctx.send_event(
                        HumanAnswerEvent(response=response_data.get("response", ""))
                    )

        # Get final result
        result = await handler

        final_result = result.final_result or ""
        if not result.error:
            await websocket.send_json(
                {
                    "type": "status",
                    "data": {
                        "label": "Writing answer",
                        "detail": "Composing the final response",
                    },
                }
            )
            await websocket.send_json({"type": "answer_start", "data": {}})
            streamed_parts: list[str] = []
            agent = get_agent()
            async for chunk in agent.stream_final_answer(fallback_answer=final_result):
                streamed_parts.append(chunk)
                await websocket.send_json(
                    {"type": "answer_delta", "data": {"text": chunk}}
                )
            streamed_final = "".join(streamed_parts).strip()
            if streamed_final:
                final_result = streamed_final
            cited_sources = extract_cited_sources(final_result)
            referenced_documents = trace.sorted_documents()
            cited_source_links = _source_links(
                cited_sources=cited_sources,
                referenced_documents=referenced_documents,
                root_directory=str(folder_path),
            )
            await websocket.send_json(
                {
                    "type": "answer_done",
                    "data": {
                        "final_result": final_result,
                        "cited_sources": cited_sources,
                        "cited_source_links": cited_source_links,
                    },
                }
            )
        else:
            cited_sources = []
            referenced_documents = trace.sorted_documents()
            cited_source_links = {}

        # Get token usage
        agent = get_agent()
        usage = agent.token_usage
        input_cost, output_cost, total_cost = usage._calculate_cost()

        await websocket.send_json(
            {
                "type": "complete",
                "data": {
                    "final_result": final_result,
                    "error": result.error,
                    "stats": {
                        "steps": step_number,
                        "api_calls": usage.api_calls,
                        "documents_scanned": usage.documents_scanned,
                        "documents_parsed": usage.documents_parsed,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "tool_result_chars": usage.tool_result_chars,
                        "estimated_cost": round(total_cost, 6),
                    },
                    "trace": {
                        "step_path": trace.step_path,
                        "referenced_documents": referenced_documents,
                        "cited_sources": cited_sources,
                        "cited_source_links": cited_source_links,
                    },
                },
            }
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"message": str(e)}})
    finally:
        if index_storage is not None:
            index_storage.close()
        set_search_flags(enable_semantic=False, enable_metadata=False)
        clear_index_context()


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
