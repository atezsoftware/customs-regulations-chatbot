"""
FastAPI server for FsExplorer web UI.

Provides a WebSocket endpoint for real-time workflow streaming
and serves the single-page HTML interface.
"""

import re
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent import clear_index_context, set_index_context, set_search_flags
from fs_explorer_shared.auth import internal_token_valid, require_internal_token
from fs_explorer_shared.embeddings import EmbeddingProvider
from .exploration_trace import ExplorationTrace, extract_cited_sources
from fs_explorer_shared.index_config import (
    corpus_root as resolve_corpus_root,
    resolve_database_url,
)
from .search import IndexedQueryEngine
from fs_explorer_shared.storage import PostgresStorage
from .workflow import (
    AskHumanEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ToolCallEvent,
    get_agent,
    reset_agent,
    set_agent_llm_config,
    workflow,
)

app = FastAPI(title="FsExplorer", description="AI-powered filesystem exploration")


class TaskRequest(BaseModel):
    """Request model for task submission."""

    task: str
    folder: str = "."
    use_index: bool = False
    database_url: str | None = None
    model: str | None = None
    temperature: float | None = None


class SearchRequest(BaseModel):
    """Request model for search queries."""

    corpus_folder: str
    query: str
    filters: str | None = None
    limit: int = 5
    database_url: str | None = None


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
            "label": "Searching indexed chunks",
            "detail": str(query) if query else "Running indexed retrieval",
        }
    if tool_name in {"parse_file", "preview_file", "get_document", "read"}:
        target = (
            tool_input.get("file_path")
            or tool_input.get("doc_id")
            or tool_input.get("directory")
            or "document"
        )
        return {
            "label": "Reading indexed chunks",
            "detail": _display_target(str(target)),
        }
    if tool_name == "scan_folder":
        return {
            "label": "Scanning indexed documents",
            "detail": _display_target(str(tool_input.get("directory", ""))),
        }
    if tool_name in {"grep", "glob"}:
        return {
            "label": "Searching indexed text",
            "detail": _display_target(str(tool_input.get("pattern", ""))),
        }
    return {"label": "Using indexed retrieval", "detail": ""}


def _display_target(value: str) -> str:
    """Keep user-facing status copy free from local paths and raw storage details."""
    if not value:
        return ""
    cleaned = value.replace("\\", "/")
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    cleaned = re.sub(r"^\d+-", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    if cleaned.startswith("doc_"):
        return "indexed document"
    return cleaned[:120]


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


@app.post("/api/search", dependencies=[Depends(require_internal_token)])
async def search_index(request: SearchRequest):
    """Search an indexed corpus and return ranked hits."""
    try:
        corpus_root = resolve_corpus_root(request.corpus_folder)

        resolved_database_url = resolve_database_url(request.database_url)
        storage = PostgresStorage(
            resolved_database_url, read_only=True, initialize=False
        )
        corpus_id = storage.get_corpus_id(corpus_root)
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
            "corpus_folder": corpus_root,
            "query": request.query,
            "hits": [
                {
                    "doc_id": hit.doc_id,
                    "relative_path": hit.relative_path,
                    "absolute_path": hit.absolute_path,
                    "chunk_id": hit.chunk_id,
                    "position": hit.position,
                    "text": hit.text,
                    "chunk_type": hit.chunk_type,
                    "metadata": hit.metadata,
                    "semantic_score": hit.semantic_score,
                    "metadata_score": hit.metadata_score,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                }
                for hit in hits
            ],
        }
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
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
    index_storage: PostgresStorage | None = None

    try:
        # Receive the task
        data = await websocket.receive_json()

        if not internal_token_valid(data.get("internal_token")):
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": "Invalid or missing internal token."},
                }
            )
            return

        task = data.get("task", "")
        original_task = task
        folder = data.get("folder", ".")
        use_index = bool(data.get("use_index", False))
        raw_index_folders = data.get("index_folders")
        database_url = data.get("database_url")
        enable_semantic = bool(data.get("enable_semantic", False))
        enable_metadata = bool(data.get("enable_metadata", False))
        conversation_context = data.get("conversation_context")
        model = data.get("model")
        temperature = data.get("temperature")

        if not task:
            await websocket.send_json(
                {"type": "error", "data": {"message": "No task provided"}}
            )
            return

        # Validate folder only for raw filesystem mode. Indexed chat can use
        # virtual corpus keys backed entirely by PostgreSQL.
        folder_path = Path(folder).resolve()
        index_folder_values: list[str] = []
        if isinstance(raw_index_folders, list):
            index_folder_values = [
                str(item) for item in raw_index_folders if isinstance(item, str)
            ]
        elif isinstance(raw_index_folders, str):
            index_folder_values = [raw_index_folders]
        if not use_index and (not folder_path.exists() or not folder_path.is_dir()):
            await websocket.send_json(
                {"type": "error", "data": {"message": f"Invalid folder: {folder}"}}
            )
            return

        clear_index_context()
        if use_index:
            resolved_database_url = resolve_database_url(
                database_url if isinstance(database_url, str) else None
            )
            storage = PostgresStorage(resolved_database_url)
            index_folders = index_folder_values or [str(folder_path)]

            available_index_folders: list[str] = []
            for index_folder in index_folders:
                corpus_root = resolve_corpus_root(index_folder)
                if storage.get_corpus_id(corpus_root) is not None:
                    available_index_folders.append(corpus_root)

            if not available_index_folders:
                storage.close()
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
            set_index_context(available_index_folders, resolved_database_url)

        set_search_flags(
            enable_semantic=enable_semantic and use_index,
            enable_metadata=enable_metadata and use_index,
        )

        task = _task_with_context(str(task), conversation_context)
        trace = ExplorationTrace(root_directory=str(folder_path))

        # Reset agent for fresh state
        reset_agent()
        set_agent_llm_config(
            model=model if isinstance(model, str) else None,
            temperature=float(temperature)
            if isinstance(temperature, (int, float))
            else None,
        )

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
            cited_source_links = (
                {}
                if use_index
                else _source_links(
                    cited_sources=cited_sources,
                    referenced_documents=referenced_documents,
                    root_directory=str(folder_path),
                )
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
                        "thinking_tokens": usage.thinking_tokens,
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
